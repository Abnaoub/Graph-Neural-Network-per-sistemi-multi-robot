"""Controllore di flotta: la stessa logica di controllo per il simulatore 2D e per Webots.

Ad ogni passo di controllo (DT = 0,08 s) FleetController.step riceve posizioni e
velocita' di tutti i robot e restituisce le velocita' (vx, vy) da eseguire:

  1. goal locali lungo i percorsi A* (in linea di vista)        core/planning
  2. (opzionale) retromarcia dei robot in stallo                 core/safety.StallEscape
  3. policy: GNN / MLP (numpy) oppure esperto                    core/environment
  4. attracco alla piazzola nell'ultimo metro (regolatore P)     qui sotto, goal_docking
  5. evitamento ostacoli statici                                 core/safety.obstacle_repulsion
  6. prenotazione delle zone critiche -> chi aspetta si ferma     core/safety.ZoneManager
  7. filtro d'emergenza tra robot                                core/safety.emergency_filter
  8. robot arrivati (entro 0,30 m) e non ancora partiti: fermi

Il supervisore Webots e il rollout 2D non contengono logica di controllo: chiamano step().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core import warehouse
from core.environment import (COMM_RADIUS, MAX_SPEED, build_edge_features, clip_action, observation,
                              safety_filter, warehouse_expert_action)
from core.metrics import GOAL_TOLERANCE
from core.safety import StallEscape, ZoneManager, emergency_filter, obstacle_repulsion


@dataclass
class ControlOptions:
    policy: str = "gnn"            # gnn | mlp | esperto
    con_corsie: bool = True        # corsie a senso unico rigide
    zone: bool = True              # prenotazione delle zone critiche
    repulsione: bool = True        # evitamento ostacoli statici
    filtro: str = "emergenza"      # emergenza | simmetrico | nessuno
    retromarcia: bool = True       # rete di sicurezza finale
    scaglionato: float = 0.0       # secondi tra una partenza e la successiva (0 = tutti insieme)


DOCK_RANGE = 1.0        # m: nell'ultimo metro verso la piazzola
DOCK_MIN, DOCK_MAX, DOCK_GAIN = 0.25, 0.8, 1.2
PARK_TOLERANCE = 0.06   # m: un robot "arrivato" (entro GOAL_TOLERANCE = 0,30 m, la metrica) continua
                        # l'attracco a bassa velocita' fino a 6 cm dal centro, poi si parcheggia
PARK_SPEED = 0.30       # m/s massimi durante l'attracco finale


def goal_docking(position: np.ndarray, action: np.ndarray, goals: np.ndarray, arrived: np.ndarray) -> np.ndarray:
    """Attracco alla piazzola: regolatore proporzionale nell'ultimo metro.

    La policy appresa decelera troppo vicino al goal (comportamento ereditato
    dall'esperto, che frena sotto 0,6 m) e puo' fermarsi a 0,4-0,5 m senza mai
    entrare nei 0,30 m richiesti. Nell'ultimo metro si garantisce una velocita'
    minima verso la piazzola; i filtri a valle (ostacoli, zone, emergenza) restano
    attivi, quindi l'attracco non puo' causare contatti.
    """
    out = np.asarray(action, dtype=np.float32).copy()
    for i in range(len(position)):
        if arrived[i]:
            continue
        delta = goals[i] - position[i]
        dist = float(np.linalg.norm(delta))
        if dist >= DOCK_RANGE or dist < 1e-6:
            continue
        unit = delta / dist
        wanted = min(DOCK_MAX, max(DOCK_MIN, DOCK_GAIN * dist))
        if float(np.dot(out[i], unit)) < wanted:
            out[i] = wanted * unit
    return out


class FleetController:
    def __init__(self, scenario, options: ControlOptions, policy_fn=None):
        self.scenario = scenario
        self.opt = options
        self.policy_fn = policy_fn
        n = len(scenario.starts)
        self.n = n
        self.tracker = warehouse.WaypointTracker(scenario.starts, scenario.goals, scenario.obstacles,
                                                 one_way=options.con_corsie)
        self.zones = ZoneManager(warehouse.critical_zones(), self.tracker, n) if options.zone else None
        self.escape = StallEscape(n) if options.retromarcia else None
        self.arrived = np.zeros(n, dtype=bool)     # entro 0,30 m: conta per le metriche e per gli altri robot
        self.parked = np.zeros(n, dtype=bool)      # entro 6 cm: fermo definitivamente
        self.blocked = np.zeros(n, dtype=bool)
        self.released = np.zeros(n, dtype=bool)

    # ---- stato esportato per log e metriche ---------------------------------------
    @property
    def reverse_events(self) -> int:
        return self.escape.triggered if self.escape is not None else 0

    @property
    def forced_entries(self) -> int:
        return self.zones.forced if self.zones is not None else 0

    def initial_headings(self) -> np.ndarray:
        """Orientamento iniziale (solo estetico in Webots): verso il primo goal locale."""
        first = self.tracker.local_goals(self.scenario.starts)
        d = first - self.scenario.starts
        return np.arctan2(d[:, 1], d[:, 0])

    # ---- un passo di controllo ----------------------------------------------------
    def step(self, position: np.ndarray, velocity: np.ndarray, now: float) -> np.ndarray:
        n = self.n
        sc = self.scenario
        dist_goal = np.linalg.norm(position - sc.goals, axis=1)
        self.arrived |= dist_goal < GOAL_TOLERANCE
        self.parked |= self.arrived & (dist_goal < PARK_TOLERANCE)
        self.released = now >= np.arange(n) * self.opt.scaglionato
        goals_now = self.tracker.local_goals(position)
        waiting = self.blocked          # del passo precedente (per non contare l'attesa come stallo)
        if self.escape is not None:
            goals_now = self.escape.local_goals(self.tracker, position, goals_now, self.arrived, waiting, now)
        if self.opt.policy == "esperto" or self.policy_fn is None:
            action = warehouse_expert_action(position, goals_now, sc.obstacles, sc.goals)
        else:
            features, adjacency = observation(position, velocity, goals_now, sc.obstacles, COMM_RADIUS)
            edges = build_edge_features(position, velocity, COMM_RADIUS)
            action = self.policy_fn(features, adjacency, edges) * MAX_SPEED
        action = clip_action(np.asarray(action, dtype=np.float32))
        action = goal_docking(position, action, sc.goals, self.arrived)
        if self.opt.repulsione:
            action = obstacle_repulsion(position, action, sc.obstacles)
        if self.zones is not None:
            self.blocked = self.zones.update(position, self.arrived, now)
            rank = self.zones.rank(position, now)
        else:
            self.blocked = np.zeros(n, dtype=bool)
            rank = np.arange(n, dtype=float)
        rank[self.arrived] += 1e6          # i parcheggiati non hanno mai la precedenza
        if self.opt.filtro == "emergenza":
            action = emergency_filter(position, action, rank, parked=self.arrived)
        elif self.opt.filtro == "simmetrico":
            action = safety_filter(position, action)
        action[self.blocked] = 0.0
        action[self.arrived | ~self.released] = 0.0
        # attracco finale: dal raggio di arrivo (0,30 m) al centro della piazzola, piano
        docking = self.arrived & ~self.parked
        if docking.any():
            delta = sc.goals[docking] - position[docking]
            d = np.linalg.norm(delta, axis=1, keepdims=True)
            action[docking] = delta / np.maximum(d, 1e-6) * np.minimum(PARK_SPEED, 2.0 * d)
        return action
