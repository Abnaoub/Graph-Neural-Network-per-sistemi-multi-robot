"""Strati di sicurezza indipendenti dalla policy (usati identici nel 2D e in Webots).

Ordine di applicazione in core/controller.py:
  1. obstacle_repulsion  - ostacoli statici (scaffali, muri, pilastri)
  2. ZoneManager         - prenotazione delle zone critiche (incroci): chi deve aspettare si ferma
  3. emergency_filter    - ultima istanza tra robot: sotto 0,70 m il meno prioritario si ferma,
                           sotto 0,62 m entrambi si allontanano
  4. StallEscape         - retromarcia: solo rete di sicurezza finale, con contatore nel log
"""

from __future__ import annotations

import numpy as np

from core.environment import ROBOT_RADIUS, clip_action
from core.warehouse import _inside

STOP_DISTANCE = 0.70        # centro-centro: sotto, il robot meno prioritario si ferma
SEPARATE_DISTANCE = 0.62    # centro-centro: sotto, entrambi si allontanano (contatto a 0,56)
# sotto (STOP + SEPARATE)/2 = 0,66 m il robot meno prioritario si ferma del tutto
OBSTACLE_MARGIN = 0.45      # dal bordo dell'ostacolo: si annulla la componente verso l'ostacolo
OBSTACLE_PUSH = ROBOT_RADIUS + 0.08


def obstacle_repulsion(position: np.ndarray, action: np.ndarray, obstacles: np.ndarray,
                       margin: float = OBSTACLE_MARGIN, gain: float = 1.0) -> np.ndarray:
    """Evitamento degli ostacoli statici (analogo del campo di protezione di un lidar di sicurezza).

    La GNN osserva solo l'ostacolo piu' vicino: per ogni ostacolo entro `margin` dal bordo
    si toglie all'azione la componente diretta verso l'ostacolo (il robot scivola lungo la
    parete invece di spingerci contro) e sotto OBSTACLE_PUSH si aggiunge una spinta verso
    l'esterno.
    """
    out = np.asarray(action, dtype=np.float32).copy()
    for i, p in enumerate(position):
        for ox, oy, ow, oh in obstacles:
            closest = np.array([np.clip(p[0], ox - ow / 2, ox + ow / 2), np.clip(p[1], oy - oh / 2, oy + oh / 2)])
            delta = p - closest
            d = float(np.linalg.norm(delta))
            if d >= margin:
                continue
            unit = delta / max(d, 1e-6)
            into = float(np.dot(out[i], -unit))
            if into > 0.0:
                out[i] += into * unit * min(1.0, (margin - d) / (margin - ROBOT_RADIUS))
            if d < OBSTACLE_PUSH:
                out[i] += gain * (OBSTACLE_PUSH - d) / 0.08 * unit
    return clip_action(out)


def emergency_filter(position: np.ndarray, action: np.ndarray, rank: np.ndarray,
                     parked: np.ndarray | None = None,
                     stop_distance: float = STOP_DISTANCE,
                     separate_distance: float = SEPARATE_DISTANCE) -> np.ndarray:
    """Filtro di ultima istanza tra robot (asimmetrico: niente stalli faccia a faccia).

    rank: valore piu' basso = priorita' piu' alta. Per ogni coppia sotto `stop_distance`
    che si sta avvicinando (velocita' relativa):
      - ciascuno perde la parte della propria velocita' diretta verso l'altro che eccede la
        velocita' dell'altro (un robot in coda adegua la velocita' a chi lo precede, due robot
        frontali si fermano: si passa di lato, non attraverso);
      - sotto STOP_FULL il robot con rank maggiore si ferma del tutto;
      - sotto `separate_distance` entrambi ricevono una spinta di allontanamento.
    Un robot `parked` (gia' arrivato) e' trattato come un ostacolo fermo: chi gli passa
    accanto scivola intorno a lui e non si ferma (altrimenti nessuno raggiungerebbe una
    piazzola vicina a un robot parcheggiato).
    """
    out = np.asarray(action, dtype=np.float32).copy()
    n = len(position)
    parked = np.zeros(n, dtype=bool) if parked is None else parked
    stop = np.zeros(n, dtype=bool)
    push = np.zeros_like(out)
    stop_full = (stop_distance + separate_distance) / 2
    for i in range(n):
        for j in range(i + 1, n):
            delta = position[i] - position[j]
            d = float(np.linalg.norm(delta))
            if d >= stop_distance:
                continue
            unit = delta / max(d, 1e-6)                     # da j verso i
            if d < separate_distance:
                strength = 0.6 * (separate_distance - d) / (separate_distance - 2 * ROBOT_RADIUS + 1e-6) + 0.2
                if not parked[i]:
                    push[i] += strength * unit
                if not parked[j]:
                    push[j] -= strength * unit
            for a, b, toward in ((i, j, -unit), (j, i, unit)):     # toward: da a verso b
                if parked[a]:
                    continue
                own = float(np.dot(out[a], toward))
                relative = float(np.dot(out[a] - out[b], toward))
                if own <= 0.0 or relative <= 0.0:
                    continue
                out[a] -= min(own, relative) * toward
                if not parked[b] and rank[a] > rank[b] and d < stop_full:
                    stop[a] = True
    out[stop] = 0.0
    out += push
    return clip_action(out)


class ZoneManager:
    """Prenotazione delle zone critiche (incroci tra corridoi verticali e passaggi).

    Regole:
    - un robot "occupa" una zona quando il suo centro e' nel rettangolo allargato del raggio;
    - un robot "chiede" la zona in cui entra il punto del suo percorso 0,8 m piu' avanti;
    - ogni attraversamento e' classificato H (orizzontale, lungo il passaggio) o V (verticale,
      cambio di passaggio) dal tratto di percorso dentro la zona. Attraversamenti della stessa
      classe sono compatibili (convoglio nello stesso senso: ci pensa la GNN a tenere le
      distanze); classi diverse si incrociano e NON possono coesistere;
    - la zona viene concessa a un richiedente alla volta: prima chi e' PIU' VICINO alla zona
      (e' davanti nella coda: dare la precedenza a chi aspetta da piu' tempo ma sta dietro un
      altro robot creava uno stallo tra prenotazione e coda), poi l'indice. La concessione
      viene ricalcolata a ogni passo. Un robot a cui la zona e' negata si ferma all'ingresso
      (wait_time nel log);
    - antistallo: dopo FORCE_AFTER secondi di attesa il robot entra comunque (contato nel log);
      il filtro d'emergenza impedisce il contatto.
    """

    FORCE_AFTER = 12.0
    APPROACH = 8            # punti di percorso (0,1 m l'uno) guardati avanti: 0,8 m

    def __init__(self, zones: list, tracker, n: int):
        self.zones = [(cx, cy, w, h) for _, cx, cy, w, h in zones]
        self.names = [name for name, *_ in zones]
        self.n = n
        self.tracker = tracker
        self.waiting_since = np.full(n, -1.0)
        self.reserved: dict[int, int] = {}          # zona -> robot a cui e' stata concessa
        self.forced = 0
        self.blocked = np.zeros(n, dtype=bool)
        # classe di attraversamento per (robot, zona), dal percorso A*
        self.classes: dict[tuple[int, int], str] = {}
        for i, path in enumerate(tracker.paths):
            for z, rect in enumerate(self.zones):
                inside = [p for p in path if _inside(p, rect, ROBOT_RADIUS)]
                if len(inside) >= 2:
                    dx, dy = inside[-1][0] - inside[0][0], inside[-1][1] - inside[0][1]
                    self.classes[(i, z)] = "H" if abs(dx) >= abs(dy) else "V"
                elif inside:
                    self.classes[(i, z)] = "H"

    def distance_to_zone(self, point, z: int) -> float:
        cx, cy, w, h = self.zones[z]
        dx = max(abs(point[0] - cx) - w / 2, 0.0)
        dy = max(abs(point[1] - cy) - h / 2, 0.0)
        return float(np.hypot(dx, dy))

    def zone_of(self, point) -> int | None:
        for z, rect in enumerate(self.zones):
            if _inside(point, rect, ROBOT_RADIUS):
                return z
        return None

    def update(self, position: np.ndarray, arrived: np.ndarray, now: float) -> np.ndarray:
        """Restituisce la maschera dei robot che devono fermarsi (zona negata)."""
        n = self.n
        inside = [None if arrived[i] else self.zone_of(position[i]) for i in range(n)]
        occupants: dict[int, list[int]] = {}
        for i, z in enumerate(inside):
            if z is not None:
                occupants.setdefault(z, []).append(i)
        self.reserved = {}            # ricalcolata a ogni passo
        requests: dict[int, list[int]] = {}
        wanted = [None] * n
        for i in range(n):
            if arrived[i]:
                continue
            path = self.tracker.paths[i]
            ahead = path[min(len(path) - 1, self.tracker.progress[i] + self.APPROACH)]
            z = self.zone_of(ahead)
            if z is not None and z != inside[i]:
                wanted[i] = z
                requests.setdefault(z, []).append(i)
        blocked = np.zeros(n, dtype=bool)
        for z, robots in requests.items():
            robots.sort(key=lambda i: (self.distance_to_zone(position[i], z), i))
            for i in robots:
                my_class = self.classes.get((i, z), "H")
                others = [j for j in occupants.get(z, []) if j != i]
                if z in self.reserved and self.reserved[z] != i:
                    others.append(self.reserved[z])
                compatible = all(self.classes.get((j, z), "H") == my_class for j in others)
                if compatible and z not in self.reserved:
                    self.reserved[z] = i
                if not compatible:
                    if self.waiting_since[i] < 0:
                        self.waiting_since[i] = now
                    elif now - self.waiting_since[i] > self.FORCE_AFTER:
                        self.forced += 1
                        self.waiting_since[i] = now
                        continue                     # entra comunque (antistallo)
                    blocked[i] = True
        for i in range(n):
            if not blocked[i]:
                self.waiting_since[i] = -1.0
        self.blocked = blocked
        return blocked

    def rank(self, position: np.ndarray, now: float) -> np.ndarray:
        """Priorita' per il filtro d'emergenza: dentro una zona > aspetta da piu' tempo > indice."""
        r = np.arange(self.n, dtype=float)
        for i in range(self.n):
            if self.zone_of(position[i]) is not None:
                r[i] -= 1000.0
            elif self.waiting_since[i] >= 0:
                r[i] -= 10.0 * (now - self.waiting_since[i])
        return r


class StallEscape:
    """Retromarcia automatica, SOLO come rete di sicurezza finale (contata nel log).

    Un robot che non avanza lungo il percorso per `patience` secondi (e non sta
    aspettando una zona) riceve per `duration` secondi come goal locale un punto
    `back` passi indietro sul suo percorso, poi riprova.
    """

    def __init__(self, n: int, patience: float = 6.0, duration: float = 2.0, back: int = 8):
        self.patience, self.duration, self.back = patience, duration, back
        self.anchor_progress = np.zeros(n, dtype=int)
        self.since = np.zeros(n)
        self.until = np.zeros(n)
        self.triggered = 0
        self.started = False

    def local_goals(self, tracker, position: np.ndarray, goals_now: np.ndarray, arrived: np.ndarray,
                    waiting: np.ndarray, now: float) -> np.ndarray:
        n = len(position)
        if not self.started:
            self.anchor_progress[:] = tracker.progress
            self.since[:] = now
            self.started = True
        out = goals_now.copy()
        for i in range(n):
            if arrived[i] or waiting[i] or tracker.remaining(i) < 1.0:
                self.anchor_progress[i], self.since[i] = tracker.progress[i], now
                continue
            if self.until[i] > now:
                out[i] = tracker.paths[i][max(0, tracker.progress[i] - self.back)]
                continue
            if tracker.progress[i] - self.anchor_progress[i] >= 2:
                self.anchor_progress[i], self.since[i] = tracker.progress[i], now
            elif now - self.since[i] > self.patience:
                self.anchor_progress[i], self.since[i] = tracker.progress[i], now
                self.until[i] = now + self.duration
                self.triggered += 1
        return out

    def reversing(self, now: float) -> np.ndarray:
        return self.until > now
