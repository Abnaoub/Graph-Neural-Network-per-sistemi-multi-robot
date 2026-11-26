"""Rollout 2D del magazzino con FleetController e le metriche condivise (core/metrics.py).

Stessa integrazione di sempre (environment.advance: passo DT, scaffali e muri invalicabili),
stesso controllore e stesse metriche del supervisore Webots: le righe prodotte hanno le
stesse colonne di results/webots_log.csv.
"""

from __future__ import annotations

import numpy as np

from core import warehouse
from core.controller import ControlOptions, FleetController
from core.environment import DT, advance, make_scenario
from core.metrics import EpisodeMetrics


def rollout_warehouse(n: int, seed: int, options: ControlOptions, policy_fn=None,
                      max_time: float = 90.0, trajectory: bool = False) -> dict:
    warehouse.ONE_WAY = options.con_corsie
    scenario = make_scenario(n, seed, "warehouse")
    ctrl = FleetController(scenario, options, policy_fn)
    metrics = EpisodeMetrics(n, scenario.goals, scenario.obstacles)
    position = scenario.starts.copy()
    velocity = np.zeros_like(position)
    steps = int(round(max_time / DT))
    traj = [position.copy()]
    for k in range(steps):
        now = k * DT
        action = ctrl.step(position, velocity, now)
        velocity = action
        position, _ = advance(scenario, position, velocity)
        metrics.update(position, DT, ctrl.blocked)
        if trajectory:
            traj.append(position.copy())
        if metrics.arrived.all() and ctrl.parked.all():
            break
    row = {"policy": options.policy, "filtro": options.filtro, "zone": options.zone,
           "repulsione": options.repulsione, "retromarcia": options.retromarcia,
           "corsie": options.con_corsie, "scaglionato": options.scaglionato, "robots": n, "seed": seed}
    row.update(metrics.row(position, ctrl.reverse_events, ctrl.forced_entries))
    if trajectory:
        row["trajectory"] = np.asarray(traj)
    return row
