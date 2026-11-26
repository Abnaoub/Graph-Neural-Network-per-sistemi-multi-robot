"""Metriche di episodio identiche per il simulatore 2D e per Webots.

Colonne prodotte da EpisodeMetrics.row():
  arrived, success, time_s, final_error, min_separation, contact_events (< 0,56 m),
  near_miss_events (< 0,72 m), path_length, shelf_contact_time_s (somma sui robot del
  tempo a contatto con scaffali/muri), shelf_contact_max_s (contatto continuo piu' lungo
  di un singolo robot), wait_time_s (tempo fermo per prenotazione delle zone),
  reverse_events, forced_entries.
Un "contatto" con uno scaffale e' centro a meno di ROBOT_RADIUS + 0,03 m dal bordo:
nel 2D resolve_motion ferma il robot esattamente a ROBOT_RADIUS, in Webots la fisica
lo ferma al contatto del cilindro (raggio ROBOT_RADIUS).
"""

from __future__ import annotations

import numpy as np

from core.environment import ROBOT_RADIUS, SAFE_DISTANCE
from core.warehouse import distance_to_obstacles

GOAL_TOLERANCE = 0.30
SHELF_TOUCH = ROBOT_RADIUS + 0.03


class EpisodeMetrics:
    def __init__(self, n: int, goals: np.ndarray, obstacles: np.ndarray):
        self.n = n
        self.goals = np.asarray(goals, dtype=np.float32)
        self.obstacles = np.asarray(obstacles, dtype=np.float32)
        self.min_sep = 99.0
        self.contacts = 0
        self.near_misses = 0
        self.active_contacts: set = set()
        self.active_near: set = set()
        self.path_length = 0.0
        self.last_pos = None
        self.shelf_time = 0.0
        self.shelf_streak = np.zeros(n)
        self.shelf_max = 0.0
        self.wait_time = 0.0
        self.arrived = np.zeros(n, dtype=bool)
        self.time = 0.0
        self.arrival_time = None        # istante in cui l'ultimo robot entra nei 0,30 m

    def update(self, position: np.ndarray, dt: float, blocked: np.ndarray | None = None) -> None:
        n = self.n
        self.time += dt
        if self.last_pos is not None:
            self.path_length += float(np.linalg.norm(position - self.last_pos, axis=1).sum())
        self.last_pos = position.copy()
        sep = np.linalg.norm(position[:, None] - position[None], axis=-1) + np.eye(n) * 100.0
        self.min_sep = min(self.min_sep, float(sep.min()))
        touching = {(i, j) for i in range(n) for j in range(i + 1, n) if sep[i, j] < 2 * ROBOT_RADIUS}
        close = {(i, j) for i in range(n) for j in range(i + 1, n) if sep[i, j] < SAFE_DISTANCE}
        self.contacts += len(touching - self.active_contacts)
        self.near_misses += len(close - self.active_near)
        self.active_contacts, self.active_near = touching, close
        for i in range(n):
            if not self.arrived[i] and distance_to_obstacles(position[i], self.obstacles) < SHELF_TOUCH:
                self.shelf_time += dt
                self.shelf_streak[i] += dt
                self.shelf_max = max(self.shelf_max, float(self.shelf_streak[i]))
            else:
                self.shelf_streak[i] = 0.0
        if blocked is not None:
            self.wait_time += dt * int(np.sum(blocked))
        self.arrived |= np.linalg.norm(position - self.goals, axis=1) < GOAL_TOLERANCE
        if self.arrival_time is None and self.arrived.all():
            self.arrival_time = self.time

    def row(self, position: np.ndarray, reverse_events: int = 0, forced_entries: int = 0) -> dict:
        final_error = float(np.mean(np.linalg.norm(position - self.goals, axis=1)))
        return {
            "arrived": int(self.arrived.sum()), "success": int(self.arrived.all()),
            "time_s": round(self.arrival_time if self.arrival_time is not None else self.time, 2),
            "final_error": round(final_error, 3),
            "min_separation": round(self.min_sep, 3), "contact_events": self.contacts,
            "near_miss_events": self.near_misses, "path_length": round(self.path_length, 2),
            "shelf_contact_time_s": round(self.shelf_time, 2), "shelf_contact_max_s": round(self.shelf_max, 2),
            "wait_time_s": round(self.wait_time, 2), "reverse_events": reverse_events,
            "forced_entries": forced_entries,
        }
