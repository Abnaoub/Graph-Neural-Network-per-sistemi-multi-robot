"""Pianificazione dei percorsi: A* vincolato dalle corsie, smussamento, goal locale in linea di vista.

Le funzioni vivono in core/warehouse.py (insieme alla geometria, perche' la griglia di A*
dipende dalla mappa); questo modulo le espone con il nome richiesto dalla struttura del
progetto e documenta le regole:

- griglia GRID_RESOLUTION = 0,1 m, inflazione INFLATION = raggio + 0,15 m (= 0,43 m):
  un percorso non passa mai a meno di 0,43 m dal bordo di uno scaffale o di un muro;
- corsie a senso unico (ONE_WAY) come vincolo rigido: le celle della meta' sbagliata
  della zona scaffali sono ostacoli, non costi maggiorati;
- i percorsi sono calcolati UNA volta per episodio (WaypointTracker) e poi smussati
  (segmenti rettilinei che restano nelle celle libere);
- il goal locale e' il punto piu' lontano del percorso entro 1 m che sia visibile dal robot
  con franco LOS_CLEARANCE = raggio + 0,10 m (WaypointTracker.local_goals).
"""

from core.warehouse import (GRID_RESOLUTION, INFLATION, LOS_CLEARANCE, WaypointTracker,  # noqa: F401
                            distance_to_obstacles, lane_for, plan_path)

__all__ = ["GRID_RESOLUTION", "INFLATION", "LOS_CLEARANCE", "WaypointTracker",
           "distance_to_obstacles", "lane_for", "plan_path"]
