"""Test della mappa e delle regole: dimensioni, percorsi, zone critiche, linea di vista."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import warehouse  # noqa: E402
from core.environment import make_scenario  # noqa: E402
from core.safety import ZoneManager, emergency_filter  # noqa: E402

MIN_PASSAGE = 1.0            # ogni passaggio usato dai robot >= 1,0 m
MIN_PATH_CLEARANCE = 0.43    # raggio + 0,15 m


def test_passaggi_larghi_almeno_un_metro():
    for name, lo, hi, w, _ in warehouse.passages():
        assert w >= MIN_PASSAGE, f"{name}: {w:.2f} m"
    cols = sorted(warehouse.SHELF_COLUMNS)
    for a, b in zip(cols[:-1], cols[1:]):
        assert b - a - warehouse.SHELF_SIZE[0] >= MIN_PASSAGE


def test_pilastri_fuori_dai_corridoi():
    obs = warehouse.warehouse_obstacles()
    for x, y, r in warehouse.PILLARS:
        others = np.asarray([o for o in obs if not (abs(o[0] - x) < 1e-6 and abs(o[1] - y) < 1e-6)])
        # distanza bordo-bordo dal resto >= 1,0 m oppure e' appoggiato a un muro/angolo (nessun passaggio da chiudere)
        d = warehouse.distance_to_obstacles((x, y), others) - r
        corner = min(abs(abs(x) - warehouse.ARENA_W / 2), abs(abs(y) - warehouse.ARENA_H / 2)) < 0.6
        assert d >= MIN_PASSAGE or corner


def test_piazzole_distanti():
    slots = np.asarray(warehouse.SLOTS_LEFT + warehouse.SLOTS_RIGHT)
    d = np.linalg.norm(slots[:, None] - slots[None], axis=-1) + np.eye(len(slots)) * 9
    assert d.min() >= 1.0 - 1e-6
    assert (np.abs(slots[:, 0]) <= warehouse.ARENA_W / 2 - 1.0 + 1e-6).all()
    assert (np.abs(slots[:, 1]) <= warehouse.ARENA_H / 2 - 1.0 + 1e-6).all()
    shelf_edge = max(warehouse.SHELF_COLUMNS) + warehouse.SHELF_SIZE[0] / 2
    assert (np.abs(slots[:, 0]) - shelf_edge >= 1.5 - 1e-6).all()


def test_percorsi_lontani_dagli_ostacoli_e_nelle_corsie():
    warehouse.ONE_WAY = True
    obs = warehouse.warehouse_obstacles()
    for seed in (1000, 1001, 1002):
        sc = make_scenario(6, seed, "warehouse")
        for s, g in zip(sc.starts, sc.goals):
            path = warehouse.WaypointTracker._densify(warehouse.plan_path(s, g, obs))
            inner = path[5:-5]      # partenza e arrivo sono sulle piazzole, non sulla griglia
            for p in inner:
                assert warehouse.distance_to_obstacles(p, obs) >= MIN_PATH_CLEARANCE - 0.02
            zone = path[(path[:, 0] > warehouse.SHELF_ZONE_X[0]) & (path[:, 0] < warehouse.SHELF_ZONE_X[1])]
            if warehouse.lane_for(s, g) == "destra":
                assert (zone[:, 1] > warehouse.LANE_SPLIT_Y).all()
            else:
                assert (zone[:, 1] < warehouse.LANE_SPLIT_Y).all()


def test_goal_locale_in_linea_di_vista():
    warehouse.ONE_WAY = True
    sc = make_scenario(6, 1000, "warehouse")
    tr = warehouse.WaypointTracker(sc.starts, sc.goals, sc.obstacles)
    pos = sc.starts.copy()
    for _ in range(200):
        g = tr.local_goals(pos)
        for i in range(6):
            assert tr.visible(pos[i], g[i]) or np.linalg.norm(g[i] - pos[i]) < 0.3   # (fallback: torna sul percorso)
        step = g - pos
        pos = pos + 0.08 * step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-6)


def test_zone_manager_non_ammette_incroci():
    warehouse.ONE_WAY = True
    sc = make_scenario(6, 1000, "warehouse")
    tr = warehouse.WaypointTracker(sc.starts, sc.goals, sc.obstacles)
    zm = ZoneManager(warehouse.critical_zones(), tr, 6)
    _, cx, cy, w, h = warehouse.critical_zones()[0]
    # robot 0 dentro la zona con attraversamento V, robot 1 che chiede di entrare con classe H
    zm.classes[(0, 0)] = "V"
    zm.classes[(1, 0)] = "H"
    pos = sc.starts.copy()
    pos[0] = (cx, cy)
    pos[1] = (cx - w / 2 - 0.5, cy)
    tr.paths[1] = np.asarray([[cx - w / 2 - 0.5 + 0.1 * k, cy] for k in range(30)], dtype=np.float32)
    tr.progress[1] = 0
    blocked = zm.update(pos, np.zeros(6, bool), 0.0)
    assert blocked[1] and not blocked[0]
    # con la stessa classe (convoglio) puo' entrare
    zm.classes[(1, 0)] = "V"
    zm.waiting_since[:] = -1
    blocked = zm.update(pos, np.zeros(6, bool), 0.0)
    assert not blocked[1]


def test_filtro_emergenza_ferma_il_meno_prioritario():
    pos = np.asarray([[0, 0], [0.65, 0]], dtype=np.float32)
    act = np.asarray([[1, 0], [-1, 0]], dtype=np.float32)
    out = emergency_filter(pos, act, rank=np.asarray([0.0, 1.0]))
    assert out[0, 0] <= 0.0 + 1e-6          # frontali: nessuno dei due avanza verso l'altro
    assert out[1, 0] >= 0.0 - 1e-6
    assert np.linalg.norm(out[1]) <= 1e-6   # il meno prioritario e' fermo
    # in coda: chi segue adegua la velocita' a chi precede, non si ferma
    act = np.asarray([[0.5, 0], [1.0, 0]], dtype=np.float32)      # 1 insegue 0 (entrambi verso +x)
    pos = np.asarray([[0.68, 0], [0, 0]], dtype=np.float32)       # 0 davanti a x=0.68, 1 dietro a x=0
    out = emergency_filter(pos, act, rank=np.asarray([0.0, 1.0]))
    assert abs(out[1, 0] - 0.5) < 1e-5 and abs(out[0, 0] - 0.5) < 1e-5
