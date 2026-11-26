"""Geometria del magazzino (unica fonte: il 2D, l'addestramento e il mondo Webots la leggono da qui).

Mappa dimensionata sul robot (raggio ROBOT_RADIUS = 0,28 m, diametro 0,56 m) secondo
la pratica AGV / ISO 3691-4 (almeno 0,5 m di franco per lato):
  - corsia a senso unico   >= larghezza robot + 2 x 0,5 m = 1,56 m   -> passaggi tra le file: 1,8 m
  - corsia a doppio senso  >= 2 x robot + 3 x 0,5 m      = 2,62 m   -> corridoi verticali: 2,1 m
                                                                      (traffico verticale raro: solo
                                                                       cambio di passaggio, una corsia)
  - corridoi lungo i muri sopra e sotto: 2,3 m
  - piazzole: >= 1,0 m tra i centri, >= 1,0 m dai muri, >= 1,5 m dagli scaffali
Rispetto alla mappa precedente (12 x 8 m, varchi da 0,6-0,7 m, pilastro nel corridoio
superiore) ogni passaggio che un robot deve usare e' largo almeno 1,8 m.

Corsie a senso unico (ONE_WAY): nella zona scaffali i robot diretti a destra usano solo
la meta' superiore (y > LANE_SPLIT_Y), quelli diretti a sinistra solo la meta' inferiore.
La linea di separazione passa per il CENTRO della fila di scaffali centrale (y = 0),
cosi' ogni corsia ha due passaggi (uno tra le file, uno lungo il muro) e non esiste un
passaggio "a cavallo" della linea. In A* il vincolo e' rigido (celle vietate).

Coordinate: x verso destra, y verso l'alto, origine al centro del pavimento.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

# ---------------------------------------------------------------------------
# Dimensioni (metri)
# ---------------------------------------------------------------------------
ROBOT_RADIUS = 0.28         # come core/environment.py (cerchio del simulatore 2D)
ARENA_W = 15.0              # lato x del pavimento
ARENA_H = 10.0              # lato y del pavimento
WALL_T = 0.1                # spessore dei muri perimetrali

SHELF_SIZE = (0.4, 0.6, 0.8)                   # (x, y, z) di ogni scaffale
SHELF_COLUMNS = (-3.75, -1.25, 1.25, 3.75)     # x dei centri: corridoi verticali 2,5 - 0,4 = 2,1 m
SHELF_ROWS = (2.4, 0.0, -2.4)                  # y dei centri: passaggi 2,4 - 0,6 = 1,8 m;
                                               # corridoi lungo i muri 5,0 - 2,7 = 2,3 m
SHELF_ZONE_X = (min(SHELF_COLUMNS) - SHELF_SIZE[0] / 2 - 0.45,   # zona scaffali (con franco)
                max(SHELF_COLUMNS) + SHELF_SIZE[0] / 2 + 0.45)   # = (-4.4, 4.4)
LANE_SPLIT_Y = 0.0          # centro della fila centrale: sopra -> destra, sotto -> sinistra
ONE_WAY = False             # True = corsie a senso unico rigide (opzione --con-corsie)

# Pilastro: nell'angolo in basso a sinistra, fuori da ogni corridoio e a >= 1,2 m dalle piazzole.
# (Nella mappa precedente stava nel corridoio superiore e lo chiudeva: 0,34 m dallo scaffale.)
PILLARS = ((-7.05, -4.55, 0.08),)              # (x, y, raggio)

# Stazioni: piazzole colorate a terra (solo visive). Sinistra = prelievo, destra = consegna.
STATIONS = (
    # nome,            x,     y,    lato_x, lato_y, colore RGB,        lato
    ("PAD_YELLOW_1", -6.0,  2.5,   2.0,    0.8,   (1.0, 1.0, 0.0),   "sinistra"),
    ("PAD_YELLOW_2", -6.0, -0.5,   2.0,    0.8,   (1.0, 1.0, 0.0),   "sinistra"),
    ("PAD_ORANGE_L", -6.0, -3.5,   2.0,    0.8,   (1.0, 0.6, 0.0),   "sinistra"),
    ("PAD_GREEN_1",   6.0,  2.5,   2.0,    0.8,   (0.0, 0.9, 0.0),   "destra"),
    ("PAD_GREEN_2",   6.0, -0.5,   2.0,    0.8,   (0.0, 0.9, 0.0),   "destra"),
    ("PAD_ORANGE_R",  6.0, -3.5,   2.0,    0.8,   (1.0, 0.6, 0.0),   "destra"),
)

# Piazzole di partenza/arrivo: griglia 2 x 6 per lato, passo 1,0 m (centro-centro),
# x = -6.5 (1,0 m dal muro) e -5.5 (1,55 m dal bordo dello scaffale piu' vicino: un robot
# parcheggiato sulla piazzola interna lascia > 1,0 m tra se' e l'uscita dei passaggi,
# cosi' chi esce dalla zona scaffali gli passa accanto senza fermarsi).
SLOT_XS_LEFT = (-6.5, -5.5)
SLOT_YS = (4.0, 2.5, 1.0, -0.5, -2.0, -3.5)
SLOTS_LEFT = tuple((x, y) for y in SLOT_YS for x in SLOT_XS_LEFT)
SLOTS_RIGHT = tuple((-x, y) for x, y in SLOTS_LEFT)
SLOT_JITTER = 0.10          # rumore sulle partenze (le piazzole distano 1,0 m: resta >= 0,8 m)

BOUNDS = (-ARENA_W / 2 + ROBOT_RADIUS, ARENA_W / 2 - ROBOT_RADIUS,
          -ARENA_H / 2 + ROBOT_RADIUS, ARENA_H / 2 - ROBOT_RADIUS)

# Pianificazione
GRID_RESOLUTION = 0.1                    # m per cella di A*
INFLATION = ROBOT_RADIUS + 0.15          # 0,43 m: raggio + franco dalle pareti
LOS_CLEARANCE = ROBOT_RADIUS + 0.10      # il goal locale deve essere visibile con questo franco


def shelves() -> list[tuple[str, float, float]]:
    """Nome e centro (x, y) di ogni scaffale (nomi usati anche nel mondo Webots)."""
    items = []
    for c, x in enumerate(SHELF_COLUMNS, start=1):
        for r, y in enumerate(SHELF_ROWS, start=1):
            items.append((f"shelf_c{c}_r{r}", x, y))
    return items


def warehouse_obstacles() -> np.ndarray:
    """Ostacoli come rettangoli [cx, cy, larghezza, altezza] (formato del simulatore 2D)."""
    rects = [[x, y, SHELF_SIZE[0], SHELF_SIZE[1]] for _, x, y in shelves()]
    rects += [[x, y, 2 * r, 2 * r] for x, y, r in PILLARS]
    hx, hy = ARENA_W / 2 + WALL_T / 2, ARENA_H / 2 + WALL_T / 2
    rects += [
        [-hx, 0.0, WALL_T, ARENA_H + 2 * WALL_T], [hx, 0.0, WALL_T, ARENA_H + 2 * WALL_T],
        [0.0, -hy, ARENA_W + 2 * WALL_T, WALL_T], [0.0, hy, ARENA_W + 2 * WALL_T, WALL_T],
    ]
    return np.asarray(rects, dtype=np.float32)


def passages() -> list[tuple[str, float, float, float, float]]:
    """Passaggi orizzontali attraverso la zona scaffali: (nome, y_min, y_max, larghezza, corsia).

    Serve ai test di geometria e alla figura della mappa.
    """
    half = SHELF_SIZE[1] / 2
    out = []
    rows = sorted(SHELF_ROWS)
    bands = [(-ARENA_H / 2, rows[0] - half)] + [(a + half, b - half) for a, b in zip(rows[:-1], rows[1:])] \
        + [(rows[-1] + half, ARENA_H / 2)]
    for k, (lo, hi) in enumerate(bands):
        mid = (lo + hi) / 2
        lane = "destra" if mid > LANE_SPLIT_Y else "sinistra"
        out.append((f"passaggio_{k}", lo, hi, hi - lo, lane))
    return out


def critical_zones() -> list[tuple[str, float, float, float, float]]:
    """Zone critiche per la prenotazione: gli incroci tra i corridoi verticali e i passaggi.

    Ogni zona e' un rettangolo (nome, cx, cy, w, h). Nei passaggi il flusso e' a senso
    unico (corsie), quindi i conflitti restano solo dove un robot che cambia passaggio
    (moto verticale) incrocia chi percorre il passaggio: li' entra un robot alla volta.
    """
    cols = sorted(SHELF_COLUMNS)
    half_x = SHELF_SIZE[0] / 2
    corridors = [(a + half_x, b - half_x) for a, b in zip(cols[:-1], cols[1:])]   # x_min, x_max
    zones = []
    for i, (x0, x1) in enumerate(corridors):
        for name, y0, y1, _, _ in passages():
            if y1 - y0 > 2.0 and (y0 <= -ARENA_H / 2 + 1e-6 or y1 >= ARENA_H / 2 - 1e-6):
                continue        # i corridoi lungo i muri (2,3 m) sono larghi abbastanza per due robot
            zones.append((f"incrocio_{i}_{name}", (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return zones


def make_warehouse_scenario(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Partenze e goal: i robot pari vanno da sinistra a destra, i dispari al contrario.

    Cosi' i robot attraversano la zona scaffali in entrambi i sensi, che e' la
    situazione in cui corsie e cooperazione servono davvero.
    """
    if n > 2 * len(SLOTS_LEFT):
        raise ValueError(f"al massimo {2 * len(SLOTS_LEFT)} robot nel magazzino")
    rng = np.random.default_rng(seed)
    left = [SLOTS_LEFT[i] for i in rng.permutation(len(SLOTS_LEFT))]
    right = [SLOTS_RIGHT[i] for i in rng.permutation(len(SLOTS_RIGHT))]
    left_goals = [SLOTS_LEFT[i] for i in rng.permutation(len(SLOTS_LEFT))]
    right_goals = [SLOTS_RIGHT[i] for i in rng.permutation(len(SLOTS_RIGHT))]
    starts, goals = [], []
    for i in range(n):
        if i % 2 == 0:
            starts.append(left.pop()); goals.append(right_goals.pop())
        else:
            starts.append(right.pop()); goals.append(left_goals.pop())
    jitter = rng.uniform(-SLOT_JITTER, SLOT_JITTER, size=(n, 2))
    starts = np.asarray(starts, dtype=np.float32) + jitter.astype(np.float32)
    return starts, np.asarray(goals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Funzioni geometriche usate dal pianificatore e dalla fisica 2D
# ---------------------------------------------------------------------------
def _inside(point, rect, margin: float) -> bool:
    x, y = point
    cx, cy, w, h = rect
    return abs(x - cx) <= w / 2 + margin and abs(y - cy) <= h / 2 + margin


def distance_to_obstacles(point, obstacles: np.ndarray) -> float:
    """Distanza dal bordo dell'ostacolo piu' vicino (0 se dentro). Vettorizzata."""
    obstacles = np.asarray(obstacles, dtype=np.float32)
    dx = np.maximum(np.abs(point[0] - obstacles[:, 0]) - obstacles[:, 2] / 2, 0.0)
    dy = np.maximum(np.abs(point[1] - obstacles[:, 1]) - obstacles[:, 3] / 2, 0.0)
    return float(np.min(np.hypot(dx, dy)))


def segment_clear(a, b, obstacles: np.ndarray, margin: float, step: float = 0.08) -> bool:
    """True se il segmento a-b resta a distanza >= margin da tutti i rettangoli (vettorizzato)."""
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    steps = max(2, int(np.linalg.norm(b - a) / step) + 1)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    q = a + t * (b - a)                                           # (S, 2)
    inside_x = np.abs(q[:, None, 0] - obstacles[None, :, 0]) <= obstacles[None, :, 2] / 2 + margin
    inside_y = np.abs(q[:, None, 1] - obstacles[None, :, 1]) <= obstacles[None, :, 3] / 2 + margin
    return not bool(np.any(inside_x & inside_y))


_GRID_CACHE: dict = {}


def _grid(obstacles: np.ndarray, resolution: float, inflation: float):
    key = (obstacles.tobytes(), resolution, inflation)
    if key not in _GRID_CACHE:
        x0, y0 = -ARENA_W / 2, -ARENA_H / 2
        cols, rows = int(round(ARENA_W / resolution)), int(round(ARENA_H / resolution))
        xs = x0 + (np.arange(cols) + 0.5) * resolution
        ys = y0 + (np.arange(rows) + 0.5) * resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        blocked = np.zeros((cols, rows), dtype=bool)
        for cx, cy, w, h in obstacles:
            blocked |= (np.abs(gx - cx) <= w / 2 + inflation) & (np.abs(gy - cy) <= h / 2 + inflation)
        _GRID_CACHE[key] = (blocked, xs, ys)
    return _GRID_CACHE[key]


def lane_for(start, goal) -> str:
    """Senso di marcia: 'destra' se il robot va verso x crescenti, altrimenti 'sinistra'."""
    return "destra" if goal[0] > start[0] else "sinistra"


def lane_forbidden(obstacles_grid_xs: np.ndarray, ys: np.ndarray, lane: str) -> np.ndarray:
    """Celle vietate dalle corsie: nella zona scaffali, la meta' 'sbagliata' per quel senso."""
    in_zone = (obstacles_grid_xs[:, None] > SHELF_ZONE_X[0]) & (obstacles_grid_xs[:, None] < SHELF_ZONE_X[1])
    wrong = (ys[None, :] < LANE_SPLIT_Y) if lane == "destra" else (ys[None, :] > LANE_SPLIT_Y)
    return in_zone & wrong


def plan_path(start, goal, obstacles: np.ndarray, resolution: float = GRID_RESOLUTION,
              inflation: float = INFLATION, one_way: bool | None = None) -> np.ndarray:
    """Percorso A* (8 direzioni, senza tagliare gli angoli) poi semplificato.

    Con one_way=True le corsie sono un VINCOLO RIGIDO: le celle della meta' sbagliata
    della zona scaffali sono trattate come ostacoli (nessuna penalita' "soft").
    """
    if one_way is None:
        one_way = ONE_WAY
    blocked, xs, ys = _grid(obstacles, resolution, inflation)
    if one_way:
        blocked = blocked | lane_forbidden(xs, ys, lane_for(start, goal))
    cols, rows = blocked.shape

    def cell(p):
        c = (int(np.argmin(np.abs(xs - p[0]))), int(np.argmin(np.abs(ys - p[1]))))
        if not blocked[c]:
            return c
        free = np.argwhere(~blocked)
        d = (xs[free[:, 0]] - p[0]) ** 2 + (ys[free[:, 1]] - p[1]) ** 2
        k = int(np.argmin(d))
        return int(free[k, 0]), int(free[k, 1])

    src, dst = cell(start), cell(goal)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    frontier = [(0.0, src)]
    parent = {src: None}
    cost = {src: 0.0}
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == dst:
            break
        for dx, dy in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < cols and 0 <= nxt[1] < rows) or blocked[nxt]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue
            g = cost[cur] + math.hypot(dx, dy)
            if g < cost.get(nxt, float("inf")):
                cost[nxt] = g
                parent[nxt] = cur
                heapq.heappush(frontier, (g + math.hypot(dst[0] - nxt[0], dst[1] - nxt[1]), nxt))
    if dst not in parent:
        return np.asarray([start, goal], dtype=np.float32)
    cells = []
    cur = dst
    while cur is not None:
        cells.append(cur)
        cur = parent[cur]
    cells.reverse()
    pts = [np.asarray(start, dtype=float)] + [np.array([xs[c[0]], ys[c[1]]]) for c in cells[1:-1]] \
        + [np.asarray(goal, dtype=float)]

    def free_segment(a, b):
        steps = max(2, int(np.linalg.norm(b - a) / (resolution / 2)))
        for t in np.linspace(0, 1, steps):
            q = a + t * (b - a)
            i, j = int(np.argmin(np.abs(xs - q[0]))), int(np.argmin(np.abs(ys - q[1])))
            if blocked[i, j]:
                return False
        return True

    simplified = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not free_segment(pts[i], pts[j]):
            j -= 1
        simplified.append(pts[j])
        i = j
    return np.asarray(simplified, dtype=np.float32)


class WaypointTracker:
    """Fornisce a ogni robot un goal locale lungo il proprio percorso A*.

    La policy (esperto o GNN) vede come "goal" il punto piu' lontano del percorso entro
    `lookahead` metri che sia IN LINEA DI VISTA (segmento robot-punto a >= LOS_CLEARANCE
    da ogni ostacolo). Senza questo vincolo, alle svolte il goal finiva dietro lo spigolo
    dello scaffale e il robot ci andava a sbattere.
    """

    def __init__(self, starts: np.ndarray, goals: np.ndarray, obstacles: np.ndarray,
                 lookahead: float = 1.0, inflation: float = INFLATION, line_of_sight: bool = True,
                 clearance: float = LOS_CLEARANCE, one_way: bool | None = None):
        self.goals = np.asarray(goals, dtype=np.float32)
        self.lookahead = lookahead
        self.obstacles = np.asarray(obstacles, dtype=np.float32)
        self.line_of_sight = line_of_sight
        self.clearance = clearance
        self.paths = [self._densify(plan_path(s, g, obstacles, inflation=inflation, one_way=one_way))
                      for s, g in zip(starts, goals)]
        self.progress = [0] * len(self.paths)

    @staticmethod
    def _densify(path: np.ndarray, step: float = 0.1) -> np.ndarray:
        out = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out.extend(a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:])
        return np.asarray(out, dtype=np.float32)

    def visible(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True se il segmento a-b resta a distanza `clearance` da tutti gli ostacoli."""
        return segment_clear(a, b, self.obstacles, self.clearance)

    def remaining(self, i: int) -> float:
        """Metri di percorso ancora da fare (per le priorita')."""
        path = self.paths[i]
        k = self.progress[i]
        return float(np.sum(np.linalg.norm(path[k + 1:] - path[k:-1], axis=1))) if k < len(path) - 1 else 0.0

    def local_goals(self, position: np.ndarray) -> np.ndarray:
        targets = np.zeros_like(self.goals)
        for i, path in enumerate(self.paths):
            start = self.progress[i]
            window = path[start:start + 40]
            nearest = start + int(np.argmin(np.linalg.norm(window - position[i], axis=1)))
            self.progress[i] = nearest
            k = nearest
            best = nearest
            while k < len(path) - 1 and np.linalg.norm(path[k] - position[i]) < self.lookahead:
                k += 1
                if not self.line_of_sight or self.visible(position[i], path[k]):
                    best = k
                else:
                    break                   # i punti successivi stanno dietro lo stesso spigolo
            if self.line_of_sight and best == nearest and nearest < len(path) - 1:
                best = nearest + 1          # fuori strada dietro un angolo: torna verso il percorso
            targets[i] = path[best if self.line_of_sight else k]
        return targets


def resolve_motion(old: np.ndarray, new: np.ndarray, obstacles: np.ndarray,
                   radius: float = ROBOT_RADIUS) -> tuple[np.ndarray, int]:
    """Impedisce di attraversare scaffali e muri nel 2D (in Webots ci pensa la fisica).

    Se il nuovo punto entra in un ostacolo il robot prova a scivolare lungo x o lungo y;
    altrimenti resta fermo. Restituisce anche quanti robot sono stati bloccati nel passo.
    """
    x0, x1, y0, y1 = BOUNDS
    out = new.copy()
    contacts = 0
    for i in range(len(new)):
        candidates = [new[i], np.array([new[i, 0], old[i, 1]]), np.array([old[i, 0], new[i, 1]]), old[i]]
        for k, p in enumerate(candidates):
            p = np.array([np.clip(p[0], x0, x1), np.clip(p[1], y0, y1)], dtype=np.float32)
            if k == len(candidates) - 1 or not any(_inside(p, r, radius) for r in obstacles):
                out[i] = p
                contacts += int(k > 0)
                break
    return out, contacts
