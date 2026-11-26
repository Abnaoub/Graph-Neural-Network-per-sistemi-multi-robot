"""Controller Webots "gnn_supervisor": collega Webots al controllore di flotta di core/controller.py.

Questo file NON contiene logica di controllo. Ad ogni passo di controllo (DT = 0,08 s):
  1. legge posizione e velocita' di ogni AGV_i (API Supervisor);
  2. chiama FleetController.step(...) -> velocita' (vx, vy) per ogni robot;
  3. le esegue:  robot olonomo  -> node.setVelocity([vx, vy, 0, 0, 0, 0]) ad ogni passo fisico
                 Pioneer 3-DX   -> velocita' delle ruote via Emitter (agv_controller);
  4. aggiorna le metriche condivise (core/metrics.py) e disegna gli archi del grafo.
Alla fine di ogni episodio scrive una riga in results/webots_log.csv (stesse colonne del 2D).

Argomenti (campo controllerArgs del robot "gnn_supervisor"):
  --policy gnn|mlp|esperto  --checkpoint file.npz  --seed 1000  --episodes 20  --max-time 90
  --robot olonomo|pioneer   --filtro emergenza|simmetrico|nessuno
  --senza-corsie  --senza-zone  --senza-repulsione  --senza-retromarcia  --scaglionato S
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "core" / "controller.py").exists())
sys.path.insert(0, str(ROOT))
DEBUG_LOG = ROOT / "webots_debug.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [gnn_supervisor] {msg}"
    print(line, flush=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


log(f"avvio: python {sys.version.split()[0]} in {sys.executable}")

import numpy as np  # noqa: E402

from core import warehouse  # noqa: E402
from core.controller import ControlOptions, FleetController  # noqa: E402
from core.environment import COMM_RADIUS, DT, make_scenario  # noqa: E402
from core.metrics import EpisodeMetrics  # noqa: E402
from core.numpy_policy import load_numpy_policy  # noqa: E402

log("moduli del progetto caricati")

# Pioneer 3-DX (solo per --robot pioneer)
WHEEL_RADIUS, AXLE_LENGTH, MAX_WHEEL_SPEED, HEADING_GAIN, MAX_TURN_RATE = 0.0975, 0.33, 12.3, 2.0, 2.5
EDGE_Z = 0.33
HIDDEN = [0.0, 0.0, -1.0]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["gnn", "mlp", "esperto"], default="gnn")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--robot", choices=["olonomo", "pioneer"], default="olonomo")
    parser.add_argument("--filtro", choices=["emergenza", "simmetrico", "nessuno"], default="emergenza")
    parser.add_argument("--senza-corsie", action="store_true")
    parser.add_argument("--senza-zone", action="store_true")
    parser.add_argument("--senza-repulsione", action="store_true")
    parser.add_argument("--senza-retromarcia", action="store_true")
    parser.add_argument("--scaglionato", type=float, default=0.0)
    parser.add_argument("--log", default=str(ROOT / "results" / "webots_log.csv"))
    return parser.parse_args(argv)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_wheel_speeds(velocity: np.ndarray, yaw: float) -> tuple[float, float]:
    """Pioneer: da (vx, vy) nel mondo a velocita' angolari delle ruote (ruota, poi avanza)."""
    speed = min(float(np.linalg.norm(velocity)), MAX_WHEEL_SPEED * WHEEL_RADIUS)
    if speed < 0.03:
        return 0.0, 0.0
    error = wrap_angle(math.atan2(float(velocity[1]), float(velocity[0])) - yaw)
    forward = speed * max(0.0, math.cos(error))
    turn = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, HEADING_GAIN * error))
    left = (forward - turn * AXLE_LENGTH / 2) / WHEEL_RADIUS
    right = (forward + turn * AXLE_LENGTH / 2) / WHEEL_RADIUS
    scale = max(1.0, abs(left) / MAX_WHEEL_SPEED, abs(right) / MAX_WHEEL_SPEED)
    return left / scale, right / scale


class WebotsBridge:
    """Lettura dello stato e invio dei comandi: l'unica parte che parla con Webots."""

    def __init__(self, supervisor, args: argparse.Namespace):
        self.sup = supervisor
        self.args = args
        self.timestep = int(supervisor.getBasicTimeStep())
        self.control_every = max(1, round(DT * 1000 / self.timestep))
        self.emitter = supervisor.getDevice("emitter")
        self.robots = []
        while (node := supervisor.getFromDef(f"AGV_{len(self.robots)}")) is not None:
            self.robots.append(node)
        if not self.robots:
            raise RuntimeError("nessun robot AGV_0, AGV_1, ... nel mondo")
        self.goal_markers = [supervisor.getFromDef(f"GOAL_{i}") for i in range(len(self.robots))]
        n = len(self.robots)
        self.edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_node = supervisor.getFromDef("EDGE_COORD")
        self.edge_points = edge_node.getField("point") if edge_node is not None else None
        self.edge_on = [True] * len(self.edge_pairs)
        self.spawn_z = 0.0975 if args.robot == "pioneer" else 0.111
        log(f"{n} robot ({args.robot}), controllo ogni {self.control_every * self.timestep} ms")

    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.robots)
        pos, vel, yaw = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32), np.zeros(n)
        for i, node in enumerate(self.robots):
            p, r, v = node.getPosition(), node.getOrientation(), node.getVelocity()
            pos[i] = (p[0], p[1])
            vel[i] = (v[0], v[1])
            yaw[i] = math.atan2(r[3], r[0])
        return pos, vel, yaw

    def place(self, starts: np.ndarray, goals: np.ndarray, headings: np.ndarray) -> None:
        for i, node in enumerate(self.robots):
            x, y = (float(c) for c in starts[i])
            node.getField("translation").setSFVec3f([x, y, self.spawn_z])
            node.getField("rotation").setSFRotation([0, 0, 1, float(headings[i])])
            node.resetPhysics()
            if self.goal_markers[i] is not None:
                gx, gy = (float(c) for c in goals[i])
                self.goal_markers[i].getField("translation").setSFVec3f([gx, gy, 0.012])

    def apply(self, action: np.ndarray, yaw: np.ndarray, episode: int) -> None:
        if self.args.robot == "olonomo":
            for node, (vx, vy) in zip(self.robots, action):
                node.setVelocity([float(vx), float(vy), 0.0, 0.0, 0.0, 0.0])
        else:
            wheels = [to_wheel_speeds(action[i], float(yaw[i])) for i in range(len(self.robots))]
            payload = json.dumps({"ep": episode, "w": [[round(a, 4), round(b, 4)] for a, b in wheels]})
            try:
                self.emitter.send(payload)
            except TypeError:
                self.emitter.send(payload.encode("utf-8"))

    def stop_all(self, episode: int) -> None:
        self.apply(np.zeros((len(self.robots), 2), np.float32), np.zeros(len(self.robots)), episode)

    def draw_graph(self, pos: np.ndarray) -> None:
        if self.edge_points is None:
            return
        for k, (i, j) in enumerate(self.edge_pairs):
            active = float(np.linalg.norm(pos[i] - pos[j])) <= COMM_RADIUS
            if active:
                self.edge_points.setMFVec3f(2 * k, [float(pos[i, 0]), float(pos[i, 1]), EDGE_Z])
                self.edge_points.setMFVec3f(2 * k + 1, [float(pos[j, 0]), float(pos[j, 1]), EDGE_Z])
            elif self.edge_on[k]:
                self.edge_points.setMFVec3f(2 * k, HIDDEN)
                self.edge_points.setMFVec3f(2 * k + 1, HIDDEN)
            self.edge_on[k] = active


def run(supervisor, args: argparse.Namespace) -> None:
    bridge = WebotsBridge(supervisor, args)
    options = ControlOptions(policy=args.policy, con_corsie=not args.senza_corsie, zone=not args.senza_zone,
                             repulsione=not args.senza_repulsione, filtro=args.filtro,
                             retromarcia=not args.senza_retromarcia, scaglionato=args.scaglionato)
    warehouse.ONE_WAY = options.con_corsie
    policy_fn = None
    if args.policy != "esperto":
        path = Path(args.checkpoint or ROOT / "results" / "checkpoints_warehouse" / f"{args.policy}_n8.npz")
        if not path.is_absolute():
            path = ROOT / path
        policy_fn = load_numpy_policy(str(path))
        log(f"policy {args.policy} caricata da {path}")
    log(f"opzioni: {options}")
    n = len(bridge.robots)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        scenario = make_scenario(n, seed, "warehouse")
        ctrl = FleetController(scenario, options, policy_fn)
        metrics = EpisodeMetrics(n, scenario.goals, scenario.obstacles)
        bridge.place(scenario.starts, scenario.goals, ctrl.initial_headings())
        log(f"episodio {episode} (seed {seed}): robot posizionati")
        for _ in range(5):                      # lascia assestare la fisica
            bridge.stop_all(episode)
            if supervisor.step(bridge.timestep) == -1:
                return
        start_time = supervisor.getTime()
        action = np.zeros((n, 2), np.float32)
        yaw = np.zeros(n)
        tick = 0
        done = False
        while supervisor.getTime() - start_time < args.max_time and not done:
            if supervisor.step(bridge.timestep) == -1:
                return
            tick += 1
            if tick % bridge.control_every == 0:
                pos, vel, yaw = bridge.read_state()
                now = supervisor.getTime() - start_time
                action = ctrl.step(pos, vel, now)
                metrics.update(pos, DT, ctrl.blocked)
                bridge.draw_graph(pos)
                done = bool(metrics.arrived.all() and ctrl.parked.all())
            bridge.apply(action, yaw, episode)        # il robot olonomo riceve la velocita' ad ogni passo fisico
        bridge.stop_all(episode)
        pos, _, _ = bridge.read_state()
        row = {"policy": args.policy, "robot": args.robot, "filtro": options.filtro, "zone": options.zone,
               "repulsione": options.repulsione, "retromarcia": options.retromarcia, "corsie": options.con_corsie,
               "scaglionato": options.scaglionato, "robots": n, "seed": seed}
        row.update(metrics.row(pos, ctrl.reverse_events, ctrl.forced_entries))
        rows.append(row)
        log(f"episodio {episode}: {row}")
    if rows:
        path = Path(args.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            if header != list(rows[0].keys()):
                path.rename(path.with_name(path.stem + "_vecchio.csv"))
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
        ok = sum(r["success"] for r in rows)
        log(f"fine: {ok}/{len(rows)} episodi riusciti, contatti totali {sum(r['contact_events'] for r in rows)}, log in {path}")
    try:
        supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
    except AttributeError:
        pass


def main(argv: list[str] | None = None, supervisor=None) -> None:
    if supervisor is None:
        from controller import Supervisor
        supervisor = Supervisor()
        log("Supervisor creato")
    run(supervisor, parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
