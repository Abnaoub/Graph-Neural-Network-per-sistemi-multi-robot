#!/usr/bin/env python3
"""Animazione del simulatore 2D (stesso controllore e stesse metriche di Webots).

Uso:
  python scripts/anima_2d.py                          # 6 robot, seed 1000, configurazione completa
  python scripts/anima_2d.py --seed 1003 --senza-corsie --policy esperto
  python scripts/anima_2d.py --gif figures/generated/anima_seed1000.gif   # salva il GIF
  python scripts/anima_2d.py --robots 8 --policy mlp --config solo_comunicazione --gif figures/generated/mlp_n8.mp4
Senza --gif apre una finestra interattiva (chiudila per terminare). Se il file finisce in .mp4
si salva un video (serve ffmpeg installato), altrimenti un GIF.
--config usa una delle configurazioni di valida_2d.py (completa, senza_zone, senza_corsie,
filtro_simmetrico, solo_policy, solo_comunicazione) al posto delle singole opzioni.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import warehouse  # noqa: E402
from core.controller import ControlOptions  # noqa: E402
from core.environment import DT, make_scenario  # noqa: E402
from core.numpy_policy import load_numpy_policy  # noqa: E402
from core.simulation2d import rollout_warehouse  # noqa: E402

ROBOT_COLOR = "#5b8def"      # tutti uguali: si distinguono dalla sigla AMR-i
COLORS = [ROBOT_COLOR]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy", default="gnn", choices=["gnn", "mlp", "esperto"])
    parser.add_argument("--senza-corsie", action="store_true")
    parser.add_argument("--senza-zone", action="store_true")
    parser.add_argument("--senza-repulsione", action="store_true")
    parser.add_argument("--filtro", default="emergenza", choices=["emergenza", "simmetrico", "nessuno"])
    parser.add_argument("--senza-retromarcia", action="store_true")
    parser.add_argument("--config", default="", help="configurazione di valida_2d.py (sostituisce le opzioni sopra)")
    parser.add_argument("--checkpoint", default="", help="modello .npz (predefinito: results/checkpoints_warehouse/<policy>_n8.npz)")
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--gif", default="", help="se dato, salva qui il GIF (o il video, se finisce in .mp4) invece di aprire la finestra")
    parser.add_argument("--salta", type=int, default=2, help="mostra un fotogramma ogni N passi di controllo")
    args = parser.parse_args()
    if args.gif:
        matplotlib.use("Agg")
    if args.config:
        sys.path.insert(0, str(ROOT / "scripts"))
        from valida_2d import CONFIGS
        opt = ControlOptions(**{**CONFIGS[args.config].__dict__, "policy": args.policy})
    else:
        opt = ControlOptions(policy=args.policy, con_corsie=not args.senza_corsie, zone=not args.senza_zone,
                             repulsione=not args.senza_repulsione, filtro=args.filtro,
                             retromarcia=not args.senza_retromarcia)
    policy_fn = None
    if args.policy != "esperto":
        path = args.checkpoint or ROOT / "results" / "checkpoints_warehouse" / f"{args.policy}_n8.npz"
        policy_fn = load_numpy_policy(str(path))
    row = rollout_warehouse(args.robots, args.seed, opt, policy_fn, args.max_time, trajectory=True)
    traj = row.pop("trajectory")[:: args.salta]
    warehouse.ONE_WAY = opt.con_corsie
    sc = make_scenario(args.robots, args.seed, "warehouse")
    print({k: v for k, v in row.items()})

    W, H = warehouse.ARENA_W, warehouse.ARENA_H
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.add_patch(Rectangle((-W / 2, -H / 2), W, H, fc="#ececec", ec="k", lw=3))
    if opt.con_corsie:
        x0, x1 = warehouse.SHELF_ZONE_X
        ax.add_patch(Rectangle((x0, 0), x1 - x0, H / 2, fc="#dff0ff", ec="none", alpha=0.6))
        ax.add_patch(Rectangle((x0, -H / 2), x1 - x0, H / 2, fc="#ffe9d6", ec="none", alpha=0.6))
    for _, cx, cy, w, h in warehouse.critical_zones():
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="none", ec="#d62728", ls=":", lw=1))
    for _, x, y in warehouse.shelves():
        ax.add_patch(Rectangle((x - 0.2, y - 0.3), 0.4, 0.6, fc="#f28e2b", ec="#1f3a93"))
    for x, y, r in warehouse.PILLARS:
        ax.add_patch(Circle((x, y), r, fc="#ffcc00", ec="k"))
    for (x, y) in warehouse.SLOTS_LEFT + warehouse.SLOTS_RIGHT:
        ax.add_patch(Circle((x, y), 0.22, fc="none", ec="#999", ls="--", lw=0.7))
    bodies, trails, labels = [], [], []
    for i in range(args.robots):
        c = COLORS[i % len(COLORS)]
        ax.add_patch(Circle(sc.goals[i], 0.24, fc="none", ec="#333", lw=1.5))
        ax.text(sc.goals[i, 0], sc.goals[i, 1], f"AMR-{i + 1}", ha="center", va="center", fontsize=6.5, color="#333", weight="bold")
        body = Circle(traj[0, i], warehouse.ROBOT_RADIUS, fc=c, ec="k", alpha=0.9)
        ax.add_patch(body); bodies.append(body)
        labels.append(ax.text(*traj[0, i], f"AMR-{i + 1}", ha="center", va="center", fontsize=6.5, color="w", weight="bold"))
        trail, = ax.plot([], [], "-", color=c, lw=1, alpha=0.6); trails.append(trail)
    title = ax.set_title("")
    ax.set_xlim(-W / 2 - 0.2, W / 2 + 0.2); ax.set_ylim(-H / 2 - 0.2, H / 2 + 0.2); ax.set_aspect("equal")

    def update(k):
        pos = traj[k]
        sep = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(args.robots) * 9
        for i in range(args.robots):
            bodies[i].center = pos[i]
            labels[i].set_position(pos[i])
            trails[i].set_data(traj[:k + 1, i, 0], traj[:k + 1, i, 1])
        title.set_text(f"seed {args.seed} — t = {k * args.salta * DT:5.1f} s — separazione minima ora {sep.min():.2f} m "
                       f"(episodio: {row['min_separation']:.2f} m, contatti {row['contact_events']})")
        return bodies + trails + labels + [title]

    anim = FuncAnimation(fig, update, frames=len(traj), interval=args.salta * DT * 1000, blit=False, repeat=False)
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        fps = int(1 / (args.salta * DT))
        writer = FFMpegWriter(fps=fps) if args.gif.lower().endswith(".mp4") else PillowWriter(fps=fps)
        anim.save(args.gif, writer=writer)
        print(f"scritto {args.gif}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
