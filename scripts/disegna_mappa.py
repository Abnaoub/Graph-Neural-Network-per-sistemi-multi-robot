#!/usr/bin/env python3
"""Figura 2D della mappa: scaffali, piazzole, corsie, zone critiche e (opzionale) percorsi A*.

Uso: python scripts/disegna_mappa.py [--robots 6 --seed 1000 --con-corsie] [--output figures/generated/mappa_magazzino.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import warehouse  # noqa: E402

ROBOT_COLOR = "#5b8def"      # tutti uguali: si distinguono dalla sigla AMR-i
COLORS = [ROBOT_COLOR]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--con-corsie", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "figures" / "generated" / "mappa_magazzino.png"))
    args = parser.parse_args()
    warehouse.ONE_WAY = args.con_corsie
    W, H = warehouse.ARENA_W, warehouse.ARENA_H
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.add_patch(Rectangle((-W / 2, -H / 2), W, H, fc="#e9e9e9", ec="k", lw=3))
    # corsie
    if args.con_corsie:
        x0, x1 = warehouse.SHELF_ZONE_X
        ax.add_patch(Rectangle((x0, 0), x1 - x0, H / 2, fc="#dff0ff", ec="none", alpha=0.7))
        ax.add_patch(Rectangle((x0, -H / 2), x1 - x0, H / 2, fc="#ffe9d6", ec="none", alpha=0.7))
        ax.text(0, H / 2 - 0.3, "corsia  →  (verso destra)", ha="center", va="top", fontsize=10, color="#1f5f9f")
        ax.text(0, -H / 2 + 0.3, "corsia  ←  (verso sinistra)", ha="center", va="bottom", fontsize=10, color="#9f4f1f")
        ax.plot([x0, x1], [warehouse.LANE_SPLIT_Y] * 2, "--", color="#c9a400", lw=1.5)
    # zone critiche
    for name, cx, cy, w, h in warehouse.critical_zones():
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="none", ec="#d62728", ls=":", lw=1.2))
    # piazzole visive
    for name, x, y, sx, sy, rgb, _ in warehouse.STATIONS:
        ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, fc=rgb, ec="none", alpha=0.35))
    for (x, y) in warehouse.SLOTS_LEFT + warehouse.SLOTS_RIGHT:
        ax.add_patch(Circle((x, y), 0.22, fc="none", ec="#666", lw=0.8, ls="--"))
    # scaffali e pilastri
    for _, x, y in warehouse.shelves():
        ax.add_patch(Rectangle((x - warehouse.SHELF_SIZE[0] / 2, y - warehouse.SHELF_SIZE[1] / 2),
                               warehouse.SHELF_SIZE[0], warehouse.SHELF_SIZE[1], fc="#f28e2b", ec="#1f3a93", lw=1.5))
    for x, y, r in warehouse.PILLARS:
        ax.add_patch(Circle((x, y), r, fc="#ffcc00", ec="k"))
    # quote dei passaggi
    for name, y0, y1, w, lane in warehouse.passages():
        ax.annotate("", xy=(-4.8, y1), xytext=(-4.8, y0), arrowprops=dict(arrowstyle="<->", color="#333"))
        ax.text(-4.7, (y0 + y1) / 2, f"{w:.1f} m", fontsize=8, va="center")
    cols = sorted(warehouse.SHELF_COLUMNS)
    cw = cols[1] - cols[0] - warehouse.SHELF_SIZE[0]
    ax.annotate("", xy=(cols[1] - 0.2, 3.9), xytext=(cols[0] + 0.2, 3.9), arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text((cols[0] + cols[1]) / 2, 4.0, f"{cw:.1f} m", fontsize=8, ha="center")
    # scenario e percorsi
    from core.environment import make_scenario
    sc = make_scenario(args.robots, args.seed, "warehouse")
    for i, (s, g) in enumerate(zip(sc.starts, sc.goals)):
        path = warehouse.plan_path(s, g, sc.obstacles)
        c = COLORS[i % len(COLORS)]
        ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=1.5, alpha=0.9)
        ax.add_patch(Circle(s, warehouse.ROBOT_RADIUS, fc=c, ec="k", alpha=0.9))
        ax.text(s[0], s[1], f"AMR-{i + 1}", ha="center", va="center", fontsize=6.5, color="w", weight="bold")
        ax.add_patch(Circle(g, 0.24, fc="none", ec="#333", lw=1.5))
        ax.text(g[0], g[1], f"AMR-{i + 1}", ha="center", va="center", fontsize=6.5, color="#333", weight="bold")
    ax.set_xlim(-W / 2 - 0.3, W / 2 + 0.3); ax.set_ylim(-H / 2 - 0.3, H / 2 + 0.3)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Magazzino {W:g}×{H:g} m — {args.robots} robot, seed {args.seed}"
                 + (" — corsie a senso unico" if args.con_corsie else "") + " — zone critiche tratteggiate in rosso")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.output, dpi=140)
    print(f"scritto {args.output}")


if __name__ == "__main__":
    main()
