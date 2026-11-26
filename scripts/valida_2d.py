#!/usr/bin/env python3
"""Validazione nel simulatore 2D: N robot, 20 seed, confronto delle configurazioni.

Uso: python scripts/valida_2d.py --robots 6 --seeds 1000-1019 --policy gnn
Configurazioni: completa, senza_zone, senza_corsie, solo_policy (= "senza sicurezza" nella tesi),
filtro_simmetrico, solo_comunicazione.
Scrive results/validazione_2d.csv (o --output) e stampa un riepilogo con i criteri dell'obiettivo:
success >= 95%, contact_events = 0, min_separation >= 0,60 m, shelf_contact_max_s <= 2 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.controller import ControlOptions  # noqa: E402
from core.numpy_policy import load_numpy_policy  # noqa: E402
from core.simulation2d import rollout_warehouse  # noqa: E402

CONFIGS = {
    "completa": ControlOptions(),
    "senza_zone": ControlOptions(zone=False),
    "senza_corsie": ControlOptions(con_corsie=False),
    "solo_policy": ControlOptions(zone=False, repulsione=False, filtro="nessuno", retromarcia=False),
    "filtro_simmetrico": ControlOptions(zone=False, filtro="simmetrico"),
    "solo_comunicazione": ControlOptions(con_corsie=False, zone=False, filtro="nessuno", retromarcia=False),
}


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--seeds", default="1000-1019")
    parser.add_argument("--policy", default="gnn", choices=["gnn", "mlp", "esperto"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--output", default=str(ROOT / "results" / "validazione_2d.csv"))
    args = parser.parse_args()
    policy_fn = None
    if args.policy != "esperto":
        path = Path(args.checkpoint or ROOT / "results" / "checkpoints_warehouse" / f"{args.policy}_n8.npz")
        policy_fn = load_numpy_policy(str(path))
    rows = []
    for name in args.configs:
        opt = ControlOptions(**{**CONFIGS[name].__dict__, "policy": args.policy})
        for seed in parse_seeds(args.seeds):
            row = rollout_warehouse(args.robots, seed, opt, policy_fn, args.max_time)
            row["config"] = name
            rows.append(row)
            print(f"{name:18s} seed {seed}: arrivati {row['arrived']}/{args.robots} t={row['time_s']:5.1f}s "
                  f"contatti={row['contact_events']} minsep={row['min_separation']:.2f} "
                  f"scaffali_max={row['shelf_contact_max_s']:.1f}s attesa={row['wait_time_s']:.1f}s "
                  f"retro={row['reverse_events']}", flush=True)
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summary = df.groupby("config").agg(success=("success", "mean"), arrivati=("arrived", "mean"),
                                       time_s=("time_s", "mean"), path_length=("path_length", "mean"),
                                       contatti=("contact_events", "mean"), contatti_max=("contact_events", "max"),
                                       min_sep=("min_separation", "min"), scaffali_max_s=("shelf_contact_max_s", "max"),
                                       attesa_s=("wait_time_s", "mean"), retro=("reverse_events", "mean"),
                                       forzati=("forced_entries", "sum"))
    summary["success"] = (100 * summary["success"]).round(0)
    print("\nRiepilogo (success in %, medie sui seed; contatti_max, min_sep e scaffali_max_s sono i casi peggiori):")
    print(summary.round(2).to_string())
    print(f"\nscritto {args.output}")


if __name__ == "__main__":
    main()
