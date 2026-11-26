#!/usr/bin/env python3
"""DAgger: la GNN migliora facendo girare SE STESSA tante volte e chiedendo all'esperto cosa avrebbe fatto.

La rete NON impara durante le prove (i pesi sono congelati). Impara solo quando viene
riaddestrata. Questo script fa il ciclo completo (Ross, Gordon, Bagnell 2011):

  iterazione 0  dataset dell'esperto sulla mappa attuale (se manca, viene generato)
  iterazione k  1. la GNN corrente guida i robot nel 2D sui seed di addestramento (con tutte
                   le regole: corsie, zone, filtri) e si registrano gli stati che attraversa;
                2. per ogni stato si salva la velocita' che l'ESPERTO avrebbe scelto (etichetta);
                3. si aggiungono questi esempi al dataset e si riaddestra (scripts/train_policy.py);
                4. si esporta il nuovo .npz e si valuta su seed MAI VISTI (scripts/valida_2d).
  alla fine     il checkpoint migliore (meno contatti, poi piu' successi) viene copiato in
                results/checkpoints_warehouse/gnn_dagger.npz

Uso (serve torch; dalla cartella del progetto):
  python scripts/dagger.py --robots 6 --seeds-train 2000-2019 --seeds-eval 1000-1019 --iterazioni 3
Per "insistere" su un seed preciso: --seeds-train 1003 (ma poi valuta su altri seed, altrimenti
misuri una memorizzazione, non un miglioramento).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core import warehouse  # noqa: E402
from core.controller import ControlOptions, FleetController  # noqa: E402
from core.environment import (COMM_RADIUS, DT, MAX_SPEED, SAFE_DISTANCE, advance, build_edge_features,  # noqa: E402
                              make_scenario, observation, warehouse_expert_action)
from core.numpy_policy import load_numpy_policy  # noqa: E402
from core.simulation2d import rollout_warehouse  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def collect(n: int, seeds: list[int], policy_fn, options: ControlOptions, max_time: float) -> dict:
    """Fa guidare la GNN e registra (osservazione, grafo, archi) + etichetta dell'esperto."""
    rows = {k: [] for k in ("features", "adjacency", "edge_features", "actions", "positions", "episode_step", "rewards")}
    obstacles = None
    for ep, seed in enumerate(seeds):
        print(f"  [raccolta] n={n} seed {seed} ({ep + 1}/{len(seeds)})", flush=True)
        warehouse.ONE_WAY = options.con_corsie
        sc = make_scenario(n, seed, "warehouse")
        obstacles = sc.obstacles
        ctrl = FleetController(sc, options, policy_fn)
        pos, vel = sc.starts.copy(), np.zeros_like(sc.starts)
        for t in range(int(max_time / DT)):
            now = t * DT
            goals_now = ctrl.tracker.local_goals(pos)
            feats, adj = observation(pos, vel, goals_now, sc.obstacles, COMM_RADIUS)
            edges = build_edge_features(pos, vel, COMM_RADIUS)
            label = warehouse_expert_action(pos, goals_now, sc.obstacles, sc.goals)   # cosa farebbe l'esperto QUI
            rows["features"].append(feats); rows["adjacency"].append(adj); rows["edge_features"].append(edges)
            rows["actions"].append(label / MAX_SPEED); rows["positions"].append(pos.copy())
            rows["episode_step"].append((ep, t))
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            rows["rewards"].append(float(-np.linalg.norm(sc.goals - pos, axis=1).mean() - 2.0 * np.triu(d < SAFE_DISTANCE, 1).sum()))
            action = ctrl.step(pos, vel, now)              # ma si muove con la GNN (stati "suoi")
            vel = action
            pos, _ = advance(sc, pos, vel)
            if ctrl.parked.all():
                break
    return {
        "features": np.asarray(rows["features"], np.float32), "adjacency": np.asarray(rows["adjacency"], np.float32),
        "edge_features": np.asarray(rows["edge_features"], np.float32), "actions": np.asarray(rows["actions"], np.float32),
        "positions": np.asarray(rows["positions"], np.float32), "episode_step": np.asarray(rows["episode_step"], np.int32),
        "rewards": np.asarray(rows["rewards"], np.float32), "obstacles": np.asarray(obstacles, np.float32),
        "layout": np.asarray("warehouse"), "expert": np.asarray("dagger"), "robot_count": np.asarray(n, np.int32),
        "seed": np.asarray(seeds[0], np.int32),
    }


def evaluate(n: int, seeds: list[int], policy_fn, options: ControlOptions, max_time: float) -> dict:
    rows = []
    for k, s in enumerate(seeds):
        r = rollout_warehouse(n, s, options, policy_fn, max_time)
        rows.append(r)
        print(f"  [valutazione] n={n} seed {s} ({k + 1}/{len(seeds)}): arrivati {r['arrived']}/{n} "
              f"contatti {r['contact_events']} t={r['time_s']:.1f}s", flush=True)
    return {"success": float(np.mean([r["success"] for r in rows])), "contatti": float(np.sum([r["contact_events"] for r in rows])),
            "quasi": float(np.mean([r["near_miss_events"] for r in rows])), "time_s": float(np.mean([r["time_s"] for r in rows])),
            "min_sep": float(np.min([r["min_separation"] for r in rows])), "retro": float(np.sum([r["reverse_events"] for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", type=int, nargs="+", default=[6])
    parser.add_argument("--seeds-train", default="2000-2019", help="seed su cui la GNN gira e raccoglie esempi")
    parser.add_argument("--seeds-eval", default="1000-1019", help="seed MAI usati per imparare: misurano il miglioramento")
    parser.add_argument("--iterazioni", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodi-base", type=int, default=40, help="episodi dell'esperto per il dataset di partenza")
    parser.add_argument("--steps-base", type=int, default=500)
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints_warehouse" / "gnn_n8.npz"))
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "dagger"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "checkpoints_dagger"))
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    options = ControlOptions(policy="gnn")
    seeds_train, seeds_eval = parse_seeds(args.seeds_train), parse_seeds(args.seeds_eval)
    python = sys.executable

    # iterazione 0: dataset dell'esperto sulla mappa attuale
    base_files = []
    for n in args.robots:
        path = data_dir / f"esperto_warehouse_n{n}.npz"
        if not path.exists():
            print(f"[dagger] genero il dataset dell'esperto con {n} robot ({args.episodi_base} episodi)...", flush=True)
            from generate_dataset import generate
            np.savez_compressed(path, **generate(n, args.episodi_base, args.steps_base, 100, "warehouse"))
        base_files.append(str(path))
    data_files = list(base_files)

    policy_path = Path(args.checkpoint)
    policy_fn = load_numpy_policy(str(policy_path))
    history = []
    print(f"[dagger] iterazione 0: valuto la policy di partenza su {len(seeds_eval)} seed...", flush=True)
    score = evaluate(args.robots[0], seeds_eval, policy_fn, options, args.max_time)
    history.append({"iterazione": 0, "checkpoint": str(policy_path), **score})
    print(f"[dagger] iterazione 0 (policy di partenza): {score}", flush=True)
    for k in range(1, args.iterazioni + 1):
        for n in args.robots:
            print(f"[dagger] iterazione {k}: la GNN guida {n} robot su {len(seeds_train)} seed e l'esperto etichetta...", flush=True)
            data = collect(n, seeds_train, policy_fn, options, args.max_time)
            path = data_dir / f"dagger_warehouse_n{n}_iter{k}.npz"
            np.savez_compressed(path, **data)
            data_files.append(str(path))
            print(f"[dagger]   {len(data['features'])} nuovi esempi -> {path.name}", flush=True)
        iter_dir = out_dir / f"iter{k}"
        print(f"[dagger] iterazione {k}: riaddestro la GNN su {len(data_files)} file...", flush=True)
        subprocess.run([python, str(ROOT / "scripts" / "train_policy.py"), "--data", *data_files, "--method", "gnn",
                        "--epochs", str(args.epochs), "--output", str(iter_dir)], check=True)
        pt = next(iter_dir.glob("gnn_n*.pt"))
        npz = iter_dir / (pt.stem + ".npz")
        subprocess.run([python, str(ROOT / "scripts" / "export_numpy_checkpoint.py"), "--input", str(pt), "--output", str(npz)], check=True)
        policy_fn = load_numpy_policy(str(npz))
        policy_path = npz
        score = evaluate(args.robots[0], seeds_eval, policy_fn, options, args.max_time)
        history.append({"iterazione": k, "checkpoint": str(npz), **score})
        print(f"[dagger] iterazione {k}: {score}", flush=True)

    best = min(history, key=lambda h: (h["contatti"], -h["success"], h["quasi"], h["time_s"]))
    final = ROOT / "results" / "checkpoints_warehouse" / "gnn_dagger.npz"
    shutil.copy(best["checkpoint"], final)
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "dagger_storia.csv", index=False)
    print("\n[dagger] riepilogo sui seed di valutazione (mai visti in addestramento):")
    print(df.round(3).to_string(index=False))
    print(f"\n[dagger] migliore: iterazione {best['iterazione']} -> copiato in {final}")
    print("          usalo con: --checkpoint results/checkpoints_warehouse/gnn_dagger.npz (valida_2d, anima_2d, genera_mondo_webots)")


if __name__ == "__main__":
    main()
