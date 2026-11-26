#!/usr/bin/env python3
"""Confronto 2D <-> Webots sugli stessi seed: tabella (csv + markdown) e figura.

Legge results/webots_log.csv (scritto dal supervisore), riesegue nel simulatore 2D ogni
episodio con le stesse opzioni e lo stesso seed (core/simulation2d.py) e confronta le metriche
comuni (core/metrics.py). Scrive:
  results/confronto_2d_webots.csv        una riga per (seed, simulatore)
  results/confronto_2d_webots.md         tabella riassuntiva pronta per la tesi
  figures/generated/confronto_2d_webots.png

Uso: python scripts/evaluate_webots_vs_2d.py [--webots-log results/webots_log.csv] [--filtro-robot olonomo]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.controller import ControlOptions  # noqa: E402
from core.numpy_policy import load_numpy_policy  # noqa: E402
from core.simulation2d import rollout_warehouse  # noqa: E402

METRICS = ["success", "arrived", "time_s", "path_length", "min_separation", "contact_events",
           "near_miss_events", "shelf_contact_max_s", "wait_time_s", "reverse_events", "forced_entries"]
LABELS = {"success": "successo", "arrived": "robot arrivati", "time_s": "tempo [s]", "path_length": "percorso totale [m]",
          "min_separation": "separazione minima [m]", "contact_events": "contatti", "near_miss_events": "quasi-contatti",
          "shelf_contact_max_s": "contatto scaffale max [s]", "wait_time_s": "attesa zone [s]",
          "reverse_events": "retromarce", "forced_entries": "ingressi forzati"}


def to_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "si", "yes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webots-log", default=str(ROOT / "results" / "webots_log.csv"))
    parser.add_argument("--filtro-robot", default="", help="usa solo le righe con questo tipo di robot (olonomo/pioneer)")
    parser.add_argument("--output", default=str(ROOT / "results" / "confronto_2d_webots"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "generated" / "confronto_2d_webots.png"))
    args = parser.parse_args()
    web = pd.read_csv(args.webots_log)
    required = ["robot", "zone", "filtro", "repulsione", "retromarcia", "corsie", "scaglionato", "shelf_contact_max_s"]
    missing = [c for c in required if c not in web.columns]
    if missing:
        raise SystemExit(
            f"{args.webots_log} e' nel formato vecchio (mancano le colonne {missing}).\n"
            "E' il log del supervisore precedente: esegui prima la simulazione nuova in Webots\n"
            "(apri webots/worlds/warehouse_gnn.wbt e premi play): il supervisore nuovo rinomina il\n"
            "log vecchio in webots_log_vecchio.csv e ne scrive uno con le colonne giuste.")
    if args.filtro_robot:
        web = web[web["robot"] == args.filtro_robot]
    if web.empty:
        raise SystemExit("nessuna riga nel log Webots")
    web = web.drop_duplicates(subset=["seed", "robots", "policy", "robot", "filtro", "zone", "corsie"], keep="last")
    policies = {}
    rows = []
    for _, r in web.iterrows():
        opt = ControlOptions(policy=r["policy"], con_corsie=to_bool(r["corsie"]), zone=to_bool(r["zone"]),
                             repulsione=to_bool(r["repulsione"]), filtro=str(r["filtro"]),
                             retromarcia=to_bool(r["retromarcia"]), scaglionato=float(r["scaglionato"]))
        if opt.policy != "esperto" and opt.policy not in policies:
            policies[opt.policy] = load_numpy_policy(str(ROOT / "results" / "checkpoints_warehouse" / f"{opt.policy}_n8.npz"))
        sim = rollout_warehouse(int(r["robots"]), int(r["seed"]), opt, policies.get(opt.policy), max_time=90.0)
        sim["simulatore"] = "2D"
        wrow = {k: r[k] for k in METRICS if k in r}
        wrow.update({"seed": int(r["seed"]), "robots": int(r["robots"]), "policy": r["policy"], "simulatore": f"Webots ({r['robot']})"})
        sim = {k: sim[k] for k in METRICS + ["seed", "robots", "policy", "simulatore"]}
        rows += [sim, wrow]
        print(f"seed {int(r['seed'])}: 2D t={sim['time_s']:.1f}s contatti={sim['contact_events']} | "
              f"Webots t={float(r['time_s']):.1f}s contatti={int(r['contact_events'])}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    agg = df.groupby("simulatore")[METRICS].agg(["mean", "min", "max"])
    summary = pd.DataFrame({"metrica": [LABELS[m] for m in METRICS]})
    for sim_name in df["simulatore"].unique():
        summary[f"{sim_name} media"] = [agg.loc[sim_name, (m, "mean")] for m in METRICS]
        summary[f"{sim_name} min-max"] = [f"{agg.loc[sim_name, (m, 'min')]:.2f} – {agg.loc[sim_name, (m, 'max')]:.2f}" for m in METRICS]
    sims = list(df["simulatore"].unique())
    if len(sims) == 2:
        a, b = sims
        summary["differenza Webots-2D (%)"] = [
            (100 * (agg.loc[b, (m, "mean")] - agg.loc[a, (m, "mean")]) / agg.loc[a, (m, "mean")]) if agg.loc[a, (m, "mean")] else np.nan
            for m in METRICS]
    md = ["# Confronto simulatore 2D ↔ Webots", "",
          f"Episodi: {df['seed'].nunique()} seed ({df['seed'].min()}–{df['seed'].max()}), {int(df['robots'].iloc[0])} robot, policy {df['policy'].iloc[0]}.", "",
          summary.round(2).to_markdown(index=False), "",
          "Criteri dell'obiettivo: successo ≥ 95 %, contatti = 0, separazione minima ≥ 0,60 m, contatto scaffale ≤ 2 s, "
          "tempo e percorso Webots entro +30 % del 2D."]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    # figura: per-seed, 2D vs Webots
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m in zip(axs, ["time_s", "path_length", "min_separation", "contact_events"]):
        for sim_name, marker in zip(sims, ("o", "s")):
            d = df[df["simulatore"] == sim_name].sort_values("seed")
            ax.plot(d["seed"], d[m], marker=marker, lw=1, label=sim_name)
        ax.set_title(LABELS[m]); ax.set_xlabel("seed"); ax.grid(alpha=0.3)
        if m == "min_separation":
            ax.axhline(0.60, color="r", ls="--", lw=1, label="soglia 0,60 m")
        if m == "contact_events":
            ax.axhline(0, color="r", ls="--", lw=1)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, dpi=140)
    print(summary.round(2).to_string(index=False))
    print(f"\nscritti {out.with_suffix('.csv')}, {out.with_suffix('.md')}, {args.figure}")


if __name__ == "__main__":
    main()
