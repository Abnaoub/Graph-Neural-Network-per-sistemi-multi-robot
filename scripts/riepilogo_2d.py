#!/usr/bin/env python3
"""Riepilogo dei risultati 2D per le tabelle della tesi.

Legge i file <cartella>/val_{gnn,mlp}_n{N}.csv scritti da valida_2d.py e stampa:
  - per la configurazione completa: completamento, robot arrivati, tempo (sui soli episodi
    completati), contatti, distanza minima e retromarce (Tabella tab:confronto);
  - per tutte le configurazioni: completamento e contatti per episodio (Tabelle tab:ablazione
    e tab:ablazione_contatti).
Scrive anche <cartella>/riepilogo.csv con tutte le metriche.

Uso: python scripts/riepilogo_2d.py --cartella results/val8 --robots 4 6 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ["completa", "senza_zone", "senza_corsie", "filtro_simmetrico", "solo_policy", "solo_comunicazione"]


def riassumi(g: pd.DataFrame) -> pd.Series:
    completati = g[g.success == 1]
    return pd.Series({
        "episodi": len(g),
        "completamento_%": 100 * g.success.mean(),
        "arrivati_%": 100 * (g.arrived / g.n_robot).mean(),
        "tempo_s": completati.time_s.mean() if len(completati) else float("nan"),
        "contatti": g.contact_events.mean(),
        "contatti_per_robot": (g.contact_events / g.n_robot).mean(),
        "dist_min_media_m": g.min_separation.mean(),
        "dist_min_peggiore_m": g.min_separation.min(),
        "attesa_s": g.wait_time_s.mean(),
        "retromarce": g.reverse_events.mean(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartella", default=str(ROOT / "results" / "val8"))
    parser.add_argument("--robots", type=int, nargs="+", default=[4, 6, 8])
    args = parser.parse_args()
    cartella = Path(args.cartella)
    parti = []
    for politica in ["gnn", "mlp"]:
        for n in args.robots:
            f = cartella / f"val_{politica}_n{n}.csv"
            if f.exists():
                d = pd.read_csv(f)
                d["politica"], d["N"], d["n_robot"] = politica, n, n
                parti.append(d)
            else:
                print(f"manca {f}")
    df = pd.concat(parti)
    trovate = sorted(df.config.unique())
    mancanti = [c for c in CONFIG if c not in trovate]
    if mancanti:
        print(f"ATTENZIONE: nei file mancano le configurazioni {mancanti}; presenti: {trovate}")
    r = df.groupby(["config", "N", "politica"]).apply(riassumi)
    r.to_csv(cartella / "riepilogo.csv")
    pd.set_option("display.width", 200)

    print("\n=== Configurazione completa (tab:confronto) ===")
    print(r.loc["completa", ["completamento_%", "arrivati_%", "tempo_s", "contatti", "dist_min_media_m",
                             "retromarce"]].round(2).to_string())
    for col, nome in [("completamento_%", "Completamento [%] (tab:ablazione)"),
                      ("contatti", "Contatti per episodio (tab:ablazione_contatti)")]:
        print(f"\n=== {nome} ===")
        t = r[col].unstack(["politica", "N"]).reindex(CONFIG)
        print(t.round(2).to_string())
    print(f"\nscritto {cartella / 'riepilogo.csv'}")


if __name__ == "__main__":
    main()
