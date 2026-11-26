#!/usr/bin/env python3
"""Grafici dei risultati 2D (GNN contro MLP) a partire dai CSV di valida_2d.py.

Legge <cartella>/val_{gnn,mlp}_n{N}.csv e scrive in --output tre figure (PNG e PDF):
  confronto_completa   configurazione completa: scenari completati, tempo medio dei soli
                       episodi completati, retromarce per episodio, in funzione di N
  completamento        scenari completati [%] nelle sei configurazioni, un pannello per N
  contatti             contatti tra robot per episodio nelle sei configurazioni, un pannello per N
La MLP ha le barre tratteggiate, cosi' le figure si leggono anche in bianco e nero.

Uso: python scripts/grafici_2d.py --cartella results/val8 --robots 4 6 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ["completa", "senza_zone", "senza_corsie", "filtro_simmetrico", "solo_policy", "solo_comunicazione"]
NOMI = {"completa": "completa", "senza_zone": "senza zone", "senza_corsie": "senza corsie",
        "filtro_simmetrico": "filtro simmetrico", "solo_policy": "senza sicurezza",
        "solo_comunicazione": "solo comunicazione"}
STILE = {"gnn": dict(color="#b03a2e", edgecolor="#b03a2e", label="GNN"),
         "mlp": dict(color="white", edgecolor="#1f5fa8", hatch="////", label="MLP")}


def carica(cartella: Path, robots: list[int]) -> pd.DataFrame:
    parti = []
    for politica in ["gnn", "mlp"]:
        for n in robots:
            d = pd.read_csv(cartella / f"val_{politica}_n{n}.csv")
            d["politica"], d["N"] = politica, n
            parti.append(d)
    return pd.concat(parti)


def stile_assi(ax) -> None:
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    ax.set_axisbelow(True)


def salva(fig, output: Path, nome: str) -> None:
    for est in ("png", "pdf"):
        fig.savefig(output / f"{nome}.{est}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"scritto {output / nome}.png e .pdf")


def confronto_completa(df: pd.DataFrame, robots: list[int], output: Path) -> None:
    d = df[df.config == "completa"]
    misure = [("Scenari completati [%]", lambda g: 100 * g.success.mean(), 0),
              ("Tempo medio [s]", lambda g: g[g.success == 1].time_s.mean(), 1),
              ("Retromarce per episodio", lambda g: g.reverse_events.mean(), 1)]
    fig, assi = plt.subplots(1, 3, figsize=(10, 3.3))
    x = np.arange(len(robots)); w = 0.36
    for ax, (titolo, f, decimali) in zip(assi, misure):
        for k, politica in enumerate(["gnn", "mlp"]):
            valori = [f(d[(d.politica == politica) & (d.N == n)]) for n in robots]
            barre = ax.bar(x + (k - 0.5) * (w + 0.02), valori, w, linewidth=1, **STILE[politica])
            for b, v in zip(barre, valori):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.{decimali}f}".replace(".", ","),
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [str(n) for n in robots]); ax.set_xlabel("Numero di robot N")
        ax.set_title(titolo, fontsize=10); ax.yaxis.grid(True, color="#dddddd"); stile_assi(ax)
    assi[0].set_ylim(0, 115)
    gestori, etichette = assi[0].get_legend_handles_labels()
    fig.legend(gestori, etichette, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    salva(fig, output, "confronto_completa")


def per_configurazione(df: pd.DataFrame, robots: list[int], output: Path, nome: str, titolo: str, f) -> None:
    # stessa scala orizzontale nei tre pannelli, cosi' si vede l'effetto del numero di robot
    fig, assi = plt.subplots(1, len(robots), figsize=(10, 3.6), sharey=True, sharex=True)
    y = np.arange(len(CONFIG)); h = 0.38
    for ax, n in zip(np.atleast_1d(assi), robots):
        for k, politica in enumerate(["gnn", "mlp"]):
            valori = [f(df[(df.config == c) & (df.politica == politica) & (df.N == n)]) for c in CONFIG]
            ax.barh(y + (k - 0.5) * (h + 0.02), valori, h, linewidth=1, **STILE[politica])
        ax.set_title(f"N = {n}", fontsize=10); ax.invert_yaxis()
        ax.xaxis.grid(True, color="#dddddd"); stile_assi(ax)
    np.atleast_1d(assi)[0].set_yticks(y, [NOMI[c] for c in CONFIG])
    np.atleast_1d(assi)[len(robots) // 2].set_xlabel(titolo)
    gestori, etichette = np.atleast_1d(assi)[0].get_legend_handles_labels()
    fig.legend(gestori, etichette, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salva(fig, output, nome)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartella", default=str(ROOT / "results" / "val8"))
    parser.add_argument("--robots", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--output", default=str(ROOT / "figures" / "generated"))
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    df = carica(Path(args.cartella), args.robots)
    confronto_completa(df, args.robots, output)
    per_configurazione(df, args.robots, output, "completamento", "Scenari completati [%]",
                       lambda g: 100 * g.success.mean())
    per_configurazione(df, args.robots, output, "contatti", "Contatti per episodio",
                       lambda g: g.contact_events.mean())


if __name__ == "__main__":
    main()
