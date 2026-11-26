#!/bin/bash
# Grafici delle tabelle e animazioni 2D per 4, 6 e 8 robot, GNN e MLP.
# Uscita in figures/generated/ (grafici .png e .pdf, animazioni .gif o .mp4).
# Da lanciare in Git Bash dalla cartella del progetto, dopo gli esperimenti:
#     bash esegui_figure_2d.sh                            # risultati e modelli della tesi
#     FORMATO=mp4 bash esegui_figure_2d.sh                # video invece dei GIF (serve ffmpeg)
#     CONFIG=solo_comunicazione bash esegui_figure_2d.sh  # animazioni in un'altra configurazione
#     CK=results/checkpoints_nuovi OUT=results/val8_nuovi FIG=figures/nuovi bash esegui_figure_2d.sh
CK=${CK:-results/checkpoints_warehouse}
OUT=${OUT:-results/val8}
FIG=${FIG:-figures/generated}
FORMATO=${FORMATO:-gif}
CONFIG=${CONFIG:-completa}
SEED=${SEED:-1000}

echo "== Grafici dai risultati in $OUT"
python scripts/grafici_2d.py --cartella $OUT --robots 4 6 8 --output $FIG || exit 1

echo "== Animazioni (configurazione $CONFIG, scenario $SEED)"
for N in 4 6 8; do
  for P in gnn mlp; do
    python scripts/anima_2d.py --robots $N --policy $P --config $CONFIG --seed $SEED \
      --checkpoint $CK/${P}_n8.npz --gif $FIG/${P}_n${N}_${CONFIG}.$FORMATO || exit 1
  done
done
