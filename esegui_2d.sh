#!/bin/bash
# Esperimenti 2D della tesi (Tabelle 4.3, 4.4, 4.5 e colonne 2D della 4.6).
# Modelli addestrati con 4, 6 e 8 robot (results/checkpoints_warehouse/gnn_n8.npz, mlp_n8.npz),
# prove con N = 4, 6, 8: 6 configurazioni x 20 scenari (semi 1000-1019) per ciascuna rete.
# Nessun addestramento. Risultati in results/val8/.
# Da lanciare in Git Bash (o in un terminale Linux/macOS) dalla cartella del progetto:
#     bash esegui_2d.sh
# Con altri modelli (per esempio riaddestrati in results/checkpoints_nuovi):
#     CK=results/checkpoints_nuovi OUT=results/val8_nuovi bash esegui_2d.sh
CK=${CK:-results/checkpoints_warehouse}
OUT=${OUT:-results/val8}
for f in $CK/gnn_n8.npz $CK/mlp_n8.npz scripts/riepilogo_2d.py; do
  [ -f "$f" ] || { echo "ERRORE: manca $f"; exit 1; }
done
mkdir -p $OUT
for N in 4 6 8; do
  for P in gnn mlp; do
    python scripts/valida_2d.py --robots $N --seeds 1000-1019 --policy $P \
      --checkpoint $CK/${P}_n8.npz --output $OUT/val_${P}_n${N}.csv \
      > $OUT/log_${P}_n${N}.txt 2>&1 &
  done
  wait
  echo "   finito N = $N"
done
python scripts/riepilogo_2d.py --cartella $OUT --robots 4 6 8
