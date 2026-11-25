#!/bin/bash
# Addestramento + esperimenti 2D + riepilogo per la tesi.
# Da lanciare in Git Bash dalla cartella Progetto_Tesi-GNN:
#     bash esegui_tutto.sh
# Per usare modelli già presenti in results/checkpoints_warehouse senza riaddestrare:
#     SALTA_ADDESTRAMENTO=1 bash esegui_tutto.sh

DATI="data/generated/orca_warehouse_n6.npz data/generated/orca_warehouse_n12.npz data/generated/orca_warehouse_n18.npz"
CK=results/checkpoints_warehouse

# controlli iniziali
for f in $DATI scripts/riepilogo_2d.py; do
  [ -f "$f" ] || { echo "ERRORE: manca $f"; exit 1; }
done

if [ -z "$SALTA_ADDESTRAMENTO" ]; then
  echo "== 1. Addestramento ed esportazione"
  python scripts/train_policy.py --method gnn --data $DATI --output $CK || exit 1
  python scripts/train_policy.py --method mlp --data $DATI --output $CK || exit 1
  python scripts/export_numpy_checkpoint.py --input $CK/gnn_n18.pt --output $CK/gnn_n18.npz || exit 1
  python scripts/export_numpy_checkpoint.py --input $CK/mlp_n18.pt --output $CK/mlp_n18.npz || exit 1
fi
for f in $CK/gnn_n18.npz $CK/mlp_n18.npz; do
  [ -f "$f" ] || { echo "ERRORE: manca $f"; exit 1; }
done

echo "== 2. Esperimenti 2D: 6 configurazioni x 20 scenari, GNN e MLP, N = 6, 12, 18"
mkdir -p results/val18
for N in 6 12 18; do
  for P in gnn mlp; do
    python scripts/valida_2d.py --robots $N --seeds 1000-1019 --policy $P \
      --checkpoint $CK/${P}_n18.npz --output results/val18/val_${P}_n${N}.csv \
      > results/val18/log_${P}_n${N}.txt 2>&1 &
  done
  wait
  echo "   finito N = $N (log in results/val18/log_*_n$N.txt)"
done

echo "== 3. Riepilogo per le tabelle"
python scripts/riepilogo_2d.py
