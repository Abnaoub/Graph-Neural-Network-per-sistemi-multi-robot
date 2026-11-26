#!/bin/bash
# VERSIONE ALTERNATIVA (non usata nella tesi): addestramento su 6, 12 e 18 robot,
# esperimenti 2D (6 configurazioni x 20 scenari, N = 6, 12, 18) e riepilogo.
# Richiede data/generated/orca_warehouse_n6.npz, n12.npz, n18.npz.
# Modelli e risultati vanno in results/versione_6_12_18/.
# Da lanciare in Git Bash dalla cartella del progetto:
#     bash esegui_versione_6_12_18.sh
# Per usare i modelli gia' presenti senza riaddestrare:
#     SALTA_ADDESTRAMENTO=1 bash esegui_versione_6_12_18.sh

DATI="data/generated/orca_warehouse_n6.npz data/generated/orca_warehouse_n12.npz data/generated/orca_warehouse_n18.npz"
CK=results/versione_6_12_18/checkpoints
OUT=results/versione_6_12_18/val18

for f in $DATI; do
  [ -f "$f" ] || { echo "ERRORE: manca $f"; exit 1; }
done

if [ -z "$SALTA_ADDESTRAMENTO" ]; then
  echo "== 1. Addestramento ed esportazione"
  python scripts/train_policy.py --method gnn --data $DATI --output $CK || exit 1
  python scripts/train_policy.py --method mlp --data $DATI --output $CK || exit 1
  python scripts/export_numpy_checkpoint.py --input $CK/gnn_n18.pt --output $CK/gnn_n18.npz || exit 1
  python scripts/export_numpy_checkpoint.py --input $CK/mlp_n18.pt --output $CK/mlp_n18.npz || exit 1
fi

echo "== 2. Esperimenti 2D"
mkdir -p $OUT
for N in 6 12 18; do
  for P in gnn mlp; do
    python scripts/valida_2d.py --robots $N --seeds 1000-1019 --policy $P \
      --checkpoint $CK/${P}_n18.npz --output $OUT/val_${P}_n${N}.csv \
      > $OUT/log_${P}_n${N}.txt 2>&1 &
  done
  wait
  echo "   finito N = $N"
done

echo "== 3. Riepilogo"
python scripts/riepilogo_2d.py --cartella $OUT --robots 6 12 18
