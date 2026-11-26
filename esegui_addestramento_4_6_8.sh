#!/bin/bash
# NUOVO ADDESTRAMENTO con 4, 6 e 8 robot, poi gli esperimenti 2D e il riepilogo delle tabelle.
# Non tocca i modelli della tesi: i dati vanno in data/nuovi_4_6_8/, i modelli in
# results/checkpoints_nuovi/ e i risultati in results/val8_nuovi/.
# Stessi parametri dei modelli della tesi (esperto "orca" = campi di potenziale, magazzino,
# 30 epoche, hidden 128, 3 livelli, seme 0: i valori predefiniti di train_policy.py).
# ATTENZIONE: i dati rigenerati non sono identici a quelli originali, quindi i nuovi modelli
# e i nuovi risultati non coincidono con quelli della tesi.
# Da lanciare in Git Bash dalla cartella del progetto:
#     bash esegui_addestramento_4_6_8.sh
# Per rifare solo gli esperimenti con i modelli nuovi gia' addestrati:
#     SALTA_ADDESTRAMENTO=1 bash esegui_addestramento_4_6_8.sh

DIR_DATI=data/nuovi_4_6_8
CK=results/checkpoints_nuovi
OUT=results/val8_nuovi
DATI="$DIR_DATI/orca_warehouse_n4.npz $DIR_DATI/orca_warehouse_n6.npz $DIR_DATI/orca_warehouse_n8.npz"

if [ -z "$SALTA_ADDESTRAMENTO" ]; then
  echo "== 1. Dati dell'esperto (4, 6, 8 robot)"
  python scripts/generate_dataset.py --robots 4 6 8 --layout warehouse --expert orca --output $DIR_DATI || exit 1
  echo "== 2. Addestramento GNN e MLP ed esportazione in NumPy"
  python scripts/train_policy.py --method gnn --data $DATI --output $CK || exit 1
  python scripts/train_policy.py --method mlp --data $DATI --output $CK || exit 1
  python scripts/export_numpy_checkpoint.py --input $CK/gnn_n8.pt --output $CK/gnn_n8.npz || exit 1
  python scripts/export_numpy_checkpoint.py --input $CK/mlp_n8.pt --output $CK/mlp_n8.npz || exit 1
fi

echo "== 3. Esperimenti 2D (720 episodi) e riepilogo"
CK=$CK OUT=$OUT bash esegui_2d.sh
