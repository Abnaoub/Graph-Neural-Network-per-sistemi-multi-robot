#!/bin/bash
# Le 6 esecuzioni in Webots della Tabella 4.6 (configurazione completa, 20 scenari, semi 1000-1019):
# N = 4, 6, 8 per GNN e MLP, con i modelli results/checkpoints_warehouse/{gnn,mlp}_n8.npz.
# Versione SENZA GRAFICA per Linux (richiede Webots R2025a e xvfb-run). Su Windows si fanno
# le stesse esecuzioni a mano: genera_mondo_webots.py, poi apri il mondo in Webots in modalita veloce.
# Da lanciare dalla cartella del progetto:   WEBOTS_HOME=/percorso/webots bash scripts/esegui_webots_headless.sh
WEBOTS=${WEBOTS_HOME:-/opt/webots}/webots
mkdir -p results/webots_runs
[ -f results/webots_log.csv ] && mv results/webots_log.csv results/webots_log_precedente.csv
for N in 4 6 8; do
  for P in gnn mlp; do
    python3 scripts/genera_mondo_webots.py --robots $N --policy $P --episodes 20 > /dev/null
    L=results/webots_runs/webots_${P}_n${N}.txt
    timeout 3000 xvfb-run -a "$WEBOTS" --batch --mode=fast --no-rendering --stdout --stderr --minimize \
      webots/worlds/warehouse_gnn.wbt > $L 2>&1 &
    WP=$!
    # il supervisore mette in pausa la simulazione alla fine: si chiude Webots quando compare "fine:"
    until grep -q "fine:\|Traceback" $L 2>/dev/null; do sleep 2; kill -0 $WP 2>/dev/null || break; done
    sleep 2; pkill -f webots-bin; wait $WP 2>/dev/null
    echo "webots $P N=$N: $(grep 'fine:' $L | sed 's/.*fine: //')"
  done
done
echo "righe nel log: $(($(wc -l < results/webots_log.csv) - 1))"
