# Navigazione cooperativa di AMR in un magazzino automatizzato con Graph Neural Network

Codice della parte sperimentale della tesi triennale *Graph Neural Networks per sistemi multi-robot:
applicazione e valutazione nella navigazione cooperativa in un magazzino automatizzato*
(Sapienza Università di Roma, Ingegneria Informatica e Automatica).

## Cosa contiene

Una flotta di AMR (6-24) attraversa un magazzino 15 × 10 m con scaffali, corsie a senso unico e
piazzole di prelievo/consegna. Il percorso di ogni robot viene da A\*; la velocità istantanea la
decide una **Graph Neural Network** che usa solo informazioni locali (vicini entro 3,4 m), addestrata
per imitazione da un esperto ORCA e migliorata con DAgger; strati di sicurezza indipendenti dalla
policy (evitamento scaffali, prenotazione degli incroci, filtro d'emergenza) rendono il sistema
eseguibile in Webots con un robot olonomo e fisica reale.

```
core/        geometria (warehouse), pianificazione (planning), sicurezza (safety), controllore di
             flotta (controller), metriche (metrics), GNN e feature (environment, models, numpy_policy)
scripts/     generate_dataset, train_policy, export_numpy_checkpoint, dagger   -> addestramento
             valida_2d, anima_2d, disegna_mappa                                -> simulatore 2D
             genera_mondo_webots, evaluate_webots_vs_2d                        -> Webots e confronto
webots/      mondo .wbt generato, controller del supervisore, guida (LEGGIMI_WEBOTS.md)
tests/       test di geometria, regole e supervisore (girano senza Webots)
results/     modelli (.pt/.npz), validazione 2D, log Webots, confronto 2D <-> Webots
docs/        regole_rifacimento_webots.md: regole, dimensioni e motivazioni
```

## Riproduzione rapida

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/valida_2d.py --robots 6 --seeds 1000-1019          # validazione 2D
python scripts/anima_2d.py --robots 12                            # animazione
python scripts/genera_mondo_webots.py --robots 12 --episodes 20   # poi apri webots/worlds/warehouse_gnn.wbt
python scripts/evaluate_webots_vs_2d.py                           # tabella 2D <-> Webots
python scripts/dagger.py --robots 12 --iterazioni 3               # miglioramento iterativo (serve torch)
```

La guida completa per Webots è in `webots/LEGGIMI_WEBOTS.md`; le regole e le dimensioni con le
motivazioni in `docs/regole_rifacimento_webots.md`.

## Attribuzione

I modelli (message passing GNN, MLP, attention, transformer in `core/models.py`), il simulatore 2D di
base, l'esperto ORCA/RVO e le procedure di addestramento (`generate_dataset.py`, `train_policy.py`)
derivano dal repository pubblico *Graph Neural Network-Based Cooperative Path Planning and Collision
Avoidance for Multi-Robot Systems* vedi `LICENSE`; completare con autori e URL:
Xuelin Hu, multi-robot-gnn-cooperative-navigation, https://github.com/XuelinHu/multi-robot-gnn-cooperative-navigation (consultato il 29/10/2026).

Il magazzino parametrico, la pianificazione con corsie rigide e goal in linea di vista, gli strati di
sicurezza, il controllore di flotta condiviso 2D/Webots, le metriche comuni, l'integrazione Webots con
robot olonomo, DAgger e la validazione sono il contributo di questa tesi.
