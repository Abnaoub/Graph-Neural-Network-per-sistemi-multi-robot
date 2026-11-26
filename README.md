# GNN per la navigazione cooperativa in un magazzino automatizzato

Codice della tesi triennale *Graph Neural Network per sistemi multi-robot* (Sapienza Università
di Roma, relatore Prof. Antonio Pietrabissa).

Una flotta di robot attraversa un magazzino con scaffali, corsie a senso unico e incroci. La
velocità di ogni robot è decisa da una **GNN**, che comunica con i robot vicini, addestrata
imitando un esperto; come confronto c'è una **MLP**, senza comunicazione. Le prove si fanno in
un simulatore 2D e in Webots, con 4, 6 e 8 robot.

I comandi si lanciano in **Git Bash** dalla cartella del progetto. Tutti i comandi, con le
opzioni, sono in **`LEGGIMI_COMANDI.md`**.

## Installazione

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## Risultati della tesi

```bash
bash esegui_2d.sh            # simulatore 2D: Tabelle 4.3, 4.4, 4.5 e colonne 2D della 4.6
bash esegui_figure_2d.sh     # grafici delle tabelle e 6 GIF
```

Usano i modelli della tesi (`results/checkpoints_warehouse/`). Le colonne Webots della
Tabella 4.6 si ottengono con Webots R2025a

Un'animazione singola:

```bash
python scripts/anima_2d.py --robots 6 --policy gnn --config completa --gif figures/generated/gnn_n6.gif
```

## Nuovo addestramento (facoltativo)

```bash
bash esegui_addestramento_4_6_8.sh
```

Raccoglie i dati dall'esperto, addestra GNN e MLP e rifà gli esperimenti, in cartelle separate
(`data/nuovi_4_6_8/`, `results/checkpoints_nuovi/`, `results/val8_nuovi/`). I dati originali dei
modelli della tesi non sono disponibili: i nuovi modelli danno risultati diversi da quelli della tesi.

## Cartelle

| Cartella | Contenuto |
|---|---|
| `core/` | magazzino, reti, controllore, strati di sicurezza, simulatore 2D, metriche |
| `scripts/` | dati, addestramento, esperimenti, grafici, animazioni, Webots |
| `webots/` | mondo e controller di Webots |
| `results/checkpoints_warehouse/` | modelli della tesi |
| `results/val8/` | risultati 2D della tesi |
| `results/webots_log.csv` | risultati Webots della tesi |
| `results/versione_6_12_18/` | prova con 6, 12, 18 robot, non usata nella tesi |

