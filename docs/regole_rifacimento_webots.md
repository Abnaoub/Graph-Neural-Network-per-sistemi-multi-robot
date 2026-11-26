# Rifacimento della simulazione Webots: regole eliminate, regole aggiunte, dimensioni

Una pagina riutilizzabile nel capitolo dei risultati. Tutto quello che segue vale sia per il
simulatore 2D (`core/simulation2d.py`) sia per Webots (`webots/controllers/gnn_supervisor`):
i due usano lo stesso controllore (`core/controller.py`) e le stesse metriche (`core/metrics.py`).

## Perché la prima versione in Webots andava male (diagnosi)

| sintomo in Webots | causa |
|---|---|
| robot che spingono contro gli scaffali | i varchi tra le file erano larghi 0,6–0,7 m (robot: 0,56 m); il goal locale della policy stava oltre lo spigolo dello scaffale; nessuno strato evitava gli ostacoli statici (la GNN vede solo l'ostacolo più vicino) |
| incontri frontali nei varchi | il pilastro a (−2,6, 3,5) chiudeva il corridoio superiore (0,34 m dallo scaffale), quindi i robot diretti a destra non avevano alcun passaggio nella loro corsia e andavano contromano; le corsie erano solo una penalità di costo in A* |
| stalli faccia a faccia / ingorghi | filtro di sicurezza simmetrico (entrambi si fermano) e poi priorità fissa per indice (code che non si sbloccano) |
| contatti tra robot (min 0,30 m) | la policy olonoma (vx, vy) veniva eseguita da un Pioneer differenziale: ritardo di orientamento e sbandate, dinamica mai vista in addestramento |

## Regole eliminate

1. **Pilastro nel corridoio** e qualsiasi passaggio obbligato più stretto di 1,0 m (test `test_passaggi_larghi_almeno_un_metro`).
2. **Varchi da 0,6–0,7 m**: la mappa non ne ha più.
3. **Penalità "soft" delle corsie** (costo ×9): le corsie sono un vincolo rigido di A* (celle vietate).
4. **Priorità fissa per indice** come meccanismo principale: resta solo come ultimo criterio di spareggio.
5. **Lookahead cieco**: il goal locale è sempre in linea di vista (`WaypointTracker`, `test_goal_locale_in_linea_di_vista`).
6. **Fattore di velocità** `--velocita`: riduceva gli arrivi senza ridurre i contatti.
7. **Retromarcia a tempo** come meccanismo di coordinamento: resta solo come rete di sicurezza finale (conteggiata in `reverse_events` nel log).

## Regole aggiunte

| regola | dove | perché |
|---|---|---|
| **A. Mappa dimensionata sul robot** (15 × 10 m): passaggi tra le file 1,8 m, corridoi verticali 2,1 m, corridoi lungo i muri 2,3 m, piazzole a 1,0 m tra loro, 1,0 m dai muri, 1,5 m dagli scaffali; pilastro in un angolo | `core/warehouse.py` | pratica AGV / ISO 3691-4: almeno 0,5 m di franco per lato del veicolo (0,56 m) ⇒ corsia singola ≥ 1,56 m, doppia ≥ 2,62 m. Le piazzole a 1,5 m dagli scaffali evitano che un robot parcheggiato chiuda l'uscita di un passaggio |
| **B. A\* su griglia 0,1 m, inflazione 0,43 m, corsie rigide, percorso calcolato una volta per episodio e smussato, goal locale in linea di vista** | `core/warehouse.py` / `core/planning.py` | ogni percorso resta a ≥ 0,43 m dagli ostacoli (`test_percorsi_lontani_dagli_ostacoli_e_nelle_corsie`); la linea di separazione delle corsie passa per il centro della fila centrale, così ogni corsia ha due passaggi e nessun passaggio è "a cavallo" della linea |
| **C. Evitamento degli ostacoli statici** (`obstacle_repulsion`): entro 0,45 m dal bordo si annulla la componente di velocità verso l'ostacolo, sotto raggio+0,08 m spinta verso l'esterno | `core/safety.py` | analogo del campo di protezione del lidar di sicurezza; indipendente dalla policy (nessun riaddestramento). Contatto con scaffali: da 741 passi/episodio (12 robot, mappa vecchia) a 0 |
| **D. Prenotazione delle zone critiche** (`ZoneManager`): gli incroci tra corridoi verticali e passaggi; attraversamenti classificati H (lungo il passaggio) o V (cambio passaggio); classi diverse non coesistono; concessione a chi è più vicino alla zona, poi indice; antistallo dopo 12 s (`forced_entries`) | `core/safety.py` | rende impossibili gli incroci dentro la zona; i convogli nello stesso senso non vengono fermati (ci pensa la GNN). La precedenza "a chi aspetta da più tempo" è stata provata e scartata: chi aspetta da più tempo ma sta dietro un altro robot bloccava la coda |
| **E. Filtro d'emergenza tra robot** (`emergency_filter`): sotto 0,70 m chi si avvicina perde la parte di velocità che eccede quella dell'altro (chi segue adegua la velocità, i frontali si fermano), sotto 0,66 m il meno prioritario si ferma, sotto 0,62 m spinta di allontanamento; priorità: dentro una zona > in attesa > indice; i robot parcheggiati sono trattati come ostacoli fermi | `core/safety.py` | ultima istanza, asimmetrico: niente stalli faccia a faccia. Risultato: separazione minima ≥ 0,60 m in tutti gli episodi |
| **F. Robot olonomo in Webots** (F.1): cilindro r 0,28 m, h 0,22 m, massa 20 kg, materiale `agv` senza attrito col pavimento; il supervisore impone (vx, vy) con `setVelocity` a ogni passo fisico; la fisica di Webots gestisce gli urti con scaffali, muri e robot | `scripts/genera_mondo_webots.py`, `gnn_supervisor.py` | la policy (vx, vy) viene eseguita esattamente come nel 2D: nessun riaddestramento, le feature non cambiano. Il Pioneer resta disponibile con `--robot pioneer` |
| **G. Partenze simultanee** | `ControlOptions.scaglionato = 0` | le piazzole distano 1,0 m, non serve scaglionare (resta come opzione) |
| **H. Metriche identiche 2D/Webots** (`EpisodeMetrics`) | `core/metrics.py` | un'unica tabella di confronto (`scripts/evaluate_webots_vs_2d.py`) |
| **I. Attracco alla piazzola** (`goal_docking`): nell'ultimo metro velocità minima 0,25 m/s verso la piazzola | `core/controller.py` | la policy appresa frena troppo vicino al goal (eredità dell'esperto) e si fermava a 0,4–0,5 m senza entrare nei 0,30 m richiesti; i filtri a valle restano attivi, quindi l'attracco non può causare contatti |

## Dimensioni (tutte in `core/warehouse.py`, con il "perché" nei commenti)

| grandezza | valore | criterio |
|---|---|---|
| raggio robot | 0,28 m | come nel 2D (feature della GNN invariate) |
| arena | 15 × 10 m | ricavata dalle larghezze sotto |
| passaggi tra le file di scaffali | 1,8 m | ≥ 1,56 m (robot + 2 × 0,5 m), un solo senso |
| corridoi verticali tra le colonne | 2,1 m | traffico raro (cambio passaggio), ≥ 1,56 m |
| corridoi lungo i muri | 2,3 m | ≥ 1,56 m, spazio per due robot affiancati |
| inflazione A\* | 0,43 m | raggio + 0,15 m |
| franco del goal locale (linea di vista) | 0,38 m | raggio + 0,10 m |
| piazzole | passo 1,0 m, 1,0 m dai muri, 1,5 m dagli scaffali | ≥ 0,72 m (distanza di sicurezza) + rumore 0,1 m |
| soglie tra robot | 0,70 / 0,66 / 0,62 m | sicurezza 0,72 m, contatto 0,56 m |

## Protocollo di validazione

- 2D: `python scripts/valida_2d.py --robots 6 --seeds 1000-1019` (configurazione completa più ablazioni:
  senza zone, senza corsie, solo policy, filtro simmetrico, solo comunicazione) → `results/validazione_2d.csv`;
  per tutti gli esperimenti della tesi: `bash esegui_2d.sh` → `results/val8/`.
- Webots: `python scripts/genera_mondo_webots.py --robots 6 --episodes 20`, poi ▶ in Webots → `results/webots_log.csv`.
- Confronto: `python scripts/evaluate_webots_vs_2d.py` → `results/confronto_2d_webots.{csv,md}`, `figures/generated/confronto_2d_webots.png`.
- Criteri: success ≥ 95 %, contatti = 0, separazione minima ≥ 0,60 m, contatto con scaffali ≤ 2 s, tempo e percorso Webots entro +30 % del 2D.
