# Collegare la GNN a Webots: guida passo per passo

Testato con Webots R2025a, Python 3.10+ e numpy 2.x. Le regole di controllo e le dimensioni
della mappa sono spiegate in `docs/regole_rifacimento_webots.md`.

## Come funziona il collegamento

```
                 Webots (fisica, ogni 16 ms)
   ┌─────────────────────────────────────────────────────────────┐
   │  gnn_supervisor  (Robot con supervisor TRUE)                 │
   │   ogni 80 ms:                                                │
   │   1. legge posizione/velocità di AGV_i  (API Supervisor)     │
   │   2. core/controller.FleetController.step(pos, vel, t)       │
   │        A* con corsie -> goal locale in linea di vista        │
   │        policy GNN/MLP (numpy, .npz) o esperto                │
   │        attracco piazzola, evitamento scaffali,               │
   │        prenotazione zone critiche, filtro d'emergenza        │
   │   3. metriche condivise (core/metrics.py)                    │
   │   ogni 16 ms:                                                │
   │   4. AGV_i.setVelocity([vx, vy, 0, 0, 0, 0])   (olonomo)      │
   │      oppure Emitter -> agv_controller          (pioneer)     │
   └──────────────────────────────────────────────────────────────┘
```

- **Robot olonomo (default).** Base cilindrica r 0,28 m con fisica; la velocità (vx, vy) della
  policy viene imposta direttamente, come nel simulatore 2D; gli urti con scaffali, muri e robot
  sono gestiti dalla fisica di Webots. Nessun riaddestramento.
- **Pioneer 3-DX (`--robot pioneer`).** Variante storica: robot differenziale con `agv_controller`.
- **Un solo file per la geometria.** `core/warehouse.py` genera sia il 2D sia il mondo `.wbt`.
- **Niente PyTorch dentro Webots.** Policy dal file `.npz` (`core/numpy_policy.py`).

## Passo 1: preparare il progetto (dalla cartella del progetto)

```bash
python -m venv .venv
source .venv/Scripts/activate        # su Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_geometria.py tests/test_supervisore.py -q   # test che girano senza Webots
```

`tests/test_supervisore.py` esegue il supervisore vero su un finto Webots (`tests/fake_webots.py`):
se passa, il collegamento con `core/` funziona.

## Passo 2: validazione nel 2D (stesso controllore, stesse metriche)

```bash
python scripts/valida_2d.py --robots 6 --seeds 1000-1019
```

Scrive `results/validazione_2d.csv` e stampa il riepilogo per la configurazione completa e le
ablazioni (senza zone, senza corsie, solo policy = "senza sicurezza" nella tesi, filtro simmetrico,
solo comunicazione).

## Passo 3: generare il mondo Webots

```bash
python scripts/genera_mondo_webots.py --robots 6 --policy gnn --episodes 20
```

Crea `webots/worlds/warehouse_gnn.wbt` con il magazzino 15 × 10 m, 6 AGV olonomi (`AGV_0`…`AGV_5`),
i marcatori dei goal, le frecce delle corsie, i contorni rossi delle zone critiche e il supervisore.
Opzioni: `--robots N` (fino a 24), `--policy gnn|mlp|esperto`, `--seed`, `--episodes`, `--max-time`,
`--robot olonomo|pioneer`, `--filtro emergenza|simmetrico|nessuno`, `--senza-corsie`, `--senza-zone`,
`--senza-repulsione`, `--senza-retromarcia`, `--scaglionato S`, `--senza-archi`.

## Passo 4: dire a Webots quale Python usare

**Tools → Preferences → General → Python command**: il Python del venv (quello con numpy), per esempio
`C:\Users\<tu>\...\progetto_gnn_webots\.venv\Scripts\python.exe`.

## Passo 5: avviare

1. **File → Open World…** → `webots/worlds/warehouse_gnn.wbt` (se era già aperto: **File → Revert**).
2. ▶ (tempo reale) oppure ⏩ (veloce).
3. In console: `6 robot (olonomo), controllo ogni 80 ms`, poi una riga per episodio con le metriche.
4. Alla fine la simulazione va in pausa e i risultati sono in `results/webots_log.csv`.

Le opzioni si cambiano rigenerando il mondo oppure nel nodo `Robot "gnn_supervisor"`, campo
`controllerArgs` (poi File → Revert).

Su Linux, senza grafica, le sei esecuzioni della tesi (4, 6 e 8 robot, GNN e MLP, configurazione
completa) si fanno con `bash scripts/esegui_webots_headless.sh` (serve `xvfb-run`).

## Passo 6: confronto 2D ↔ Webots (il capitolo dei risultati)

```bash
python scripts/evaluate_webots_vs_2d.py
```

Riesegue nel 2D gli stessi seed del log Webots e scrive `results/confronto_2d_webots.csv`,
`results/confronto_2d_webots.md` (tabella pronta per la tesi) e `figures/generated/confronto_2d_webots.png`.

Colonne del log (uguali nel 2D e in Webots):

| colonna | significato |
|---|---|
| arrived / success | robot arrivati entro 0,30 m dal goal / tutti arrivati |
| time_s | tempo simulato dell'episodio |
| final_error | distanza media finale dal goal |
| min_separation | distanza minima tra i centri di due robot (contatto sotto 0,56 m) |
| contact_events / near_miss_events | coppie scese sotto 0,56 m / sotto 0,72 m |
| path_length | metri percorsi in totale |
| shelf_contact_time_s / shelf_contact_max_s | tempo a contatto con scaffali o muri (totale / continuo massimo) |
| wait_time_s | tempo fermo in attesa di una zona critica |
| reverse_events / forced_entries | retromarce di emergenza / ingressi forzati nelle zone dopo 12 s di attesa |

## Problemi frequenti

| sintomo | causa e soluzione |
|---|---|
| `ModuleNotFoundError: numpy` | Webots non usa il venv: rifai il Passo 4 |
| `nessun robot AGV_0...` | hai aperto un altro mondo: usa quello generato al Passo 3 |
| `FileNotFoundError ... gnn_n8.npz` | manca il checkpoint: vedi README (Passo 2 originale) oppure passa `--checkpoint` |
| i robot restano fermi | controlla che il supervisore stampi `episodio ...`; con `--robot pioneer` Receiver ed Emitter devono essere sul canale 1 |
| un robot resta fermo davanti a una zona | sta aspettando (wait_time_s); dopo 12 s entra comunque (forced_entries) |
| va lento | usa ⏩ e `--senza-archi` |
