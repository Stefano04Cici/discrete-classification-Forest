# Tipi di file generati in `test_output/`

La cartella `test_output/` è la directory di output del programma
(`config.TEST_OUTPUT_DIR`). A runtime il sistema può salvarvi **tre tipi
di file**, a seconda dell'operazione eseguita dall'utente.

## Tipi di file generabili

### `.joblib` — Modello Machine Learning

| Dettaglio | Valore |
|---|---|
| Nome file | `model_path.joblib` |
| Origine | `preprocessing.py` → `train_model()` con `joblib.dump(...)` |
| Quando | All'avvio del programma e da **Menu → 4) TRAIN THE AI MODEL** (riaddestramento) |
| Contenuto | Payload serializzato: modello Random Forest calibrato (CalibratedClassifierCV), `label_encoder`, `class_names`, `features`, `test_data` (holdout), metadati di calibrazione |

### `.json` — Diamanti e Knowledge Base

| Tipo | Nome(i) | Origine | Quando |
|---|---|---|---|
| KB semplice | `rules.json` | `threshold_system.py` → `MiniKB.save_to_json()` | **Menu 2 → 6) Save the knowledge base** |
| Regole composite | `composite_rules.json` | `threshold_system.py` → `ExtendedKB.save_to_json()` | **Menu 2 → 6) Save the knowledge base** |
| Diamanti casuali | `diamond_random_1.json`, `diamond_random_2.json`, ... | `ui.py` (opzione 2) → `random_diamond()` | **Menu 1 → 2) Generate RANDOM diamond** (da 1 a 10) |
| Diamante di test | `diamond_evaluation.json` | `ui.py` → `random_diamond()` | **Menu 2 → 2) Evaluate a RANDOM diamond** |
| Diamante salvato | nome libero (`{nome}.json`) | `ui.py`, comando `save` | **Menu 1, dopo una predizione** (salva l'ultimo diamante testato) |

Nota: `config.RANDOM_DIAMOND` definisce anche `random_diamond.json`, costante
disponibile ma non usata dai menu attuali.

### `.ttl` — Conoscenza semantica (RDF/Turtle)

| Tipo | Nome(i) | Origine | Quando |
|---|---|---|---|
| Export KB | `{base}_kb.ttl` (default: `diamonds_ai_system_kb.ttl`) | `rdf_exporter.py` → `export_kb_rdf()` | **Menu 3 → 1) Export the Knowledge Base in RDF** |
| Report diamante | `{nome}.ttl` (default: `diamond_report.ttl`) | `rdf_exporter.py` → `generate_diamond_rdf_report()` | **Menu 3 → 3) Generate RDF reports for a specific diamond** |

## File che il programma può rileggere a runtime

- `.json` dentro `test_output/` → caricati da **Menu 1 → 3) Load a diamond from a JSON file**
- `.ttl` dentro `test_output/` → caricati da **Menu 3 → 2) Load a Knowledge Base from an RDF file** (KB o report di diamondi); su questi si lanciano le query SPARQL
- `.joblib` → `model_path.joblib` viene caricato a ogni avvio (`prediction.load_payload`)

## Non genera qui

- **`.csv`**: il dataset categorico `diamonds_categorical.csv` viene scritto
  nella cartella `dataset/` (`config.CATEGORICAL_CSV`), non in `test_output/`.
- **immagini PNG**: i grafici (EDA, confusion matrix, reliability, learning
  curve) sono mostrati a video (`plt.show()`), mai salvati su disco.