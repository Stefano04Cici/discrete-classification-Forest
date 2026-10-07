# Types of files generated in `test_output/`

The `test_output/` folder is the output directory of the program
(`config.TEST_OUTPUT_DIR`). At runtime the system can save **three types
of files** in it, depending on the operation performed by the user.

## Types of files that can be generated

### `.joblib` — Machine Learning Model

| Detail | Value |
|---|---|
| File name | `model_path.joblib` |
| Origin | `preprocessing.py` → `train_model()` with `joblib.dump(...)` |
| When | At the program startup and from **Menu → 4) TRAIN THE AI MODEL** (retraining) |
| Content | Serialized payload: calibrated Random Forest model (CalibratedClassifierCV), `label_encoder`, `class_names`, `features`, `test_data` (holdout), calibration metadata |

### `.json` — Diamonds and Knowledge Base

| Type | Name(s) | Origin | When |
|---|---|---|---|
| Simple KB | `rules.json` | `threshold_system.py` → `MiniKB.save_to_json()` | **Menu 2 → 6) Save the knowledge base** |
| Composite rules | `composite_rules.json` | `threshold_system.py` → `ExtendedKB.save_to_json()` | **Menu 2 → 6) Save the knowledge base** |
| Random diamonds | `diamond_random_1.json`, `diamond_random_2.json`, ... | `ui.py` (option 2) → `random_diamond()` | **Menu 1 → 2) Generate RANDOM diamond** (from 1 to 10) |
| Test diamond | `diamond_evaluation.json` | `ui.py` → `random_diamond()` | **Menu 2 → 2) Evaluate a RANDOM diamond** |
| Saved diamond | free name (`{name}.json`) | `ui.py`, `save` command | **Menu 1, after a prediction** (saves the last tested diamond) |

### `.ttl` — Semantic knowledge (RDF/Turtle)

| Type | Name(s) | Origin | When |
|---|---|---|---|
| KB export | `{base}_kb.ttl` (default: `diamonds_ai_system_kb.ttl`) | `rdf_exporter.py` → `export_kb_rdf()` | **Menu 3 → 1) Export the Knowledge Base in RDF** |
| Diamond report | `{name}.ttl` (default: `diamond_report.ttl`) | `rdf_exporter.py` → `generate_diamond_rdf_report()` | **Menu 3 → 3) Generate RDF reports for a specific diamond** |

## Files the program can re-read at runtime

- `.json` inside `test_output/` → loaded by **Menu 1 → 3) Load a diamond from a JSON file**
- `.ttl` inside `test_output/` → loaded by **Menu 3 → 2) Load a Knowledge Base from an RDF file** (KB or diamond reports); SPARQL queries are run on these
- `.joblib` → `model_path.joblib` is loaded at every startup (`prediction.load_payload`)

## Does not generate here

- **`.csv`**: the categorical dataset `diamonds_categorical.csv` is written
  in the `dataset/` folder (`config.CATEGORICAL_CSV`), not in `test_output/`.
- **PNG images**: the graphs (EDA, confusion matrix, reliability, learning
  curve) are shown on screen (`plt.show()`), never saved on disk.