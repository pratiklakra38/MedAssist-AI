# Project Timeline Log

This file tracks the completion timings of each phase, detailing the state before the phase began and the exact changes and additions made during the phase execution.

---

## Phase 1: Project Scaffolding & Dataset
- **Completed:** 2026-04-07 14:18 (IST)
- **State Before:** Empty repository containing only the `GEMINI.md` and `Project_workflow.md` documentation. 
- **What was done:** 
  - Created the full hierarchical directory structure.
  - Setup core Python dependencies and isolated them inside an active `.venv` environment to prevent global package pollution.
  - Developed a script to programmatically ingest the Kaggle symptom-disease dataset.
  - Analyzed the raw dataset through an Exploratory Data Analysis (EDA) notebook.
- **Files Added:**
  - `requirements.txt`
  - `src/__init__.py`
  - `src/download_dataset.py`
  - `README.md`
  - `notebooks/explore_data.py`
  - `data/raw/Training.csv`, `data/raw/Testing.csv`

---

## Phase 2: Data Preprocessing & Feature Engineering
- **Completed:** 2026-04-07 14:24 (IST)
- **State Before:** Raw datasets containing duplicate or empty junk columns (`Unnamed: 133`, `fluid_overload.1`) and raw textual target labels.
- **What was done:**
  - Standardized feature string names and removed junk columns across both training and testing datasets.
  - Encoded string disease targets into numerical indices using `LabelEncoder`.
  - Saved feature distributions natively as CSV matrices, along with ordered JSON mapping dictionaries for features and targets for future inference routing.
- **Files Added:**
  - `src/preprocessing.py`
  - `data/processed/X_train.csv`, `data/processed/X_test.csv`
  - `data/processed/y_train.csv`, `data/processed/y_test.csv`
  - `data/processed/symptom_list.json`
  - `data/processed/disease_list.json`

---

## Phase 3: Rule-Based Reasoning Engine
- **Completed:** 2026-04-07 14:35 (IST)
- **State Before:** No deterministic rule capabilities. Relying entirely on raw datasets.
- **What was done:** 
  - Extracted average feature frequencies locally to form empirically-grounded heuristic rules for 18 core diseases.
  - Implemented an exact deterministic IF-THEN simulation evaluator in `rule_engine.py`.
  - Added thorough unit tests to validate edge cases including multiple matches, precise matches, and matching thresholds.
- **Files Added:**
  - `src/rule_engine.py`
  - `tests/test_rule_engine.py`
  - `extracted_rules.json` (intermediary analysis export)

---

## Phase 4: Machine Learning Model (Random Forest)
- **Completed:** 2026-04-16 22:10 (IST)
- **State Before:** No machine learning predictive model. Data was preprocessed but unused for probabilistic inference.
- **What was done:** 
  - Designed the `MLModel` class to train a `RandomForestClassifier` mapping binary symptom arrays to diseases.
  - Used balanced class weights, default max depth, and split tunings to accurately handle the categorical patterns, achieving 97.6% accuracy.
  - Formatted the predictions to output as an ordered dictionary mapping the `disease_list` to its discrete probability.
  - Pickled the fitted model via `joblib` into `models/random_forest.pkl`.
  - Authored a test suite simulating raw input validations.
- **Files Added:**
  - `src/ml_model.py`
  - `tests/test_ml_model.py`
  - `models/random_forest.pkl`

---

## Phase 5: Dual Inference & Confidence Fusion
- **Completed:** 2026-04-16 22:20 (IST)
- **State Before:** Rule engine and ML engine existed individually but were not cohesively synthesizing outputs or measuring medical risk.
- **What was done:** 
  - Implemented mathematical confidence interpolation inside `fusion.py` using weighted strengths (Rules 40%, ML 60%).
  - Created `TriageEngine` in `triage.py` to classify urgencies securely without hallucinating treatments.
  - Linked all models efficiently into `InferencePipeline` in `inference.py`, dynamically injecting standard disclaimers.
  - Asserted system logic comprehensively using unittest suites.
- **Files Added:**
  - `src/fusion.py`
  - `src/triage.py`
  - `src/inference.py`
  - `tests/test_fusion.py`

---

## Phase 6: Web Interface (Streamlit)
- **Completed:** 2026-04-16 23:05 (IST)
- **State Before:** Headless API/backend logic accessible only via terminal scripts and mock inference runs.
- **What was done:** 
  - Designed `app.py` utilizing Streamlit, directly importing `InferencePipeline`.
  - Injected an advanced CSS template mimicking a Google/Inter-inspired sleek dark-grey medical interface (`#202124`).
  - Implemented 4D CSS keyframe animations representing triage urgency (Pulse alerts) and dynamic Glassmorphism styling (slide up/hover scale).
  - Formatted ugly system feature string variables into human-readable Title Cased formats (e.g., `nodal_skin_eruptions` -> `Nodal Skin Eruptions`).
  - Enforced the hard-coded ethical disclaimer visually via a caution banner.
- **Files Added:**
  - `app.py`

---
