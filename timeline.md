# Project Timeline Log

This file tracks the completion timings of each phase, detailing the state before the phase began and the exact changes and additions made during the phase execution.

---

## Phase 1: Project Scaffolding & Dataset
- **Completed:** 2026-04-22 23:43 (IST)
- **State Before:** Empty repository containing only `.md` files, `requirements.txt`, and `.venv`.
- **What was done:**
  - Created the directory structure: `data/raw/`, `data/processed/`, `models/`, `src/`, `tests/`.
  - Created `src/__init__.py` to establish the Python package.
  - Developed `src/download_dataset.py` to fetch the Kaggle disease-prediction dataset.
  - Downloaded `Training.csv` (4920 samples) and `Testing.csv` (42 samples).
- **Files Added:**
  - `src/__init__.py`
  - `src/download_dataset.py`
  - `data/raw/Training.csv`, `data/raw/Testing.csv`

---

## Phase 2: Data Preprocessing & Feature Engineering
- **Completed:** 2026-04-22 23:44 (IST)
- **State Before:** Raw CSVs with junk columns (`Unnamed: 133`, `.1` duplicates) and string disease labels.
- **What was done:**
  - Cleaned column names, dropped junk columns.
  - Encoded disease targets into numerical indices.
  - Saved structured CSVs and JSON mapping files.
  - Result: 131 symptoms, 41 disease classes.
- **Files Added:**
  - `src/preprocessing.py`
  - `data/processed/X_train.csv`, `data/processed/X_test.csv`
  - `data/processed/y_train.csv`, `data/processed/y_test.csv`
  - `data/processed/symptom_list.json`, `data/processed/disease_list.json`

---

## Phase 3: Rule-Based Reasoning Engine
- **Completed:** 2026-04-22 23:44 (IST)
- **State Before:** No deterministic inference. Only raw data existed.
- **What was done:**
  - Hand-curated IF-THEN rules for 20 diseases with known symptom profiles.
  - Implemented `RuleEngine` class that evaluates symptom match strength as a ratio.
  - Rules ensure deterministic, explainable outputs that the ML model cannot override.
- **Files Added:**
  - `src/rule_engine.py`

---

## Phase 4: Machine Learning Model (Random Forest)
- **Completed:** 2026-04-22 23:45 (IST)
- **State Before:** Rule engine existed but no probabilistic prediction capability.
- **What was done:**
  - Trained a `RandomForestClassifier` (200 estimators, balanced class weights).
  - Achieved **97.62% test accuracy**.
  - Model outputs probability distributions across all 41 diseases.
  - Pickled model saved to `models/random_forest.pkl`.
- **Files Added:**
  - `src/ml_model.py`
  - `models/random_forest.pkl`

---

## Phase 5: Dual Inference & Confidence Fusion
- **Completed:** 2026-04-22 23:46 (IST)
- **State Before:** Rule engine and ML model existed independently.
- **What was done:**
  - Implemented `fusion.py` with dynamic weighting: strong rule matches (>70%) get 60% weight to prevent ML from overriding clear deterministic diagnoses.
  - Created `TriageEngine` in `triage.py` for urgency classification (High/Moderate/Low).
  - Built `InferencePipeline` in `inference.py` linking all modules.
  - Verified: Malaria symptoms → Malaria at Rank #1 (100.0%), no Heatstroke confusion.
- **Files Added:**
  - `src/fusion.py`
  - `src/triage.py`
  - `src/inference.py`

---

## Phase 6: Agentic AI Layer (CO4)
- **Completed:** 2026-04-22 23:47 (IST)
- **State Before:** Passive inference pipeline with no autonomous behavior.
- **What was done:**
  - Implemented 3 Agentic AI modules in `src/agents.py`:
    1. **ClarificationAgent** — Detects ambiguous symptom sets and asks follow-up questions.
    2. **SelfCheckAgent** — Validates ML predictions against rule evidence; swaps rankings if ML contradicts rules.
    3. **TaskExecutionAgent** — Generates a `Doctor_Referral.txt` file for high-urgency cases.
  - Each agent follows the Observe → Think → Act pattern.
- **Files Added:**
  - `src/agents.py`

---

## Phase 7: Web Interface (Streamlit)
- **Completed:** 2026-04-22 23:49 (IST)
- **State Before:** Backend logic accessible only via terminal.
- **What was done:**
  - Built `app.py` with dark glassmorphism UI, Inter font, gradient title.
  - Integrated all 3 Agentic AI features into the UI:
    - Clarification Agent prompts appear as interactive Yes/No questions.
    - Self-Check warnings displayed as orange alert boxes.
    - Task Execution Agent provides a downloadable referral file.
  - Symptom names displayed as human-readable Title Case.
  - Mandatory ethical disclaimer shown on every prediction.
- **Files Added:**
  - `app.py`

---

## Phase 8: Testing & Validation
- **Completed:** 2026-04-22 23:49 (IST)
- **State Before:** No automated tests.
- **What was done:**
  - Created `tests/test_rule_engine.py` (4 tests) and `tests/test_fusion.py` (3 tests).
  - Critical test: `test_malaria_beats_heatstroke` — ensures Malaria always ranks #1 when Malaria symptoms are provided.
  - All **7 tests passed**.
- **Files Added:**
  - `tests/test_rule_engine.py`
  - `tests/test_fusion.py`

---
