# MedAssist AI — System Workflow

> **Disclaimer**: This is a preliminary screening tool developed as an academic project.
> It does **not** provide medical diagnoses. Consult a registered medical practitioner for accurate diagnosis.

---

## Overview

MedAssist AI is an **Autonomous Medical Triage Agent** that uses a hybrid dual-inference architecture combining **Forward Chaining (Rule-Based Reasoning)** with a **Random Forest ML Classifier** to screen patients for probable diseases based on reported symptoms.

The system follows an **Agentic AI** pattern:

```
PERCEPTION  →  TOOL ORCHESTRATION  →  SYNTHESIS  →  ACTION
```

---

## End-to-End Pipeline (6 Steps)

### Step 1 — Data Collection & Preprocessing (`src/preprocessing.py`)

**What happens:**
- Raw dataset is loaded from `data/raw/` (Kaggle Disease-Symptom Dataset)
- 132 binary symptom features (1 = present, 0 = absent) across 41+ diseases
- Junk columns (`Unnamed: 133`, `fluid_overload.1`) are dropped
- Column names are standardized (whitespace stripped)
- Disease labels are encoded via `LabelEncoder` into numerical indices
- Processed splits saved to `data/processed/`

**Outputs:**
| File | Description |
|------|-------------|
| `X_train.csv` / `X_test.csv` | Binary symptom feature matrices |
| `y_train.csv` / `y_test.csv` | Encoded disease labels |
| `symptom_list.json` | Ordered list of 132 symptom feature names |
| `disease_list.json` | Ordered list of disease class names |

---

### Step 2 — Knowledge Representation (`src/rule_engine.py`)

**What happens:**
- Medical expertise is encoded as **IF-THEN rules** derived from WHO ICD-11 diagnostic criteria and ICMR protocols
- Each disease maps to a list of required diagnostic symptom markers
- 22 diseases are covered in the rule base (18 from Kaggle + 4 custom)

**Example Rule:**
```
IF [high_fever AND cough AND fatigue AND loss_of_smell AND breathlessness]
THEN → COVID-19
```

**Knowledge Base Format:**
```python
rules = {
    "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness"],
    "Malaria":  ["chills", "high_fever", "sweating", "headache", "muscle_pain"],
    ...
}
```

---

### Step 3 — Rule-Based Inference via Forward Chaining (`src/rule_engine.py`)

**What happens:**
- The Rule Engine applies **Forward Chaining** (data-driven inference)
- Starts from **known facts** (patient-reported symptoms)
- Iterates through every rule in the knowledge base
- Computes an **overlap ratio** between patient facts and required symptoms

**Forward Chaining Algorithm:**
```
1. KNOWN FACTS = set(patient_symptoms)
2. FOR each rule R in knowledge_base:
       required = set(R.symptoms)
       overlap  = KNOWN FACTS ∩ required
       match_ratio = |overlap| / |required|
       IF match_ratio >= threshold (default 50%) THEN fire rule
3. SORT all fired rules by match_strength DESC
4. RETURN ranked candidate list
```

**Output:** List of `RuleMatch` objects, each containing:
- Disease name
- Match strength (0.0 – 1.0)
- List of matched symptoms

---

### Step 4 — Machine Learning Prediction (`src/ml_model.py`)

**What happens:**
- A **Random Forest Classifier** (200 trees, balanced class weights) processes the same binary symptom vector
- The model was trained on the preprocessed dataset achieving **~98.8% accuracy**
- Outputs a **probability distribution** across all disease classes

**Why Random Forest?**
- Handles 132 binary features efficiently
- Built-in feature importance (explainability)
- Robust to noise and class imbalance
- No GPU required (offline-capable on low-end hardware)

**Output:** Dictionary mapping disease names to probability scores (0.0 – 1.0)

---

### Step 5 — Confidence Fusion (`src/fusion.py`)

**What happens:**
- Outputs from both inference tracks are merged using a **weighted scoring formula**:

```
Final Score = (0.4 × Rule Match Strength) + (0.6 × ML Probability)
```

- All diseases from either track are included
- If a disease appears only in rules (no ML data), it gets 100% rule strength
- Results are ranked by fused confidence score
- **Top 3** candidates are returned

**Why this weighting?**
- ML gets 60% because it captures non-obvious symptom co-occurrence patterns from the full 132-feature space
- Rules get 40% because they encode verified, interpretable diagnostic criteria
- Neither system alone dominates the final prediction

---

### Step 6 — Triage, Reasoning & Output (`src/triage.py`, `src/inference.py`, `app.py`)

**What happens:**

#### 6a. Triage Classification (`src/triage.py`)
The top prediction is classified into urgency levels:

| Level | Condition | Action |
|-------|-----------|--------|
| **High** | Confidence ≥ 55% AND disease is critical (e.g., Heart Attack, TB, Dengue, Pneumonia, COVID-19) | Seek emergency medical attention |
| **Moderate** | Confidence ≥ 35% | Schedule a doctor's appointment |
| **Low** | Confidence < 35% | Monitor symptoms, rest |

#### 6b. Agent Reasoning (`src/inference.py`)
The `MedicalTriageAgent` generates a **transparent reasoning chain** explaining:
- How many symptoms were analyzed
- Which tools contributed (Rule Engine, ML Classifier, or both)
- The exact contribution percentages from each tool
- Key symptoms that triggered rule matches
- Alternative diagnoses considered (from the top 3 list)

#### 6c. Explainable Output (`app.py`)
The Streamlit UI displays:
1. **Top 3 probable diseases** with confidence percentages and animated bars
2. **Key contributing symptoms** as visual tags on each card
3. **Urgency level** with pulsing CSS animation (High = red, Moderate = amber, Low = green)
4. **Agent's decision reasoning** in a dedicated panel
5. **Recommended action plan** based on triage level
6. **Mandatory disclaimer** on every output

---

## Agentic AI Architecture

The `MedicalTriageAgent` class in `src/inference.py` implements the Agentic AI pattern:

```
┌──────────────────────────────────────────────┐
│              PERCEPTION LAYER                │
│  Symptom Input → Binary Feature Vector       │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│           TOOL ORCHESTRATION                 │
│                                              │
│  ┌─────────────┐    ┌──────────────────┐     │
│  │ Tool 1:     │    │ Tool 2:          │     │
│  │ Rule Engine │    │ ML Classifier    │     │
│  │ (Forward    │    │ (Random Forest)  │     │
│  │  Chaining)  │    │                  │     │
│  └──────┬──────┘    └────────┬─────────┘     │
│         └───────┬────────────┘               │
│        ┌────────▼──────────┐                 │
│        │ Confidence Fusion │                 │
│        │ (Weighted Merge)  │                 │
│        └────────┬──────────┘                 │
└─────────────────┼────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────┐
│              ACTION LAYER                    │
│                                              │
│  • Top 3 Ranked Diagnoses                    │
│  • Transparent Reasoning Chain               │
│  • Triage Urgency Classification             │
│  • Actionable Next Steps                     │
│  • Ethical Guardrails (Disclaimer)            │
└──────────────────────────────────────────────┘
```

### Agentic Concepts Demonstrated:

| Concept | Implementation |
|---------|---------------|
| **Tool Use** | Agent orchestrates 4 specialized tools (Rule Engine, ML Classifier, Fusion, Triage) |
| **Autonomous Reasoning** | Agent generates its own explanation chain without external prompting |
| **Goal-Directed Behavior** | Agent's goal is accurate triage, not just raw prediction |
| **Guardrails** | Agent enforces ethical boundaries (disclaimer, no treatment advice, no emergency triggers) |
| **Perception-Action Loop** | Symptoms → Analysis → Diagnosis + Action Plan |

---

## Project Structure (Final)

```
AI-CA-3/
├── app.py                          # Streamlit UI (Symptom Collector + Diagnosis Output)
├── requirements.txt                # Python dependencies
├── data/
│   ├── raw/                        # Original Kaggle dataset
│   │   ├── Training.csv
│   │   └── Testing.csv
│   └── processed/                  # Cleaned & encoded data
│       ├── X_train.csv / X_test.csv
│       ├── y_train.csv / y_test.csv
│       ├── symptom_list.json
│       └── disease_list.json
├── models/
│   └── random_forest.pkl           # Trained ML model (serialized)
├── src/
│   ├── __init__.py
│   ├── download_dataset.py         # Kaggle dataset acquisition
│   ├── preprocessing.py            # Data cleaning & encoding
│   ├── rule_engine.py              # Forward Chaining IF-THEN engine
│   ├── ml_model.py                 # Random Forest training & inference
│   ├── fusion.py                   # Confidence Aggregator (weighted fusion)
│   ├── triage.py                   # Urgency classification + action plans
│   ├── inference.py                # MedicalTriageAgent (agentic orchestrator)
│   └── inject_custom_diseases.py   # Synthetic data injection for custom diseases
└── tests/
    ├── test_rule_engine.py         # Forward Chaining unit tests
    ├── test_ml_model.py            # ML model validation tests
    └── test_fusion.py              # Fusion + Agent integration tests
```

---

## How to Run

```bash
# 1. Activate virtual environment
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select symptoms, click "Run Agent Diagnosis", and observe the full agentic pipeline in action.

---

## Test Suite

```bash
python -m pytest tests/ -v
```

All 12 tests validate:
- Forward Chaining rule matching (exact, partial, threshold)
- ML model loading, probability format, and prediction accuracy
- Fusion formula correctness (weighted scoring)
- Full agent integration (tool registry, top-3 output, reasoning, guardrails)
