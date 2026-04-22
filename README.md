# MedAssist AI

> **Disclaimer**: This is a preliminary screening tool developed as an academic project. It does **not** provide medical diagnoses. Consult a registered medical practitioner for accurate diagnosis.

## Overview

MedAssist AI is an offline-first clinical decision support system designed to assist in early-stage medical triage using patient-reported symptoms. It combines **deterministic rule-based reasoning** with **probabilistic machine learning** (Random Forest) to deliver outputs that are both interpretable and data-driven.

### Key Features

- **Hybrid Intelligence** — Symbolic AI (rule-based) + ML (Random Forest) working in parallel
- **Confidence Fusion** — Weighted combination of both engines for reliable predictions
- **Triage Classification** — Low / Moderate / High urgency indicators
- **Explainability** — Rule match details and contributing symptoms displayed
- **Offline-First** — Fully functional without internet connectivity

---

## Architecture

```
┌─────────────────┐
│  Symptom Input   │  Binary vector (132 symptoms: 1=present, 0=absent)
│  (+ Age/Gender)  │  Optional contextual inputs
└────────┬────────┘
         │
    ┌────▼────┐
    │  DUAL   │
    │INFERENCE│
    └─┬────┬──┘
      │    │
┌─────▼─┐ ┌▼──────────┐
│ Rules  │ │ Random    │
│ Engine │ │ Forest    │
│(IF-THEN│ │(Trained   │
│ logic) │ │ Ensemble) │
└────┬───┘ └─────┬─────┘
     │           │
  ┌──▼───────────▼──┐
  │ Confidence      │  Weighted fusion of rule strength + ML probability
  │ Fusion          │
  └───────┬─────────┘
          │
  ┌───────▼─────────┐
  │ Triage + Output │  Ranked predictions, urgency level, explanations
  │ + Disclaimer    │
  └─────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd AI-CA-3

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the dataset from Kaggle:

```bash
python src/download_dataset.py
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) and place CSVs in `data/raw/`.

### Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
AI-CA-3/
├── GEMINI.md                   # System specification
├── Project_workflow.md         # Constraints & guardrails
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned & encoded data
├── models/
│   └── random_forest.pkl       # Trained ML model
├── src/
│   ├── __init__.py
│   ├── download_dataset.py     # Dataset acquisition
│   ├── preprocessing.py        # Data loading & cleaning
│   ├── rule_engine.py          # Rule-based reasoning
│   ├── ml_model.py             # Random Forest training & inference
│   ├── fusion.py               # Confidence fusion logic
│   ├── triage.py               # Urgency classification
│   └── inference.py            # Unified inference pipeline
├── app.py                      # Streamlit web UI
├── notebooks/
│   └── exploration.ipynb       # EDA & experimentation
└── tests/
    ├── test_rule_engine.py
    ├── test_ml_model.py
    └── test_fusion.py
```

---

## Limitations

- Limited to ~40 disease classes from the training dataset
- Symptoms are binary (present/absent) — no severity or duration modeling
- No lab reports, imaging, or medical history integration
- Rule coverage is manually curated (~15-20 diseases)
- Model performance depends on training data quality

---

## Ethical Considerations

- **Not a diagnostic tool** — All outputs are framed as "probable conditions"
- **Human-in-the-loop** — Final decisions must be made by qualified medical professionals
- **No critical decision automation** — Does not trigger emergency services or suggest treatments
- **Data privacy** — No personally identifiable information required; local-only processing
- **Transparency** — Explanation signals provided for all predictions

---

## License

Academic project — for educational purposes only.
