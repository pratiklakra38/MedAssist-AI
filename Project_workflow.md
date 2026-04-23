# MedAssist AI — Constraints & Guardrails

## Project Nature

MedAssist AI is a **mini academic project** developed as part of an Artificial Intelligence coursework focused on structured knowledge representation.

The system is designed to demonstrate the practical application of:

* Rule-based reasoning (symbolic AI)
* Machine learning (Random Forest)
* Hybrid inference systems

The primary goal is **conceptual clarity and functional demonstration**, not real-world clinical deployment.

---

## Intended Scope

* Educational prototype
* Demonstration of AI techniques in healthcare
* Offline-capable decision support simulation

The system is **not designed, tested, or validated for real clinical use**.

---

## Core Constraints

### 1. Dataset Limitations

* Limited to ~40 disease classes
* Based on a predefined symptom dataset
* Does not cover full medical spectrum
* May not generalize to rare or complex conditions

---

### 2. Input Simplification

* Symptoms are binary (present/absent)
* No severity, duration, or progression modeling
* No lab reports, imaging, or medical history depth

---

### 3. Model Constraints

* Random Forest trained on structured dataset only
* No real-time learning or adaptation
* Performance depends entirely on training data quality

---

### 4. Rule-Based System Constraints

* Rules are manually defined
* Limited coverage of medical knowledge
* Cannot handle unseen or ambiguous combinations

---

### 5. Offline-First Design

* No real-time external validation
* No integration with medical databases or APIs
* Updates must be manually deployed

---

## Guardrails

### 1. Non-Diagnostic Positioning

The system **does not provide medical diagnoses**.

All outputs are framed as:

* “Probable conditions”
* “Preliminary screening results”

---

### 2. Mandatory Disclaimer

Every output must include:

> “This is a preliminary screening tool. Consult a registered medical practitioner for accurate diagnosis.”

---

### 3. Human-in-the-Loop Expectation

* Final decisions must always be made by qualified medical professionals
* The system is only an assistive layer

---

### 4. Risk Communication

* Triage levels are indicative, not definitive
* High-risk outputs encourage immediate consultation
* No treatment or medication suggestions are provided

---

### 5. No Critical Decision Automation

* The system does not trigger emergency services
* Does not replace clinical judgment
* Does not perform autonomous actions

---

## Ethical Considerations

### 1. Safety First

* Avoids overconfidence in predictions
* Prioritizes caution in high-risk scenarios

---

### 2. Data Privacy

* No personally identifiable sensitive data required
* Local storage minimizes exposure risk

---

### 3. Transparency

* Provides explanation signals for outputs
* Avoids black-box-only decisions

---

## Summary

MedAssist AI is a **controlled, educational implementation** of a hybrid AI system for symptom-based disease prediction.

It is intentionally constrained to:

* Demonstrate AI concepts clearly
* Maintain interpretability
* Avoid unsafe real-world misuse

The system emphasizes **learning, explainability, and responsible AI usage** over clinical accuracy or deployment readiness.

---

## Evaluation (ESE Rubrics)

The project is evaluated based on the following End Semester Examination (ESE) rubrics:

| Criteria | Description | CO | Marks |
| :--- | :--- | :---: | :---: |
| **Understanding of AI Concepts, Presentation + Viva /Q&A** | Conceptual clarity, quality of presentation/demonstration, ability to explain approach, justify design decisions, and answer questions | CO1 | 5 |
| **Problem Solving & Knowledge Representation Design** | Case study understanding, problem formulation, and selection/design of appropriate AI techniques/models | CO2 | 3 |
| **Data Preparation & Feature Engineering** | Data preparation, preprocessing, transformation, and feature extraction/selection relevant to the problem | CO3 | 3 |
| **Implementation & Result Analysis** | Code implementation, execution, correctness, performance evaluation, and interpretation of results, **ensuring incorporation of Agentic AI concepts within the case study implementation.** | CO4 | 4 |
| **Total** | | | **15 Marks** |
