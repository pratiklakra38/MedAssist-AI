# MedAssist AI

## Overview

MedAssist AI is an offline-first clinical decision support system designed to assist in early-stage medical triage using patient-reported symptoms. The system processes structured symptom inputs and produces a ranked list of probable diagnoses along with confidence scores and urgency levels.

It combines deterministic medical reasoning with probabilistic machine learning to deliver outputs that are both interpretable and data-driven.

---

## Core Objective

Transform raw symptom inputs into clinically meaningful insights by:

* Identifying likely diseases
* Estimating confidence levels
* Classifying urgency for medical attention

The system acts as an intelligent first layer of screening, enabling faster and more informed decision-making.

---

## Input Representation

The system operates on a fixed-length binary symptom vector:

* Each symptom is represented as:

  * `1` → present
  * `0` → absent
* Total features: 100+ symptoms
* Input is structured, consistent, and model-ready

Optional contextual inputs:

* Age
* Gender

---

## Intelligence Layer

### 1. Rule-Based Reasoning

* Encodes domain knowledge using IF-THEN logic
* Evaluates symptom combinations deterministically
* Produces:

  * Matching diseases
  * Rule match strength scores

This layer ensures interpretability and alignment with medical logic.

---

### 2. Machine Learning Model

* Uses a trained ensemble classifier (Random Forest)
* Learns symptom co-occurrence patterns from data
* Outputs:

  * Probability distribution across diseases

This layer captures non-obvious patterns and improves predictive performance.

---

### 3. Dual Inference Strategy

Both systems operate on the same input simultaneously.

* Rule-based system → structured reasoning
* ML model → probabilistic prediction

Outputs are complementary:

* Rules provide explainability
* ML provides generalization

---

## Confidence Fusion

The system merges outputs from both inference tracks into a unified score.

* Weighted combination of:

  * Rule-based confidence
  * ML probability

* Produces:

  * Final confidence score per disease
  * Ranked list of top predictions

This ensures balanced decision-making between logic and data.

---

## Output Generation

The system generates a structured response containing:

### 1. Diagnosis

* Single most probable disease
* Displayed with its confidence percentage

### 2. Explanation Signals

* Key contributing symptoms
* Rule matches (if applicable)

### 3. Triage Classification

* Low → non-critical
* Moderate → medical consultation recommended
* High → urgent attention required

---

## System Behavior

* Fully functional without internet connectivity
* Runs inference locally
* Lightweight and fast execution
* Deterministic + probabilistic hybrid reasoning

---

## Design Philosophy

* Interpretability over black-box complexity
* Simplicity in input, richness in output
* Hybrid intelligence instead of single-model dependency
* Structured reasoning aligned with real-world decision flows

---

## Key Strength

The system bridges symbolic AI and machine learning:

* Symbolic layer → "why this diagnosis"
* ML layer → "how likely this diagnosis"

This combination enables reliable, explainable, and practical decision support from minimal input.

---

## Agentic AI Integration (Basic Concepts for Mini-Project)

To fulfill **CO4**, we need to show that our system has "agency" (the ability to observe, think, and take actions on its own) rather than just passively taking input and giving output. 

Here are 2 simple ways we demonstrate Agentic AI in our project:

### 1. The "Clarification" Agent (Observe & Ask)
*   **What it does:** Instead of relying entirely on the user typing everything perfectly, the AI acts like a virtual doctor. 
*   **How it shows Agency:** If the user only inputs "fever" and "headache", the Agent *Observes* the ambiguity, *Thinks* "This could be Malaria or Heatstroke", and takes the *Action* to ask the user: "Are you also experiencing chills or sweating?" to narrow it down.

### 2. The "Self-Check" Agent (Reflection)
*   **What it does:** The system double-checks its own predictions before showing them to the user.
*   **How it shows Agency:** The Agent *Observes* the final disease prediction. If the ML model guesses something serious but no major symptoms match, the Agent *Thinks* "This prediction feels wrong", and takes the *Action* to lower the confidence percentage and warn the user.

