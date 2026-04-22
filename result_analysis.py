"""
MedAssist AI — Result Analysis & Discussion Report Generator

Generates comprehensive performance evaluation with:
1. ML Model Accuracy & Classification Report (Table)
2. Per-Disease Precision/Recall/F1 Bar Charts
3. Confusion Matrix Heatmap
4. Rule Engine Coverage Analysis (Table + Bar Chart)
5. Confidence Fusion Comparison (Rule vs ML vs Fused) for sample cases
6. Triage Distribution Pie Chart

All outputs are saved to results/ directory as PNG images and printed as tables.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from src.ml_model import MLModel
from src.rule_engine import RuleEngine
from src.inference import MedicalTriageAgent

# ─── Setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Dark theme for all charts
plt.style.use("dark_background")
COLORS = {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "bg": "#0a0e17",
    "card": "#161b22",
    "text": "#e6edf3",
    "muted": "#6b7280",
}

def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [OK] Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════
# 1. ML MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. ML MODEL PERFORMANCE EVALUATION")
print("=" * 70)

ml = MLModel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

with open(os.path.join(PROCESSED_DIR, "disease_list.json"), "r") as f:
    disease_list = json.load(f)

y_pred = ml.model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Overall Accuracy: {accuracy * 100:.2f}%")
print(f"  Total Test Samples: {len(y_test)}")
print(f"  Total Disease Classes: {len(disease_list)}")

# Classification Report as DataFrame
report_dict = classification_report(
    y_test, y_pred,
    target_names=[disease_list[i] for i in sorted(set(y_test))],
    output_dict=True, zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.round(3)

print("\n  Classification Report (per disease):")
print(report_df.to_string())

# Save as CSV
report_df.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
print(f"\n  [OK] Saved: {os.path.join(RESULTS_DIR, 'classification_report.csv')}")

# ── Chart 1: Per-Disease F1 Scores ────────────────────────────────────
disease_rows = {k: v for k, v in report_dict.items()
                if k not in ("accuracy", "macro avg", "weighted avg")}
diseases_sorted = sorted(disease_rows.items(), key=lambda x: x[1]["f1-score"])

fig, ax = plt.subplots(figsize=(12, max(8, len(diseases_sorted) * 0.35)))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

names = [d[0] for d in diseases_sorted]
f1s = [d[1]["f1-score"] for d in diseases_sorted]
bar_colors = [COLORS["success"] if f >= 0.9 else COLORS["warning"] if f >= 0.7 else COLORS["danger"] for f in f1s]

bars = ax.barh(names, f1s, color=bar_colors, edgecolor="none", height=0.7)
ax.set_xlabel("F1-Score", color=COLORS["text"], fontsize=12)
ax.set_title("Per-Disease F1 Scores (Random Forest)", color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, 1.05)
ax.tick_params(colors=COLORS["muted"], labelsize=9)
ax.axvline(x=0.9, color=COLORS["muted"], linestyle="--", alpha=0.4, label="0.9 threshold")

for bar, val in zip(bars, f1s):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
            va="center", fontsize=8, color=COLORS["text"])

ax.legend(loc="lower right", fontsize=9)
save_fig(fig, "01_f1_scores.png")

# ── Chart 2: Confusion Matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
class_names = [disease_list[i] for i in sorted(set(y_test))]

fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor=COLORS["card"],
            cbar_kws={"shrink": 0.8})
ax.set_xlabel("Predicted", color=COLORS["text"], fontsize=12)
ax.set_ylabel("Actual", color=COLORS["text"], fontsize=12)
ax.set_title("Confusion Matrix — Random Forest Classifier", color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=COLORS["muted"], labelsize=7)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
save_fig(fig, "02_confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════════════
# 2. RULE ENGINE COVERAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. RULE ENGINE COVERAGE ANALYSIS")
print("=" * 70)

re = RuleEngine()
rule_data = []
for disease, symptoms in re.rules.items():
    rule_data.append({
        "Disease": disease,
        "Required Symptoms": len(symptoms),
        "Symptom List": ", ".join(symptoms)
    })

rule_df = pd.DataFrame(rule_data)
print(f"\n  Total diseases with rules: {len(rule_df)}")
print(f"\n  Rule Coverage Table:")
print(rule_df[["Disease", "Required Symptoms"]].to_string(index=False))

# ── Chart 3: Rule Symptoms Count ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

rule_df_sorted = rule_df.sort_values("Required Symptoms", ascending=True)
bars = ax.barh(rule_df_sorted["Disease"], rule_df_sorted["Required Symptoms"],
               color=COLORS["accent"], edgecolor="none", height=0.65)
ax.set_xlabel("Number of Required Symptoms", color=COLORS["text"], fontsize=12)
ax.set_title("Rule Engine — Symptoms per Disease Rule", color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=COLORS["muted"], labelsize=9)

for bar, val in zip(bars, rule_df_sorted["Required Symptoms"]):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, str(val),
            va="center", fontsize=9, color=COLORS["text"])

save_fig(fig, "03_rule_coverage.png")

# ═══════════════════════════════════════════════════════════════════════
# 3. CONFIDENCE FUSION COMPARISON
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. CONFIDENCE FUSION COMPARISON (Sample Cases)")
print("=" * 70)

agent = MedicalTriageAgent()

test_cases = {
    "Malaria": ["chills", "high_fever", "sweating", "headache", "muscle_pain"],
    "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness"],
    "Tuberculosis": ["cough", "weight_loss", "sweating", "blood_in_sputum"],
    "Pneumonia": ["cough", "breathlessness", "phlegm", "fast_heart_rate"],
    "Dengue": ["skin_rash", "high_fever", "joint_pain", "pain_behind_the_eyes"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    "Common Cold": ["continuous_sneezing", "chills"],
}

fusion_rows = []
for case_name, syms in test_cases.items():
    res = agent.run(syms)
    top = res["top_predictions"][0]
    fusion_rows.append({
        "Test Case": case_name,
        "Predicted": top["disease"],
        "Rule Contribution": f"{top['rule_contribution'] * 100:.1f}%",
        "ML Contribution": f"{top['ml_contribution'] * 100:.1f}%",
        "Fused Confidence": f"{top['confidence'] * 100:.1f}%",
        "Triage": res["triage"]["level"],
        "Correct": "Y" if top["disease"] == case_name else "N"
    })

fusion_df = pd.DataFrame(fusion_rows)
print(f"\n{fusion_df.to_string(index=False)}")

# ── Chart 4: Fusion Comparison Grouped Bar ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

x = np.arange(len(test_cases))
width = 0.25

rule_vals = [float(r["Rule Contribution"].strip("%")) for r in fusion_rows]
ml_vals = [float(r["ML Contribution"].strip("%")) for r in fusion_rows]
fused_vals = [float(r["Fused Confidence"].strip("%")) for r in fusion_rows]

ax.bar(x - width, rule_vals, width, label="Rule Engine (40%)", color=COLORS["warning"], edgecolor="none")
ax.bar(x, ml_vals, width, label="ML Classifier (60%)", color=COLORS["primary"], edgecolor="none")
ax.bar(x + width, fused_vals, width, label="Fused Confidence", color=COLORS["success"], edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(test_cases.keys(), rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Contribution (%)", color=COLORS["text"], fontsize=12)
ax.set_title("Confidence Fusion — Rule vs ML vs Fused Score", color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=COLORS["muted"])
ax.legend(fontsize=10)
ax.set_ylim(0, 110)

save_fig(fig, "04_fusion_comparison.png")

# ═══════════════════════════════════════════════════════════════════════
# 4. TRIAGE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. TRIAGE DISTRIBUTION ACROSS TEST CASES")
print("=" * 70)

triage_counts = {"High": 0, "Moderate": 0, "Low": 0}
for row in fusion_rows:
    triage_counts[row["Triage"]] += 1

print(f"\n  High:     {triage_counts['High']} cases")
print(f"  Moderate: {triage_counts['Moderate']} cases")
print(f"  Low:      {triage_counts['Low']} cases")

# ── Chart 5: Triage Pie Chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

labels = list(triage_counts.keys())
sizes = list(triage_counts.values())
colors_pie = [COLORS["danger"], COLORS["warning"], COLORS["success"]]
explode = (0.05, 0.02, 0.02)

wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=labels, autopct="%1.0f%%",
    colors=colors_pie, startangle=90, textprops={"color": COLORS["text"], "fontsize": 13},
    wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2}
)
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title("Triage Level Distribution", color=COLORS["text"], fontsize=14, fontweight="bold", pad=20)

save_fig(fig, "05_triage_distribution.png")

# ═══════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE (Top 20)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. TOP 20 FEATURE IMPORTANCES (Random Forest)")
print("=" * 70)

with open(os.path.join(PROCESSED_DIR, "symptom_list.json"), "r") as f:
    symptom_list = json.load(f)

importances = ml.model.feature_importances_
imp_df = pd.DataFrame({
    "Symptom": symptom_list,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(f"\n{imp_df.head(20).to_string(index=False)}")

# ── Chart 6: Feature Importance ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

top20 = imp_df.head(20).sort_values("Importance", ascending=True)
pretty_names = [s.replace("_", " ").title() for s in top20["Symptom"]]
bars = ax.barh(pretty_names, top20["Importance"], color=COLORS["secondary"], edgecolor="none", height=0.65)
ax.set_xlabel("Feature Importance", color=COLORS["text"], fontsize=12)
ax.set_title("Top 20 Most Important Symptoms (Random Forest)", color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=COLORS["muted"], labelsize=9)

for bar, val in zip(bars, top20["Importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
            va="center", fontsize=8, color=COLORS["text"])

save_fig(fig, "06_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULT ANALYSIS COMPLETE")
print("=" * 70)
print(f"\n  ML Accuracy:          {accuracy * 100:.2f}%")
print(f"  Macro F1-Score:       {report_dict['macro avg']['f1-score']:.3f}")
print(f"  Weighted F1-Score:    {report_dict['weighted avg']['f1-score']:.3f}")
print(f"  Rule Coverage:        {len(re.rules)} diseases")
print(f"  Agent Fusion Cases:   {len(test_cases)} tested, {sum(1 for r in fusion_rows if r['Correct'] == 'Y')}/{len(test_cases)} correct")
print(f"\n  All charts saved to: {RESULTS_DIR}/")
print(f"  Files generated:")
print(f"    01_f1_scores.png")
print(f"    02_confusion_matrix.png")
print(f"    03_rule_coverage.png")
print(f"    04_fusion_comparison.png")
print(f"    05_triage_distribution.png")
print(f"    06_feature_importance.png")
print(f"    classification_report.csv")
