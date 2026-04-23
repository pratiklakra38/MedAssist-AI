import json
import os
from src.rule_engine import rules

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class QuestionBot:
    """Checks if the input is ambiguous and asks a follow-up question."""

    def __init__(self):
        with open(os.path.join(data_path, "symptom_list.json")) as f:
            self.symptoms = json.load(f)

    def check(self, user_symptoms):
        active = {self.symptoms[i] for i, v in enumerate(user_symptoms) if v == 1}

        candidates = {}
        for disease, needed in rules.items():
            needed_set = set(needed)
            hit = needed_set & active
            missing = needed_set - active
            if len(hit) >= 2 and missing:
                ratio = len(hit) / len(needed)
                if 0.3 <= ratio <= 0.7:
                    candidates[disease] = (ratio, sorted(missing))

        if len(candidates) < 2:
            return None

        top = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:2]
        name1, (_, missing1) = top[0]
        name2, _ = top[1]

        unique = set(missing1) - set(rules[name2])
        ask = sorted(unique)[0] if unique else missing1[0]

        return {
            "confused_between": [name1, name2],
            "question": f"Are you also experiencing '{ask.replace('_', ' ')}'?",
            "symptom_key": ask,
            "symptom_index": self.symptoms.index(ask) if ask in self.symptoms else None
        }


class ReviewerBot:
    """Double-checks the prediction before showing it."""

    def check(self, results):
        preds = results["predictions"]
        if len(preds) < 2:
            return results

        first, second = preds[0], preds[1]

        if first["matched_symptoms"]:
            return results

        if not first["matched_symptoms"] and second["matched_symptoms"]:
            results["warning"] = (
                f"'{first['disease']}' was predicted by ML but has no rule-based evidence. "
                f"'{second['disease']}' has stronger symptom alignment."
            )
            preds[0], preds[1] = preds[1], preds[0]

        elif abs(first["confidence"] - second["confidence"]) < 5:
            results["warning"] = (
                f"'{first['disease']}' and '{second['disease']}' have very close scores. "
                f"More symptoms are needed for a reliable prediction."
            )

        return results
