import pandas as pd
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
model_file = os.path.join(model_dir, "random_forest.pkl")


class DiseasePredictor:
    def __init__(self):
        with open(os.path.join(data_path, "disease_list.json")) as f:
            self.diseases = json.load(f)
        self.model = None

    def train(self):
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()

        self.model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
        self.model.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f"Accuracy: {acc * 100:.2f}%")

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, model_file)
        return acc

    def load(self):
        self.model = joblib.load(model_file)

    def predict(self, user_symptoms):
        if self.model is None:
            self.load()
        probs = self.model.predict_proba([user_symptoms])[0]
        results = {}
        for i, cls in enumerate(self.model.classes_):
            results[self.diseases[cls]] = round(probs[i] * 100, 2)
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    m = DiseasePredictor()
    m.train()
