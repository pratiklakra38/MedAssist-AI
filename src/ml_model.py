import os
import json
import joblib
import pandas as pd
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class MLModel:
    """
    Probabilistic Machine Learning Model for Disease Prediction.
    Uses Random Forest to capture symptom co-occurrence patterns.
    """
    def __init__(self, model_path: str = None, data_dir: str = None):
        self.model = None
        self.disease_list = []
        self.symptom_list = []
        
        # Determine paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(self.project_root, "data", "processed")
        self.model_path = model_path or os.path.join(self.project_root, "models", "random_forest.pkl")
        
        self.load_metadata()
        if os.path.exists(self.model_path):
            self.load_model()

    def load_metadata(self):
        disease_path = os.path.join(self.data_dir, "disease_list.json")
        symptom_path = os.path.join(self.data_dir, "symptom_list.json")
        if os.path.exists(disease_path):
            with open(disease_path, "r") as f:
                self.disease_list = json.load(f)
        if os.path.exists(symptom_path):
            with open(symptom_path, "r") as f:
                self.symptom_list = json.load(f)

    def train(self):
        """Trains the Random Forest classifier and saves it."""
        print("Loading training data...")
        X_train = pd.read_csv(os.path.join(self.data_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(self.data_dir, "y_train.csv"))
        X_test = pd.read_csv(os.path.join(self.data_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(self.data_dir, "y_test.csv"))
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit model
        self.model.fit(X_train, y_train.values.ravel())
        
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.disease_list))
        
        self.save_model()
        print(f"Model saved to {self.model_path}")
        return acc

    def save_model(self):
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def predict_proba(self, active_symptoms: List[str]) -> Dict[str, float]:
        """
        Predicts disease probabilities based on active symptoms.
        Returns a dictionary mapping disease names to probability scores.
        """
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
            
        vector = [0] * len(self.symptom_list)
        for sym in active_symptoms:
            if sym in self.symptom_list:
                idx = self.symptom_list.index(sym)
                vector[idx] = 1
                
        df_input = pd.DataFrame([vector], columns=self.symptom_list)
        probabilities = self.model.predict_proba(df_input)[0]
        
        result = {}
        for idx, prob in enumerate(probabilities):
            if prob > 0:
                disease_name = self.disease_list[self.model.classes_[idx]]
                result[disease_name] = round(float(prob), 4)
                
        # Sort descending
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    ml = MLModel()
    ml.train()
