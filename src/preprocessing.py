import os
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

def run_preprocessing():
    """
    Data Preprocessing for MedAssist AI.
    - Loads raw datasets.
    - Cleans columns and drops unnecessary features.
    - Encodes target labels.
    - Saves processed split datasets and feature maps.
    """
    print("Starting data preprocessing...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    
    # Load raw data
    train_path = os.path.join(raw_dir, "Training.csv")
    test_path = os.path.join(raw_dir, "Testing.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded raw Training data: {train_df.shape}")
    print(f"Loaded raw Testing data: {test_df.shape}")
    
    # Clean the data: Drop junk columns
    cols_to_drop = ["Unnamed: 133", "fluid_overload.1"]
    for col in cols_to_drop:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
            print(f"Dropped column from Training: {col}")
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)
            print(f"Dropped column from Testing: {col}")
            
    # Clean column names (strip whitespace)
    train_df.columns = [col.strip() for col in train_df.columns]
    test_df.columns = [col.strip() for col in test_df.columns]
    
    # Separate features and target
    target_col = "prognosis"
    X_train_raw = train_df.drop(columns=[target_col])
    y_train_raw = train_df[target_col]
    
    X_test_raw = test_df.drop(columns=[target_col])
    y_test_raw = test_df[target_col]
    
    # Ensure all features are integers
    X_train = X_train_raw.astype(int)
    X_test = X_test_raw.astype(int)
    
    # Encode the target labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    # Save the processed splits
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    
    pd.DataFrame(y_train, columns=[target_col]).to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=[target_col]).to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    print(f"Saved processed data splits to {processed_dir}")
    
    # Save symptom list for future reference (ordered)
    symptom_list = list(X_train.columns)
    with open(os.path.join(processed_dir, "symptom_list.json"), "w") as f:
        json.dump(symptom_list, f, indent=4)
        
    # Save disease class mapping (index corresponding to encoded label)
    disease_list = list(le.classes_)
    with open(os.path.join(processed_dir, "disease_list.json"), "w") as f:
        json.dump(disease_list, f, indent=4)
        
    print(f"Exported {len(symptom_list)} symptoms and {len(disease_list)} disease classes.")
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    run_preprocessing()
