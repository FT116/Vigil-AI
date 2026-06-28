import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def main():
    print("=" * 100)
    print("Training XGBoost baseline")
    print("=" * 100)
    
    # 1. Load data
    balanced_data_path = "data/training_balanced_clean.csv"
    augmented_data_path = "data/training_augmented_clean.csv"
    if os.path.exists(balanced_data_path):
        print(f"Loading data from {balanced_data_path}")
        df = pd.read_csv(balanced_data_path)
    else:
        print(f"Loading data from {augmented_data_path}")
        df = pd.read_csv(augmented_data_path)
    
    # Clean data
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    
    # 2. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    X = df["text"]
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    # 5. Create pipeline
    print("\nCreating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2),
        stop_words="english"
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")
    
    # 6. Train XGBoost
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    
    # Compute sample weights for XGBoost
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    model.fit(X_train_tfidf, y_train, sample_weight=sample_weights)
    
    # 7. Evaluate
    print("\nEvaluating XGBoost model...")
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Print report
    print("\n" + "=" * 100)
    print("XGBoost Evaluation Report")
    print("=" * 100)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save report
    report_path = "outputs/reports/xgboost_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("XGBoost Evaluation Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    # Save model and components
    print(f"\nSaving model to outputs/models/xgboost_model.pkl")
    model_save_dict = {
        "model": model,
        "tfidf": tfidf,
        "label_encoder": label_encoder
    }
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model_save_dict, "outputs/models/xgboost_model.pkl")
    
    print(f"Saved report to {report_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
