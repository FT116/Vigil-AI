import os
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import TrainingArguments, Trainer
import evaluate
import shutil
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MODELS, BALANCED_DATA_PATH, OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_LOGS_DIR,
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE
)
from src.build_dataset import build_final_dataset
from src.data_preprocessing import load_and_prepare_data
from src.dataset import create_hf_dataset
from src.model import load_tokenizer, load_classification_model
from src.trainer import WeightedTrainer, compute_custom_class_weights

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def main():
    for d in [OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(BALANCED_DATA_PATH):
        print(f"{BALANCED_DATA_PATH} not found.")
        raise FileNotFoundError("Please run create_balanced_training_data.py first!")

    train_df, val_df, test_df, label_encoder, label2id, id2label = load_and_prepare_data()
    num_labels = len(label2id)

    # Only train roberta
    model_key = "roberta"
    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f" Training {model_key}")
    print(f"{'='*60}")

    # Load model and tokenizer
    tokenizer = load_tokenizer(model_name)
    model = load_classification_model(model_name, num_labels, id2label, label2id)

    # Prepare datasets
    train_dataset = create_hf_dataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = create_hf_dataset(val_df, tokenizer, MAX_LENGTH)
    test_dataset = create_hf_dataset(test_df, tokenizer, MAX_LENGTH)

    # Compute class weights
    class_weights = compute_custom_class_weights(train_df, num_labels)

    # Training arguments
    temp_model_dir = os.path.join(OUTPUT_MODELS_DIR, f"{model_key}_temp")
    training_args = TrainingArguments(
        output_dir=temp_model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_LOGS_DIR, model_key),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Evaluate on test set
    print(f"\n Evaluating {model_key} on test set...")
    start_inference = time.time()
    test_results = trainer.predict(test_dataset)
    inference_time = (time.time() - start_inference) / len(test_dataset)

    metrics = test_results.metrics
    print(f"\n Test metrics: {metrics}")
    preds = np.argmax(test_results.predictions, axis=-1)
    labels = test_dataset["label"]

    print("\n" + classification_report(labels, preds, target_names=label_encoder.classes_))
    report_path = os.path.join(OUTPUT_REPORTS_DIR, f"{model_key}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(labels, preds, target_names=label_encoder.classes_))

    # Check previous model comparison
    existing_comparison_path = os.path.join(OUTPUT_REPORTS_DIR, "model_comparison.csv")
    existing_df = pd.DataFrame()
    if os.path.exists(existing_comparison_path):
        existing_df = pd.read_csv(existing_comparison_path)

    # Get previous best metrics
    prev_best_f1 = 0
    if len(existing_df) > 0:
        prev_best_f1 = existing_df["f1_macro"].max()

    current_f1 = metrics["test_f1_macro"]
    current_accuracy = metrics["test_accuracy"]

    # Decide if we should save this model
    should_save = False
    if current_f1 > prev_best_f1:
        print(f"\n New best F1 macro ({current_f1:.4f} > {prev_best_f1:.4f})! Saving model.")
        should_save = True
    elif current_accuracy > 0.69:
        print(f"\n Accuracy above 0.69 ({current_accuracy:.4f})! Saving model.")
        should_save = True
    else:
        print(f"\n Model doesn't meet criteria (F1: {current_f1:.4f}, Acc: {current_accuracy:.4f}). Not saving.")

    if should_save:
        # Save final model
        final_model_dir = os.path.join(OUTPUT_MODELS_DIR, model_key)
        if os.path.exists(final_model_dir):
            shutil.rmtree(final_model_dir)
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # Save best model to outputs/models/best_model
        best_model_dir = os.path.join(OUTPUT_MODELS_DIR, "best_model")
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        shutil.copytree(final_model_dir, best_model_dir)
        joblib.dump(label_encoder, os.path.join(best_model_dir, "label_encoder.pkl"))

        # Update model comparison
        new_result = {
            "model_key": model_key,
            "model_name": model_name,
            "accuracy": metrics["test_accuracy"],
            "f1_macro": metrics["test_f1_macro"],
            "f1_weighted": metrics["test_f1_weighted"],
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time
        }
        if len(existing_df) > 0:
            existing_df = existing_df[existing_df["model_key"] != model_key]
        updated_df = pd.concat([existing_df, pd.DataFrame([new_result])], ignore_index=True)
        updated_df.to_csv(existing_comparison_path, index=False)
        print(f"\n Saved model comparison to {existing_comparison_path}")

    # Clean up temp model
    if os.path.exists(temp_model_dir):
        shutil.rmtree(temp_model_dir)

if __name__ == "__main__":
    main()
