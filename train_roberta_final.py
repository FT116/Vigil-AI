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
    MODELS, OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_LOGS_DIR,
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE
)
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

    # Load and split data
    train_df, val_df, test_df, label_encoder, label2id, id2label = load_and_prepare_data()
    num_labels = len(label2id)

    # Only train roberta
    model_key = "roberta"
    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f" Training RoBERTa")
    print(f"{'='*60}")

    # Load tokenizer and model
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

    # Trainer uses WeightedTrainer with class weights, no Focal Loss!
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
    print(f"\n Evaluating RoBERTa on test set...")
    test_results = trainer.predict(test_dataset)

    metrics = test_results.metrics
    preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    # Print and save reports
    print("\n" + "=" * 80)
    print("Final RoBERTa Test Metrics")
    print("=" * 80)
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {metrics['test_f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['test_f1_weighted']:.4f}")
    
    print("\n" + classification_report(test_labels, preds, target_names=label_encoder.classes_))
    
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "roberta_final_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Final RoBERTa Test Metrics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['test_f1_macro']:.4f}\n")
        f.write(f"Weighted F1: {metrics['test_f1_weighted']:.4f}\n\n")
        f.write(classification_report(test_labels, preds, target_names=label_encoder.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(test_labels, preds)))

    # Save best model to outputs/models/best_model
    best_model_dir = os.path.join(OUTPUT_MODELS_DIR, "best_model")
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    joblib.dump(label_encoder, os.path.join(best_model_dir, "label_encoder.pkl"))
    print(f"\n Best model saved to {best_model_dir}")
    print(f" Report saved to {report_path}")

    # Clean up temp directory
    if os.path.exists(temp_model_dir):
        shutil.rmtree(temp_model_dir)

if __name__ == "__main__":
    main()
