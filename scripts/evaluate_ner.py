"""
Entity-Level NER Evaluation Script
Provides proper entity-level metrics instead of token-level accuracy.
Uses cyber-only label schema; reports cyber entity F1 summary.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset, DatasetDict
from collections import defaultdict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

# Configuration (paths relative to project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "secbert_ner_final")
CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "cyber", "cyberner_clean.csv")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "datasets", "cyber", "ner_cyber_labels.json")
if not os.path.exists(SCHEMA_PATH):
    SCHEMA_PATH = os.path.join(PROJECT_ROOT, "config", "ner_cyber_labels.json")

# Core cyber entity types for summary (B- prefix for seqeval report keys)
CYBER_ENTITY_TYPES = ["APT", "MALWARE", "VULNERABILITY", "TOOL", "EXPLOIT", "THREAT_ACTOR", "METHOD", "CAMPAIGN", "INDICATOR", "HASH", "IP", "URL", "FILE", "SOFTWARE", "INFRASTRUCTURE"]

print("="*80)
print("ENTITY-LEVEL NER EVALUATION")
print("="*80)
print()

# Load model and tokenizer (use model's cyber-only label config)
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()
label2id = model.config.label2id
id2label = model.config.id2label
print("[OK] Model loaded (cyber-only labels from config)")
print()

# Load schema for tag mapping (CSV may have raw 75 tags)
if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    tag_mapping = schema.get("tag_mapping", {})
else:
    tag_mapping = {}

# Load and prepare dataset
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df['Word'] = df['Word'].fillna("#")
df['Tag'] = df['Tag'].fillna("O")

# Map tags to cyber-only schema (unmapped -> O)
def map_tag(t):
    return tag_mapping.get(t, "O")
df["Tag"] = df["Tag"].apply(map_tag)

# Group by Sentence_ID
grouped = df.groupby("Sentence_ID").agg({
    "Word": list,
    "Tag": list
}).reset_index()

print(f"[OK] Loaded {len(grouped)} sentences from {os.path.basename(CSV_PATH)}")
print()

# Convert tags to IDs using model's label2id
def convert_tags_to_ids(tag_list):
    return [label2id.get(t, label2id["O"]) for t in tag_list]

grouped["ner_tags"] = grouped["Tag"].apply(convert_tags_to_ids)
grouped = grouped.rename(columns={"Word": "tokens"})

# Create train/validation split (same as training)
full_ds = Dataset.from_pandas(grouped[["tokens", "ner_tags"]])
split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
validation_dataset = split_ds["test"]

print(f"Validation set size: {len(validation_dataset)} sentences")
print()

# Tokenization with offset mapping
def tokenize_and_align_labels_with_offsets(examples):
    """Tokenize with offset mapping to track word boundaries"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        return_offsets_mapping=False  # We'll track word_ids instead
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First subtoken gets label
            else:
                label_ids.append(-100)  # Other subtokens masked
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing validation set...")
tokenized_validation = validation_dataset.map(
    tokenize_and_align_labels_with_offsets, 
    batched=True
)
print("[OK] Tokenization complete")
print()

# Run inference
print("Running inference on validation set...")
all_predictions = []
all_labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for example in tokenized_validation:
    input_ids = torch.tensor([example["input_ids"]]).to(device)
    attention_mask = torch.tensor([example["attention_mask"]]).to(device)
    labels = example["labels"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    # Extract only non-masked predictions
    pred_labels = []
    true_labels = []
    
    for pred, label in zip(predictions, labels):
        if label != -100:  # Skip masked tokens
            pred_labels.append(id2label[pred])
            true_labels.append(id2label[label])
    
    all_predictions.append(pred_labels)
    all_labels.append(true_labels)

print("[OK] Inference complete")
print()

# ============================================================================
# ENTITY-LEVEL METRICS (What your client wants!)
# ============================================================================

print("="*80)
print("ENTITY-LEVEL METRICS")
print("="*80)
print()

# Calculate entity-level metrics using seqeval
entity_f1 = f1_score(all_labels, all_predictions, mode='strict', scheme=IOB2)
entity_precision = precision_score(all_labels, all_predictions, mode='strict', scheme=IOB2)
entity_recall = recall_score(all_labels, all_predictions, mode='strict', scheme=IOB2)

print("Overall Entity-Level Performance:")
print(f"  Precision: {entity_precision:.4f} ({entity_precision*100:.2f}%)")
print(f"  Recall:    {entity_recall:.4f} ({entity_recall*100:.2f}%)")
print(f"  F1-Score:  {entity_f1:.4f} ({entity_f1*100:.2f}%)")
print()

# Detailed per-label metrics
print("="*80)
print("PER-LABEL ENTITY METRICS")
print("="*80)
print()

report = classification_report(
    all_labels, 
    all_predictions, 
    mode='strict', 
    scheme=IOB2,
    output_dict=True,
    zero_division=0
)

# Print per-entity metrics
print(f"{'Label':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print("-"*80)

for label in sorted(report.keys()):
    if label in ['micro avg', 'macro avg', 'weighted avg']:
        continue
    
    metrics = report[label]
    print(f"{label:<30} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
          f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<12}")

print("-"*80)
print()

# Cyber entity summary (F1 for core cyber types only)
print("="*80)
print("CYBER ENTITY SUMMARY (core security types)")
print("="*80)
cyber_keys = [k for k in report if k not in ("micro avg", "macro avg", "weighted avg", "O")
             and (k.split("-")[-1] if "-" in k else k) in CYBER_ENTITY_TYPES]
if cyber_keys:
    for k in sorted(cyber_keys):
        m = report[k]
        print(f"  {k:<28} F1={m['f1-score']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  support={int(m['support'])}")
    cyber_f1_avg = np.mean([report[k]["f1-score"] for k in cyber_keys]) if cyber_keys else 0
    print(f"  {'(avg cyber types)':<28} F1={cyber_f1_avg:.4f}")
else:
    print("  (no cyber entity types in report)")
print()

# Macro averages
print("Macro Averages (across all entity types):")
print(f"  Precision: {report['macro avg']['precision']:.4f}")
print(f"  Recall:    {report['macro avg']['recall']:.4f}")
print(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}")
print()

# ============================================================================
# CONFUSION BREAKDOWN: O vs Entity Tokens
# ============================================================================

print("="*80)
print("CONFUSION BREAKDOWN: O vs ENTITY TOKENS")
print("="*80)
print()

# Flatten all predictions and labels
flat_predictions = [item for sublist in all_predictions for item in sublist]
flat_labels = [item for sublist in all_labels for item in sublist]

# Count O vs Entity tokens
true_o_count = sum(1 for label in flat_labels if label == 'O')
true_entity_count = sum(1 for label in flat_labels if label != 'O')

pred_o_count = sum(1 for pred in flat_predictions if pred == 'O')
pred_entity_count = sum(1 for pred in flat_predictions if pred != 'O')

# Confusion matrix: O vs Entity
true_o_pred_o = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                    if true == 'O' and pred == 'O')
true_o_pred_entity = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                         if true == 'O' and pred != 'O')
true_entity_pred_o = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                         if true != 'O' and pred == 'O')
true_entity_pred_entity = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                              if true != 'O' and pred != 'O')

print("Token Distribution:")
print(f"  Total tokens: {len(flat_labels)}")
print(f"  True O tokens: {true_o_count} ({true_o_count/len(flat_labels)*100:.1f}%)")
print(f"  True Entity tokens: {true_entity_count} ({true_entity_count/len(flat_labels)*100:.1f}%)")
print()

print("Confusion Matrix (O vs Entity):")
print(f"{'':>20} {'Predicted O':>15} {'Predicted Entity':>18}")
print("-"*55)
print(f"{'True O':>20} {true_o_pred_o:>15} {true_o_pred_entity:>18}")
print(f"{'True Entity':>20} {true_entity_pred_o:>15} {true_entity_pred_entity:>18}")
print()

# Calculate metrics for O vs Entity classification
if true_entity_count > 0:
    entity_detection_precision = true_entity_pred_entity / (true_entity_pred_entity + true_o_pred_entity) if (true_entity_pred_entity + true_o_pred_entity) > 0 else 0
    entity_detection_recall = true_entity_pred_entity / true_entity_count
    entity_detection_f1 = 2 * (entity_detection_precision * entity_detection_recall) / (entity_detection_precision + entity_detection_recall) if (entity_detection_precision + entity_detection_recall) > 0 else 0
    
    print("Entity Detection Metrics (O vs Any Entity):")
    print(f"  Precision: {entity_detection_precision:.4f} ({entity_detection_precision*100:.2f}%)")
    print(f"  Recall:    {entity_detection_recall:.4f} ({entity_detection_recall*100:.2f}%)")
    print(f"  F1-Score:  {entity_detection_f1:.4f} ({entity_detection_f1*100:.2f}%)")
    print()


import json

evaluation_results = {
    "model_path": MODEL_PATH,
    "validation_size": len(validation_dataset),
    "total_tokens_evaluated": len(flat_labels),
    "entity_level_metrics": {
        "overall": {
            "precision": float(entity_precision),
            "recall": float(entity_recall),
            "f1_score": float(entity_f1)
        },
        "per_label": {}
    },
    "confusion_breakdown": {
        "true_o_tokens": int(true_o_count),
        "true_entity_tokens": int(true_entity_count),
        "true_o_pred_o": int(true_o_pred_o),
        "true_o_pred_entity": int(true_o_pred_entity),
        "true_entity_pred_o": int(true_entity_pred_o),
        "true_entity_pred_entity": int(true_entity_pred_entity),
        "entity_detection_precision": float(entity_detection_precision) if true_entity_count > 0 else 0,
        "entity_detection_recall": float(entity_detection_recall) if true_entity_count > 0 else 0,
        "entity_detection_f1": float(entity_detection_f1) if true_entity_count > 0 else 0
    }
}

# Add per-label metrics
for label in report.keys():
    if label not in ['micro avg', 'macro avg', 'weighted avg']:
        evaluation_results["entity_level_metrics"]["per_label"][label] = {
            "precision": float(report[label]['precision']),
            "recall": float(report[label]['recall']),
            "f1_score": float(report[label]['f1-score']),
            "support": int(report[label]['support'])
        }

# Save to JSON
output_path = os.path.join(PROJECT_ROOT, "evaluation_results.json")
with open(output_path, 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("="*80)
print(f"[OK] Evaluation results saved to: {output_path}")
print("="*80)
print()

print("SUMMARY FOR CLIENT:")
print("-"*80)
print(f"Entity-Level F1: {entity_f1:.4f} ({entity_f1*100:.2f}%)")
print(f"Entity Detection Rate: {entity_detection_recall:.4f} ({entity_detection_recall*100:.2f}%)")
print(f"Entity Precision: {entity_precision:.4f} ({entity_precision*100:.2f}%)")
print(f"Total Entity Types: {len([k for k in report.keys() if k not in ['micro avg', 'macro avg', 'weighted avg']])}")
print("-"*80)
