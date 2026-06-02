"""
Re-run Mini-CyBERT NER evaluation with confusion matrices and MCC.

Matches the training notebook split strategy (seed=42):
  80% train -> 80/20 train/val  => validation
  20% held out                   => test

Reports entity-level P/R/F1 (seqeval), macro-F1, token accuracy, and:
  - Binary entity-detection MCC (O vs any entity token)
  - Token-level multiclass MCC (31 labels)
  - Per-entity-family binary MCC (optional export)

Outputs JSON/CSV/LaTeX snippets under final-edits/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mini_cybert_final"
DEFAULT_CSV = PROJECT_ROOT / "datasets" / "cyber" / "cyberner_clean.csv"
DEFAULT_AUGMENT = PROJECT_ROOT / "datasets" / "cyber" / "ner_attack_type_augment.csv"
DEFAULT_SCHEMA = PROJECT_ROOT / "config" / "ner_cyber_labels.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CYBER_ENTITY_TYPES = [
    "APT", "MALWARE", "VULNERABILITY", "TOOL", "EXPLOIT", "THREAT_ACTOR",
    "METHOD", "CAMPAIGN", "INDICATOR", "HASH", "IP", "URL", "FILE",
    "SOFTWARE", "INFRASTRUCTURE",
]


def load_schema(schema_path: Path) -> tuple[list[str], dict[str, str]]:
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    return schema["label_list"], schema.get("tag_mapping", {})


def load_ner_dataset(
    csv_path: Path,
    augment_path: Path | None,
    tag_mapping: dict[str, str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if augment_path and augment_path.exists():
        df_aug = pd.read_csv(augment_path)
        df = pd.concat([df, df_aug], ignore_index=True)

    df["Word"] = df["Word"].fillna("#")
    df["Tag"] = df["Tag"].fillna("O").apply(lambda t: tag_mapping.get(t, "O"))

    grouped = (
        df.groupby("Sentence_ID")
        .agg({"Word": list, "Tag": list})
        .reset_index()
        .rename(columns={"Word": "tokens", "Tag": "tags"})
    )
    return grouped


def build_splits(grouped: pd.DataFrame, label2id: dict[str, int], seed: int = 42) -> DatasetDict:
    grouped = grouped.copy()
    grouped["ner_tags"] = grouped["tags"].apply(
        lambda tags: [label2id.get(t, label2id["O"]) for t in tags]
    )

    full_ds = Dataset.from_pandas(grouped[["tokens", "ner_tags"]])
    outer = full_ds.train_test_split(test_size=0.2, seed=seed)
    inner = outer["train"].train_test_split(test_size=0.2, seed=seed)
    return DatasetDict(
        {
            "train": inner["train"],
            "validation": inner["test"],
            "test": outer["test"],
        }
    )


def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    labels = []
    for i, tag_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag_ids[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def run_inference(model, tokenizer, dataset, id2label, device) -> tuple[list[list[str]], list[list[str]]]:
    model.eval()
    model.to(device)

    all_predictions: list[list[str]] = []
    all_labels: list[list[str]] = []

    for example in dataset:
        input_ids = torch.tensor([example["input_ids"]]).to(device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(device)
        labels = example["labels"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        pred_labels: list[str] = []
        true_labels: list[str] = []
        for pred, label in zip(preds, labels):
            if label != -100:
                pred_labels.append(id2label[int(pred)])
                true_labels.append(id2label[int(label)])

        all_predictions.append(pred_labels)
        all_labels.append(true_labels)

    return all_predictions, all_labels


def flatten_tokens(
    predictions: list[list[str]], labels: list[list[str]]
) -> tuple[list[str], list[str]]:
    flat_pred = [p for sent in predictions for p in sent]
    flat_true = [t for sent in labels for t in sent]
    return flat_pred, flat_true


def binary_entity_confusion(flat_true: list[str], flat_pred: list[str]) -> dict[str, int]:
    true_is_entity = [t != "O" for t in flat_true]
    pred_is_entity = [p != "O" for p in flat_pred]

    tn = sum(1 for t, p in zip(true_is_entity, pred_is_entity) if not t and not p)
    fp = sum(1 for t, p in zip(true_is_entity, pred_is_entity) if not t and p)
    fn = sum(1 for t, p in zip(true_is_entity, pred_is_entity) if t and not p)
    tp = sum(1 for t, p in zip(true_is_entity, pred_is_entity) if t and p)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def mcc_from_binary_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def token_accuracy(flat_true: list[str], flat_pred: list[str]) -> float:
    if not flat_true:
        return 0.0
    return sum(t == p for t, p in zip(flat_true, flat_pred)) / len(flat_true)


def entity_family(label: str) -> str | None:
    if label == "O":
        return None
    return label.split("-", 1)[-1]


def per_entity_binary_mcc(
    flat_true: list[str], flat_pred: list[str], entity_types: list[str]
) -> dict[str, dict[str, float | int]]:
    results = {}
    for etype in entity_types:
        true_bin = [1 if entity_family(t) == etype else 0 for t in flat_true]
        pred_bin = [1 if entity_family(p) == etype else 0 for p in flat_pred]
        tp = sum(1 for t, p in zip(true_bin, pred_bin) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_bin, pred_bin) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_bin, pred_bin) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_bin, pred_bin) if t == 1 and p == 0)
        support = sum(true_bin)
        mcc = mcc_from_binary_counts(tp, tn, fp, fn) if support > 0 else 0.0
        results[etype] = {
            "mcc": mcc,
            "support_tokens": support,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
    return results


def evaluate_split(
    split_name: str,
    predictions: list[list[str]],
    labels: list[list[str]],
    label_list: list[str],
) -> dict:
    flat_pred, flat_true = flatten_tokens(predictions, labels)

    entity_precision = precision_score(labels, predictions, mode="strict", scheme=IOB2)
    entity_recall = recall_score(labels, predictions, mode="strict", scheme=IOB2)
    entity_f1 = f1_score(labels, predictions, mode="strict", scheme=IOB2)

    report = classification_report(
        labels,
        predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    acc = token_accuracy(flat_true, flat_pred)

    binary_counts = binary_entity_confusion(flat_true, flat_pred)
    entity_mcc = mcc_from_binary_counts(
        binary_counts["TP"], binary_counts["TN"], binary_counts["FP"], binary_counts["FN"]
    )

    true_ids = [label_list.index(t) if t in label_list else label_list.index("O") for t in flat_true]
    pred_ids = [label_list.index(p) if p in label_list else label_list.index("O") for p in flat_pred]
    token_mcc = float(matthews_corrcoef(true_ids, pred_ids))

    cm_multiclass = confusion_matrix(true_ids, pred_ids, labels=list(range(len(label_list))))

    classwise = {}
    for label in report:
        if label in {"micro avg", "macro avg", "weighted avg"}:
            continue
        classwise[label] = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }

    cyber_classwise = {}
    for etype in CYBER_ENTITY_TYPES:
        b_label = f"B-{etype}"
        i_label = f"I-{etype}"
        supports = []
        f1s = []
        for lbl in (b_label, i_label):
            if lbl in classwise:
                supports.append(classwise[lbl]["support"])
                f1s.append(classwise[lbl]["f1"])
        if supports:
            cyber_classwise[etype] = {
                "precision_b": classwise.get(b_label, {}).get("precision"),
                "recall_b": classwise.get(b_label, {}).get("recall"),
                "f1_b": classwise.get(b_label, {}).get("f1"),
                "support_b": classwise.get(b_label, {}).get("support", 0),
                "precision_i": classwise.get(i_label, {}).get("precision"),
                "recall_i": classwise.get(i_label, {}).get("recall"),
                "f1_i": classwise.get(i_label, {}).get("f1"),
                "support_i": classwise.get(i_label, {}).get("support", 0),
                "entity_f1_avg": float(np.mean(f1s)) if f1s else 0.0,
                "entity_support_sum": int(sum(supports)),
            }

    per_entity_mcc = per_entity_binary_mcc(flat_true, flat_pred, CYBER_ENTITY_TYPES)

    return {
        "split": split_name,
        "sentences": len(labels),
        "tokens_evaluated": len(flat_true),
        "entity_level": {
            "precision": float(entity_precision),
            "recall": float(entity_recall),
            "f1": float(entity_f1),
            "macro_f1": float(macro_f1),
            "token_accuracy": float(acc),
        },
        "mcc": {
            "entity_detection_binary": float(entity_mcc),
            "token_multiclass_31_labels": float(token_mcc),
        },
        "confusion_binary_o_vs_entity": binary_counts,
        "confusion_multiclass_labels": label_list,
        "confusion_multiclass_matrix": cm_multiclass.tolist(),
        "classwise_token_metrics": classwise,
        "cyber_entity_summary": cyber_classwise,
        "per_entity_family_binary_mcc": per_entity_mcc,
    }


def save_confusion_csv(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "true_label"
    df.to_csv(path)


def save_main_results_table(results: dict, path: Path) -> None:
    rows = []
    for split in ("validation", "test"):
        r = results[split]
        e = r["entity_level"]
        m = r["mcc"]
        rows.append(
            {
                "Model": "Mini-CyBERT NER",
                "Split": split.capitalize(),
                "Precision": round(e["precision"], 4),
                "Recall": round(e["recall"], 4),
                "F1_Entity_Level": round(e["f1"], 4),
                "Macro_F1": round(e["macro_f1"], 4),
                "Token_Accuracy": round(e["token_accuracy"], 4),
                "MCC_Entity_Detection": round(m["entity_detection_binary"], 4),
                "MCC_Token_Multiclass": round(m["token_multiclass_31_labels"], 4),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_latex_snippets(results: dict, path: Path) -> None:
    val = results["validation"]["entity_level"]
    test = results["test"]["entity_level"]
    val_mcc = results["validation"]["mcc"]["entity_detection_binary"]
    test_mcc = results["test"]["mcc"]["entity_detection_binary"]
    val_mcc_mc = results["validation"]["mcc"]["token_multiclass_31_labels"]
    test_mcc_mc = results["test"]["mcc"]["token_multiclass_31_labels"]

    val_bin = results["validation"]["confusion_binary_o_vs_entity"]
    test_bin = results["test"]["confusion_binary_o_vs_entity"]

    lines = [
        "% Auto-generated by final-edits/evaluate_with_mcc.py",
        "% Main NER table with MCC (entity-detection binary MCC recommended for imbalanced reporting)",
        "",
        "% --- Table: tab:main_ner (replace in paper) ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main NER performance on validation and test splits (entity-level P/R/F1; macro-F1; token accuracy; MCC).}",
        "\\label{tab:main_ner}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Split} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{Macro-F1} & \\textbf{Acc.} & \\textbf{MCC} \\\\",
        "\\midrule",
        f"Validation & {val['precision']:.3f} & {val['recall']:.3f} & {val['f1']:.3f} & {val['macro_f1']:.3f} & {val['token_accuracy']:.3f} & {val_mcc:.3f} \\\\",
        f"Test         & {test['precision']:.3f} & {test['recall']:.3f} & {test['f1']:.3f} & {test['macro_f1']:.3f} & {test['token_accuracy']:.3f} & {test_mcc:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\vspace{0.25em}",
        "",
        f"\\noindent{{\\scriptsize "
        f"\\textbf{{Notes:}} F1 is entity-level (seqeval strict IOB2). Acc.\\ is token-level. "
        f"MCC is binary entity-detection (\\texttt{{O}} vs.\\ any entity token), computed from the confusion matrix below; "
        f"token-level 31-class MCC: validation={val_mcc_mc:.3f}, test={test_mcc_mc:.3f}.}}",
        "\\end{table}",
        "",
        "% --- Table: tab:confusion_binary (new quantitative confusion matrix) ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Token-level confusion matrix for entity detection (\\texttt{O} vs.\\ entity) on the test split.}",
        "\\label{tab:confusion_binary}",
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        " & \\textbf{Predicted O} & \\textbf{Predicted Entity} \\\\",
        "\\midrule",
        f"True O & {test_bin['TN']:,} & {test_bin['FP']:,} \\\\",
        f"True Entity & {test_bin['FN']:,} & {test_bin['TP']:,} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mini-CyBERT NER with MCC and confusion matrices.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--augment-path", type=Path, default=DEFAULT_AUGMENT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batched inference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        return 1
    if not args.csv_path.exists():
        print(f"Dataset not found: {args.csv_path}", file=sys.stderr)
        return 1

    label_list, tag_mapping = load_schema(args.schema_path)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    grouped = load_ner_dataset(args.csv_path, args.augment_path, tag_mapping)
    ner_dataset = build_splits(grouped, label2id, seed=args.seed)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Splits: train={len(ner_dataset['train'])}, "
        f"validation={len(ner_dataset['validation'])}, "
        f"test={len(ner_dataset['test'])}"
    )

    tokenized = {}
    for split in ("validation", "test"):
        tokenized[split] = ner_dataset[split].map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id),
            batched=True,
        )

    all_results = {}
    for split in ("validation", "test"):
        print(f"\nEvaluating {split}...")
        predictions, labels = run_inference(model, tokenizer, tokenized[split], id2label, device)
        all_results[split] = evaluate_split(split, predictions, labels, label_list)

        bin_counts = all_results[split]["confusion_binary_o_vs_entity"]
        print(
            f"  Entity F1={all_results[split]['entity_level']['f1']:.4f}, "
            f"Macro-F1={all_results[split]['entity_level']['macro_f1']:.4f}, "
            f"Acc={all_results[split]['entity_level']['token_accuracy']:.4f}"
        )
        print(
            f"  MCC (entity detection)={all_results[split]['mcc']['entity_detection_binary']:.4f}, "
            f"MCC (31-class tokens)={all_results[split]['mcc']['token_multiclass_31_labels']:.4f}"
        )
        print(
            f"  Binary confusion: TP={bin_counts['TP']}, TN={bin_counts['TN']}, "
            f"FP={bin_counts['FP']}, FN={bin_counts['FN']}"
        )

        save_confusion_csv(
            all_results[split]["confusion_multiclass_matrix"],
            label_list,
            args.results_dir / f"confusion_matrix_{split}_31class.csv",
        )

    payload = {
        "model_path": str(args.model_path),
        "csv_path": str(args.csv_path),
        "seed": args.seed,
        "splits": {
            "train": len(ner_dataset["train"]),
            "validation": len(ner_dataset["validation"]),
            "test": len(ner_dataset["test"]),
        },
        "validation": all_results["validation"],
        "test": all_results["test"],
    }

    json_path = args.results_dir / "evaluation_with_mcc.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_main_results_table(all_results, args.results_dir / "main_results_with_mcc.csv")
    save_latex_snippets(all_results, args.results_dir / "paper_table_updates.tex")

    from generate_tables_8_9_mcc import main as generate_tables_8_9

    if generate_tables_8_9() != 0:
        print("Warning: could not regenerate Tables 8/9 with MCC.", file=sys.stderr)

    print(f"\nSaved results to {args.results_dir}")
    print(f"  - {json_path.name}")
    print("  - main_results_with_mcc.csv")
    print("  - paper_table_updates.tex")
    print("  - confusion_matrix_validation_31class.csv")
    print("  - confusion_matrix_test_31class.csv")
    print("  - table8_classwise_with_mcc.csv")
    print("  - table9_epoch_with_mcc.csv")
    print("  - paper_tables_8_9_with_mcc.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
