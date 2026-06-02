#!/usr/bin/env python3
"""Add MCC columns to paper Tables 8 and 9 (class-wise test NER + per-epoch validation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
JSON_PATH = RESULTS_DIR / "evaluation_with_mcc.json"


def mcc_from_binary_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

# Table 8 — test split, selected cybersecurity-centric entity types (paper Table 8).
TABLE_8_ROWS = [
    ("VULNERABILITY", 0.791, 0.775, 0.783, 258),
    ("EXPLOIT", 0.840, 0.944, 0.889, 288),
    ("MALWARE", 0.680, 0.716, 0.698, 1125),
    ("THREAT_ACTOR", 0.359, 0.465, 0.405, 170),
    ("APT", 0.777, 0.813, 0.795, 717),
    ("TOOL", 0.609, 0.652, 0.630, 1396),
    ("CAMPAIGN", 0.333, 0.317, 0.325, 41),
    ("METHOD", 0.346, 0.436, 0.386, 397),
    ("HASH", 0.792, 0.927, 0.854, 82),
    ("IP", 0.846, 0.880, 0.863, 50),
    ("INDICATOR", 0.637, 0.766, 0.696, 312),
    ("INFRASTRUCTURE", 0.560, 0.618, 0.587, 144),
]

# Table 9 — validation metrics by training epoch (paper Table 9).
TABLE_9_ROWS = [
    (1, 0.4133, 0.4058, 0.4095, 0.3453, 0.9233),
    (2, 0.5736, 0.5297, 0.5508, 0.5361, 0.9380),
    (3, 0.6161, 0.6083, 0.6122, 0.5856, 0.9443),
    (4, 0.5782, 0.6717, 0.6215, 0.6100, 0.9451),
    (5, 0.5915, 0.6926, 0.6381, 0.6298, 0.9468),
    (6, 0.6543, 0.6854, 0.6695, 0.6338, 0.9525),
    (7, 0.6420, 0.7069, 0.6729, 0.6485, 0.9521),
    (8, 0.6520, 0.7105, 0.6800, 0.6591, 0.9528),
    (9, 0.6594, 0.7177, 0.6873, 0.6654, 0.9538),
    (10, 0.6577, 0.7152, 0.6853, 0.6654, 0.9539),
]


def mcc_per_entity_from_pr(
    precision: float,
    recall: float,
    gold_token_count: int,
    n_tokens: int,
) -> tuple[float, int, int, int, int]:
    """Token-level one-vs-rest MCC for an entity family from P/R and gold token count."""
    if gold_token_count <= 0:
        return 0.0, 0, n_tokens, 0, 0
    tp = int(round(recall * gold_token_count))
    fn = gold_token_count - tp
    if precision > 0:
        fp = int(round(tp / precision - tp))
    else:
        fp = 0
    tn = n_tokens - tp - fp - fn
    mcc = mcc_from_binary_counts(tp, tn, fp, fn)
    return mcc, tp, tn, fp, fn


def mcc_entity_detection_epoch(
    precision_entity: float,
    recall_entity: float,
    n_tokens: int,
    gold_entity_tokens: int,
    calibrate_precision: float,
    calibrate_recall: float,
    tp_ref: int,
    fp_ref: int,
    fn_ref: int,
) -> tuple[float, int, int, int, int]:
    """
    Binary O-vs-entity token MCC per epoch from entity-level validation P/R.

    Maps entity-level P/R to token-level counts using calibration ratios from the
    final validation confusion matrix (same split, seed=42).
    """
    p_tok_ref = tp_ref / (tp_ref + fp_ref) if (tp_ref + fp_ref) else 0.0
    r_tok_ref = tp_ref / (tp_ref + fn_ref) if (tp_ref + fn_ref) else 0.0
    kp = p_tok_ref / calibrate_precision if calibrate_precision > 0 else 1.0
    kr = r_tok_ref / calibrate_recall if calibrate_recall > 0 else 1.0

    tp = int(round(recall_entity * kr * gold_entity_tokens))
    fn = gold_entity_tokens - tp
    p_tok = max(min(precision_entity * kp, 0.999), 1e-6)
    fp = int(round(tp / p_tok - tp)) if tp > 0 else 0
    tn = n_tokens - tp - fp - fn
    mcc = mcc_from_binary_counts(tp, tn, fp, fn)
    return mcc, tp, tn, fp, fn


def build_table_8(test_eval: dict) -> pd.DataFrame:
    n_tokens = int(test_eval["tokens_evaluated"])
    per_entity = test_eval["per_entity_family_binary_mcc"]
    rows = []
    for entity, prec, rec, f1, support in TABLE_8_ROWS:
        gold_tokens = int(per_entity[entity]["TP"] + per_entity[entity]["FN"])
        mcc, tp, tn, fp, fn = mcc_per_entity_from_pr(prec, rec, gold_tokens, n_tokens)
        mcc_actual = float(per_entity[entity]["mcc"])
        rows.append(
            {
                "Entity_Type": entity,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Support": support,
                "MCC": round(mcc, 3),
                "MCC_From_Confusion": round(mcc_actual, 3),
                "Gold_Tokens": gold_tokens,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )
    return pd.DataFrame(rows)


def build_table_9(val_eval: dict) -> pd.DataFrame:
    n_tokens = int(val_eval["tokens_evaluated"])
    bin_counts = val_eval["confusion_binary_o_vs_entity"]
    tp_ref = int(bin_counts["TP"])
    fp_ref = int(bin_counts["FP"])
    fn_ref = int(bin_counts["FN"])
    gold_entity_tokens = tp_ref + fn_ref
    ent = val_eval["entity_level"]
    cal_p = float(ent["precision"])
    cal_r = float(ent["recall"])

    rows = []
    for epoch, prec, rec, f1, macro_f1, acc in TABLE_9_ROWS:
        mcc, tp, tn, fp, fn = mcc_entity_detection_epoch(
            prec, rec, n_tokens, gold_entity_tokens, cal_p, cal_r, tp_ref, fp_ref, fn_ref
        )
        rows.append(
            {
                "Epoch": epoch,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Macro_F1": macro_f1,
                "Accuracy": acc,
                "MCC": round(mcc, 3),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )
    return pd.DataFrame(rows)


def latex_table_8(df: pd.DataFrame) -> str:
    lines = [
        "% --- Table 8: class-wise test NER with MCC (auto-generated) ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Selected cybersecurity-centric class-wise NER results on the test split.}",
        "\\label{tab:classwise_ner_test}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "\\textbf{Entity type} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{Support} & \\textbf{MCC} \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        name = row["Entity_Type"].replace("_", "\\_")
        lines.append(
            f"{name} & {row['Precision']:.3f} & {row['Recall']:.3f} & "
            f"{row['F1']:.3f} & {int(row['Support'])} & {row['MCC']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.25em}",
            "",
            "{\\scriptsize \\textbf{Notes:} MCC is token-level one-vs-rest (entity family vs.\\ all other tokens), "
            "computed from reported precision/recall and gold token counts on the test split.}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def latex_table_9(df: pd.DataFrame) -> str:
    lines = [
        "% --- Table 9: per-epoch validation with MCC (auto-generated) ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{NER validation metrics by training epoch.}",
        "\\label{tab:ner_epoch_validation}",
        "\\footnotesize",
        "\\begin{tabular}{@{}ccccccc@{}}",
        "\\toprule",
        "\\textbf{Epoch} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{Macro-F1} & \\textbf{Acc.} & \\textbf{MCC} \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['Epoch'])} & {row['Precision']:.4f} & {row['Recall']:.4f} & "
            f"{row['F1']:.4f} & {row['Macro_F1']:.4f} & {row['Accuracy']:.4f} & {row['MCC']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.25em}",
            "",
            "{\\scriptsize \\textbf{Notes:} MCC is binary entity-detection (\\texttt{O} vs.\\ any entity token) on the "
            "validation split, derived from epoch entity-level P/R with token-level calibration from the final checkpoint.}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if not JSON_PATH.exists():
        print(f"Missing {JSON_PATH}. Run evaluate_with_mcc.py first.")
        return 1

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    table8 = build_table_8(data["test"])
    table9 = build_table_9(data["validation"])

    table8_path = RESULTS_DIR / "table8_classwise_with_mcc.csv"
    table9_path = RESULTS_DIR / "table9_epoch_with_mcc.csv"
    table8.to_csv(table8_path, index=False)
    table9.to_csv(table9_path, index=False)

    tex_path = RESULTS_DIR / "paper_tables_8_9_with_mcc.tex"
    tex_path.write_text(latex_table_8(table8) + latex_table_9(table9), encoding="utf-8")

    print(table8.to_string(index=False))
    print()
    print(table9.to_string(index=False))
    print(f"\nWrote {table8_path.name}, {table9_path.name}, {tex_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
