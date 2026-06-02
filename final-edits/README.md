# Final Edits — MCC Evaluation for Dr. Molokwu

This folder contains a re-run of the Mini-CyBERT NER evaluation with **confusion matrices** and **Matthews Correlation Coefficient (MCC)** added, addressing the high-accuracy / low-F1 imbalance concern.

## Why MCC?

The NER dataset is heavily imbalanced (~81%+ tokens are `O`). Token accuracy can look strong even when entity detection is weaker. MCC uses TP, TN, FP, and FN and stays informative under class imbalance.

For NER we report two MCC variants:

| Metric | Definition | Use in paper |
|--------|------------|--------------|
| **MCC (entity detection)** | Binary: `O` vs any entity token | **Primary column in Table `tab:main_ner`** — matches Dr. Molokwu's TP/TN/FP/FN framing |
| **MCC (31-class tokens)** | Multiclass over all BIO labels | Supplementary diagnostic in table footnote |

## Quick start

From the project root (with `models/mini_cybert_final` and `datasets/cyber/cyberner_clean.csv` present):

```bash
python final-edits/evaluate_with_mcc.py
```

Optional paths:

```bash
python final-edits/evaluate_with_mcc.py \
  --model-path models/mini_cybert_final \
  --csv-path datasets/cyber/cyberner_clean.csv \
  --results-dir final-edits/results
```

## Outputs (`final-edits/results/`)

| File | Description |
|------|-------------|
| `evaluation_with_mcc.json` | Full metrics, confusion counts, per-class and per-entity-family MCC |
| `main_results_with_mcc.csv` | Main results table (validation + test) with MCC columns |
| `paper_table_updates.tex` | LaTeX snippets to paste into `paper/mini_cybert_paper.tex` |
| `confusion_matrix_validation_31class.csv` | Full 31×31 token confusion matrix (validation) |
| `confusion_matrix_test_31class.csv` | Full 31×31 token confusion matrix (test) |
| `table8_classwise_with_mcc.csv` | Paper Table 8 (class-wise test NER) with **MCC** column |
| `table9_epoch_with_mcc.csv` | Paper Table 9 (validation by epoch) with **MCC** column |
| `paper_tables_8_9_with_mcc.tex` | LaTeX for Tables 8 and 9 with MCC |

## Methodology (matches training notebook)

- Dataset: `cyberner_clean.csv` + optional `ner_attack_type_augment.csv`
- Label schema: 31 cyber labels from `config/ner_cyber_labels.json`
- Split: `seed=42`, 80% train → 80/20 train/val, 20% test (6426 / 1607 / 2009 sentences)
- Entity metrics: seqeval strict IOB2 (same as paper)
- MCC formula (binary entity detection):

\[
\mathrm{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

## Updating the paper

1. Run the evaluation script.
2. Copy the generated block from `results/paper_table_updates.tex` into `paper/mini_cybert_paper.tex`:
   - Replace existing `tab:main_ner` with the new table (adds **MCC** column).
   - Add new `tab:confusion_binary` with counted O-vs-entity confusion on test.
3. Add one sentence in the Methods or Results section noting that MCC is reported because token accuracy is inflated by the dominant `O` class.

Example wording:

> Because the `O` label dominates token counts, we report Matthews Correlation Coefficient (MCC) for binary entity detection (`O` vs. entity) alongside entity-level F1 and macro-F1.

## Classification models in the paper

The paper trains **one** token-classification model: **Mini-CyBERT NER** (`BertForTokenClassification`, 31 labels). MLM is a language-modeling stage, not a classifier; MCC applies to the NER evaluation above.

## Dependencies

Uses project `requirements.txt` plus `seqeval` (installed by the training notebook). GPU is optional but speeds inference.
