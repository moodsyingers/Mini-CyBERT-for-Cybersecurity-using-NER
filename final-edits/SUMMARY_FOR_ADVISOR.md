# MCC Re-Evaluation Summary (Dr. Molokwu Request)

**Date:** Re-run with `models/mini_cybert_final` on validation/test splits (seed=42, same split logic as training notebook).

## What was done

1. Re-ran **Mini-CyBERT NER** inference on validation (1,608 sentences) and test (2,009 sentences).
2. Computed **binary O-vs-entity confusion matrices** (token-level).
3. Calculated **Matthews Correlation Coefficient (MCC)** using TP, TN, FP, FN.
4. Exported full **31×31 token confusion matrices** and updated results tables.

## Main results (re-run)

| Split | Prec. | Rec. | F1 | Macro-F1 | Token Acc. | **MCC (entity detection)** |
|-------|-------|------|-----|----------|------------|----------------------------|
| Validation | 0.702 | 0.714 | 0.708 | 0.682 | 0.954 | **0.774** |
| Test | 0.698 | 0.703 | 0.701 | 0.677 | 0.952 | **0.776** |

**Token-level 31-class MCC (supplementary):** validation 0.731, test 0.734.

## Interpreting high accuracy vs. lower macro-F1

- ~**95%** of evaluated tokens are label `O` (outside any entity).
- Token **accuracy ≈ 0.95** is therefore dominated by correct `O` predictions.
- **Macro-F1 ≈ 0.68** averages all label types equally, so rare classes (e.g., CAMPAIGN, METHOD) pull it down.
- **MCC (entity detection) ≈ 0.78** shows the model is **not** behaving like a trivial “always predict O” classifier:
  - A always-O model would get high accuracy but **MCC ≈ 0** on entity detection.
  - MCC near **+1** indicates strong balanced correlation; **0** is random; **−1** is inverted.

## Test split confusion matrix (O vs entity tokens)

|  | Predicted O | Predicted Entity |
|--|-------------|------------------|
| **True O** | TN = 102,642 | FP = 1,842 |
| **True Entity** | FN = 2,630 | TP = 8,783 |

MCC from these counts:

\[
\frac{(8783)(102642) - (1842)(2630)}{\sqrt{(8783+1842)(8783+2630)(102642+1842)(102642+2630)}} \approx 0.776
\]

## Note on paper vs. re-run numbers

The current `paper/mini_cybert_paper.tex` reports slightly different entity F1 (e.g., test F1 **0.678** vs re-run **0.701**). Token accuracy matches closely (0.952). Use the **re-run numbers in `final-edits/results/`** for the MCC submission unless you reconcile against a specific exported checkpoint bundle.

## Files for the paper

| File | Purpose |
|------|---------|
| `results/main_results_with_mcc.csv` | Spreadsheet-ready main table |
| `results/paper_table_updates.tex` | LaTeX to paste into `tab:main_ner` + new `tab:confusion_binary` |
| `results/confusion_matrix_test_31class.csv` | Full multiclass confusion matrix |
| `results/per_entity_mcc.csv` | Per entity-family binary MCC |
| `results/evaluation_with_mcc.json` | Complete machine-readable report |

## Suggested paper text (Methods)

> Because the `O` label dominates token counts, we report Matthews Correlation Coefficient (MCC) for binary entity detection (`O` versus any entity token) alongside entity-level F1, macro-F1, and token accuracy. MCC uses true positives, true negatives, false positives, and false negatives and remains informative under class imbalance.

## Classification models evaluated

Only **one** trained classifier appears in the paper: **Mini-CyBERT NER** (`BertForTokenClassification`, 31 BIO labels). MLM is unsupervised adaptation, not a classification head; MCC applies to the NER evaluation above.
