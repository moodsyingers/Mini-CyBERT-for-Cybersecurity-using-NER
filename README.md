# Mini-CyBERT

A domain-adaptive BERT-based pipeline for **cybersecurity Named Entity Recognition (NER)** and **Masked Language Modeling (MLM)**, with a Flask API and React UI for interactive analysis.

**Repository:** [github.com/moodsyingers/Mini-CyBERT](https://github.com/moodsyingers/Mini-CyBERT)

---

## Overview

Mini-CyBERT extends BERT with cybersecurity vocabulary and uses a two-stage training recipe:

1. **MLM adaptation** on NVD CVE descriptions (`datasets/cyber/corpus.txt`)
2. **NER fine-tuning** on CyNER-style annotations mapped to **31 BIO labels** (15 entity families + `O`)

**Entity families:** APT, CAMPAIGN, EXPLOIT, FILE, HASH, INDICATOR, INFRASTRUCTURE, IP, MALWARE, METHOD, SOFTWARE, THREAT_ACTOR, TOOL, URL, VULNERABILITY

---

## Results (NER)

Re-evaluated with confusion matrices and **Matthews Correlation Coefficient (MCC)** for imbalanced token classification. See [`final-edits/SUMMARY_FOR_ADVISOR.md`](final-edits/SUMMARY_FOR_ADVISOR.md) for interpretation.

| Metric | Validation | Test |
|--------|------------|------|
| Entity F1 | 0.708 | 0.701 |
| Macro-F1 | 0.682 | 0.677 |
| Token accuracy | 0.954 | 0.952 |
| **MCC (entity detection)** | **0.774** | **0.776** |

Token accuracy is high because most tokens are label `O`; MCC uses TP, TN, FP, and FN and remains informative under class imbalance.

Full tables: [`final-edits/results/main_results_with_mcc.csv`](final-edits/results/main_results_with_mcc.csv)

---

## Quick start

### Prerequisites

- Python 3.8+
- Node.js 18+ and npm

### 1. Clone and install

```bash
git clone https://github.com/moodsyingers/Mini-CyBERT.git
cd Mini-CyBERT

pip install -r requirements.txt
pip install seqeval datasets evaluate

cd frontend && npm install && cd ..
```

### 2. Model weights (required)

Model checkpoints are **not** included in this repository (file size). Download trained weights and place them locally:

```
models/
├── mlm_final/              # MLM checkpoint
└── mini_cybert_final/      # NER checkpoint
```

Train your own with [`model_training_sheiley.ipynb`](model_training_sheiley.ipynb), or add a download link here once weights are hosted (Google Drive, Hugging Face, etc.).

### 3. NER data

Obtain the [CyNER dataset](https://huggingface.co/datasets/CynerAI/CyNER), save as `datasets/cyber/cyberner.csv`, then run:

```bash
python scripts/clean_cyberner_dataset.py
```

Output: `datasets/cyber/cyberner_clean.csv`

### 4. Run the application

**Backend** (from project root):

```bash
python backend/ner_api.py
```

API: http://localhost:5001

**Frontend** (new terminal):

```bash
cd frontend
npm run dev
```

UI: http://localhost:5173

For detailed setup, see [`SETUP_GUIDE.md`](SETUP_GUIDE.md).

---

## Evaluation

**Standard NER evaluation:**

```bash
python scripts/evaluate_ner.py
```

**MCC re-evaluation** (confusion matrices + paper tables):

```bash
python final-edits/evaluate_with_mcc.py
```

Outputs are written to `final-edits/results/` (JSON, CSV, LaTeX snippets, confusion matrices).

---

## Training

End-to-end training is documented in [`model_training_sheiley.ipynb`](model_training_sheiley.ipynb):

1. Vocabulary extension (TF-IDF cyber terms)
2. MLM on NVD corpus (3 epochs)
3. NER fine-tuning (10 epochs, 31-label head)

**MLM data pipeline:**

```bash
python scripts/mlm_data_collection.py
python scripts/mlm_data_cleaning.py
```

---

## Project structure

```
Mini-CyBERT/
├── backend/                 # Flask inference API
├── frontend/                # React UI (NER + MLM)
├── scripts/                 # Data prep, training helpers, evaluation
├── config/                  # 31-label NER schema
├── datasets/cyber/          # Corpus, labels, vocabulary
├── final-edits/             # MCC evaluation + updated result tables
├── paper/                   # LaTeX manuscript
├── model_training_sheiley.ipynb
├── README.md
└── SETUP_GUIDE.md
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/ner/analyze` | NER — JSON body `{ "text": "..." }` |
| `POST` | `/api/mlm/predict` | Fill-mask — text with `[MASK]` token |

---

## Paper

LaTeX source: [`paper/mini_cybert_paper.tex`](paper/mini_cybert_paper.tex)

MCC table updates for the manuscript: [`final-edits/results/paper_table_updates.tex`](final-edits/results/paper_table_updates.tex)

---

## Authors

**Sheiley Patel** — Department of Computer Science, California State University, Sacramento  
**Bonaventure Chidube Molokwu** — Department of Computer Science, California State University, Sacramento

---

## References

1. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019
2. Ranade et al., *CyBERT: Contextualized Embeddings for the Cybersecurity Domain*, IEEE Big Data 2021
3. Lal et al., [CyNER benchmark](https://huggingface.co/datasets/CynerAI/CyNER), 2022
4. [National Vulnerability Database (NVD)](https://nvd.nist.gov/)

---

## License

Developed for academic research. See repository license for terms.
