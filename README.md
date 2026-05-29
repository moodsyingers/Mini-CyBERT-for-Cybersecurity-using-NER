# Mini-CyBERT

A domain-adaptive BERT-based pipeline for **cybersecurity Named Entity Recognition (NER)** and **Masked Language Modeling (MLM)**, with a Flask API and React UI for interactive analysis.

**Repository:** github.com/moodsyingers/Mini-CyBERT

---

## Overview

Mini-CyBERT extends BERT with cybersecurity vocabulary and uses a two-stage training recipe:

1. **MLM adaptation** on NVD CVE descriptions (`datasets/cyber/corpus.txt`)
2. **NER fine-tuning** on CyNER-style annotations mapped to **31 BIO labels** (15 entity families + `O`)

**Entity families:** APT, CAMPAIGN, EXPLOIT, FILE, HASH, INDICATOR, INFRASTRUCTURE, IP, MALWARE, METHOD, SOFTWARE, THREAT_ACTOR, TOOL, URL, VULNERABILITY

---

## Results (NER)

Re-evaluated with confusion matrices and **Matthews Correlation Coefficient (MCC)** for imbalanced token classification. See `final-edits/SUMMARY_FOR_ADVISOR.md` for interpretation.

| Metric | Validation | Test |
|--------|------------|------|
| Entity F1 | 0.708 | 0.701 |
| Macro-F1 | 0.682 | 0.677 |
| Token accuracy | 0.954 | 0.952 |
| **MCC (entity detection)** | **0.774** | **0.776** |

Token accuracy is high because most tokens are label `O`; MCC uses TP, TN, FP, and FN and remains informative under class imbalance.

Full tables: `final-edits/results/main_results_with_mcc.csv`

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

Train your own using `model_training_sheiley.ipynb` (open locally in Jupyter or Google Colab after cloning—do not rely on GitHub’s notebook preview). Host weights separately (Google Drive, Hugging Face, etc.) and document the download steps here.

### 3. NER data

Obtain the CyNER dataset (see reference [6]), save as `datasets/cyber/cyberner.csv`, then run:

```bash
python scripts/clean_cyberner_dataset.py
```

Output: `datasets/cyber/cyberner_clean.csv`

### 4. Run the application

**Backend** (from project root):

```bash
python backend/ner_api.py
```

API: `http://localhost:5001`

**Frontend** (new terminal):

```bash
cd frontend
npm run dev
```

UI: `http://localhost:5173`

For detailed setup, see `SETUP_GUIDE.md`.

---

## Evaluation

**Standard NER evaluation:**

```bash
python scripts/evaluate_ner.py
```

**MCC re-evaluation** (confusion matrices + result tables):

```bash
python final-edits/evaluate_with_mcc.py
```

Outputs are written to `final-edits/results/` (JSON, CSV, confusion matrices).

---

## Training

End-to-end training steps:

1. Vocabulary extension (TF-IDF cyber terms)
2. MLM on NVD corpus (3 epochs)
3. NER fine-tuning (10 epochs, 31-label head)

Clone the repo, then open `model_training_sheiley.ipynb` in Jupyter or Google Colab to run the full pipeline.

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
├── final-edits/             # MCC evaluation + result tables
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

## Authors

**Sheiley Patel** — Department of Computer Science, California State University, Sacramento  
**Bonaventure Chidube Molokwu** — Department of Computer Science, California State University, Sacramento

---

## References

[1] S. Zhou, J. Liu, X. Zhong, and W. Zhao, "Named Entity Recognition Using BERT with Whole Word Masking in Cybersecurity Domain," in *Proc. IEEE 6th International Conference on Big Data Analytics (ICBDA)*, 2021, pp. 316–320.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[3] P. Ranade, A. Piplai, S. Mittal, A. Joshi, and T. Finin, "CyBERT: Contextualized embeddings for the cybersecurity domain," in *Proc. IEEE Big Data*, 2021, pp. 3334–3342.

[4] M. Bayer et al., "CySecBERT: A domain-adapted language model for the cybersecurity domain," *ACM Trans. Privacy Security*, vol. 26, no. 3, pp. 1–27, 2023.

[5] National Institute of Standards and Technology, "National Vulnerability Database," 2024. [Online]. Available: https://nvd.nist.gov/

[6] S. Lal et al., "CyNER: A benchmark dataset for cybersecurity named entity recognition," Hugging Face Datasets, 2022. [Online]. Available: https://huggingface.co/datasets/CynerAI/CyNER

[7] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.

[8] Z. Lan et al., "ALBERT: A lite BERT for self-supervised learning of language representations," in *Proc. ICLR*, 2020.

[9] V. Sanh et al., "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.

[10] J. Lee et al., "BioBERT: A pre-trained biomedical language representation model for biomedical text mining," *Bioinformatics*, vol. 36, no. 4, pp. 1234–1240, 2020.

[11] I. Beltagy, K. Lo, and A. Cohan, "SciBERT: A pretrained language model for scientific text," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3615–3620.

[12] E. Tjong Kim Sang and F. De Meulder, "Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition," in *Proc. CoNLL*, 2003, pp. 142–147.

---

## License

Developed for academic research. See repository license for terms.
