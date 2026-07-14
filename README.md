# SecBERT Cybersecurity NER

Fine-tuning **SecBERT** for Cybersecurity Named Entity Recognition (NER) using Hugging Face Transformers, with a Flask + React demo application.

---

## Project Overview

This project fine-tunes [`jackaduma/SecBERT`](https://huggingface.co/jackaduma/SecBERT) — a BERT model that was **already pre-trained with Masked Language Modeling (MLM) on cybersecurity corpora by its original authors** (APTnotes, Stucco-Data, CASIE, SemEval-2018 Task 8) — for the cybersecurity NER task.

**We do not perform any MLM or domain-adaptive pre-training ourselves.** Our contribution is:

1. **Fine-tuning** SecBERT for token classification (NER) on the CyberNER dataset with a 31-label BIO schema (15 cyber entity types).
2. **Evaluating** the fine-tuned model with standard entity-level NER metrics (Precision, Recall, F1 via seqeval).
3. **Serving** the model in a web demo (Flask API + React frontend).

### Methodology

```
jackaduma/SecBERT  (pre-trained with MLM by its original authors)
        │
        ▼
Fine-tune for Cybersecurity NER (Token Classification, 31-label BIO schema)
        │
        ▼
Evaluate (Precision, Recall, F1 — seqeval, strict IOB2)
        │
        ▼
Demo (Flask API + React frontend)
```

---

## Dataset

**CyberNER** (`datasets/cyber/cyberner_clean.csv`) — a token-per-row CSV (~610k rows) derived from the CyNER dataset (Hugging Face), with columns `Word`, `Tag`, `Sentence_ID`.

- **Label schema:** `config/ner_cyber_labels.json` defines 31 BIO labels — `O` plus `B-`/`I-` of 15 cyber entity types: APT, CAMPAIGN, EXPLOIT, FILE, HASH, INDICATOR, INFRASTRUCTURE, IP, MALWARE, METHOD, SOFTWARE, THREAT_ACTOR, TOOL, URL, VULNERABILITY. A `tag_mapping` collapses the raw dataset tags into this schema; unmapped tags become `O`.
- **Splits:** 80/20 train/test (seed 42), then 80/20 of train → train/validation (seed 42). The same splits are re-derived by `scripts/evaluate_ner.py` for local reproduction.

---

## Model Training (Fine-Tuning)

Training is done in **`train_secbert_ner.ipynb`** (Google Colab, GPU runtime):

1. Load `jackaduma/SecBERT` with a token-classification head (`AutoModelForTokenClassification`). The classification head is newly initialized — this is the part fine-tuning trains; the pre-trained SecBERT encoder is reused.
2. Tokenize with the SecBERT tokenizer (`is_split_into_words=True`); only the first sub-token of each word carries the label, the rest are masked with `-100`.
3. Fine-tune with Hugging Face `Trainer`: learning rate 3e-5, batch size 16, 10 epochs, weight decay 0.01, best checkpoint selected by validation F1.
4. The trained model is saved to Google Drive (`MyDrive/models/secbert_ner_final.zip`); extract it to **`models/secbert_ner_final/`** in this repository.

---

## Evaluation

Metrics on the held-out test set (2,009 sentences), computed with **seqeval**:

| Metric | Value |
|---|---|
| Token Accuracy | **94.20%** |
| Precision (entity-level, strict IOB2) | 0.6281 |
| Recall (entity-level, strict IOB2) | 0.6231 |
| F1 Score (entity-level, strict IOB2) | 0.6256 |

Token accuracy measures per-token label correctness; the entity-level scores use strict matching, where an entity counts as correct only if its full span and type match exactly — a substantially harder criterion, on which published models fine-tuned on the CyNER data report comparable results.

Strongest entity types: EXPLOIT (F1 0.84), HASH (0.77), IP (0.76), VULNERABILITY (0.72), APT (0.71). Weakest: CAMPAIGN (0.18, only 40 test instances), METHOD (0.29) — low-support and semantically diffuse classes.

- The notebook reports test-set Precision/Recall/F1 plus a per-entity classification report.
- Locally, run `python scripts/evaluate_ner.py` — it re-derives the same held-out test split, evaluates `models/secbert_ner_final`, and writes `evaluation_results.json` (full per-entity breakdown).

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Node.js and npm

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Required packages: Flask, Flask-CORS, transformers, torch.

### Frontend Setup

```bash
cd frontend
npm install
```

### Model

Run `train_secbert_ner.ipynb` on Google Colab and extract the resulting `secbert_ner_final.zip` to `models/secbert_ner_final/`.

---

## Running the Application

### Start Backend Server

From the project root:

```bash
python backend/ner_api.py
```

The backend loads the fine-tuned SecBERT NER model and starts on http://localhost:5001.

### Start Frontend Interface

In a new terminal:

```bash
cd frontend
npm start
```

Access the application at: http://localhost:3000

### Using the Demo

Enter cybersecurity text (e.g., vulnerability descriptions, threat reports) or pick one of the built-in examples, then click "Analyze text". Detected entities are highlighted inline in the annotated text and summarized by type.

**Example input:**

```
APT28 exploited CVE-2023-12345 in a phishing campaign targeting Windows systems.
```

**Example output:**

- APT28: APT
- CVE-2023-12345: VULNERABILITY
- phishing: METHOD
- Windows: SOFTWARE

> Note: the API additionally enhances model predictions with a post-processing dictionary lookup built from the training data (`datasets/cyber/entity_vocabulary.json`) to catch terms the model may miss.

---

## Project Structure

```
shiley-project/
|
|-- backend/
|   |-- ner_api.py                  # Flask API server (NER)
|   |-- requirements.txt
|
|-- frontend/
|   |-- src/
|   |   |-- App.jsx                 # React application (NER demo)
|   |   |-- components/Header.jsx
|   |-- package.json
|
|-- scripts/
|   |-- evaluate_ner.py             # Entity-level evaluation (seqeval P/R/F1)
|   |-- run_ner.py                  # Interactive NER inference CLI
|   |-- build_dataset_vocabulary.py # Builds entity_vocabulary.json
|
|-- config/
|   |-- ner_cyber_labels.json       # 31-label cyber BIO schema + tag mapping
|
|-- datasets/
|   |-- cyber/
|       |-- cyberner_clean.csv      # CyberNER training data
|       |-- entity_vocabulary.json  # Post-processing lookup vocabulary
|
|-- models/
|   |-- secbert_ner_final/          # Fine-tuned SecBERT NER model (from Colab)
|
|-- train_secbert_ner.ipynb         # Colab fine-tuning notebook
|-- README.md
|-- SETUP_GUIDE.md
```

---

## API Endpoints

### NER Endpoint

```
POST http://localhost:5001/api/ner/analyze
Content-Type: application/json

{
  "text": "your cybersecurity text here"
}
```

### Health Check

```
GET http://localhost:5001/api/health
```

---

## References

1. SecBERT — pre-trained BERT for the security domain: https://huggingface.co/jackaduma/SecBERT (pre-trained with MLM on APTnotes, Stucco-Data, CASIE, and SemEval-2018 Task 8 corpora by its original authors).
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. Alam, M. T., et al. (2022). CyNER: A Python Library for Cybersecurity Named Entity Recognition.

---

## License

This project is developed for academic research purposes.

---

**Developed as part of: Fine-Tuning SecBERT for Cybersecurity Named Entity Recognition**
