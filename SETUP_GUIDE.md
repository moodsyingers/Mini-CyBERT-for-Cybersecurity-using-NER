# SecBERT Cybersecurity NER — Setup Guide

Step-by-step instructions for setting up and running the SecBERT NER application.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Getting the Model](#2-getting-the-model)
3. [Backend Setup](#3-backend-setup)
4. [Frontend Setup](#4-frontend-setup)
5. [Running the Application](#5-running-the-application)
6. [Evaluating the Model](#6-evaluating-the-model)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

**Required software:**

- Python 3.8 or higher
- Node.js and npm
- Git (optional, for version control)

**Verify installations:**

```bash
python --version
node --version
npm --version
```

---

## 2. Getting the Model

The demo needs the fine-tuned SecBERT NER model at `models/secbert_ner_final/`.

1. Open `train_secbert_ner.ipynb` in Google Colab (GPU runtime).
2. Upload `datasets/cyber/cyberner_clean.csv` and `config/ner_cyber_labels.json` when prompted.
3. Run all cells. The notebook fine-tunes `jackaduma/SecBERT` for token classification (the base model was already MLM pre-trained on cybersecurity corpora by its original authors — no MLM training happens here) and evaluates it on the held-out test set.
4. Download `secbert_ner_final.zip` from the final cell and extract it so that the files land at:

```
models/secbert_ner_final/
|-- config.json
|-- model.safetensors
|-- tokenizer_config.json
|-- vocab.txt / tokenizer.json
```

---

## 3. Backend Setup

The backend provides the NER API endpoint.

```bash
cd backend
pip install -r requirements.txt
```

**Required packages:** Flask, Flask-CORS, transformers, torch.

---

## 4. Frontend Setup

```bash
cd frontend
npm install
```

---

## 5. Running the Application

Two servers run simultaneously.

### Terminal 1 — Backend

From the project root:

```bash
python backend/ner_api.py
```

**Expected output:**

```
Loading NER model from: models/secbert_ner_final
[SUCCESS] NER model loaded successfully!
Starting Flask server on http://localhost:5001
```

### Terminal 2 — Frontend

```bash
cd frontend
npm start
```

Open the browser at **http://localhost:3000**.

### Using the Application

1. Enter cybersecurity text in the input field.
2. Click "Analyze Text".
3. View extracted entities with their classifications.

**Example:**

Input:

```
APT28 exploited CVE-2023-12345 in Windows
```

Output:

```
APT28 -> THREAT_ACTOR
CVE-2023-12345 -> VULNERABILITY
Windows -> SOFTWARE
```

---

## 6. Evaluating the Model

To reproduce the test-set metrics locally (Precision, Recall, F1 — seqeval, strict IOB2):

```bash
pip install -r requirements.txt   # project root; includes seqeval + datasets
python scripts/evaluate_ner.py
```

This re-derives the same held-out test split used in the notebook (seed 42), evaluates `models/secbert_ner_final`, prints overall and per-entity metrics, and writes `evaluation_results.json`.

---

## 7. Troubleshooting

### Backend won't start

- Verify Python dependencies are installed.
- Check the model exists at `models/secbert_ner_final/` (see section 2).
- Ensure port 5001 is not already in use.

### Frontend won't connect

- Ensure the backend is running on port 5001.
- Check the browser console for errors.
- Verify CORS is enabled in the backend (it is by default).

### Models loading slowly

- First load takes ~20-30 seconds; subsequent requests are fast.

### Out of memory

- Close other applications.
- Inference runs fine on CPU; no GPU is required locally.

### Port already in use

- Backend (5001): change the port at the bottom of `backend/ner_api.py`.
- Frontend (3000): set the `PORT` environment variable before `npm start`.

---

## Quick Start Summary

```bash
# 1. Produce the model (Google Colab)
#    Run train_secbert_ner.ipynb, extract zip to models/secbert_ner_final/

# 2. Install dependencies
cd backend && pip install -r requirements.txt && cd ..
cd frontend && npm install && cd ..

# 3. Start backend
python backend/ner_api.py

# 4. Start frontend (new terminal)
cd frontend && npm start

# 5. Open browser
# http://localhost:3000
```

---

For project details, methodology, and evaluation results, see `README.md`.
