# Mini-CyBERT Setup Guide

This guide provides step-by-step instructions for setting up and running the Mini-CyBERT cybersecurity language model application.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Collection and Preparation](#data-collection-and-preparation)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Running the Application](#running-the-application)
6. [Using the Application](#using-the-application)
7. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Before starting, ensure you have the following installed:

**Required Software:**

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

## 2. Data Collection and Preparation

The project uses vulnerability data from the National Vulnerability Database.

### Step 1: Collect NVD Data

From the project root directory:

```bash
python scripts/mlm_data_collection.py
```

**This will:**

- Fetch 10,000 CVE records from NVD API
- Save to `datasets/nvd/nvd_cves.json`
- Take approximately 5-7 minutes

### Step 2: Clean and Process Data

After collection completes:

```bash
python scripts/mlm_data_cleaning.py
```

**This will:**

- Process the collected CVE data
- Create clean text corpus
- Save to `datasets/cyber/corpus.txt`
- Take approximately 1-2 minutes

**Output Files:**

```
datasets/nvd/nvd_cves.json           (Raw CVE data)
datasets/cyber/corpus.txt             (Processed corpus)
datasets/cyber/cleaning_report.json   (Statistics)
```

---

## 3. Backend Setup

The backend provides API endpoints for the NER and MLM models.

### Step 1: Install Python Dependencies

From the **project root** (not only `backend/`):

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install seqeval datasets evaluate
```

**Required packages include:** Flask, Flask-CORS, PyTorch, transformers, pandas, numpy.

### Step 2: Verify Model Files

Ensure model weights exist in:

```
models/mlm_final/
models/mini_cybert_final/
```

These folders are **not** in Git (file size). Either train them with `model_training_sheiley.ipynb` or copy checkpoints from a trained environment. See `models/README.md`.

Verify setup:

```bash
python scripts/verify_setup.py
```

---

## 4. Frontend Setup

The frontend provides a web interface for interacting with the models.

### Step 1: Install Node.js Dependencies

Navigate to frontend directory:

```bash
cd frontend
```

Install packages:

```bash
npm install
```

**This will install:**

- React
- Vite
- Required UI libraries

---

## 5. Running the Application

Two servers need to be running simultaneously.

### Terminal 1 - Backend Server

From project root:

```bash
python backend/ner_api.py
```

**Expected output:**

```
Loading NER model from: models/mini_cybert_final
NER model loaded successfully!
Loading MLM model from: models/mlm_final
MLM model loaded successfully!
Starting Flask server on http://localhost:5001
```

Wait for both models to load (20-30 seconds).

### Terminal 2 - Frontend Server

Open a new terminal. From project root:

```bash
cd frontend
npm run dev
```

**Expected output:**

```
VITE ready in XXX ms
Local: http://localhost:5173
```

### Access Application

Open browser and navigate to: **http://localhost:5173**

---

## 6. Using the Application

### NER Model (Named Entity Recognition)

**Purpose:** Extract and classify cybersecurity entities from text

**Steps:**

1. Select "NER Model" radio button
2. Enter cybersecurity text in the input field
3. Click "Analyze Text" button
4. View extracted entities with their classifications

**Example:**

**Input:**

```
APT28 exploited CVE-2023-12345 in Windows
```

**Output:**

```
APT28 -> THREAT_ACTOR
CVE-2023-12345 -> VULNERABILITY
Windows -> SOFTWARE
```

### MLM Model (Masked Language Modeling)

**Purpose:** Predict masked words in cybersecurity context

**Steps:**

1. Select "MLM Model" radio button
2. Enter text with `[MASK]` token (BERT fill-mask)
3. Click "Analyze Text" button
4. View top 5 word predictions

**Example:**

**Input:**

```
The attacker used a [MASK] exploit
```

**Output:**

```
1. zero-day
2. remote
3. buffer
4. privilege
5. sql
```

---

## 7. Troubleshooting

### Issue: Backend won't start

**Solution:**

- Verify Python dependencies are installed
- Check that model files exist in `models/` directory
- Ensure port 5001 is not already in use

### Issue: Frontend won't connect

**Solution:**

- Ensure backend is running on port 5001
- Check browser console for errors
- Verify CORS is enabled in backend

### Issue: Models loading slowly

**This is normal behavior:**

- First load takes 20-30 seconds (loading 500MB+ models)
- Subsequent requests are fast
- Be patient during initial load

### Issue: API rate limit (data collection)

**Solution:**

- Script already implements 6-second delays
- For faster collection, obtain NVD API key
- Visit: https://nvd.nist.gov/developers

### Issue: Out of memory

**Solution:**

- Close other applications
- Reduce batch size if training
- Use CPU instead of GPU for inference

### Issue: Port already in use

**Backend (port 5001):**

- Change port in `backend/ner_api.py`

**Frontend (port 5173):**

- Change port in `frontend/vite.config.js`

---

## Quick Start Summary

```bash
# 1. Install dependencies (from project root)
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install seqeval datasets evaluate
cd frontend && npm install && cd ..

# 2. Verify data + model checkpoints
python scripts/verify_setup.py

# Optional: rebuild NVD corpus
python scripts/mlm_data_collection.py
python scripts/mlm_data_cleaning.py

# 3. Start backend
python backend/ner_api.py

# 4. Start frontend (new terminal)
cd frontend && npm run dev

# 5. Open browser
# http://localhost:5173
```

---

For additional help or documentation, refer to `README.md` in the project root.
