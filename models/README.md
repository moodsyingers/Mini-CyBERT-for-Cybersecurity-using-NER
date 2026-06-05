# Model checkpoints

Trained weights are **not** stored in Git (each checkpoint is ~400MB+). After cloning the repository, choose one of the following.

## Option A — Train locally (recommended for reproduction)

1. Install dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   pip install seqeval datasets evaluate
   ```
2. Ensure `datasets/cyber/cyberner_clean.csv` exists (included in this repo).
3. Open `model_training_sheiley.ipynb` in Jupyter or Google Colab and run all cells.
4. Save outputs to:
   ```
   models/mlm_final/              # Stage 1 MLM checkpoint
   models/mini_cybert_final/      # Stage 2 NER checkpoint
   ```

Foundation model: **`bert-base-uncased`** (Hugging Face).

## Option B — Copy checkpoints from a trained environment

If you already trained Mini-CyBERT, copy the two folders above into this `models/` directory. The Flask API (`backend/ner_api.py`) expects exactly these paths:

| Path | Purpose |
|------|---------|
| `models/mlm_final/` | Masked language modeling (fill-mask) |
| `models/mini_cybert_final/` | Named entity recognition |

## Verify setup

From the project root:

```bash
python scripts/verify_setup.py
```

Then start the API:

```bash
python backend/ner_api.py
```

Visit `http://localhost:5001/api/health` — both `ner_loaded` and `mlm_loaded` should be `true`.
