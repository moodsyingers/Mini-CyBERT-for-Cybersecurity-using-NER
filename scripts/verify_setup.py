#!/usr/bin/env python3
"""Check that Mini-CyBERT can run: dependencies, data, and model checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    ROOT / "models" / "mlm_final",
    ROOT / "models" / "mini_cybert_final",
]

REQUIRED_FILES = [
    ROOT / "datasets" / "cyber" / "cyberner_clean.csv",
    ROOT / "config" / "ner_cyber_labels.json",
    ROOT / "datasets" / "cyber" / "corpus.txt",
    ROOT / "datasets" / "cyber" / "entity_vocabulary.json",
]

PYTHON_PACKAGES = [
    "flask",
    "flask_cors",
    "torch",
    "transformers",
    "pandas",
    "numpy",
]


def check_packages() -> list[str]:
    missing = []
    for pkg in PYTHON_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    return missing


def main() -> int:
    print("Mini-CyBERT setup verification")
    print("=" * 50)
    ok = True

    missing_pkgs = check_packages()
    if missing_pkgs:
        ok = False
        print("\n[FAIL] Missing Python packages:")
        for pkg in missing_pkgs:
            print(f"  - {pkg}")
        print("\nInstall from project root:")
        print("  pip install -r requirements.txt")
        print("  pip install seqeval datasets evaluate")
    else:
        print("\n[OK] Core Python packages installed")

    print("\nData files:")
    for path in REQUIRED_FILES:
        if path.exists():
            print(f"  [OK] {path.relative_to(ROOT)}")
        else:
            ok = False
            print(f"  [MISSING] {path.relative_to(ROOT)}")

    print("\nModel checkpoints:")
    for path in REQUIRED_DIRS:
        config = path / "config.json"
        if path.is_dir() and config.exists():
            print(f"  [OK] {path.relative_to(ROOT)}")
        else:
            ok = False
            print(f"  [MISSING] {path.relative_to(ROOT)}")
            print("         Train via model_training_sheiley.ipynb or see models/README.md")

    print("\n" + "=" * 50)
    if ok:
        print("All checks passed. Run: python backend/ner_api.py")
        return 0
    print("Setup incomplete. See README.md and models/README.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
