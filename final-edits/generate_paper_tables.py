#!/usr/bin/env python3
"""Regenerate CSV/LaTeX outputs from evaluation_with_mcc.json without re-running inference."""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_with_mcc import save_latex_snippets, save_main_results_table
from generate_tables_8_9_mcc import main as generate_tables_8_9

JSON_PATH = SCRIPT_DIR / "results" / "evaluation_with_mcc.json"
RESULTS_DIR = SCRIPT_DIR / "results"


def main() -> int:
    if not JSON_PATH.exists():
        print(f"Missing {JSON_PATH}. Run evaluate_with_mcc.py first.")
        return 1
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    results = {"validation": data["validation"], "test": data["test"]}
    save_main_results_table(results, RESULTS_DIR / "main_results_with_mcc.csv")
    save_latex_snippets(results, RESULTS_DIR / "paper_table_updates.tex")
    generate_tables_8_9()
    print(f"Updated {RESULTS_DIR / 'main_results_with_mcc.csv'}")
    print(f"Updated {RESULTS_DIR / 'paper_table_updates.tex'}")
    print(f"Updated table8/9 CSV and paper_tables_8_9_with_mcc.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
