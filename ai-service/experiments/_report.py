"""Shared helpers for the experiment scripts: save results where the AI Lab page can find them."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ARTIFACT_DIR = os.path.join(ROOT, "ml", "artifacts")


def save_report(name: str, report: dict) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def print_table(columns, rows):
    widths = [max(len(str(c)), *(len(f"{r[i]:.4f}" if isinstance(r[i], float) else str(r[i])) for r in rows))
              for i, c in enumerate(columns)]
    line = "  ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w) for v, w in zip(r, widths)))
