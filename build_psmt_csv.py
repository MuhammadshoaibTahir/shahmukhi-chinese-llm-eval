#!/usr/bin/env python3
"""Rebuild psmt_eval.csv from the PSMT-eval dataset on Hugging Face (Niazi/PSMT_EVAL).

Each subset folder holds English.txt and Punjabi.txt with one sentence per line.
Subsets are written in a FIXED order so that row numbers (the sentence IDs used
in outputs/*.jsonl) are always the same.

    pip install huggingface_hub pandas
    python build_psmt_csv.py            # writes psmt_eval.csv (5,182 rows)
"""
import os
import pandas as pd
from huggingface_hub import snapshot_download

ORDER = ["FLORES-101", "Maan Ki Baat", "tatoeba-test.eng-urd", "IN22 test set"]

path = snapshot_download("Niazi/PSMT_EVAL", repo_type="dataset")
found = {}
for root, _, files in os.walk(path):
    if "English.txt" in files and "Punjabi.txt" in files:
        found[os.path.relpath(root, path)] = root

rows = []
for subset in ORDER:
    root = found[subset]
    en = open(os.path.join(root, "English.txt"), encoding="utf-8").read().strip().split("\n")
    pa = open(os.path.join(root, "Punjabi.txt"), encoding="utf-8").read().strip().split("\n")
    assert len(en) == len(pa), f"{subset}: line counts differ"
    rows += [{"english": e.strip(), "shahmukhi": p.strip(), "subset": subset} for e, p in zip(en, pa)]

pd.DataFrame(rows).to_csv("psmt_eval.csv", index=False)
print("Wrote psmt_eval.csv with", len(rows), "rows")
