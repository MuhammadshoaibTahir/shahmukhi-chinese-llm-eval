#!/usr/bin/env python3
"""
Shahmukhi MT evaluation of Chinese LLMs on PSMT-eval
====================================================

Replicates the setup of PSMT-eval (Basit & Azeemi, WAT 2026) so results
are directly comparable with the paper:
  - Direction: English -> Punjabi Shahmukhi
  - Same prompt (system + user instruction)
  - temperature = 0.1
  - Metrics: SacreBLEU (default) and chrF++ (sacrebleu CHRF word_order=2)

Extra (your contribution):
  - Script-fidelity check: how often output is Shahmukhi vs Urdu vs Gurmukhi vs Latin

All models are called through OpenAI-compatible APIs. The easiest route is
OpenRouter (one key for every model). You can also call each provider directly.

USAGE
-----
  pip install openai sacrebleu datasets pandas tqdm

  # 1) Set your API key(s)
  export OPENROUTER_API_KEY="sk-or-..."

  # 2) Pilot run: 50 sentences, 2 models
  python shahmukhi_eval.py --models qwen deepseek --limit 50

  # 3) Full run, all models
  python shahmukhi_eval.py --models all

  # 4) Only re-score what is already saved (no API calls)
  python shahmukhi_eval.py --score-only

  # Use a local CSV instead of Hugging Face (needs 'english' and 'shahmukhi' columns,
  # optional 'subset' column):
  python shahmukhi_eval.py --csv psmt_eval.csv --models qwen

  # Test the pipeline without spending money (model echoes the reference):
  python shahmukhi_eval.py --models mock --limit 20

Outputs go to ./results/ :
  - outputs_<model>.jsonl   raw translations (resumable; reruns skip done items)
  - scores.csv              BLEU / chrF++ per model per subset
  - script_fidelity.csv     % Shahmukhi / Urdu-like / Gurmukhi / Latin per model
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. MODEL CONFIG  — check model IDs on openrouter.ai/models before running;
#    providers rename/upgrade models often. Record the exact ID in your paper.
# ---------------------------------------------------------------------------
OPENROUTER = {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"}

MODELS = {
    # --- Free models (OpenRouter ":free", no credit needed; daily limits apply) ---
    "ling":           ("inclusionai/ling-3.2-flash:free",      OPENROUTER),  # Ant Group
    "ernie":          ("baidu/ernie-5.5-flash-preview:free",   OPENROUTER),  # Baidu
    "dots":           ("dots-studio/dots-3-note-preview:free", OPENROUTER),  # Xiaohongshu
    "gemma":          ("google/gemma-4-31b-it:free",           OPENROUTER),  # Google (non-Chinese baseline)
    # --- Paid models (need OpenRouter credit) ---
    "qwen":           ("qwen/qwen3.8-max-0902",             OPENROUTER),
    "qwen-27b":       ("qwen/qwen3.8-27b",                  OPENROUTER),
    "deepseek":       ("deepseek/deepseek-v4-pro-0813",     OPENROUTER),
    "deepseek-flash": ("deepseek/deepseek-v4.1-flash-1017", OPENROUTER),
    "kimi":           ("moonshotai/kimi-k3.1-pro",          OPENROUTER),
    "glm":            ("z-ai/glm-5.3",                      OPENROUTER),
    "minimax":        ("minimax/minimax-m3",                OPENROUTER),
    "mimo":           ("xiaomi/mimo-v2.6-pro",              OPENROUTER),
    "qwen3-32b":      ("qwen/qwen3-32b",                    OPENROUTER),  # paper baseline
}

# ---------------------------------------------------------------------------
# 2. PROMPT — same as PSMT-eval paper
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert English-to-Punjabi translator. Output only the translation "
    "in Punjabi Shahmukhi script (Perso-Arabic based), with no explanations or notes. "
    "Preserve the original meaning while keeping the translation natural in Punjabi."
)
USER_TEMPLATE = (
    "Please translate the English sentence into Punjabi Shahmukhi script. "
    "Only return the translation.\n\n{src}"
)
TEMPERATURE = 0.1
MAX_TOKENS = 2048   # reasoning models may need more; raise if outputs come back empty

# ---------------------------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------------------------
HF_DATASET = "Niazi/PSMT_EVAL"

EN_CANDIDATES = ["english", "en", "eng", "source", "src", "English", "EN"]
PA_CANDIDATES = ["shahmukhi", "punjabi", "pnb", "pa", "target", "tgt", "reference",
                 "Punjabi", "Shahmukhi", "punjabi_shahmukhi"]


def _pick(cols, candidates, label):
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    sys.exit(f"Could not find the {label} column. Columns are: {list(cols)}\n"
             f"Pass it with --{'src' if label == 'English' else 'ref'}-col NAME")


def load_data(args):
    import pandas as pd
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        from datasets import load_dataset, get_dataset_config_names
        frames = []
        try:
            configs = get_dataset_config_names(HF_DATASET)
        except Exception:
            configs = [None]
        for cfg in configs:
            ds = load_dataset(HF_DATASET, cfg) if cfg else load_dataset(HF_DATASET)
            for split in ds:
                f = ds[split].to_pandas()
                if "subset" not in f.columns:
                    f["subset"] = cfg if (cfg and cfg != "default") else split
                frames.append(f)
        df = pd.concat(frames, ignore_index=True)

    src = args.src_col or _pick(df.columns, EN_CANDIDATES, "English")
    ref = args.ref_col or _pick(df.columns, PA_CANDIDATES, "Shahmukhi")
    if "subset" not in df.columns:
        df["subset"] = "all"
    df = df[[src, ref, "subset"]].rename(columns={src: "english", ref: "shahmukhi"})
    df = df.dropna().reset_index(drop=True)
    df["id"] = df.index

    if args.limit:
        # stratified sample so every subset is represented in pilots
        per = max(1, args.limit // df["subset"].nunique())
        df = (df.groupby("subset", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per), random_state=42))
                .reset_index(drop=True))
    print(f"Loaded {len(df)} sentences | subsets: {dict(Counter(df['subset']))}")
    return df

# ---------------------------------------------------------------------------
# 4. TRANSLATION
# ---------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_output(text):
    text = THINK_RE.sub("", text or "").strip()
    text = text.strip('"“”\'').strip()
    return text.splitlines()[0].strip() if text else ""


def make_client(provider):
    from openai import OpenAI
    key = os.environ.get(provider["key_env"])
    if not key:
        sys.exit(f"Missing API key: set the {provider['key_env']} environment variable.")
    return OpenAI(base_url=provider["base_url"], api_key=key)


def translate(client, model_id, src, retries=5):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": USER_TEMPLATE.format(src=src)}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if 'Authentication' in type(e).__name__ or 'PermissionDenied' in type(e).__name__:
                sys.exit(f'API key rejected: {e}')
            wait = 2 ** attempt
            print(f"  ! {model_id}: {type(e).__name__}: {str(e)[:120]} — retry in {wait}s")
            time.sleep(wait)
    return "__ERROR__"


def run_model(name, df, outdir, delay):
    from tqdm import tqdm
    path = outdir / f"outputs_{name}.jsonl"
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, filter(str.strip, f)) if r["raw"] != "__ERROR__"}

    todo = df[~df["id"].isin(done)]
    if todo.empty:
        print(f"[{name}] all {len(df)} already done")
        return
    if name == "mock":
        client, model_id = None, "mock"
    else:
        model_id, provider = MODELS[name]
        client = make_client(provider)

    print(f"[{name}] translating {len(todo)} sentences with {model_id}")
    failed = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in tqdm(todo.itertuples(), total=len(todo)):
            raw = row.shahmukhi if name == "mock" else translate(client, model_id, row.english)
            if raw == "__ERROR__":
                failed += 1
                if failed >= 5:
                    print(f"[{name}] 5 failed requests in a row (probably the daily free limit). "
                          f"Stopping; run the same command again later to continue.")
                    break
                continue
            failed = 0
            rec = {"id": int(row.id), "subset": row.subset, "model": model_id,
                   "src": row.english, "ref": row.shahmukhi,
                   "raw": raw, "hyp": clean_output(raw)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if delay:
                time.sleep(delay)

# ---------------------------------------------------------------------------
# 5. SCRIPT FIDELITY (heuristic) — is the output really Shahmukhi?
# ---------------------------------------------------------------------------
GURMUKHI = re.compile(r"[਀-੿]")
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LATIN = re.compile(r"[A-Za-z]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")          # Hindi script
ARABIC_ONLY_LETTERS = re.compile(r"[ةأإى]")           # used in Arabic, rare in Urdu/Punjabi
ARABIC_WORDS = {"في", "من", "على", "إلى", "أو", "التي", "الذي", "هذا", "هذه", "مع", "عن", "أن", "كان", "بعض"}

# Common function words. Punjabi (Shahmukhi) vs Urdu share a script, so we
# tell them apart by grammar words. This is a heuristic — validate it on a
# sample with a native speaker and report its accuracy in your paper.
PUNJABI_WORDS = {"دا", "دی", "دے", "دیاں", "نوں", "اے", "ایہہ", "اوہ", "وچ", "توں",
                 "نال", "سی", "سن", "ہاں", "ہو", "رہیا", "رہی", "کیتا", "کیتی",
                 "جاندا", "جاندی", "اوہدا", "اوہدی", "ساڈا", "تہاڈا", "میرا", "ایتھے",
                 "اوتھے", "کتھے", "ہُن", "ہن", "ناں", "نئیں", "ویلے"}
URDU_WORDS = {"کا", "کی", "کے", "کو", "ہے", "ہیں", "میں", "سے", "تھا", "تھی", "تھے",
              "یہ", "وہ", "نہیں", "اور", "کیا", "گیا", "رہا", "رہی", "ہوں", "یہاں",
              "وہاں", "کہاں", "اب", "جاتا", "جاتی", "اس", "ان"}


def classify_script(text):
    if not text or text == "__ERROR__":
        return "empty/error"
    g, a, l = len(GURMUKHI.findall(text)), len(ARABIC.findall(text)), len(LATIN.findall(text))
    d = len(DEVANAGARI.findall(text))
    total = g + a + l + d
    if total == 0:
        return "other"
    if g / total > 0.5:
        return "gurmukhi"
    if d / total > 0.5:
        return "devanagari"
    if l / total > 0.5:
        return "latin"
    tokens = re.findall(r"[؀-ۿ]+", text)
    arabic_hits = (sum(t in ARABIC_WORDS for t in tokens)
                   + sum(t.startswith("ال") and len(t) > 3 for t in tokens)
                   + len(ARABIC_ONLY_LETTERS.findall(text)))
    if arabic_hits >= 3:
        return "arabic"
    p = sum(t in PUNJABI_WORDS for t in tokens)
    u = sum(t in URDU_WORDS for t in tokens)
    if u > p:
        return "urdu-like"
    return "shahmukhi"

# ---------------------------------------------------------------------------
# 6. SCORING
# ---------------------------------------------------------------------------
def score(outdir):
    import pandas as pd
    from sacrebleu.metrics import BLEU, CHRF

    files = sorted(outdir.glob("outputs_*.jsonl"))
    if not files:
        sys.exit("No outputs found to score.")
    rows, fid_rows = [], []
    for fp in files:
        name = fp.stem.replace("outputs_", "")
        recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r.get("raw") != "__ERROR__"]
        if not recs:
            print(f"(skipping {name}: no translations yet)")
            continue
        df = pd.DataFrame(recs)
        groups = list(df.groupby("subset")) + [("ALL", df)]
        for subset, g in groups:
            hyps, refs = g["hyp"].tolist(), [g["ref"].tolist()]
            bleu = BLEU().corpus_score(hyps, refs).score
            chrf = CHRF(word_order=2).corpus_score(hyps, refs).score   # chrF++
            rows.append({"model": name, "subset": subset, "n": len(g),
                         "BLEU": round(bleu, 2), "chrF++": round(chrf, 2)})
        cats = Counter(classify_script(h) for h in df["hyp"])
        fid = {"model": name, "n": len(df)}
        for c in ["shahmukhi", "urdu-like", "gurmukhi", "devanagari", "arabic", "latin", "other", "empty/error"]:
            fid[c + " %"] = round(100 * cats.get(c, 0) / len(df), 1)
        fid_rows.append(fid)

    scores = pd.DataFrame(rows)
    fid = pd.DataFrame(fid_rows)
    scores.to_csv(outdir / "scores.csv", index=False)
    fid.to_csv(outdir / "script_fidelity.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== BLEU / chrF++ ===")
    print(scores.pivot(index="model", columns="subset", values=["BLEU", "chrF++"]).round(2))
    print("\n=== Script fidelity (heuristic) ===")
    print(fid.to_string(index=False))
    print(f"\nSaved: {outdir/'scores.csv'}, {outdir/'script_fidelity.csv'}")

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen", "deepseek"],
                    help=f"'all' or any of: {', '.join(MODELS)} , mock")
    ap.add_argument("--limit", type=int, help="number of sentences (pilot); default = all")
    ap.add_argument("--csv", help="local CSV instead of Hugging Face dataset")
    ap.add_argument("--src-col", help="English column name (auto-detected if omitted)")
    ap.add_argument("--ref-col", help="Shahmukhi column name (auto-detected if omitted)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between API calls")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    if not args.score_only:
        models = list(MODELS) if args.models == ["all"] else args.models
        for m in models:
            if m != "mock" and m not in MODELS:
                sys.exit(f"Unknown model '{m}'. Choose from: {', '.join(MODELS)}, mock")
        df = load_data(args)
        for m in models:
            run_model(m, df, outdir, args.delay)
    score(outdir)


if __name__ == "__main__":
    main()
