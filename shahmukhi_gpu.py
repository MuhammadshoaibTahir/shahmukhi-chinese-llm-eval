#!/usr/bin/env python3
"""
Run small open Chinese LLMs on a free Colab GPU (T4) for the Shahmukhi study.
Free: no API key and no daily limits.

Uses the same prompt as PSMT-eval / shahmukhi_eval.py and writes outputs in the
same format, so you score everything together with:
    python shahmukhi_eval.py --score-only --outdir <same folder>

COLAB SETUP
-----------
Runtime -> Change runtime type -> T4 GPU
    !pip install -U transformers accelerate bitsandbytes pandas tqdm
    !python shahmukhi_gpu.py --models qwen3-8b --limit 50 --outdir /content/drive/MyDrive/shahmukhi_results

Run ONE model per command. Each 7-9B model downloads ~5-18 GB the first time.
Outputs are saved after every batch, so if Colab disconnects just run again.
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Small Chinese models that fit on a free T4 (16 GB) in 4-bit.
# name : (Hugging Face model id, needs trust_remote_code)
# Check each model card on huggingface.co before running, and record the
# exact id + date in your paper.
# ---------------------------------------------------------------------------
MODELS = {
    "qwen3-8b":     ("Qwen/Qwen3-8B",                        False),  # Alibaba
    "hunyuan-7b":   ("tencent/Hunyuan-7B-Instruct",          False),  # Tencent
    "glm4-9b":      ("zai-org/glm-4-9b-chat-hf",             False),  # Zhipu (may also be THUDM/glm-4-9b-chat-hf)
    "deepseek-7b":  ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", False),  # DeepSeek (Qwen-based reasoning distill; slow)
    "minicpm5-2b":  ("openbmb/MiniCPM5-2B",                  True),   # OpenBMB / Tsinghua (tiny baseline)
}

# Per-model fixes for "thinking" models:
#  - NO_THINK_PREFIX: models that switch thinking off with "/no_think" in the message
#  - ALWAYS_THINK: models that cannot switch thinking off -> give them room to finish
NO_THINK_PREFIX = {"hunyuan-7b"}
ALWAYS_THINK = {"deepseek-7b": 2048}   # max_new_tokens for these

SYSTEM_PROMPT = (
    "You are an expert English-to-Punjabi translator. Output only the translation "
    "in Punjabi Shahmukhi script (Perso-Arabic based), with no explanations or notes. "
    "Preserve the original meaning while keeping the translation natural in Punjabi."
)
USER_TEMPLATE = (
    "Please translate the English sentence into Punjabi Shahmukhi script. "
    "Only return the translation.\n\n{src}"
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)


def clean_output(text):
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>")[-1]      # keep only the answer after reasoning
    elif "<think>" in text:
        return ""                              # reasoning never finished -> no answer
    text = THINK_RE.sub("", text).strip()
    m = ANSWER_RE.search(text)               # Hunyuan wraps answers in <answer>...</answer>
    if m:
        text = m.group(1).strip()
    text = re.sub(r"</?answer>", "", text).strip()
    if not text:
        return ""
    return text.splitlines()[0].strip().strip('"“”\'').strip()


def load_data(csv_path, limit):
    import pandas as pd
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    df["id"] = df.index
    if limit:
        per = max(1, limit // df["subset"].nunique())
        df = (df.groupby("subset", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per), random_state=42))
                .reset_index(drop=True))
    return df


def load_model(model_id, trust):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        sys.exit("No GPU found. In Colab: Runtime -> Change runtime type -> T4 GPU.")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=trust)
    model.eval()
    return tok, model


def build_prompt(tok, src, name=""):
    user = USER_TEMPLATE.format(src=src)
    if name in NO_THINK_PREFIX:
        user = "/no_think " + user
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    try:
        # enable_thinking=False turns off "thinking" for Qwen3-style models;
        # other chat templates ignore it.
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except Exception:
        # some templates reject a system role -> merge into user message
        msgs = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run(name, df, outdir, batch_size, max_new_tokens):
    import torch
    from tqdm import tqdm

    model_id, trust = MODELS[name]
    if name in ALWAYS_THINK:
        max_new_tokens = max(max_new_tokens, ALWAYS_THINK[name])
        batch_size = min(batch_size, 4)
    path = outdir / f"outputs_{name}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    todo = df[~df["id"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print(f"[{name}] all {len(df)} already done")
        return

    print(f"[{name}] loading {model_id} (4-bit) ...")
    t0 = time.time()
    tok, model = load_model(model_id, trust)
    print(f"[{name}] loaded in {time.time() - t0:.0f}s — translating {len(todo)} sentences")

    with open(path, "a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(todo), batch_size)):
            batch = todo.iloc[start:start + batch_size]
            prompts = [build_prompt(tok, s, name) for s in batch["english"]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False,  # greedy ≈ paper's temperature 0.1
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for row, raw in zip(batch.itertuples(), texts):
                rec = {"id": int(row.id), "subset": row.subset, "model": model_id,
                       "src": row.english, "ref": row.shahmukhi,
                       "raw": raw,
                       # always-thinking model that never finished reasoning -> no answer
                       "hyp": "" if (name in ALWAYS_THINK and "</think>" not in raw) else clean_output(raw)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{name}] done -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help=", ".join(MODELS))
    ap.add_argument("--csv", default="psmt_eval.csv")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    for m in args.models:
        if m not in MODELS:
            sys.exit(f"Unknown model '{m}'. Choose from: {', '.join(MODELS)}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.csv, args.limit)
    print(f"{len(df)} sentences")
    for m in args.models:
        try:
            run(m, df, outdir, args.batch_size, args.max_new_tokens)
        except Exception as e:
            print(f"[{m}] FAILED: {type(e).__name__}: {e}\n  -> check the model id on huggingface.co")
    print("\nNow score everything with:\n  python shahmukhi_eval.py --score-only --outdir", outdir)


if __name__ == "__main__":
    main()
