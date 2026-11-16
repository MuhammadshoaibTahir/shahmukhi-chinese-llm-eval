# Evaluating Chinese LLMs on Punjabi Shahmukhi

Code, model outputs and evaluation data for our study of how well Chinese-developed large language models translate English into **Punjabi written in the Shahmukhi script**. Shahmukhi is the Perso-Arabic script used for Punjabi in Pakistan.

We test five models on the [PSMT-eval](https://aclanthology.org/2026.wat-1.1/) benchmark (Basit & Azeemi, WAT 2026). We score them with BLEU and chrF++, a script and language classifier, and human evaluation.

## Main finding

Only the API-served models write Shahmukhi Punjabi reliably. The small open-weight models mostly answer in Urdu, Gurmukhi or Hindi.

| Model | Developer | Sentences | BLEU | chrF++ | Shahmukhi output |
|---|---|---|---|---|---|
| Ling 3.2 Flash | Ant Group | 500 | 13.95 | 39.53 | 91.4% |
| ERNIE 5.5 Flash (preview) | Baidu | 100* | 11.01 | 37.95 | 76.0% |
| Qwen3-8B | Alibaba | 500 | 0.93 | 13.14 | 19.0% (50% Urdu, 30% Gurmukhi) |
| GLM-4-9B-Chat | Zhipu AI | 500 | 0.33 | 2.70 | 4.4% (85% Gurmukhi) |
| Hunyuan-7B-Instruct | Tencent | 500 | 0.21 | 3.73 | 7.4% (50% Hindi) |

\* ERNIE is still running because of free-tier API limits. Its 100 sentences so far are all from FLORES-101, so its scores are not yet comparable with the others' 500-sentence scores.

## Repository contents

| File | What it is |
|---|---|
| `build_psmt_csv.py` | Builds `psmt_eval.csv` from the Hugging Face dataset |
| `shahmukhi_eval.py` | Runs API models (OpenRouter), scores BLEU/chrF++, classifies script |
| `shahmukhi_gpu.py` | Runs open-weight models on a GPU (4-bit) |
| `outputs_<model>.jsonl` | One line per sentence: id, subset, model, src, ref, raw, hyp |
| `scores.csv` | BLEU and chrF++ per model and subset |
| `script_fidelity.csv` | Share of outputs per script/language label |
| `primary_ratings.csv` | 200 rated outputs (50 per model) |
| `second_rater.csv` | 50 outputs rated by an independent second rater |

In the output files, `id` is the row number in `psmt_eval.csv`, `raw` is the model's full response and `hyp` is the cleaned translation that we score. All models translated the same 500 sentences, 125 from each subset (stratified random sample, seed 42).

## How to reproduce

We ran everything in Google Colab. API models used the OpenRouter free tier; open-weight models used a free T4 GPU.

```bash
pip install openai sacrebleu pandas tqdm huggingface_hub
python build_psmt_csv.py                     # creates psmt_eval.csv (5,182 rows)

# API models (needs an OpenRouter key)
export OPENROUTER_API_KEY="sk-or-..."
python shahmukhi_eval.py --csv psmt_eval.csv --models ling ernie --limit 500 --delay 4 --outdir .

# Open-weight models (needs a GPU)
pip install -U transformers accelerate bitsandbytes
python shahmukhi_gpu.py --csv psmt_eval.csv --models qwen3-8b glm4-9b hunyuan-7b --limit 500 --outdir .

# Score everything already in the folder
python shahmukhi_eval.py --score-only --outdir .
```

Both scripts skip sentences that are already saved, so an interrupted run can simply be restarted. The open-weight runs are deterministic (greedy decoding): running them twice gave identical outputs.

## Human evaluation

Each output was rated for **adequacy** on a 1–5 scale (5 = perfect, 1 = unacceptable) and for **language** (is it Punjabi in Shahmukhi?). An output in the wrong language or script scores at most 2.

- **`primary_ratings.csv`:** An LLM (Claude) proposed a score and language label for each item. A native Punjabi speaker reviewed all 200 items. `pre_review_*` columns are the sheet before the final review; `final_*` columns are the ratings we report, which follow the LLM's proposal in every item. These are therefore **AI-assisted** ratings.
- **`second_rater.csv`:** A second native speaker rated 50 random items without seeing any proposed scores. In round 1 the rater accepted fluent Urdu as Punjabi; after re-briefing, round 2 added an explicit language label. Language agreement with the primary ratings in round 2 was 96% (Cohen's κ = 0.90). Two items have no round-2 score.

## Notes and limitations

- We used 500 of PSMT-eval's 5,182 sentences, so scores are close to, but not the same as, full-test-set results.
- Ling and ERNIE are free preview models on OpenRouter and may change. They were queried in November 2026.
- Open-weight models ran in 4-bit precision.
- The script classifier is rule-based. It agrees with human language labels on about 87% of items.
If you use this repository, please cite: > Muhammad Shoaib Tahir. 2026. Evaluating Chinese Large Language Models on Punjabi Shahmukhi Translation

Contact: [your name and email]
