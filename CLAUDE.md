# Project 3: Fine-Tuned Small Model for Text-to-SQL (QLoRA + DPO)

## Goal
Fine-tune an open model for Text-to-SQL, then show honestly how it compares with a large paid model on accuracy, cost and latency. The result should plug into `text2sql/` by changing a base URL and model name.

## Stack (fixed)
- Base model: **Qwen2.5-Coder-7B-Instruct**.
- Training: **Unsloth + TRL** with 4-bit QLoRA. `SFTTrainer` for supervised fine-tuning, then `DPOTrainer` for preference tuning. Target GPUs with 16-24 GB VRAM (Colab, Kaggle, or a rented GPU). Use gradient checkpointing and a modest max sequence length to fit memory.
- Data: Spider and BIRD training splits. Check each dataset's license and terms before use and record them in `docs/DATA.md`. Never train on any dev/test example that is used for evaluation.
- Evaluation: the `t2s_eval` package from `../text2sql/eval/` (execution accuracy). Do not re-implement a second scorer.
- Serving: merge the adapter, quantize with AWQ 4-bit, and serve with **vLLM** through its OpenAI-compatible server. Tooling for quantization changes often, so verify the current recommended library and versions before writing the script and pin them.
- Dashboard: Next.js (App Router) + TypeScript + Tailwind + Lucide in `web/`, deployable to Vercel. It shows the results tables and charts and has a playground that calls an OpenAI-compatible endpoint, with a mock mode when no endpoint is configured. vLLM itself runs on a GPU host, not on Vercel.
- Tracking: Weights & Biases or plain JSON logs (record the choice in PROGRESS.md).

## Layout
```
sql-finetune/
  CLAUDE.md  PROGRESS.md  README.md  .env.example
  data/               # scripts and processed formats (raw data not committed)
  train/              # sft.py, dpo.py, configs/
  notebooks/          # Colab/Kaggle notebooks that run the scripts
  eval/               # wrappers that call t2s_eval on each model
  serve/              # merge, quantize, vLLM launch and load-test scripts
  web/                # Next.js results dashboard and playground
  docs/               # DATA.md, RESULTS.md, MODEL_CARD.md
```

## Milestones (do in order)
- **M1 Data:** download and clean Spider and BIRD train data, build prompts with schema serialization, remove duplicates and any overlap with eval splits, produce SFT-format data. Document sizes and license notes in `docs/DATA.md`. Can start immediately.
- **M2 Training scripts:** `train/sft.py` with config files, runnable in Colab/Kaggle. Test it on a tiny subset first to confirm it runs end to end.
- **M3 Baselines:** needs `text2sql/eval/` (Project 1, M1) to exist. Measure zero-shot Qwen2.5-Coder-7B-Instruct and one large API model on the same eval split and sample.
- **M4 SFT run and ablations:** train, then ablate at least data size and LoRA rank (for example r = 8, 16, 32), plus learning rate if budget allows. Evaluate every variant with the same harness.
- **M5 DPO:** build preference pairs from execution feedback (generate multiple SQL candidates per question; executed-correct is chosen, executed-wrong is rejected). Train with `DPOTrainer` and compare against SFT-only.
- **M6 Serve and benchmark:** merge, quantize (AWQ 4-bit), serve with vLLM, measure accuracy after quantization, throughput, p95 latency, and cost per 1,000 queries versus the large API model.
- **M7 Publish:** `docs/RESULTS.md`, model card, upload adapter/model to Hugging Face, `web/` dashboard deployed to Vercel, README with honest limitations.
- **M8 Integration (optional):** point `text2sql/` at the served model and record the end-to-end result.

## Definition of done and numbers to produce
- Execution accuracy for: zero-shot base, SFT, SFT+DPO, quantized model, and the large API model, all on the same split.
- Ablation table (data size, LoRA rank).
- Cost per 1,000 queries and p95 latency for the served model vs the API model, with hardware and pricing assumptions stated.

## Rules
- **Compute:** Claude Code in this repo cannot train on a free GPU by itself. Write scripts and notebooks, then tell the human exactly what to run on Colab, Kaggle or a rented GPU and what logs or files to bring back. Mark anything not yet run as "not yet measured".
- Never fabricate or estimate metrics. Only report values from runs that were executed and whose logs are saved. Do not quote numbers from papers or model cards as your own.
- Guard against leakage: train and eval data must never overlap. Check and document it.
- Do not claim the fine-tuned model "beats" a larger model unless the measured table shows it. Report whatever the results are, including a loss.
- Never commit raw datasets, checkpoints or secrets. Pin library versions.
- Conventional commits, one per milestone step. Update PROGRESS.md after each milestone.
- Use the subagents in `.claude/agents/`: builder writes, tester tests, evaluator runs benchmarks, reviewer checks before a milestone is marked done.
