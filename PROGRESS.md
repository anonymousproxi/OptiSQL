# Progress: sql-finetune

Last updated: 2026-10-21. Status: setup done, no work started.

## Milestones
| ID | Milestone | Status | Notes |
|---|---|---|---|
| M1 | Data preparation | Not started | Can start now |
| M2 | Training scripts (SFT) | Not started | |
| M3 | Baselines | Not started | Needs text2sql M1 (eval harness) |
| M4 | SFT run + ablations | Not started | Needs GPU run by human |
| M5 | DPO | Not started | Needs GPU run by human |
| M6 | Quantize, serve (vLLM), benchmark | Not started | Needs GPU |
| M7 | Publish (HF, model card, dashboard) | Not started | |
| M8 | Integrate into text2sql (optional) | Not started | |

## Metrics (measured only)
| Model | Exec accuracy | Split / sample size | Cost per 1,000 queries | p95 latency | Date |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct zero-shot | not yet measured | | | | |
| Large API model | not yet measured | | | | |
| SFT | not yet measured | | | | |
| SFT + DPO | not yet measured | | | | |
| SFT + DPO, AWQ 4-bit | not yet measured | | | | |

## Ablations
| Variant | Exec accuracy | Notes |
|---|---|---|
| not yet run | | |

## Decisions
(none yet)

## Blockers / needs from human
- GPU runs (M4, M5, M6) must be executed by the human on Colab/Kaggle/rented GPU; Claude supplies scripts and exact commands.

## Log
- 2026-10-21: repo scaffolded with CLAUDE.md, subagents, hooks.
