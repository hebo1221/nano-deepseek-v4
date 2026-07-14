# P3 natural-language suite freeze

Status: amended and frozen before execution. Exact-revision snapshot verification
corrected transcription errors in the RULER scorer and tokenizer digests before
any natural prediction; the amendments are recorded in the authoritative manifest.
Only the already frozen model revision was prefetched and hash-verified. Benchmark
data/source acquisition, baseline selection, generated datasets, and inference
remain blocked on the P2 causal sequence gate.

## Outcome

The natural transfer stage is now an executable, digest-bound protocol rather
than a list of benchmark names. The authoritative contract is
`manifests/p3-natural-suite-v1.json`, and
`scripts/validate_p3_natural_suite_manifest.py` rejects task selection, revision
drift, sample-count drift, or silent prompt truncation.

The primary compatible model is
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its frozen config declares a
262,144-token context and the snapshot is 8,060,917,568 bytes. The existing
Qwen3-1.7B RULER tier remains useful small-model evidence but cannot cover the
64K and 128K protocol points.

## Frozen evaluation volume

| Benchmark | Required coverage | Predictions per arm |
|---|---:|---:|
| RULER | 13 tasks × 500 samples × 8K/16K/32K/64K/128K | 32,500 |
| SCBench | 12 tasks, 922 contexts, 5,143 turns, both modes | 10,286 |
| LongBench v2 | all multiple-choice items | 503 |
| LongMemEval | all LongMemEval_S cleaned questions | 500 |
| MRCR | 2/4/8 needles, 100 samples in each bin through 128K | 1,500 |
| **Minimum total** | no selected-task substitution | **45,289** |

The SCBench counts were verified from the Parquet footers and `multi_turns`
columns at the pinned Hub revision. MRCR retains all 2,400 released examples in
the source inventory even though the primary Qwen comparison stops at its five
bins through 128K.

## Provenance and scoring boundaries

- [SCBench](https://huggingface.co/datasets/microsoft/SCBench) data and the
  [MInference SCBench code](https://github.com/microsoft/MInference/tree/main/scbench)
  are pinned separately. Every official task-specific scorer and both
  multi-turn and multi-request modes are retained.
- [LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2) uses all
  503 questions and the pinned direct 0-shot prompt and answer parser. The
  upstream head-tail truncation branch is explicitly disabled; oversized items
  are reported as unsupported.
- [LongMemEval cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
  uses all 500 S questions with full histories. The official
  `gpt-4o-2024-08-06` judge contract is retained and raw judge exchanges must be
  logged. A missing judge blocks the official metric instead of being silently
  replaced.
- [OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr) uses the exact
  alphanumeric-prefix gate and `difflib.SequenceMatcher` ratio from the pinned
  dataset card.

FlashMemory and IndexCache are not compatible Qwen baselines. The
[IndexCache repository](https://github.com/THUDM/IndexCache) is pinned at
`08d22d69b1aa2aa0a3de23df6d6b88dbd5b5d044`, but it is evaluated only in a
supported DeepSeek Sparse Attention tier. The same separation applies to the
existing official FlashMemory feasibility manifest. No projected Qwen number
may stand in for either DSA result.

The same architecture boundary applies to this project's controller. Qwen3
always runs native and memory-matched fixed KV baselines. `fixed+pins` and
`calibrated+pins` are not assigned Qwen proxy implementations: they become
eligible only if a separate architecture-preserving port passes prediction and
cache equivalence. Otherwise their natural-suite cells are reported as
incompatible while the baseline study continues.

## Conflict-free continuation

`scripts/prepare_p3_natural_datasets.py` verifies all 20 source files and writes
a digest-bound inventory, but it calls the P3 sequence gate before any download.
It therefore cannot contend with or bypass the 4,500-shard P2 run. After P2 and
the 9,000-shard causal audit finish, it acquires 2,391,718,372 bytes of frozen
natural data and validates row counts, schemas, unique IDs, answer labels, and
SCBench turn totals before model inference begins.
