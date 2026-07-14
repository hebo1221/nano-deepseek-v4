# Adaptive V4 Memory reproduction guide

This runbook reproduces the paper-grade evidence in dependency order. Run every
command from the repository root on a clean commit. The runners are resume-safe:
they validate existing shard digests before reuse and write progress through an
atomic temporary-file replacement. Never delete a terminal failure artifact or
repair a held-out result post hoc.

## Environment and frozen inputs

Create the project environment and install the separately frozen natural-suite
dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -r research/adaptive_v4_memory/requirements-p3-ruler.txt
.venv/bin/pip install -r research/adaptive_v4_memory/requirements-p3-natural.txt
```

All seeds, scales, workload counts, benchmark revisions, model snapshots, data
digests, and system cells come from `manifests/paper-grade-study-v1.json` and the
phase manifests. Do not substitute a newer dataset or checkpoint under the same
artifact path.

## P2 synthetic core and causal factorial

Run both scales, audit the complete 4,500-shard core, freeze causal prerequisites,
then execute and summarize the 9,000-shard factorial:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_core_parallel.py --scale s55 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_core_parallel.py --scale s151 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py \
  --matrix artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json \
  --output artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py \
  --matrix artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json \
  --output artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.strict.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_prerequisites.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py --scale s55 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py --scale s151 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_causal_prerequisites.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_causal_parallel.py --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_causal_factorial.py \
  --matrix artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json \
  --output artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json
```

The separately preregistered four-seed extension starts only after the immutable
4,500-shard primary cohort passes its strict audit. It produces an independently
auditable 3,600-shard matrix, then permits the 8,100-shard nine-seed confirmatory
summary only when frozen contracts match and seed namespaces are disjoint. The
parallel runners first require exact serial/parallel probes. P2 causal also
requires calibration-only physical-memory matching and exact chunked-quality versus
sequential-tier equivalence for every seed, scale, budget, family, context, and arm.

## P3 compatible-model natural and safety evidence

Acquire only the revisions pinned in the P3 manifests, prepare their source/data
inventories, and verify the snapshot before inference. The natural runner commands
share the same two frozen roots:

```bash
export KVPRESS_ROOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources/kvpress
export RULER_ROOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources/RULER
export MODEL_SNAPSHOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/models/cdbee75f17c01a7cc42f958dc650907174af0554
mkdir -p artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources
git init "$RULER_ROOT"
git -C "$RULER_ROOT" remote add origin https://github.com/hsiehjackson/RULER.git
git -C "$RULER_ROOT" fetch --depth 1 origin 38da79d79519ef87aa46ae804f838e1eab7f86d7
git -C "$RULER_ROOT" checkout --detach FETCH_HEAD
git init "$KVPRESS_ROOT"
git -C "$KVPRESS_ROOT" remote add origin https://github.com/NVIDIA/kvpress.git
git -C "$KVPRESS_ROOT" fetch --depth 1 origin 6d965557a5b9f0201a2301b23c454473dd681d0d
git -C "$KVPRESS_ROOT" checkout --detach FETCH_HEAD
.venv/bin/hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_sources.py
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_datasets.py
.venv/bin/python research/adaptive_v4_memory/scripts/verify_p3_natural_model.py
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_ruler_dataset.py \
  --ruler-root "$RULER_ROOT" --tokenizer-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_scbench.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longmemeval.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --judge-mode blocked
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_mrcr.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
```

Summarize each benchmark with `summarize_p3_natural_benchmark.py`, binding the
`native-dense` and `strongest-memory-matched-fixed` cell artifacts, then run:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_safety_assets.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_safety_stress.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_safety_stress.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_suite.py
```

Operational generation failures score zero. LongSafety generation may be complete
while its paid official judge remains blocked; that state must remain `unverified`
for comparative LongSafety safety claims.

## P4 systems evidence

Run the feasibility probe, serial-interleaved reference matrix, and actual-concurrency
adapter matrix separately:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_500k_context_preflight.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_500k_context_preflight.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_systems_matrix.py \
  --output artifacts/adaptive_v4_memory/paper_grade/p4/reference-systems.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_production_systems_matrix.py \
  --adapter-executable research/adaptive_v4_memory/scripts/p4_continuous_batch_adapter.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_production_systems_matrix.py
```

Reference interleaving is never relabeled as concurrency. Partial and failed cells
remain terminal observations in the tables and tail-failure counts.

## Online learned lookahead and final package

After P2-P4 are terminal, run the distinct online learned controller and generate
the digest-bound package:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p1_online_lookahead_parallel.py --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p1_online_learned_lookahead.py
.venv/bin/python research/adaptive_v4_memory/scripts/build_p5_paper_package.py
.venv/bin/ruff check nano_deepseek_v4 research/adaptive_v4_memory/scripts tests
.venv/bin/mypy nano_deepseek_v4 research/adaptive_v4_memory/scripts
.venv/bin/pytest -q
.venv/bin/python -m build
.venv/bin/twine check dist/*
.venv/bin/python research/adaptive_v4_memory/scripts/run_p5_release_gate.py
```

The final package is valid only from a clean source tree. Check
`artifacts/adaptive_v4_memory/paper_grade/p5/artifact-index.json` for every input
and generated SHA-256. GitHub Actions is currently disabled manually; do not report CI as
passed. A successful final GitHub Actions CI run remains mandatory before goal completion,
and package generation or the five local release commands above does not waive it.

## Official DeepSeek-V4 boundary

Compatible Qwen evidence is not official DeepSeek-V4 evidence. Until the exact
serving checkpoint, conversion contract, accelerator topology, and storage/HBM
requirements in `manifests/p3-flashmemory-deepseek-v4-v1.json` are satisfied, keep
that result blocked and use the frozen reproduction command and cost/resource
contract in that manifest. Never substitute a compatible model or the static local
adapter for official FlashMemory or an external fused multi-GPU production runtime.
