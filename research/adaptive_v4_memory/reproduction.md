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
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal_prerequisites.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal.py --scale s55 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal.py --scale s151 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_seed_extension_causal.py
```

The separately preregistered four-seed extension starts only after the immutable
4,500-shard primary cohort passes its strict audit. It produces an independently
auditable 3,600-shard matrix, then permits the 8,100-shard nine-seed confirmatory
core summary only when frozen contracts match and seed namespaces are disjoint.
After the independently audited 9,000-shard primary causal matrix, it repeats
physical matching and all-arm equivalence for every extension checkpoint, runs
7,200 causal shards, and permits the 16,200-shard combined causal summary under
the same strict pooling rule. The
parallel runners first require exact serial/parallel probes. P2 causal also
requires calibration-only physical-memory matching and exact chunked-quality versus
sequential-tier equivalence for every seed, scale, budget, family, context, and arm.

## P3 compatible-model natural and safety evidence

Acquire only the revisions pinned in the P3 manifests, prepare their source/data
inventories, and verify the snapshot before inference. The natural runner commands
share the same two frozen roots:

Before the P2 sequence gate, the preregistered exception permits only immutable
model and public-code prefetch plus cryptographic verification. The following
command clones and verifies the pinned SCBench, LongBench v2, and LongMemEval code,
writes `p3-natural-source-prefetch-v1`, and explicitly records that it acquired no
benchmark dataset, selected no baseline, generated no dataset, and ran no inference:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_sources.py \
  --prefetch-only
```

Do not run the default source-inventory mode, any dataset preparation, baseline
selection, or prediction command until the primary P2 matrix, its five-seed strict
core audit, the nine-seed confirmatory core audit, and both five- and nine-seed
causal audits all pass the frozen sequence gate. The common executable gate reopens
and hashes all five dependencies, including both core audits' raw-matrix bindings,
so direct runner invocation cannot bypass this order. After that gate, the default source
command re-verifies every prefetched checkout and binds the prefetch inventory
digest into the final source inventory.

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

After the terminal nine-seed P2 causal audit, fixed scorer selection, and all five
Qwen3-4B RULER datasets exist, run the separately preregistered same-global-budget
compatibility cohort. It compares `fixed+pins` with `natural-adaptive-quota+pins`;
it is not an unchanged port of the synthetic controller.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_ruler.py
```

After both the adaptive RULER audit and baseline SCBench audit are terminal, run the
separately frozen shared-context replication. Adaptive allocation occurs only during
the initial shared-context prefill; later turn tokens are appended identically.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_scbench_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_scbench.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_scbench.py
```

After the adaptive RULER audit and baseline LongBench-v2 audit are terminal, run the
separately frozen one-pass reasoning replication. Adaptive allocation applies only to
the document-context prefill; the question is appended identically under both arms.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_longbench_v2_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_longbench_v2.py
```

After the adaptive RULER audit and baseline LongMemEval audit are terminal, run the
separately frozen judge-blocked generation cohort. It retains all raw responses and
physical quota audits, but emits no quality, non-inferiority, or negative-result claim.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_longmemeval_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longmemeval.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --judge-mode blocked
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_longmemeval.py
```

After the adaptive RULER audit and baseline MRCR audit are terminal, run the
separately frozen multi-needle replication. Adaptive allocation applies only to
the long context prefill; the final query is appended identically under both arms.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_mrcr_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_mrcr.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_mrcr.py
```

After all four adaptive Qwen summaries are terminal, run the separately frozen
cross-benchmark audit. It preserves each benchmark's interval and multiplicity
correction; it never pools heterogeneous scores or p-values. LongMemEval remains
outside this adaptive suite while its paid official judge is unavailable.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_suite_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_suite.py
```

Run the separately reported Phi-4-mini cross-family transfer only after the
nine-seed P2 causal audit and Qwen fixed-baseline selection are terminal:

```bash
export PHI_SNAPSHOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/models/cfbefacb99257ffa30c83adab238a50856ac3083
.venv/bin/hf download microsoft/Phi-4-mini-instruct \
  --revision cfbefacb99257ffa30c83adab238a50856ac3083 \
  --local-dir "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/verify_p3_natural_model.py \
  --manifest research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json \
  --snapshot "$PHI_SNAPSHOT" \
  --output artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-model-verification.json \
  --experiment-id p3-cross-family-model-snapshot-verification-v1
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_cross_family_ruler_dataset.py \
  --ruler-root "$RULER_ROOT" --tokenizer-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_cross_family_ruler.py \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_cross_family_ruler.py
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_cross_family_adaptive_quota_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_cross_family_ruler.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_cross_family_adaptive_quota_ruler.py
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_cross_family_adaptive_quota_longbench_v2_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py --cohort cross-family-adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_cross_family_adaptive_quota_longbench_v2.py
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
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_adaptive_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_adaptive_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p4_adaptive_production_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_adaptive_production_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_adaptive_production_systems_matrix.py
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
and generated SHA-256. During generation, every declared `path`/`sha256` pair in the
P3 evidence graph is reopened and rehashed, including bindings reached through linked
JSON cell artifacts; a missing half-binding, missing file, or nested digest drift fails
closed. GitHub Actions remains disabled by user request and must not be
reported as passed. It is outside the completion gate; the five digest-bound local release
checks above are the final source-verification contract. The release-gate artifact also
requires `HEAD` to have an `origin/*` upstream with zero locally tracked ahead/behind
commits. This is a push-synchronization check against the local tracking ref; it does not
fetch, establish network freshness, open a PR, or imply CI success.

## Official DeepSeek-V4 boundary

Compatible Qwen evidence is not official DeepSeek-V4 evidence. Until the exact
serving checkpoint, conversion contract, accelerator topology, and storage/HBM
requirements in `manifests/p3-flashmemory-deepseek-v4-v1.json` are satisfied, keep
that result blocked and use the frozen reproduction command and cost/resource
contract in that manifest. Never substitute a compatible model or the static local
adapter for official FlashMemory or an external fused multi-GPU production runtime.
The static adapter also cannot verify non-contiguous residency-layout costs,
position-aware cache-miss recomputation, or a fused attention-kernel cost model;
those boundaries are machine-readable in the P4 blocker and final P5 audit.
