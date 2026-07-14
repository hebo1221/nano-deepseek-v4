# P3 official FlashMemory-DeepSeek-V4 feasibility audit

## Decision

The official-weight experiment is **not executable from the current public release
on this host**. This is not only a compute-capacity problem. The published retriever
artifact and the released SGLang serving integration currently expose incompatible
checkpoint contracts. No official V4 result may be claimed until both blockers are
resolved.

The machine-readable reproduction record is
[`p3-flashmemory-deepseek-v4-v1.json`](../manifests/p3-flashmemory-deepseek-v4-v1.json).

## Frozen upstream state

| Component | Frozen identity | Audited payload |
|---|---|---:|
| Paper | arXiv `2606.09079` | FlashMemory-DeepSeek-V4 |
| Serving code | Git commit `39fe54def633496cb2b1bd44898135e3547058b3` | Mode A and Mode B scripts hashed in the manifest |
| Base model | `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` | 46 safetensors, 159,617,149,040 bytes |
| Public retriever | `libertywing/FlashMemory-Deepseek-V4@70431ba57bfcce00ffd9d0174aed1b8ca5c32a2e` | 509,633,992-byte safetensors weight |

## Hard blocker 1: checkpoint/runtime contract

The release's Mode A and Mode B launch scripts default to
`checkpoints/top3_R930_joint.pt`. The integrated scorer loads it with
`torch.load(..., weights_only=True)` and hard-codes 64 heads with Q-LoRA rank 1024.
The pinned public Hugging Face snapshot contains no PT checkpoint; it publishes
`weights/flashmemory_ds_v4.safetensors`, whose model card describes 128 heads and
Q-LoRA rank 2048. The release does not include a documented conversion or a parity
fixture connecting these two artifacts.

Therefore, renaming or ad-hoc converting the public file would not be a valid
reproduction. The required resolution is either the exact serving checkpoint or an
upstream conversion plus a golden logit/chunk-selection parity test.

## Hard blocker 2: compute topology

The base tensor payload is 159.62 GB. The local GB10 system has 130.60 GB physical
unified memory and had 138.94 GB available memory plus swap at audit time, so the
weights alone do not fit. Disk capacity is sufficient, but storage does not resolve
the load-time or runtime memory envelope.

The two released execution modes have different evidentiary meaning:

- **Mode A** uses TP4 and retains the full KV cache on GPU. It validates retriever
  selection quality only and cannot establish memory or concurrency savings.
- **Mode B** launches separate TP8 prefill and TP8 decode servers plus a router,
  requiring 16 accelerator slots and NIXL/UCX cross-machine connectivity. This is
  the path needed for physical offload and concurrency claims.

## Cost envelope

Public H20 self-service pricing was not available in the audited sources. Using
Lambda H100 SXM prices on 2026-07-14 strictly as a planning proxy, TP4 Mode A is
approximately **$16.36/hour**, and two 8-GPU Mode B servers are approximately
**$63.84/hour**. An 8-hour Mode B allocation is about **$510.72** and 24 hours about
**$1,532.16**, before tax and any network or storage surcharge. H100 runtime
compatibility has not been established, so these numbers are not an executable quote.

## Execution boundary

Once the checkpoint contract is resolved and suitable hardware is provisioned, use
the pinned acquisition, post-download verification, and launch contracts in the
manifest. Mode A covers quality only across RULER, SCBench, LongBench-v2,
LongMemEval, and MRCR. Mode B freezes the physical systems matrix at contexts
8K/32K/128K/512K, batch 1/4/8/16, concurrency 1/8/32, generation
128/512/2048, five warmups, and 30 paired measurements per cell. Record native and
FlashMemory runs with identical prompts, decoding, admission limits, load order,
warmup, and measurement windows, retaining every partial or failed cell. Until
then, the Qwen3 experiments remain separate compatible-model transfer results and
must never be labeled official DeepSeek-V4 evidence.
