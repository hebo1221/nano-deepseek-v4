# P3 cross-family RULER transfer freeze

Status: frozen before any P3 model prediction, generated RULER dataset,
fixed-baseline selection, or natural-benchmark outcome existed.

## Decision

The study adds a second instruction-tuned model family rather than increasing
only within-model rows. The authoritative contract is
`manifests/p3-cross-family-ruler-transfer-v1.json`.

The transfer model is
[`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct)
at revision `cfbefacb99257ffa30c83adab238a50856ac3083`. It is ungated, MIT-licensed,
declares `Phi3ForCausalLM`, and supports 131,072 tokens. The 21 top-level
snapshot files total 7,694,054,130 bytes and have canonical digest-set
`102f70c901015c02c6157e947ffbe2e1aa5667c8b28a0b3df17f4bc269e9889c`.
Local Transformers 5.2 resolves the snapshot without remote code, and the
pinned KVPress checkout explicitly supports `Phi3ForCausalLM`.

## Frozen matrix

| Axis | Values |
|---|---|
| Model | Phi-4-mini-instruct, one exact revision |
| RULER tasks | all 13 canonical tasks |
| Context | 8K, 32K, 128K |
| Examples | 100 per task-context |
| Arms | native; Qwen-screen-selected method at 50% KV |
| Total | 3,900 predictions per arm; 7,800 paired predictions |

The fixed comparator is selected once by the complete Qwen3-1.7B 57-cell
screen and transferred unchanged. No Phi result may select a different method,
compression ratio, prompt, task subset, or stopping rule.

## Statistical and claim boundary

The report includes the overall paired bootstrap interval, per-length exact
sign-flip tests over 13 task means, Holm adjustment across three lengths,
every task-by-length effect, worst regression, operational failures, and
realized KV bytes. Bootstrap seed `9171501` is frozen. The 51% memory gate is
applied to the maximum task-by-length ratio of summed fixed bytes to summed
native bytes over pairs where both arms measured positive resident bytes; a
cell with no measurable pair fails that component. Passing supports only
bounded transfer of the selected
operating point across Qwen3 and Phi-4. Failure is retained as a negative
transfer result. This cohort is not pooled with Qwen rows, is not a second full
natural suite, and provides no evidence about official DeepSeek-V4.

The deterministic seed and memory aggregation were added as a second recorded
amendment after runner implementation but before any cross-family dataset row
or model outcome existed.
