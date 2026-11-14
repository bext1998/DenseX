# Long-context quality: what was measured

DenseX's current evidence is a small, synthetic long-document evaluation, not
a general benchmark or a guarantee of error-free answers. Speed results and
quality results use different configurations; keep them separate.

The primary supported series is Qwen 3.8 27B GSQ-RCO across its full tier range;
Unsloth Dynamic v3.1 is the second series added later. Swift 1.5 is a GSQ-RCO
variant used as the measured sample here. Other tiers' scores are **not yet
measured** under #24 / #28; support does not transfer this sample's scores.

## Q2: 120K document, 18 questions

The historical Q2 run on 2026-11-10 used DenseX `cb04f83`, the Swift 1.5
Qwen3.8 27B GSQ-RCO IQ3_XXS-mtp model, and the measured Windows RTX 3060
12 GB / Ryzen 5 3600X / 32 GB RAM system. GPU graphics clock was fixed at
1702 MHz for this comparison.

| Setting | Q2 value |
|---|---|
| Document depth / server context | About 120000 / 131072 tokens |
| Attention | Full target attention, not KVMem sparse retrieval |
| Speculative decoding | MTP-8 |
| Main K/V | Compare q8_0/q4_0 (K8V4) with q4_0/q4_0 (K4V4) |
| FFN host placement | 27 evenly spread layers, streamed to GPU for computation |
| DenseX execution | GDN deferred commit, prefetch 4, native quantized KV path |
| Sampling | Temperature 0 |
| Thinking | Enabled, `reasoning_effort: "medium"`, budget 3072 tokens |
| Output limit | 7168 tokens, including thinking |
| Request state | Fresh server per cache arm; all recorded requests have `cache_n=0` |
| Seeds | 20260905 and 20260917 |

The 18 cases contain these tasks, each at both seeds:

| Task | Variants | Questions |
|---|---|---:|
| Aggregate ledger entries | standard L2, standard L3, hard L2 | 6 |
| Compare conflicting records | standard L3, hard L1 | 4 |
| Follow multiple relations | standard L3, hard L3 | 4 |
| Structured summary/status decisions | standard L2, standard L3 | 4 |

Each case supplies a long document, a fixed question/schema, and expected
structured fields. The client scores the final answer separately from reasoning.
An exact pass requires all scored fields to be correct. Mean field accuracy
counts partially correct answers; it is not the exact-pass rate.

## Results

| Main cache | Exact passes, all 18 | Exact passes, comparable 16 | Mean field accuracy, all 18 |
|---|---:|---:|---:|
| K8V4 | 14/18 | 13/16 | 0.929 |
| K4V4 | 15/18 | 14/16 | 0.940 |

The comparable-16 subset excludes the two standard summary L3 cases to match
earlier research comparisons, not to cherry-pick 16 passes. Sixteen of the 18 final answers matched across cache settings. Both settings
failed compare hard L1 at both seeds and summary L3 at seed 20260917. K8V4
also failed aggregate standard L3 at seed 20260905, which K4V4 answered correctly.

This small run found no observable quality regression from K4V4 relative to
K8V4. It does **not** establish equality to BF16, establish lossless KV
quantization, or show that K4V4 is more accurate in general. Temperature-zero
outputs can diverge with changed numeric paths or speculative batch sizes.

For Q2 only, median generation was 27.5 tok/s for both settings; median TTFT
was 475 s (K8V4) / 480 s (K4V4), and peak VRAM was 11909 / 10888 MiB.
The lower K4V4 memory use motivated later placement tuning.

## Do not attach these scores to the current speed preset

The [README](../../README.md#results-rtx-3060-12-gb-measured) C34 speed table
uses MTP-12, 16 host FFN layers, draft window 32768, ranked draft vocabulary
49152, thinking off, and 512 generated tokens. Its two 120K runs achieved
28.1 / 27.9 tok/s with TTFT 474 s and peak VRAM 11857 MiB.

Q2 used MTP-8, 27 host layers, and thinking medium. Its 15/18 score is not a
new quality test of the C34 configuration or of this checkout. Long-context
quality at other memory sizes, 256K, and other model quantizations is
**untested**.

## Measurement record and reproduction limits

The private research workspace recorded Q2 under `DenseX/NOTES.md`, section
"Q2: 120K long-document quality" (2026-11-10). The run used
`run_q2.ps1` / `dx_long120.py` and prepared `v2long-01/cases.jsonl` cases.
The raw result filenames are:

- `longdoc_densex_v2_120k-dxk8v4-01.jsonl`
- `longdoc_densex_v2_120k-dxk4v4-01.jsonl`

These inputs, runners, and raw results remain in the private research workspace;
they are **not bundled in this public engine repository**. This page preserves
the recorded settings and results, but a fresh clone cannot reproduce those
scores from the documentation alone. No new GPU quality run was performed here.

For a future comparable run, retain the same complete prompts, seeds, expected
fields, medium effort, thinking/output budgets, and scoring rules. Record the
engine commit, exact model, launch arguments, cache types, host placement,
TTFT, generation rate, RAM/VRAM, and errors. Change one setting at a time;
do not shorten the document, discard failures, or substitute easier retrieval
questions to improve the result. Run only when the GPU is available.

## Needle-in-a-haystack

The separate needle evaluation is tracked by
[issue #40](https://github.com/bext1998/DenseX/issues/40). Results for that task
are pending / **untested here**. No needle score is inferred from Q2's multi-record
reasoning scores or the short installation smoke check.
