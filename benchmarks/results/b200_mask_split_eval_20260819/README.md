# B200 LPT mask-split validation

Qwen3-32B was measured on B200 at default clocks using grout `b93bb0d` and
cutile-rs `e90f9b8`. The paired test used three order-alternating rounds per
length. Each process discarded JIT compilation and one additional warmup, then
averaged three measured requests with 36 generated tokens.

The old arm set `GROUT_FMHA_PREFILL_LPT_MASK_SPLIT=0`; the new arm left the
variable unset, selecting the new default. All other LPT and decode settings
matched the shipping sm_100 long-prefill profile.

| pp | old / new prefill (ms) | prefill delta | old / new e2e (ms) | e2e delta | output |
|---:|---:|---:|---:|---:|---|
| 16384 | 1170.086 / **1154.499** | **-1.332%** | 1645.321 / **1631.896** | **-0.816%** | byte-identical |
| 32768 | 3051.327 / **2977.738** | **-2.412%** | 3548.429 / **3474.549** | **-2.082%** | byte-identical |

`results.csv` contains every round mean and generated-text hash. Raw logs and
generated text are intentionally not committed.

The confirmation sweep is in `../sweep/20260819_221611`. Its median same-session
comparison is:

| pp | grout (ms) | vLLM (ms) | grout gap | prior gap |
|---:|---:|---:|---:|---:|
| 16384 | 1630.70 | **1566.66** | **+4.09%** | +5.83% |
| 32768 | 3488.40 | **3131.21** | **+11.41%** | +14.39% |

SGLang was attempted by the standard wrapper but remained unavailable because
its sm_100 extension could not load `libnuma.so.1`; it was not required for
this validation.
