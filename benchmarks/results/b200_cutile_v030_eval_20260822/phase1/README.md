# Phase 1: regression gate

Both release binaries were rebuilt, including the feature-gated
`grout_bench`. All seven GPU kernel tests passed.

The pp=2048 plus 24-token generated text has SHA-256
`407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`,
identical to the established reference.

| cell | v0.3.0 median | recorded canonical | delta |
|---|---:|---:|---:|
| pp=2048, tg=36 e2e | 563.72 ms | 586.76 ms | -3.93% |
| pp=32768, tg=36 cuTile e2e | 3258.41 ms | 3487.45 ms | -6.57% |
| pp=18, tg=128 e2e | 1638.33 ms | 1620.72 ms | +1.09% |

The tg=128 direct decode rate is 78.9 tok/s versus 79.8 tok/s recorded
(-1.13%). No cell is more than 2% slower, so the resource-usage and JIT-counter
regression escalation was not triggered.

**Regressions found: no.**
