# B200 Qwen3-32B LPT retune

This bundle records grout `3b07fd2` with cutile-rs `e90f9b8` at default B200
clocks. All paired comparisons used three order-alternating rounds after each
unique kernel form had completed a JIT run and one additional warmup.

The canonical profile exports `GROUT_FMHA_PREFILL_GQA_GROUP=0`; in the engine,
zero resolves to all eight query heads for Qwen3-32B. An initial group-4 screen
therefore did not represent the shipping shape. `default_correction.csv`
repeats the checked/twin decision at group 8 and directly pairs group 8 against
group 4. `screen_group8.csv` and `pairs_group8.csv` contain the corrected knob
screen and finalist confirmation.

The checked/twin decision remains parity: the twin changes prefill by -0.14%
at 16K and -0.06% at 32K, and all output text is identical. Checked stays.
Group 8 beats group 4 by 9.52% at 16K and 17.03% at 32K. Latency 3 and swizzle
16 were the only group-8 screen finalists; paired deltas were respectively
+0.08%/-0.22% and -0.08%/+0.02% at 16K/32K. No tuning profile changed.

The copied cubins are the canonical group-8 forms. Checked uses `REG=128`,
`STACK=200`, and 27 static LDL/STL instructions; the exact-body twin uses
`REG=128`, `STACK=0`, and no LDL/STL.
