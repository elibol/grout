#!/usr/bin/env python3
"""Kernel-level attention microbench: FlashInfer's dense single-request
kernels at grout's Qwen3 shapes.

Benches the same kernels SGLang wraps (FA2 template on sm_120; on sm_100
the batch wrappers can select trtllm-gen — see B200_RUNBOOK.md):

  prefill: single_prefill_with_kv_cache(causal=True)   [non-paged, batch 1]
  decode:  single_decode_with_kv_cache(use_tensor_cores=True)

Units are us per call, which equals per-layer per-step cost — directly
comparable to grout's GROUT_PROFILE_SYNC_OPS "Attention" row (prefill) and
nsys per-launch times for fmha_decode_gqa_split_mapped +
splitk_reduce_merge_mapped (decode; sum the pair, since single_decode does
its split-KV merge internally).

Usage (needs a venv with flashinfer; JIT-compiles on first call):
  ../bench_envs/vllm_env/bin/python benchmarks/bench_flashinfer_attn.py
"""

import argparse

import torch
import flashinfer

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.float16


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--num-qo-heads", type=int, default=NUM_QO_HEADS)
    ap.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS)
    ap.add_argument("--head-dim", type=int, default=HEAD_DIM)
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--backend", choices=["auto", "fa2", "fa3"], default="auto")
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(dev)}, flashinfer {flashinfer.__version__}")
    print(
        f"shapes: H_qo={args.num_qo_heads} H_kv={args.num_kv_heads} "
        f"D={args.head_dim} {DTYPE} backend={args.backend}"
    )

    print("\n== prefill: single_prefill_with_kv_cache(causal=True), qo_len == kv_len ==")
    for n in args.lens:
        q = torch.randn(n, args.num_qo_heads, args.head_dim, dtype=DTYPE, device=dev)
        k = torch.randn(n, args.num_kv_heads, args.head_dim, dtype=DTYPE, device=dev)
        v = torch.randn_like(k)
        us = bench(
            lambda: flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=True, backend=args.backend
            )
        )
        print(f"  len={n:6d}: {us:10.1f} us/call")

    if args.prefill_only:
        return

    print("\n== decode: single_decode_with_kv_cache(use_tensor_cores=True) ==")
    for n in args.lens:
        q = torch.randn(args.num_qo_heads, args.head_dim, dtype=DTYPE, device=dev)
        k = torch.randn(n, args.num_kv_heads, args.head_dim, dtype=DTYPE, device=dev)
        v = torch.randn_like(k)
        us = bench(
            lambda: flashinfer.single_decode_with_kv_cache(q, k, v, use_tensor_cores=True)
        )
        print(f"  kv={n:7d}: {us:10.1f} us/call")


if __name__ == "__main__":
    main()
