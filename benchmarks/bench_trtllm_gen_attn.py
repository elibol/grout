#!/usr/bin/env python3
"""Benchmark FlashInfer's Blackwell trtllm-gen context-attention wrapper."""

import argparse
import math

import torch
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


def bench(fn, warmup: int, iters: int) -> float:
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
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_one(n: int, page_size: int, args: argparse.Namespace) -> float:
    if n % page_size != 0:
        raise ValueError(f"length {n} must be divisible by page size {page_size}")
    num_pages = n // page_size
    dev = torch.device("cuda")
    q = torch.randn(
        n, args.num_qo_heads, args.head_dim, dtype=torch.float16, device=dev
    )
    k = torch.randn(
        num_pages,
        args.num_kv_heads,
        page_size,
        args.head_dim,
        dtype=torch.float16,
        device=dev,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    workspace = torch.zeros(args.workspace_mb * 1024 * 1024, dtype=torch.uint8, device=dev)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=dev)[None, :]
    seq_lens = torch.tensor([n], dtype=torch.uint32, device=dev)
    cum_seq_lens = torch.tensor([0, n], dtype=torch.int32, device=dev)

    def invoke():
        trtllm_batch_context_with_kv_cache(
            q,
            (k, v),
            workspace,
            block_tables,
            seq_lens,
            n,
            n,
            1.0 / math.sqrt(args.head_dim),
            1.0,
            1,
            cum_seq_lens,
            cum_seq_lens,
            out=out,
            kv_layout="HND",
            enable_pdl=args.enable_pdl,
        )

    return bench(invoke, args.warmup, args.iters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--dense-page", action="store_true")
    ap.add_argument("--num-qo-heads", type=int, default=64)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--workspace-mb", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--enable-pdl", action=argparse.BooleanOptionalAction, default=None
    )
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"shapes: H_qo={args.num_qo_heads} H_kv={args.num_kv_heads} "
        f"D={args.head_dim} fp16"
    )
    for n in args.lens:
        page_size = n if args.dense_page else args.page_size
        us = run_one(n, page_size, args)
        print(f"  len={n:6d} page_size={page_size:6d}: {us:10.1f} us/call")


if __name__ == "__main__":
    main()
