# B200 Qwen3-32B attention ceiling

This bundle compares one causal prefill-attention layer at Qwen3-32B's exact
shape: 64 query heads, 8 KV heads, head dimension 128, FP16, and query length
equal to KV length. External kernels use CUDA-event medians after 10 warmups
and 50 measured calls at default B200 clocks.

FlashInfer 0.6.6 selects its FA2 backend on sm_100 and is about 2x slower than
grout's checked LPT. The trtllm-gen batch-context wrapper is 28-29% faster than
grout and provides enough aggregate headroom over 64 layers to cover the
current 64.0/357.2 ms grout-vLLM gaps at 16K/32K.

The requested degenerate dense trtllm-gen layout (`page_size=kv_len`) is not
present in the installed cubin artifact: the wrapper reports a missing context
kernel with `numTokensPerPage=16384`. The production-supported page-size-16
kernel was measured instead. PDL on/off agreed within 0.1%.

An Nsight run for trtllm-gen was attempted, but Nsight remained in Python
teardown after the calls completed and generated an unbounded raw stream. It
was terminated and is not retained. Grout's synchronized-op and Nsight values
at 32K agree within 0.9%, limiting the likely event-method skew.
