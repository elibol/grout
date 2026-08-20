# Builds libgrout_trtllm.so: grout's dlopen'd bridge to the trtllm-gen
# ragged context attention kernels (CUDA C++, sm_100-class GPUs only).
#
# Inputs (override on the command line):
#   FLASHINFER_DIR  flashinfer source checkout (include/flashinfer/trtllm/...)
#   ARTIFACT_DIR    trtllm-gen fmha artifact dir: contains the .cubin files
#                   and include/flashInferMetaInfo.h. flashinfer's benchmarks
#                   populate this under ~/.cache/flashinfer/<ver>/cubins/...;
#                   find it with:
#                     python -c "from flashinfer.jit.cubin_loader import FLASHINFER_CUBIN_DIR as d; print(d)"
#                   then append the trtllm-gen fmha artifact subdirectory.
#   CUDA_HOME       CUDA toolkit (needs nvcc + driver stubs), default /usr/local/cuda
#
# Usage:
#   make FLASHINFER_DIR=$HOME/dev/flashinfer \
#        ARTIFACT_DIR=$HOME/.cache/flashinfer/.../trtllm-gen-fmha
#
# Then at runtime:
#   GROUT_TRTLLM_ATTN=1 GROUT_TRTLLM_ATTN_LIB=$(pwd)/libgrout_trtllm.so <grout ...>

CUDA_HOME ?= /usr/local/cuda
FLASHINFER_DIR ?= $(HOME)/dev/flashinfer
ARTIFACT_DIR ?= $(error set ARTIFACT_DIR to the trtllm-gen fmha artifact directory)

NVCC := $(CUDA_HOME)/bin/nvcc

libgrout_trtllm.so: grout_trtllm_shim.cu
	$(NVCC) -O2 -std=c++17 -shared -Xcompiler -fPIC \
	  -I$(FLASHINFER_DIR)/include \
	  -I$(FLASHINFER_DIR)/csrc \
	  -I$(ARTIFACT_DIR)/include \
	  -DTLLM_GEN_FMHA_CUBIN_PATH=\"$(ARTIFACT_DIR)\" \
	  -lcuda -lnvrtc \
	  -o $@ $<

clean:
	rm -f libgrout_trtllm.so
