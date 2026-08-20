// grout <-> trtllm-gen ragged context attention shim.
//
// Wraps flashinfer's TllmGenFmhaRunner (the CUDA C++ trtllm-gen kernels)
// behind a plain-C ABI so grout can dlopen it for long-context prefill
// attention on sm_100. The ragged (SeparateQkv) path takes dense K/V with
// explicit strides, so grout's [kv_heads, max_seq, D] caches bind with no
// repacking. Build with the Makefile next to this file; requires the
// flashinfer source tree and a populated trtllm-gen artifact directory
// (cubins + include/flashInferMetaInfo.h).
//
// The launcher function body below is copied verbatim from
// flashinfer/csrc/trtllm_fmha_kernel_launcher.cu (Apache-2.0) so behavior
// matches the wrapper vLLM uses; only the TVM-FFI binding layer is
// omitted.

#include <flashinfer/allocator.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common.h>
#include <flashinfer/trtllm/fmha/decoder_impl_common.h>
#include <flashinfer/trtllm/fmha/fmhaRunnerParams.h>

#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

// Cubin loader: flashinfer's python flow downloads artifacts; here we read
// them from the directory the cubin path prefix (TLLM_GEN_FMHA_CUBIN_PATH,
// set by the Makefile) points at. sha256 is not re-verified: the artifact
// dir is the same one flashinfer's own benchmarks populate and verify.
namespace flashinfer {
namespace trtllm_cubin_loader {
std::string getCubin(const std::string& path, const std::string& sha256) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "grout_trtllm_shim: cannot open cubin %s\n", path.c_str());
    return std::string();
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace trtllm_cubin_loader
}  // namespace flashinfer

#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>

namespace flashinfer {

constexpr size_t kTrtllmGenSoftmaxStatsGuardBytes = 1 * 1024 * 1024;

class TllmGenFmhaRunnerCache {
 public:
  using Key = std::tuple<Data_type, Data_type, Data_type, Data_type, int, int, int, int>;

  static std::shared_ptr<TllmGenFmhaRunner> get(Data_type q_data_type, Data_type k_data_type,
                                                Data_type v_data_type, Data_type o_data_type,
                                                int num_elts_sage_q = 0, int num_elts_sage_k = 0,
                                                int num_elts_sage_p = 0, int num_elts_sage_v = 0) {
    static std::unordered_map<Key, std::shared_ptr<TllmGenFmhaRunner>, KeyHash> cache;
    static std::mutex cache_mutex;
    Key key = std::make_tuple(q_data_type, k_data_type, v_data_type, o_data_type, num_elts_sage_q,
                              num_elts_sage_k, num_elts_sage_p, num_elts_sage_v);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto runner = std::make_shared<TllmGenFmhaRunner>(q_data_type, k_data_type, v_data_type,
                                                      o_data_type, num_elts_sage_q,
                                                      num_elts_sage_k, num_elts_sage_p,
                                                      num_elts_sage_v);
    cache.emplace(key, runner);
    return runner;
  }

 private:
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return std::hash<int>()(static_cast<int>(std::get<0>(k))) ^
             (std::hash<int>()(static_cast<int>(std::get<1>(k))) << 1) ^
             (std::hash<int>()(static_cast<int>(std::get<2>(k))) << 2) ^
             (std::hash<int>()(static_cast<int>(std::get<3>(k))) << 3) ^
             (std::hash<int>()(std::get<4>(k)) << 4) ^ (std::hash<int>()(std::get<5>(k)) << 5) ^
             (std::hash<int>()(std::get<6>(k)) << 6) ^ (std::hash<int>()(std::get<7>(k)) << 7);
    }
  };
};

// ---- copied verbatim from flashinfer csrc (see file header) ----
void trtllm_ragged_attention_launcher(
    void* out, void* query, void* key, void* value, void* workspace_buffer, int* seq_lens,
    int* cum_seq_lens_q, int* cum_seq_lens_kv, float* attention_sinks, float* lse,
    Data_type q_data_type, Data_type k_data_type, Data_type v_data_type, Data_type o_data_type,
    int64_t max_q_len, int64_t max_kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_v, int64_t sum_seq_q, int64_t sum_seq_kv,
    double bmm1_scale, double bmm2_scale, const float* bmm1_scale_log2_ptr,
    const float* bmm2_scale_ptr, double o_sf_scale, int64_t batch_size, int64_t window_left,
    int64_t sm_count, bool enable_pdl, bool is_causal, int64_t k_stride_keys_values,
    int64_t k_stride_heads, int64_t k_stride_batch, int64_t v_stride_keys_values,
    int64_t v_stride_heads, int64_t v_stride_batch, float skip_softmax_threshold_scale_factor,
    bool skips_softmax, int64_t workspace_size, const float* sage_attn_sfs_q,
    const float* sage_attn_sfs_k, const float* sage_attn_sfs_p, const float* sage_attn_sfs_v,
    int num_elts_sage_q, int num_elts_sage_k, int num_elts_sage_p, int num_elts_sage_v,
    int64_t lse_stride_tokens, int64_t lse_stride_heads, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: " << num_kv_heads
            << " and num_qo_heads: " << num_qo_heads;
    FLASHINFER_ERROR(err_msg.str());
  }
  auto fmha_runner = TllmGenFmhaRunnerCache::get(q_data_type, k_data_type, v_data_type, o_data_type,
                                                 num_elts_sage_q, num_elts_sage_k, num_elts_sage_p,
                                                 num_elts_sage_v);
  TllmGenFmhaRunnerParams runner_params;

  runner_params.qPtr = query;
  runner_params.kPtr = key;
  runner_params.vPtr = value;
  runner_params.kvPageIdxPtr = nullptr;
  runner_params.seqLensKvPtr = seq_lens;
  runner_params.oPtr = out;
  runner_params.mHeadDimQk = head_dim_qk;
  runner_params.mHeadDimV = head_dim_v;
  runner_params.mNumHeadsQ = num_qo_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_qo_heads / num_kv_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenKv = max_kv_len;
  runner_params.mQkvLayout = QkvLayout::SeparateQkv;
  runner_params.mMultiProcessorCount = sm_count;
  runner_params.stream = stream;
  // the scaleSoftmaxLog2Ptr and outputScalePtr have higher priority than the scaleSoftmaxLog2 and
  // outputScale. if they are not nullptr, then scaleSoftmaxLog2 and outputScale will be ignored
  runner_params.outputScale = bmm2_scale;
  runner_params.outputScalePtr = bmm2_scale_ptr;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  runner_params.scaleSoftmaxLog2Ptr = bmm1_scale_log2_ptr;
  runner_params.mScaleSfO = o_sf_scale;
  runner_params.mChunkedAttentionSize = INT_MAX;  // disable chunked attention by INT_MAX
  runner_params.mAttentionWindowSize =
      window_left == -1 ? INT_MAX : window_left + 1;  // disable window attention by INT_MAX
  runner_params.mMaxSeqLenQ = max_q_len;
  runner_params.mSumOfSeqLensQ = sum_seq_q;
  runner_params.mSumOfSeqLensKv = sum_seq_kv;
  runner_params.cumSeqLensKvPtr = cum_seq_lens_kv;
  runner_params.cumSeqLensQPtr = cum_seq_lens_q;
  runner_params.ptrAttentionSinks = attention_sinks;
  runner_params.enable_pdl = enable_pdl;

  runner_params.kStrideKeysValues = k_stride_keys_values;
  runner_params.kStrideHeads = k_stride_heads;
  runner_params.kStrideBatch = k_stride_batch;
  runner_params.vStrideKeysValues = v_stride_keys_values;
  runner_params.vStrideHeads = v_stride_heads;
  runner_params.vStrideBatch = v_stride_batch;

  runner_params.mKernelType = FmhaKernelType::Context;
  runner_params.mTileScheduler = TileScheduler::Persistent;
  runner_params.mMaskType =
      is_causal ? TrtllmGenAttentionMaskType::Causal : TrtllmGenAttentionMaskType::Dense;

  AlignedAllocator float_allocator(workspace_buffer, workspace_size);
  size_t max_batch_size = 8192;
  size_t max_num_qo_heads = 256;
  size_t num_semaphores =
      round_up(max_batch_size * max_num_qo_heads, 8);  // max 8MB, should align to 16 bytes
  // Workspace layout: counter | (softmax if lse) | scratch. Keep the 8MB counter at the head so
  // test guard regions around the first 8MB remain stable across LSE on/off calls.
  runner_params.multiCtasKvCounterPtr = float_allocator.aligned_alloc<int32_t>(
      num_semaphores * sizeof(uint32_t), 16, "trtllm_gen_counter_workspace");
  // Only allocate the softmax stats slab when LSE is requested; size with the same
  // tile-aligned upper bound used by the paged launcher to prevent OOB writes on
  // variable-q workloads.
  if (lse != nullptr) {
    size_t const softmax_slots = static_cast<size_t>(num_qo_heads) *
                                 static_cast<size_t>(batch_size) *
                                 static_cast<size_t>(round_up(max_q_len, int64_t{256}));
    runner_params.softmaxStatsPtr = float_allocator.aligned_alloc<float2>(
        sizeof(float2) * softmax_slots + kTrtllmGenSoftmaxStatsGuardBytes, 16,
        "trtllm_gen_softmax_workspace");
    runner_params.lsePtr = lse;
    runner_params.lseStrideTokens = lse_stride_tokens;
    runner_params.lseStrideHeads = lse_stride_heads;
  }
  // scratch takes the rest of the workspace buffer
  runner_params.multiCtasKvScratchPtr =
      float_allocator.aligned_alloc<void>(0, 16, "trtllm_gen_scratch_workspace");

  runner_params.mSkipsSoftmaxWhenPossible = skips_softmax;
  runner_params.mSkipSoftmaxThresholdScaleFactor = skip_softmax_threshold_scale_factor;

  // SageAttention scaling factors.
  runner_params.ptrSageAttnSfsQ = sage_attn_sfs_q;
  runner_params.ptrSageAttnSfsK = sage_attn_sfs_k;
  runner_params.ptrSageAttnSfsP = sage_attn_sfs_p;
  runner_params.ptrSageAttnSfsV = sage_attn_sfs_v;

  auto [foundKernels, kinfo] = fmha_runner->isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    err_msg << "Missing TRTLLM-GEN kernel ragged attention: " << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }

  fmha_runner->run(runner_params);
}
// ---- end verbatim copy ----

}  // namespace flashinfer

namespace {

thread_local std::string g_last_error;

struct DeviceScratch {
  void* workspace = nullptr;
  size_t workspace_size = 0;
  int32_t* seq_lens = nullptr;      // [1]
  int32_t* cum_q = nullptr;         // [2]
  int32_t* cum_kv = nullptr;        // [2]
  int sm_count = 0;

  bool ensure(size_t ws_bytes) {
    if (workspace_size < ws_bytes) {
      if (workspace) cudaFree(workspace);
      if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess) {
        workspace = nullptr;
        workspace_size = 0;
        return false;
      }
      workspace_size = ws_bytes;
    }
    if (!seq_lens) {
      if (cudaMalloc(&seq_lens, sizeof(int32_t)) != cudaSuccess) return false;
      if (cudaMalloc(&cum_q, 2 * sizeof(int32_t)) != cudaSuccess) return false;
      if (cudaMalloc(&cum_kv, 2 * sizeof(int32_t)) != cudaSuccess) return false;
      cudaDeviceProp prop{};
      int dev = 0;
      cudaGetDevice(&dev);
      cudaGetDeviceProperties(&prop, dev);
      sm_count = prop.multiProcessorCount;
    }
    return true;
  }
};

DeviceScratch& scratch() {
  static DeviceScratch s;
  return s;
}

}  // namespace

extern "C" {

const char* grout_trtllm_last_error() { return g_last_error.c_str(); }

// v1 ordering: grout brackets the launch with device-wide syncs instead of
// threading its stream through the FFI. Costs ~10 us per call; the stream
// plumbing is a later optimization.
int grout_trtllm_sync() { return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1; }

// Dense-cache ragged context attention, fp16 in/out, batch 1, causal.
//   q:   [q_len, num_q_heads, head_dim], contiguous
//   k,v: token stride / head stride / batch stride given in ELEMENTS
//   out: [q_len, num_q_heads, head_dim], contiguous
// stream: cudaStream_t of the producing/consuming stream (may be null for
// the default stream; the caller is responsible for ordering).
// Returns 0 on success.
int grout_trtllm_ragged_context_f16(
    void* out, void* q, void* k, void* v, int64_t q_len, int64_t kv_len,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t k_stride_tokens, int64_t k_stride_heads,
    int64_t v_stride_tokens, int64_t v_stride_heads,
    float bmm1_scale, int enable_pdl, void* stream) {
  try {
    auto& s = scratch();
    constexpr size_t kWorkspaceBytes = size_t(192) * 1024 * 1024;
    if (!s.ensure(kWorkspaceBytes)) {
      g_last_error = "workspace allocation failed";
      return 1;
    }
    cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
    int32_t h_seq[1] = {int32_t(kv_len)};
    int32_t h_cq[2] = {0, int32_t(q_len)};
    int32_t h_ckv[2] = {0, int32_t(kv_len)};
    cudaMemcpyAsync(s.seq_lens, h_seq, sizeof(h_seq), cudaMemcpyHostToDevice, cu_stream);
    cudaMemcpyAsync(s.cum_q, h_cq, sizeof(h_cq), cudaMemcpyHostToDevice, cu_stream);
    cudaMemcpyAsync(s.cum_kv, h_ckv, sizeof(h_ckv), cudaMemcpyHostToDevice, cu_stream);

    flashinfer::trtllm_ragged_attention_launcher(
        out, q, k, v, s.workspace, s.seq_lens, s.cum_q, s.cum_kv,
        /*attention_sinks=*/nullptr, /*lse=*/nullptr,
        DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16,
        /*max_q_len=*/q_len, /*max_kv_len=*/kv_len, num_q_heads, num_kv_heads,
        /*head_dim_qk=*/head_dim, /*head_dim_v=*/head_dim,
        /*sum_seq_q=*/q_len, /*sum_seq_kv=*/kv_len,
        /*bmm1_scale=*/double(bmm1_scale), /*bmm2_scale=*/1.0,
        /*bmm1_scale_log2_ptr=*/nullptr, /*bmm2_scale_ptr=*/nullptr,
        /*o_sf_scale=*/0.0, /*batch_size=*/1, /*window_left=*/-1,
        /*sm_count=*/s.sm_count, /*enable_pdl=*/enable_pdl != 0,
        /*is_causal=*/true, k_stride_tokens, k_stride_heads, /*k_stride_batch=*/0,
        v_stride_tokens, v_stride_heads, /*v_stride_batch=*/0,
        /*skip_softmax_threshold_scale_factor=*/0.0f, /*skips_softmax=*/false,
        /*workspace_size=*/int64_t(s.workspace_size),
        /*sage_attn_sfs_q=*/nullptr, /*sage_attn_sfs_k=*/nullptr,
        /*sage_attn_sfs_p=*/nullptr, /*sage_attn_sfs_v=*/nullptr,
        /*num_elts_sage_q=*/0, /*num_elts_sage_k=*/0, /*num_elts_sage_p=*/0,
        /*num_elts_sage_v=*/0, /*lse_stride_tokens=*/0, /*lse_stride_heads=*/0,
        cu_stream);
    return 0;
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return 2;
  } catch (...) {
    g_last_error = "unknown exception in trtllm launcher";
    return 3;
  }
}

}  // extern "C"
