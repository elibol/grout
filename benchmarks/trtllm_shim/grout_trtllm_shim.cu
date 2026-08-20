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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
class ShimIcheck {
 public:
  ShimIcheck(bool condition, const char* expression)
      : failed_(!condition), expression_(expression) {}

  template <typename T>
  ShimIcheck& operator<<(T&& value) {
    if (failed_) message_ << std::forward<T>(value);
    return *this;
  }

  ~ShimIcheck() noexcept(false) {
    if (!failed_) return;
    throw std::runtime_error(std::string("Check failed: ") + expression_ + ": " + message_.str());
  }

 private:
  bool failed_;
  const char* expression_;
  std::ostringstream message_;
};
}  // namespace

#define TVM_FFI_ICHECK(condition) ShimIcheck(static_cast<bool>(condition), #condition)

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

enum class TllmPagedAttentionMode {
  Context,
  ForGen,
};

// ---- copied verbatim from flashinfer csrc (see file header) ----
void trtllm_paged_attention_launcher(
    void* out, void* out_scale_factor, void* query, void* key_cache, void* value_cache,
    void* workspace_buffer, int* block_tables, const void* k_block_scales_ptr,
    const void* v_block_scales_ptr, int* seq_lens, int* cum_seq_lens_q, int* cum_seq_lens_kv,
    float* attention_sinks, float* lse, Data_type q_data_type, Data_type kv_data_type,
    Data_type o_data_type, TllmPagedAttentionMode mode, int64_t batch_size, int64_t max_q_len,
    int64_t max_kv_len, int64_t num_pages_in_mem_pool, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_vo, int64_t page_size, int64_t q_stride_tokens,
    int64_t q_stride_heads, int64_t kv_stride_keys_values, int64_t kv_stride_heads,
    int64_t kv_stride_batch, int64_t max_num_blocks_per_seq, double bmm1_scale, double bmm2_scale,
    const float* bmm1_scale_log2_ptr, const float* bmm2_scale_ptr, double o_sf_scale,
    int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t window_left, int64_t sum_seq_q,
    int64_t sparse_mla_top_k, void* sliding_window_kv_pool, int* sparse_mla_top_k_lens,
    bool has_sliding_window_kv_pool, float skip_softmax_threshold_scale_factor, bool skips_softmax,
    bool uses_shared_paged_kv_idx, int64_t sm_count, bool enable_pdl, int64_t workspace_size,
    int64_t k_sf_stride_heads, int64_t k_sf_stride_batch, int64_t v_sf_stride_heads,
    int64_t v_sf_stride_batch, bool is_causal, int64_t lse_stride_tokens, int64_t lse_stride_heads,
    cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: " << num_kv_heads
            << " and num_qo_heads: " << num_qo_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  // For paged attention, K and V have the same dtype (kv_data_type).
  auto fmha_runner =
      TllmGenFmhaRunnerCache::get(q_data_type, kv_data_type, kv_data_type, o_data_type);
  TllmGenFmhaRunnerParams runner_params;

  // Common params
  runner_params.qPtr = query;
  runner_params.kPtr = key_cache;
  runner_params.vPtr = value_cache;
  runner_params.slidingWindowKvPoolPtr = sliding_window_kv_pool;
  runner_params.kvPageIdxPtr = block_tables;
  runner_params.kSfBasePtr = k_block_scales_ptr;
  runner_params.vSfBasePtr = v_block_scales_ptr;
  runner_params.seqLensKvPtr = seq_lens;
  runner_params.oPtr = out;
  runner_params.mHeadDimQk = head_dim_qk;
  runner_params.mHeadDimV = head_dim_vo;
  runner_params.mNumHeadsQ = num_qo_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_qo_heads / num_kv_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenKv = max_kv_len;
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = page_size;
  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.mMultiProcessorCount = sm_count;
  runner_params.qStrideTokens = q_stride_tokens;
  runner_params.qStrideHeads = q_stride_heads;
  runner_params.kStrideKeysValues = kv_stride_keys_values;
  runner_params.kStrideHeads = kv_stride_heads;
  runner_params.kStrideBatch = kv_stride_batch;
  runner_params.vStrideKeysValues = kv_stride_keys_values;
  runner_params.vStrideHeads = kv_stride_heads;
  runner_params.vStrideBatch = kv_stride_batch;
  runner_params.kSfStrideHeads = k_sf_stride_heads;
  runner_params.kSfStrideBatch = k_sf_stride_batch;
  runner_params.vSfStrideHeads = v_sf_stride_heads;
  runner_params.vSfStrideBatch = v_sf_stride_batch;
  runner_params.mNumPagesInMemPool = num_pages_in_mem_pool;
  runner_params.stream = stream;
  // the scaleSoftmaxLog2Ptr and outputScalePtr have higher priority than the scaleSoftmaxLog2 and
  // outputScale. if they are not nullptr, then scaleSoftmaxLog2 and outputScale will be ignored
  runner_params.outputScale = bmm2_scale;
  runner_params.outputScalePtr = bmm2_scale_ptr;
  runner_params.mScaleSfKv = 1.0f;  // which should be fused into bmm1_scale(k)/bmm2_scale(v/o)
  runner_params.kvSfScalePtr = nullptr;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  runner_params.scaleSoftmaxLog2Ptr = bmm1_scale_log2_ptr;
  runner_params.oSfPtr = out_scale_factor;
  runner_params.mSfStartTokenIdx = o_sf_start_index;
  runner_params.mScaleSfO = o_sf_scale;
  TVM_FFI_ICHECK(o_sf_vec_size == 16 || o_sf_vec_size == -1)
      << "Only support o_sf_vec_size == 16 or -1(not used)";
  runner_params.mChunkedAttentionSize = INT_MAX;  // disable chunked attention by INT_MAX
  runner_params.mAttentionWindowSize =
      window_left == -1 ? INT_MAX : window_left + 1;  // disable window attention by INT_MAX
  runner_params.mMaxSeqLenQ = max_q_len;
  runner_params.mSumOfSeqLensQ = sum_seq_q;
  runner_params.mUsesSharedPagedKvIdx = uses_shared_paged_kv_idx;
  runner_params.ptrAttentionSinks = attention_sinks;
  runner_params.enable_pdl = enable_pdl;

  // The sparse MLA parameters.
  runner_params.mSparseMlaType =
      sparse_mla_top_k <= 0
          ? TrtllmGenSparseMlaType::None
          : (sparse_mla_top_k_lens != nullptr ? TrtllmGenSparseMlaType::DynamicTokenSparse
                                              : TrtllmGenSparseMlaType::StaticTokenSparse);
  runner_params.mSparseMlaTopK = sparse_mla_top_k;
  bool const is_dsv4_sparse_mla_decode =
      isSparseMla(runner_params.mSparseMlaType) && head_dim_qk == 512 && head_dim_vo == 512;
  bool const is_mla_decode = (head_dim_qk == 576 && head_dim_vo == 512) ||
                             (head_dim_qk == 320 && head_dim_vo == 256) ||
                             is_dsv4_sparse_mla_decode;
  runner_params.sparseMlaTopKLensPtr = sparse_mla_top_k_lens;
  runner_params.mHasSlidingWindowKvPool = has_sliding_window_kv_pool;
  TVM_FFI_ICHECK(is_mla_decode || sparse_mla_top_k <= 0) << "Only decode MLA supports sparse MLA";

  AlignedAllocator float_allocator(workspace_buffer, workspace_size);
  if (mode == TllmPagedAttentionMode::Context) {
    runner_params.mMaskType =
        is_causal ? TrtllmGenAttentionMaskType::Causal : TrtllmGenAttentionMaskType::Dense;
    runner_params.mKernelType = FmhaKernelType::Context;
    runner_params.mTileScheduler = TileScheduler::Persistent;
    runner_params.mMultiCtasKvMode = false;

    runner_params.cumSeqLensQPtr = cum_seq_lens_q;
    runner_params.cumSeqLensKvPtr = cum_seq_lens_kv;
  } else {
    // Generation.
    // Note that kernel names are still labeled as using a dense mask even when maskType is
    // specified as causal, this is expected for better performance as each CTA will only process
    // one tokenQ in those cases, so dense mask works the same as causal mask.
    runner_params.mMaskType =
        is_mla_decode ? TrtllmGenAttentionMaskType::Dense : TrtllmGenAttentionMaskType::Causal;
    runner_params.mKernelType = FmhaKernelType::Generation;
    bool use_multi_block = true;
    runner_params.mTileScheduler =
        use_multi_block ? TileScheduler::Static : TileScheduler::Persistent;
    runner_params.mMultiCtasKvMode = use_multi_block;

    runner_params.cumSeqLensQPtr = cum_seq_lens_q;
    runner_params.cumSeqLensKvPtr = nullptr;

    size_t max_batch_size = 8192;   // todo(Yingyi): get from dlfw
    size_t max_num_qo_heads = 256;  // todo(Yingyi): get from dlfw, in total 8MB
    size_t num_semaphores =
        round_up(max_batch_size * max_num_qo_heads, 8);  // max 8MB, should align to 16 bytes
    // Workspace layout for generation: counter | (softmax if lse) | scratch. The counter slab is
    // kept at a fixed 8MB so test guard regions around the first 8MB remain stable.
    runner_params.multiCtasKvCounterPtr = float_allocator.aligned_alloc<int32_t>(
        num_semaphores * sizeof(uint32_t), 16, "trtllm_gen_counter_workspace");
  }

  // Only allocate the softmax stats buffer when LSE is requested. The kernel's write layout is
  // [batchSize, mMaxNumCtasQ, numHeadsQ] float2 elements (see fmhaReduction.cu), where
  // mMaxNumCtasQ = ceil_div(max_q_len, mStepQ) is bounded above by max_q_len for all currently
  // selectable kernels. We use round_up(max_q_len, 256) as a static upper bound on mMaxNumCtasQ:
  //   * For generation kernels with mStepQ == 1 (decode, spec-decoding), this is tight up to a
  //     padding of at most 255 slots per batch/head.
  //   * For context kernels (mStepQ == 64/128), this over-allocates by roughly mStepQ. A tighter
  //     bound would require deferring the allocation until after kernel selection so mStepQ is
  //     known here.
  // TODO: revisit once kernel selection exposes mStepQ ahead of the workspace carve-out.
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

  if (mode == TllmPagedAttentionMode::ForGen) {
    // scratch takes the rest of the workspace buffer
    runner_params.multiCtasKvScratchPtr =
        float_allocator.aligned_alloc<void>(0, 16, "trtllm_gen_scratch_workspace");
  }

  // Params for skipping softmax.
  runner_params.mSkipsSoftmaxWhenPossible = skips_softmax;
  runner_params.mSkipSoftmaxThresholdScaleFactor = skip_softmax_threshold_scale_factor;

  auto [foundKernels, kinfo] = fmha_runner->isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    err_msg << "Missing TRTLLM-GEN kernel ("
            << (mode == TllmPagedAttentionMode::Context ? "context" : "decode") << "): " << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }

  fmha_runner->run(runner_params);
}

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

// Identity block table + optional packed pool for the paged context path.
struct PagedScratch {
  int32_t* block_table = nullptr;
  int64_t block_table_pages = 0;
  void* k_pool = nullptr;
  void* v_pool = nullptr;
  size_t pool_bytes = 0;

  bool ensure_table(int64_t pages, cudaStream_t stream) {
    if (pages <= block_table_pages) return true;
    if (block_table) cudaFree(block_table);
    if (cudaMalloc(&block_table, pages * sizeof(int32_t)) != cudaSuccess) {
      block_table = nullptr;
      block_table_pages = 0;
      return false;
    }
    std::vector<int32_t> host(pages);
    for (int64_t i = 0; i < pages; i++) host[i] = int32_t(i);
    cudaMemcpyAsync(block_table, host.data(), pages * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    block_table_pages = pages;
    return true;
  }

  bool ensure_pools(size_t bytes) {
    if (bytes <= pool_bytes) return true;
    if (k_pool) cudaFree(k_pool);
    if (v_pool) cudaFree(v_pool);
    if (cudaMalloc(&k_pool, bytes) != cudaSuccess || cudaMalloc(&v_pool, bytes) != cudaSuccess) {
      pool_bytes = 0;
      return false;
    }
    pool_bytes = bytes;
    return true;
  }
};

PagedScratch& paged_scratch() {
  static PagedScratch s;
  return s;
}

// Repack dense [H, max_seq, D] f16 into packed HND pages
// [pages, H, 16, D]. One thread per 8 halfs (128-bit copies).
__global__ void repack_dense_to_paged_f16(const __half* __restrict__ src,
                                          __half* __restrict__ dst, int num_heads,
                                          int max_seq, int head_dim, int kv_len) {
  int64_t vec = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t vecs_per_row = head_dim / 8;
  int64_t total = int64_t(num_heads) * kv_len * vecs_per_row;
  if (vec >= total) return;
  int64_t row = vec / vecs_per_row;
  int64_t col8 = (vec - row * vecs_per_row) * 8;
  int h = int(row / kv_len);
  int t = int(row - int64_t(h) * kv_len);
  int page = t >> 4;
  int t_in = t & 15;
  const float4* s4 = reinterpret_cast<const float4*>(src + (int64_t(h) * max_seq + t) * head_dim + col8);
  float4* d4 = reinterpret_cast<float4*>(
      dst + ((int64_t(page) * num_heads + h) * 16 + t_in) * head_dim + col8);
  *d4 = *s4;
}

extern "C" {

// Paged-KV context attention over grout's dense caches.
// repack = 0: zero-copy — the dense [H, max_seq, D] cache is presented as a
//   paged HND pool via strides (page stride 16*D, head stride max_seq*D)
//   and an identity block table.
// repack = 1: K/V are first repacked into packed HND pools (belt and
//   braces if the zero-copy strides trip a TMA constraint).
int grout_trtllm_paged_context_f16(
    void* out, void* q, void* k, void* v, int64_t q_len, int64_t kv_len,
    int64_t max_seq, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    float bmm1_scale, int enable_pdl, int repack, void* stream) {
  try {
    auto& s = scratch();
    constexpr size_t kWorkspaceBytes = size_t(192) * 1024 * 1024;
    if (!s.ensure(kWorkspaceBytes)) {
      g_last_error = "workspace allocation failed";
      return 1;
    }
    cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int64_t kPage = 16;
    int64_t pages = (kv_len + kPage - 1) / kPage;
    int64_t pool_pages = (max_seq + kPage - 1) / kPage;
    auto& ps = paged_scratch();
    if (!ps.ensure_table(pool_pages, cu_stream)) {
      g_last_error = "block table allocation failed";
      return 1;
    }

    void* k_ptr = k;
    void* v_ptr = v;
    int64_t stride_tok = head_dim;
    int64_t stride_head = max_seq * head_dim;
    int64_t stride_page = kPage * head_dim;
    if (repack != 0) {
      size_t bytes = size_t(pool_pages) * num_kv_heads * kPage * head_dim * sizeof(__half);
      if (!ps.ensure_pools(bytes)) {
        g_last_error = "paged pool allocation failed";
        return 1;
      }
      int64_t vecs = num_kv_heads * kv_len * (head_dim / 8);
      int threads = 256;
      int64_t blocks = (vecs + threads - 1) / threads;
      repack_dense_to_paged_f16<<<int(blocks), threads, 0, cu_stream>>>(
          reinterpret_cast<const __half*>(k), reinterpret_cast<__half*>(ps.k_pool),
          int(num_kv_heads), int(max_seq), int(head_dim), int(kv_len));
      repack_dense_to_paged_f16<<<int(blocks), threads, 0, cu_stream>>>(
          reinterpret_cast<const __half*>(v), reinterpret_cast<__half*>(ps.v_pool),
          int(num_kv_heads), int(max_seq), int(head_dim), int(kv_len));
      k_ptr = ps.k_pool;
      v_ptr = ps.v_pool;
      stride_tok = head_dim;
      stride_head = kPage * head_dim;
      stride_page = num_kv_heads * kPage * head_dim;
    }

    int32_t h_seq[1] = {int32_t(kv_len)};
    int32_t h_cq[2] = {0, int32_t(q_len)};
    int32_t h_ckv[2] = {0, int32_t(kv_len)};
    cudaMemcpyAsync(s.seq_lens, h_seq, sizeof(h_seq), cudaMemcpyHostToDevice, cu_stream);
    cudaMemcpyAsync(s.cum_q, h_cq, sizeof(h_cq), cudaMemcpyHostToDevice, cu_stream);
    cudaMemcpyAsync(s.cum_kv, h_ckv, sizeof(h_ckv), cudaMemcpyHostToDevice, cu_stream);

    flashinfer::trtllm_paged_attention_launcher(
        out, /*out_scale_factor=*/nullptr, q, k_ptr, v_ptr, s.workspace,
        ps.block_table, /*k_block_scales_ptr=*/nullptr, /*v_block_scales_ptr=*/nullptr,
        s.seq_lens, s.cum_q, s.cum_kv, /*attention_sinks=*/nullptr, /*lse=*/nullptr,
        DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16,
        flashinfer::TllmPagedAttentionMode::Context,
        /*batch_size=*/1, /*max_q_len=*/q_len, /*max_kv_len=*/max_seq,
        /*num_pages_in_mem_pool=*/pool_pages, num_q_heads, num_kv_heads,
        /*head_dim_qk=*/head_dim, /*head_dim_vo=*/head_dim, /*page_size=*/kPage,
        /*q_stride_tokens=*/num_q_heads * head_dim, /*q_stride_heads=*/head_dim,
        /*kv_stride_keys_values=*/stride_tok, /*kv_stride_heads=*/stride_head,
        /*kv_stride_batch=*/stride_page,
        /*max_num_blocks_per_seq=*/pages,
        /*bmm1_scale=*/double(bmm1_scale), /*bmm2_scale=*/1.0,
        /*bmm1_scale_log2_ptr=*/nullptr, /*bmm2_scale_ptr=*/nullptr,
        /*o_sf_scale=*/0.0, /*o_sf_vec_size=*/-1, /*o_sf_start_index=*/0,
        /*window_left=*/-1, /*sum_seq_q=*/q_len, /*sparse_mla_top_k=*/0,
        /*sliding_window_kv_pool=*/nullptr, /*sparse_mla_top_k_lens=*/nullptr,
        /*has_sliding_window_kv_pool=*/false,
        /*skip_softmax_threshold_scale_factor=*/0.0f, /*skips_softmax=*/false,
        /*uses_shared_paged_kv_idx=*/true, /*sm_count=*/s.sm_count,
        /*enable_pdl=*/enable_pdl != 0, /*workspace_size=*/int64_t(s.workspace_size),
        /*k_sf_stride_heads=*/0, /*k_sf_stride_batch=*/0,
        /*v_sf_stride_heads=*/0, /*v_sf_stride_batch=*/0,
        /*is_causal=*/true, /*lse_stride_tokens=*/0, /*lse_stride_heads=*/0, cu_stream);
    return 0;
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return 2;
  } catch (...) {
    g_last_error = "unknown exception in trtllm paged launcher";
    return 3;
  }
}

}  // extern "C"
