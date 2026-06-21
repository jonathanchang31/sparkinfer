#pragma once
#include <cuda_runtime.h>

namespace blackwell { namespace kernels {

// Flash decode: single-token decode attention over paged KV cache.
// q:           [num_seqs, num_heads, head_dim]  (fp16/bf16)
// k_pool:      [num_blocks, block_size, num_kv_heads, head_dim]
// v_pool:      same shape as k_pool
// block_table: [num_seqs, max_blocks_per_seq]  (int32)
// seq_lens:    [num_seqs]  (int32)
// out:         [num_seqs, num_heads, head_dim]
void launch_flash_decode(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream = nullptr
);

// Flash prefill: full causal attention for prompt processing.
// q/k/v:  [batch, seqlen, num_heads, head_dim]
// out:    same shape as q
void launch_flash_prefill(
    const void* q, const void* k, const void* v,
    void* out,
    int batch, int seqlen_q, int seqlen_kv,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, bool causal, cudaStream_t stream = nullptr
);

}} // namespace blackwell::kernels
