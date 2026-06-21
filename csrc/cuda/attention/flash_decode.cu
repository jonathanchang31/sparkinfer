#include "blackwell/kernels/attention.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Flash decode attention kernel.
// One CTA per query head; iterates over KV blocks in shared memory tiles.
// Target: minimize HBM round-trips for decode (batch_size=1..256, seqlen_kv=1..32k).
//
// References:
//   FlashAttention-2 (Dao et al. 2023)
//   FlashInfer (Ye et al. 2024)
//   PagedAttention (Kwon et al. 2023)

namespace blackwell {
namespace kernels {

template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_decode_kernel(
    const scalar_t* __restrict__ q,     // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ k_pool,// [num_blocks, block_size, num_kv_heads, head_dim]
    const scalar_t* __restrict__ v_pool,
    const int*      __restrict__ block_table, // [num_seqs, max_blocks]
    const int*      __restrict__ seq_lens,    // [num_seqs]
    scalar_t*       __restrict__ out,   // [num_seqs, num_heads, head_dim]
    const float scale,
    const int num_kv_heads,
    const int max_blocks_per_seq
) {
    const int seq_id  = blockIdx.x;
    const int head_id = blockIdx.y;
    const int kv_head = head_id / (gridDim.y / num_kv_heads); // GQA mapping

    const int seq_len = seq_lens[seq_id];
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    extern __shared__ float smem[];
    float* s_k   = smem;                          // [BLOCK_SIZE, HEAD_DIM]
    float* s_v   = s_k + BLOCK_SIZE * HEAD_DIM;  // [BLOCK_SIZE, HEAD_DIM]

    const scalar_t* q_ptr = q + (seq_id * gridDim.y + head_id) * HEAD_DIM;

    float q_reg[HEAD_DIM / 32]; // thread-local Q fragment (assumes HEAD_DIM <= 32*warpSize)
    #pragma unroll
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
        q_reg[i / blockDim.x] = __half2float(q_ptr[i]);
    }

    float m = -1e9f, l = 0.f;
    float acc[HEAD_DIM / 32] = {};

    for (int blk = 0; blk < num_blocks; blk++) {
        const int phys_block = block_table[seq_id * max_blocks_per_seq + blk];
        const scalar_t* k_blk = k_pool + phys_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM
                                       + kv_head * HEAD_DIM;
        const scalar_t* v_blk = v_pool + phys_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM
                                       + kv_head * HEAD_DIM;

        // Load KV block into shared memory
        for (int t = threadIdx.x; t < BLOCK_SIZE * HEAD_DIM; t += blockDim.x) {
            s_k[t] = __half2float(k_blk[t]);
            s_v[t] = __half2float(v_blk[t]);
        }
        __syncthreads();

        const int valid = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);
        for (int t = 0; t < valid; t++) {
            float dot = 0.f;
            #pragma unroll
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                dot += q_reg[d / blockDim.x] * s_k[t * HEAD_DIM + d];
            }
            // Warp reduce dot
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, mask);

            dot *= scale;
            const float m_new = fmaxf(m, dot);
            const float exp_m  = __expf(m - m_new);
            const float exp_qk = __expf(dot - m_new);
            l = l * exp_m + exp_qk;
            m = m_new;

            #pragma unroll
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
                acc[d / blockDim.x] = acc[d / blockDim.x] * exp_m + exp_qk * s_v[t * HEAD_DIM + d];
        }
        __syncthreads();
    }

    scalar_t* out_ptr = out + (seq_id * gridDim.y + head_id) * HEAD_DIM;
    #pragma unroll
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
        out_ptr[d] = __float2half(acc[d / blockDim.x] / l);
}

void launch_flash_decode(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(128);
    size_t smem = 2 * block_size * head_dim * sizeof(float);

    // HEAD_DIM and BLOCK_SIZE specializations for register pressure control
    flash_decode_kernel<__half, 128, 16><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k_pool),
        reinterpret_cast<const __half*>(v_pool),
        block_table, seq_lens,
        reinterpret_cast<__half*>(out),
        scale, num_kv_heads, max_blocks_per_seq
    );
}

} // namespace kernels
} // namespace blackwell
