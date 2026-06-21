#pragma once
#include <cuda_runtime.h>

namespace blackwell { namespace kernels {

// Grouped GEMM for MoE expert dispatch.
// Each token is routed to top-k experts; this kernel batches the expert GEMMs
// without padding to avoid wasted compute.
//
// input:       [num_tokens, hidden_dim]         (fp16/bf16)
// weights:     [num_experts, hidden_dim, ffn_dim]
// expert_ids:  [num_tokens, top_k]              (int32)
// expert_w:    [num_tokens, top_k]              (float, routing weights)
// output:      [num_tokens, hidden_dim]
void launch_moe_grouped_gemm(
    const void* input,
    const void* weights,
    const int*  expert_ids,
    const float* expert_weights,
    void*       output,
    int num_tokens, int top_k, int num_experts,
    int hidden_dim, int ffn_dim,
    cudaStream_t stream = nullptr
);

// Token-to-expert router: softmax + top-k selection.
// logits:      [num_tokens, num_experts]  (float)
// expert_ids:  [num_tokens, top_k]        (int32, output)
// expert_w:    [num_tokens, top_k]        (float, output, normalized weights)
void launch_moe_router(
    const float* logits,
    int* expert_ids, float* expert_weights,
    int num_tokens, int num_experts, int top_k,
    cudaStream_t stream = nullptr
);

}} // namespace blackwell::kernels
