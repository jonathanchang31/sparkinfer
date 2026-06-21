file(GLOB_RECURSE ATTN_SRC "*.cu" "*.cpp")
add_library(bw_attn STATIC ${ATTN_SRC})
target_include_directories(bw_attn PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(bw_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math -lineinfo
        -DCUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}
    >
)
