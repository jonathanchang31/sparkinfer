file(GLOB_RECURSE SRC "*.cu" "*.cpp")
add_library(bw_fused STATIC ${SRC})
target_include_directories(bw_fused PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(bw_fused PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)
