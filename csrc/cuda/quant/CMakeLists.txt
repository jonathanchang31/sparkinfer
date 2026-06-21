file(GLOB_RECURSE SRC "*.cu" "*.cpp")
add_library(bw_quant STATIC ${SRC})
target_include_directories(bw_quant PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(bw_quant PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)
