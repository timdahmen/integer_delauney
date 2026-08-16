cmake_minimum_required(VERSION 3.24)
project(delauney LANGUAGES CXX)

# Off here so a bare `cmake` configure works without a CUDA toolkit;
# pyproject.toml passes -DWITH_CUDA=ON, so pip installs do build it.
option(WITH_CUDA "Build CUDA extension" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)

    pybind11_add_module(_delauney_cuda
        src/cuda/bindings.cpp
        src/cuda/voronoi.cu
        src/cuda/triangulation.cu
        src/cuda/delaunay.cu
    )

    target_include_directories(_delauney_cuda PRIVATE src/cuda)
    target_compile_options(_delauney_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )

    install(TARGETS _delauney_cuda DESTINATION delauney)
endif()
