cmake_minimum_required(VERSION 3.24)
project(delauney LANGUAGES CXX)

option(WITH_CUDA "Build CUDA extension" OFF)
option(BUILD_PYTHON_MODULE "Build the pybind11 extension module" ON)
option(BUILD_PROFILING "Build the standalone C++ profiling executable" OFF)
option(WITH_NVTX "Emit NVTX ranges from the profiling executable" ON)

if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)

    set(DELAUNEY_CUDA_SOURCES
        src/cuda/voronoi.cu
        src/cuda/triangulation.cu
        src/cuda/incremental.cu
        src/cuda/shared_utils.cuh
        src/cuda/shared_utils.cu
    )

    # nvcc flags shared by every target that compiles the kernels.
    # CCCL (Thrust) refuses to build against MSVC's traditional preprocessor,
    # so forward /Zc:preprocessor to the host compiler there.
    set(DELAUNEY_CUDA_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(MSVC)
        list(APPEND DELAUNEY_CUDA_FLAGS
             $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()

    # -- pybind11 extension module (the wheel) ------------------------------
    if(BUILD_PYTHON_MODULE)
        find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
        find_package(pybind11 CONFIG REQUIRED)

        pybind11_add_module(_delauney_cuda
            src/cuda/bindings.cpp
            ${DELAUNEY_CUDA_SOURCES}
        )

        target_include_directories(_delauney_cuda PRIVATE src/cuda)
        target_compile_options(_delauney_cuda PRIVATE ${DELAUNEY_CUDA_FLAGS})

        install(TARGETS _delauney_cuda DESTINATION delauney)
    endif()

    # -- standalone profiling executable (no Python, no pybind11) -----------
    if(BUILD_PROFILING)
        add_executable(delauney_profiling
            src/cuda/profiling.cpp
            src/cuda/native_api.cpp
            ${DELAUNEY_CUDA_SOURCES}
        )

        target_include_directories(delauney_profiling PRIVATE src/cuda)
        target_compile_features(delauney_profiling PRIVATE cxx_std_17)
        target_compile_options(delauney_profiling PRIVATE ${DELAUNEY_CUDA_FLAGS})
        target_link_libraries(delauney_profiling PRIVATE CUDA::cudart)

        # NVTX ranges for the Nsight Systems timeline.  Only this target defines
        # DELAUNEY_WITH_NVTX, so nvtx_range.h is a no-op in the wheel build.
        if(WITH_NVTX)
            target_compile_definitions(delauney_profiling PRIVATE DELAUNEY_WITH_NVTX)
            if(WIN32)
                # nvToolsExt.h drags in windows.h
                target_compile_definitions(delauney_profiling PRIVATE
                                           NOMINMAX WIN32_LEAN_AND_MEAN)
            endif()
            if(TARGET CUDA::nvtx3)
                target_link_libraries(delauney_profiling PRIVATE CUDA::nvtx3)
            endif()
        endif()

        # Debug symbols in every configuration -- without a PDB, Nsight's CPU
        # samples cannot be resolved to function names on Windows.
        if(MSVC)
            target_compile_options(delauney_profiling PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/Zi>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
                $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            )
            target_link_options(delauney_profiling PRIVATE /DEBUG /OPT:REF /OPT:ICF)
        else()
            target_compile_options(delauney_profiling PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-g>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-g>
                $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            )
        endif()
    endif()
endif()
