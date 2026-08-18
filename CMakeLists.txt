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
    # bindings.cpp is a CXX source calling cudaMemcpy (to_host()), so it needs
    # the toolkit include dir explicitly -- CMake's CUDA language support only
    # adds it for the .cu sources compiled in the CUDA language.
    target_link_libraries(_delauney_cuda PRIVATE CUDA::cudart)
    target_compile_options(_delauney_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )

    # Symbols in the optimised build. /Zi emits a PDB and changes no code, so
    # this is not RelWithDebInfo -- that would drop /Ob2 to /Ob1 and quietly
    # reduce inlining, which is a different binary from the one we ship. A
    # sampling profiler can attribute nothing inside this module without a PDB,
    # so without it the only way to find host cost is to guess where to put a
    # timer. /DEBUG turns off two link-time optimisations by default; /OPT:REF
    # and /OPT:ICF put them back.
    if(MSVC)
        target_compile_options(_delauney_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zi>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
        )
        target_link_options(_delauney_cuda PRIVATE /DEBUG /OPT:REF /OPT:ICF)
    endif()

    install(TARGETS _delauney_cuda DESTINATION delauney)
    if(MSVC)
        install(FILES $<TARGET_PDB_FILE:_delauney_cuda>
                DESTINATION delauney OPTIONAL)
    endif()
endif()
