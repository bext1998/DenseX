# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # DenseX (issue #86 P3.2): these five tools are not part of the default `all` build.
    # They stay in the tree and can still be built by name (cmake --build . --target <name>),
    # and targets outside this list that link their *-impl libraries (app) still build them
    # as dependencies. The tools DenseX ships (server/cli/ui/mtmd/perplexity/quantize/
    # llama-bench/imatrix/tokenize/gguf-split) are untouched.
    add_subdirectory(batched-bench EXCLUDE_FROM_ALL)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion EXCLUDE_FROM_ALL)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(mtmd)
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(export-lora EXCLUDE_FROM_ALL)
    endif()
    add_subdirectory(fit-params EXCLUDE_FROM_ALL)
    add_subdirectory(results EXCLUDE_FROM_ALL)
endif()
