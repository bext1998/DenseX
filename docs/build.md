# Build llama.cpp locally

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the Code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

The following sections describe how to build with different backends and options.

* [CPU Build](#cpu-build)
* [CUDA](#cuda)
* [Arm® KleidiAI™](#arm-kleidiai)
* [Android](#android-1)
* [Notes about GPU-accelerated backends](#notes-about-gpu-accelerated-backends)

## CPU Build

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

**Notes**:

- For faster compilation, add the `-j` argument to run multiple jobs in parallel, or use a generator that does this automatically such as Ninja. For example, `cmake --build build --config Release -j 8` will run 8 jobs in parallel.
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- For debug builds, there are two cases:

    1. Single-config generators (e.g. default = `Unix Makefiles`; note that they just ignore the `--config` flag):

       ```bash
       cmake -B build -DCMAKE_BUILD_TYPE=Debug
       cmake --build build
       ```

    2. Multi-config generators (`-G` param set to Visual Studio, XCode...):

       ```bash
       cmake -B build -G "Xcode"
       cmake --build build --config Debug
       ```

    For more details and a list of supported generators, see the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).
- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:
  ```
  cmake -B build -DBUILD_SHARED_LIBS=OFF
  cmake --build build --config Release
  ```

- Building for Windows (x86, x64 and arm64) with MSVC or clang as compilers:
    - Install Visual Studio 2022, e.g. via the [Community Edition](https://visualstudio.microsoft.com/vs/community/). In the installer, select at least the following options (this also automatically installs the required additional tools like CMake,...):
    - Tab Workload: Desktop-development with C++
    - Tab Components (select quickly via search): C++-_CMake_ Tools for Windows, _Git_ for Windows, C++-_Clang_ Compiler for Windows, MS-Build Support for LLVM-Toolset (clang)
    - Please remember to always use a Developer Command Prompt / PowerShell for VS2022 for git, build, test
    - For Windows on ARM (arm64, WoA), build with:
      ```bash
      cmake --preset arm64-windows-llvm-release -D GGML_OPENMP_FETCH=ON
      cmake --build build-arm64-windows-llvm-release
      ```
      - Use `ARM64 Native Tools Command Prompt for VS 2022` if you are building on an ARM64 machine.
      - `GGML_OPENMP_FETCH` downloads the official LLVM OpenMP runtime and requires Clang, 7-Zip and network access during configuration. CMake selects the runtime from the target architecture, so this also works when cross-compiling for WoA from x64. The extracted header, import library, DLL and OpenMP license are placed under `build/_deps`. The build copies `libomp.dll` and `LICENSE-LLVM-OpenMP` to the runtime output directory and installs them together. Omit the option to use CMake's normal OpenMP detection, or pass `-D GGML_OPENMP=OFF` to disable OpenMP.
    - For building with ninja generator and clang compiler as default:
      - Set path:
        ```
        set LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\lib\x64\uwp;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64
        ```
      - Run:
        ```bash
        cmake --preset x64-windows-llvm-release
        cmake --build build-x64-windows-llvm-release
        ```
- If you want HTTPS/TLS features, you may install OpenSSL development libraries. If not installed, the project will build and run without SSL support.
  - **Debian / Ubuntu:** `sudo apt-get install libssl-dev`
  - **Fedora / RHEL / Rocky / Alma:** `sudo dnf install openssl-devel`
  - **Arch / Manjaro:** `sudo pacman -S openssl`


## CUDA

This provides GPU acceleration using an NVIDIA GPU. Make sure to have the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.

#### Download directly from NVIDIA
You may find the official downloads here: [NVIDIA developer site](https://developer.nvidia.com/cuda-downloads).


#### Compile and run inside a Fedora Toolbox Container
We also have a [guide](./backend/CUDA-FEDORA.md) for setting up CUDA toolkit in a Fedora [toolbox container](https://containertoolbx.org/).

**Recommended for:**
- ***Necessary*** for users of [Atomic Desktops for Fedora](https://fedoraproject.org/atomic-desktops/); such as: [Silverblue](https://fedoraproject.org/atomic-desktops/silverblue/) and [Kinoite](https://fedoraproject.org/atomic-desktops/kinoite/).
  - (there are no supported CUDA packages for these systems)
- ***Necessary*** for users that have a host that is not a: [Supported Nvidia CUDA Release Platform](https://developer.nvidia.com/cuda-downloads).
  - (for example, you may have [Fedora 42 Beta](https://fedoramagazine.org/announcing-fedora-linux-42-beta/) as your host operating system)
- ***Convenient*** For those running [Fedora Workstation](https://fedoraproject.org/workstation/) or [Fedora KDE Plasma Desktop](https://fedoraproject.org/spins/kde), and want to keep their host system clean.
- *Optionally* toolbox packages are available: [Arch Linux](https://archlinux.org/), [Red Hat Enterprise Linux >= 8.5](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), or [Ubuntu](https://ubuntu.com/download)


### Compilation

Make sure to read the notes about the CPU build for general instructions for e.g. speeding up the compilation.

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

To use a specific CCCL version instead of the one bundled with the installed CUDA Toolkit, add `-DGGML_CUDA_CCCL_VERSION=vMAJOR.MINOR.PATCH`. CUB DeviceTopK requires CCCL 3.4.3 or newer; older versions use the sort fallback.

Note that this also builds the CPU backend by default. On Windows on ARM, MSVC's
support for the ARM NEON intrinsics used by the CPU backend may be incomplete, so
a CUDA build produced entirely with MSVC might have a slower CPU backend. If CPU
performance matters, try following the split build used in our release workflow
([.github/workflows/release.yml](../.github/workflows/release.yml)): the CPU backend
is built with clang (`cmake/arm64-windows-llvm.cmake`) and the CUDA backend with MSVC
(`cmake/arm64-windows-msvc-cuda.cmake`), and the artifacts are merged afterwards.

### Non-Native Builds

By default llama.cpp will be built for the hardware that is connected to the system at that time.
For a build covering all CUDA GPUs, disable `GGML_NATIVE`:

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF
```

The resulting binary should run on all CUDA GPUs with optimal performance, though some just-in-time compilation may be required.

### Override Compute Capability Specifications

If `nvcc` cannot detect your gpu, you may get compile warnings such as:
 ```text
nvcc warning : Cannot find valid GPU for '-arch=native', default arch is used
```

One option is to do a non-native build as described above.
However, this will result in a large binary that takes a long time to compile.
Alternatively it is also possible to explicitly specify CUDA architectures.
This may also make sense for a non-native build, for that one should look at the logic in `ggml/src/ggml-cuda/CMakeLists.txt` as a starting point.

To override the default CUDA architectures:

#### 1. Take note of the `Compute Capability` of your NVIDIA devices: ["CUDA: Your GPU Compute > Capability"](https://developer.nvidia.com/cuda-gpus).

```text
GeForce RTX 4090      8.9
GeForce RTX 3080 Ti   8.6
GeForce RTX 3070      8.6
```

#### 2. Manually list each varying `Compute Capability` in the `CMAKE_CUDA_ARCHITECTURES` list.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

### Overriding the CUDA Version

If you have multiple CUDA installations on your system and want to compile llama.cpp for a specific one, e.g. for CUDA 11.7 installed under `/opt/cuda-11.7`:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda-11.7/bin/nvcc -DCMAKE_INSTALL_RPATH="/opt/cuda-11.7/lib64;\$ORIGIN" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

#### Fixing Compatibility Issues with Old CUDA and New glibc

If you try to use an old CUDA version (e.g. v11.7) with a new glibc version you can get errors like this:

```
/usr/include/bits/mathcalls.h(83): error: exception specification is
  incompatible with that of previous function "cospi"


  /opt/cuda-11.7/bin/../targets/x86_64-linux/include/crt/math_functions.h(5545):
  here
```

It seems the least bad solution is to patch the CUDA installation to declare the correct signatures.
Replace the following lines in `/path/to/your/cuda/installation/targets/x86_64-linux/include/crt/math_functions.h`:

```C++
// original lines
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);

// edited lines
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);
```

### Runtime CUDA environmental variables

You may set the [cuda environmental variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) at runtime.

```bash
# Use `CUDA_VISIBLE_DEVICES` to hide the first compute device.
CUDA_VISIBLE_DEVICES="-0" ./build/bin/llama-server --model /srv/models/llama.gguf
```

#### CUDA_SCALE_LAUNCH_QUEUES

The environment variable [`CUDA_SCALE_LAUNCH_QUEUES`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues) controls the size of CUDA's command buffer, which determines how many GPU operations can be queued before the CPU must wait for the GPU to catch up. A larger buffer reduces CPU-side stalls and allows more work to be queued on a GPU.

Consider setting `CUDA_SCALE_LAUNCH_QUEUES=4x`, which increases the CUDA command buffer to 4 times its default size. This optimization is particularly beneficial for **Multi-GPU setups with pipeline parallelism**, where it significantly improves prompt processing throughput by allowing more operations to be enqueued across GPUs.

#### GGML_CUDA_CUBLAS_COMPUTE_TYPE

Override default, speed-optimized compute types for cuBLAS matrix multiplications.
Legal values: `auto`, `f16`, `fp16`, `bf16`, `f32`, `fp32`.

#### GGML_CUDA_MMQ_PREC

Override the activation precision that the model requests for NVFP4 and MXFP4 matrix multiplications.
Currently supported values: `auto`, `q8`, `q4`.

NVFP4 and MXFP4 layers marked as W4A16 request 8-bit activations, so on Blackwell those layers run through the W4A8 path instead of the native W4A4 path. Set `q4` to keep the native W4A4 path for faster prompt processing at the cost of accuracy, or `q8` to use the W4A8 path for every layer, `auto` uses per-tensor prec metadata (this is the same behavior as when the environment variable is not set).

### Unified Memory

The environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` can be used to enable unified memory in Linux. This allows swapping to system RAM instead of crashing when the GPU VRAM is exhausted. In Windows this setting is available in the NVIDIA control panel as `System Memory Fallback`.

### Peer Access

The environment variable `GGML_CUDA_P2P` can be set to enable peer-to-peer access between multiple GPUs, allowing them to transfer data directly rather than to go through system memory.
Requires driver support (usually restricted to workstation/datacenter GPUs).
May cause crashes or corrupted outputs for some motherboards and BIOS settings (e.g. IOMMU).

### Performance Tuning

The following compilation options are also available to tweak performance:

| Option                        | Legal values           | Default | Description                                                                                                                                                                                                                                                                                                                                                                      |
|-------------------------------|------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GGML_CUDA_FORCE_MMQ           | Boolean                | false   | Force the use of custom matrix multiplication kernels for quantized models instead of FP16 cuBLAS even if there is no int8 tensor core implementation available (affects V100, CDNA and RDNA3+). MMQ kernels are enabled by default on GPUs with int8 tensor core support. With MMQ force enabled, speed for large batch sizes will be worse but VRAM consumption will be lower. |
| GGML_CUDA_FORCE_CUBLAS        | Boolean                | false   | Force the use of FP16 cuBLAS instead of custom matrix multiplication kernels for quantized models. There may be issues with numerical overflows (except for V100, CDNA and RDNA4 which use FP32 compute type by default) and memory use will be higher. Prompt processing may become faster on recent datacenter GPUs (the custom kernels were tuned primarily for RTX 3000/4000).   |
| GGML_CUDA_FA_QUANTS           | `all` or `type_K-type_V` list | q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16 | Select which K/V type combinations to compile the FlashAttention CUDA kernels for. `all` compiles every combination, but compilation takes much longer. Otherwise a `;`-separated list of `type_K-type_V` pairs; f16-f16 is always compiled. Combinations that were not compiled fall back to f16-f16 kernel with a warning. Legal types: f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0. |
| GGML_CUDA_FA_ALL_QUANTS       | Boolean                | false   | Deprecated alias for `GGML_CUDA_FA_QUANTS=all`.                                                                                                                                                                                                                                                                                                                               |

## Arm® KleidiAI™
KleidiAI provides optimized Arm CPU microkernels used by the ggml CPU backend. Enabling it at build time makes those kernels available; it does not force every operation to use KleidiAI. At runtime, llama.cpp selects the best compatible CPU kernel from the detected CPU features, tensor type, operation shape, and active backend priority.

Supported targets:

| Platform | Supported ABI / architecture | Notes |
| --- | --- | --- |
| Linux | AArch64 / arm64 | Runtime CPU feature detection is automatic. |
| Android | `arm64-v8a` | Use the Android NDK command below for a portable build. |
| Apple | arm64 | Runtime CPU feature detection is automatic. Non-streaming SVE vector length is treated as unavailable. |
| Windows | arm64 | Runtime CPU feature detection is automatic. SMCU count is treated as unknown until a detection path is verified. |

`GGML_CPU_KLEIDIAI=ON` is valid only for AArch64/arm64 builds. Do not enable it for x86, 32-bit Arm, or Android ABIs other than `arm64-v8a`.

### Native AArch64/arm64 build

From the llama.cpp source directory:

```bash
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```

### Android arm64-v8a NDK build

Set `ANDROID_NDK` to the Android NDK root, then run the following from the llama.cpp source directory. This command configures a portable Android `arm64-v8a` build with KleidiAI enabled and avoids Android dependencies that are not part of the NDK stable native API set.

```bash
cmake -S . -B build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-android --config Release --parallel
cmake --install build-android --prefix {install-dir} --config Release
```

Important Android options:

- `GGML_CPU_KLEIDIAI=ON` enables KleidiAI for Android `arm64-v8a`.
- `GGML_NATIVE=OFF` is required for cross-compilation because the build host CPU is not the Android target CPU.
- `GGML_OPENMP=OFF` avoids adding an OpenMP runtime dependency to this NDK command-line build.
- `GGML_LLAMAFILE=OFF` avoids the llamafile backend, which is not supported on Android.
- `LLAMA_OPENSSL=OFF` avoids depending on OpenSSL, which is not part of the Android NDK stable native API set.

The Android Studio project under `examples/llama.android` enables KleidiAI automatically for `arm64-v8a`. For Android command-line CMake builds on `arm64-v8a`, pass `-DGGML_CPU_KLEIDIAI=ON` explicitly.

Global -march flags such as `-march=armv8.7a` flag are not required for a portable Android `arm64-v8a` build. Global `-march` flags raise the baseline instruction set for generic code. No manual architecture-specific source selection is required; llama.cpp selects compatible KleidiAI kernels at runtime. The KleidiAI libraries internal CMake handles the -march flags for each particular kernel.

### Verifying the build

Run an installed or in-tree binary:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "What is a car?"
```

If KleidiAI is enabled, the output contains a line similar to:

```
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```

This confirms that the model has tensors allocated through the KleidiAI CPU buffer. It does not prove that every operation, or any specific SME-family operation, used a KleidiAI microkernel. Runtime CPU features, tensor type, operation shape, and backend priority still control dispatch.

Depending on the build target, another backend may have higher priority than the CPU backend. To force CPU execution for a run, disable higher priority backends at build time, for example `-DGGML_METAL=OFF`, or use a runtime device option such as `--device none` where supported.

### Runtime dispatch

KleidiAI microkernels use Arm CPU features such as dotprod, i8mm, SVE, and SME/SME2. Build-time configuration makes the kernels available. Runtime dispatch selects a compatible kernel for the detected CPU and operation. Older or lower-feature CPUs fall back automatically to compatible kernels.

KleidiAI accelerates selected `GGML_OP_MUL_MAT` paths for F32 and common quantized formats. Exact coverage depends on the bundled KleidiAI version and the llama.cpp runtime selector, so unsupported tensor types, unsupported operation shapes, or higher priority backends may bypass KleidiAI even when the CPU supports the required Arm feature. This is also why a model may not use SME-family kernels on SME-capable hardware.

The current llama.cpp KleidiAI SVE selector only enables SVE kernels when the runtime SVE vector length is known to be QK8_0 bytes, currently 32 bytes. Linux and Android query this at runtime. Apple reports SVE capability separately from userspace non-streaming SVE availability, so llama.cpp treats the SVE vector length as unknown there. Windows exposes SVE feature presence but not the runtime SVE vector length used by this selector, so that value is also treated as unknown. Windows arm64 also treats SMCU count as unknown until a detection mechanism is verified.

The set of available SME-family kernels depends on the bundled KleidiAI version and the detected CPU capabilities. Production configuration does not require any KleidiAI runtime environment variables.

### Diagnostics and debug overrides

KleidiAI runtime environment variables are diagnostics/debug overrides, not production configuration. Leave them unset for normal use.

`GGML_KLEIDIAI_SME` controls SME-family kernel selection and overrides the maximum number of threads assigned to selected quantized SME-family kernels:

- Not set: use automatic runtime detection.
- `0`: disable SME-family kernels.
- `<n> > 0`: enable compatible SME-family kernels and allow up to `<n>` threads for quantized SME-family kernels.

On Windows arm64, use `GGML_KLEIDIAI_SME=<n>` as the temporary diagnostics/debug override for SME thread-cap calibration until automatic SMCU count detection is verified.

If the CPU does not support the required SME-family capability for a bundled kernel, that kernel is disabled regardless of the environment variable.


## Android

To read documentation for how to build on Android, [click here](./android.md)


## IBM Z & LinuxONE

To read documentation for how to build on IBM Z & LinuxONE, [click here](./build-s390x.md)

## Notes about GPU-accelerated backends

The GPU may still be used to accelerate some parts of the computation even when using the `-ngl 0` option. You can fully disable GPU acceleration by using `--device none`.

In most cases, it is possible to build and use multiple backends at the same time. For example, you can build llama.cpp with both CUDA and Vulkan support by using the `-DGGML_CUDA=ON -DGGML_VULKAN=ON` options with CMake. At runtime, you can specify which backend devices to use with the `--device` option. To see a list of available devices, use the `--list-devices` option.

Backends can be built as dynamic libraries that can be loaded dynamically at runtime. This allows you to use the same llama.cpp binary on different machines with different GPUs. To enable this feature, use the `GGML_BACKEND_DL` option when building.
