# blake3
SYCL accelerated BLAKE3 Hash Implementation

## Motivation

In recent times I've been exploring data parallel programming domain using SYCL, which is a heterogeneous accelerator programming API. Few weeks back I completed writing Zk-STARK friendly [Rescue Prime Hash using SYCL](https://github.com/itzmeanjan/ff-gpu/), then I decided to take a look at BLAKE3, because blake3's algorithmic construction naturally lends itself for heavy parallelism. Compared to Rescue Prime Hash, BLAKE3 should be able to much better harness accelerator's compute capability when input size is relatively large ( say >= 1MB ).

SYCL -backed Rescue Prime implementation shines when there are lots of (short) indepedent inputs and multiple Rescue Prime Hashes can be executed independently on each of them, because Rescue Prime can be vectorized but doesn't provide with good scope of (multi-threaded/ OpenCL work-item based) parallelism inherently.

On the other hand SYCL implementation of BLAKE3 performs good when (single) input size is >= 1MB, then each 1KB chunk of input can be compressed parallelly --- very good fit for data parallel acceleration. After that BLAKE3 is simply Binary Merkle Tree construction, which itself is highly parallelizable, _though multi-phase kernel enqueue required (increasing host-device interaction) due to hierarchical structure of Binary Merkle Tree, which results into data dependence_.

In following implementation I heavily use SYCL2020's USM, which allows me to work with much familiar pointer arithmetics. I also use SYCL's vector intrinsics ( i.e. 4 -element array of type `sycl::uint4` ) for representing/ operating on hash state of BLAKE3. Another way to accelerate BLAKE3 (as proposed in specification) is compressing multiple chunks in parallel by distributing hash state of those participating chunks across 16 vectors, each with N -lanes, where N = # -of chunks being compressed together. N can generally be {2, 4, 8, 16}. I've implemented that scheme under namespace `blake3::v2::*`, while simpler variant is placed under namespace `blake3::v1::*`. 

**I strongly suggest you go through (hyperlinked below) BLAKE3 specification's section 5.3 for understanding where I got this idea from.**

> I followed BLAKE3 [specification](https://github.com/BLAKE3-team/BLAKE3-specs/blob/ac78a717924dd9e6f16f547baa916c6f71470b1a/blake3.pdf) and used Rust reference [implementation](https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs) as my guide while writing SYCL implementation.

> **Note,** at this moment to keep Merkle Tree construction both easy and simple, this SYCL implementation can only generate BLAKE3 digest when input has power of 2 -many chunks, given each chunk of size 1KB. That means minimum input size should be 2KB, after that it can be increased as 4KB, 8KB ....

> If input size is not >= 1MB, you probably don't want to use this implementation, because submitting job ( read enqueuing kernels ) to accelerator is not cheap and all those (required) ceremonies might defeat the whole purpose and essence of acceleration.

## Prerequisites

- Ensure you've Intel SYCL/ DPC++ compiler toolchain. See [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) for downloading precompiled binaries.
- If you happen to be interested in running on Nvidia GPU; you have to compile Intel's open-source llvm-based SYCL implementation from source; see [here](https://intel.github.io/llvm-docs/GetStartedGuide.html#prerequisites).
- For running test cases, which uses Rust Blake3 [implementation](https://docs.rs/blake3/1.2.0/blake3) for assertion, you'll need to have Rust `cargo` toolchain installed; get that [here](https://rustup.rs/)
- I'm on

```bash
$ lsb_release -d
Description:    Ubuntu 20.04.3 LTS
```

- Using Intel's SYCL/ DPC++ compiler version

```bash
$ dpcpp --version
Intel(R) oneAPI DPC++/C++ Compiler 2022.0.0 (2022.0.0.20211123)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/intel/oneapi/compiler/2022.0.1/linux/bin-llvm
```

- For CUDA backend on Nvidia Tesla V100 GPU, I used Intel's `clang++` version

```bash
$ clang++ --version
clang version 14.0.0 (https://github.com/intel/llvm dc9bd3fafdeacd28528eb4b1fef3ad9b76ef3b92)
Target: x86_64-unknown-linux-gnu
Thread model: posix
```

- I'm on `rustc` version

```bash
$ rustc --version
rustc 1.59.0-nightly (efec54529 2021-12-04)
```

- You'll also need `make` utility for running test/ benchmark etc.
- For formatting `C++` source consider using `clang-format` tool

```bash
make format
```

## Usage

This is a header only library; so clone this repo and include [blake3.hpp](./include/blake3.hpp) in your SYCL project.

```cpp
// Find full example https://github.com/itzmeanjan/blake3/blob/1de036a/test/src/main.cpp

#include "blake3.hpp"
#include <iostream>

int main() {
    sycl::device d{ sycl::default_selector{} }; // choose sycl device
    sycl::queue q{ d };                         // make sycl queue

    // @note
    // At this moment only power of 2 -many chunks are supported
    // meaning input size will be `chunk_count * chunk_size` -bytes
    //
    // chunk_size   = 1024 bytes
    // chunk_count  = 2^i, where i = {1, 2, ...}

    // allocate input/ output memory
    // fill input with data
    // see https://github.com/itzmeanjan/blake3/blob/095e80f/test/src/main.cpp#L15-L37

    // invoke hasher; last argument denotes execution doesn't need to be timed
    blake3::v1::hash(q, in_d, i_size, chunk_count, wg_size, out_d, nullptr); // either

    blake3::v2::hash(q, in_d, i_size, chunk_count, wg_size, out_d, nullptr); // or
    // see https://github.com/itzmeanjan/blake3/blob/095e80f/test/src/main.cpp#L40-L43

    // deallocate heap memory

    return 0;
}
```

## Test

For executing accompanying test cases run

```bash
BLAKE3_SIMD_LANES=2 make; make clean
BLAKE3_SIMD_LANES=4 make; make clean
BLAKE3_SIMD_LANES=8 make; make clean
BLAKE3_SIMD_LANES=16 make; make clean
```

which prepares random input of 1MB; then applies BLAKE3 using [Rust implementation](https://docs.rs/blake3/1.2.0/blake3) and both of my [SYCL implementations of BLAKE3](https://github.com/itzmeanjan/blake3/blob/b459e95539fbc203f48bccbccd356ff21c1a59b6/include/blake3.hpp). Finally both of these 32 -bytes digests are asserted. ✅

Implementation | Comment
--- | ---
`blake3::v1::hash(...)` | Each SYCL work-item compresses one and only one chunk
`blake3::v2::hash(...)` | Each SYCL work-item can compress either 2/ 4/ 8/ 16 contiguous chunks; selectable using `BLAKE3_SIMD_LANES`

## Dockerised Testing

For running test cases inside Docker container (without installing any dependencies on your host, expect `docker` itself) consider using Dockerfile provided with.

Build image

```bash
docker build -t blake3-test . # can be time consuming
```

Then run test cases inside container

```bash
docker run blake3-test
```

## Benchmark

Following benchmark results denote what was 

- kernel execution time
- time required to transfer input bytes to device
- time needed to transfer 32 -bytes digest back to host

when computing BLAKE3 hash ( v1 & v2 ) using SYCL implementation and input was of given size on first column. Input is generated on host; then explicitly transferred to accelerator because I'm using `sycl::malloc_host` and `sycl::malloc_device` for heap allocation; finally computed BLAKE3 digest ( i.e. 32 -bytes ) is transferred back to host. *None of these data transfer costs are included in kernel execution time*. For benchmarking purposes, I enable profiling in SYCL queue and sum of all differences between kernel enqueue event's start and end times are taken. I've also used a SYCL work-group size of 32 for each of these executions rounds; total of 8 rounds are executed for each row before taking average of obtained kernel execution time/ host <-> device data transfer time.

- [On Nvidia GPU](./results/nvidia_gpu.md)
- [On Intel GPU](./results/intel_gpu.md)
- [On Intel CPU](./results/intel_cpu.md)
