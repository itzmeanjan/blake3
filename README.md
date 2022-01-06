# blake3
SYCL accelerated BLAKE3 Hash Implementation

## Motivation

In recent times I've been exploring data parallel programming domain using SYCL, which is a heterogeneous accelerator programming API. Few weeks back I completed writing Zk-STARK friendly [Rescue Prime Hash using SYCL](https://github.com/itzmeanjan/ff-gpu/), then I decided to take a look at BLAKE3, because blake3's algorithmic construction naturally lends itself for heavy parallelization. Compared to Rescue Prime Hash, BLAKE3 should be able to much better harness accelerator's compute capability when input size is relatively large ( say >= 1MB ).

SYCL -backed Rescue Prime implementation shines when there are lots of (short) indepedent inputs and multiple Rescue Prime Hashes can be executed independently on each of them, because Rescue Prime can be vectorized but doesn't provide with good scope of parallelism.

On the other hand SYCL implementation of BLAKE3 performs good when (single) input size is >= 1MB, then each 1KB chunk of input can be compressed parallelly --- very good fit for data parallel acceleration. After that BLAKE3 is simply Binary Merkle Tree construction, which itself is highly parallelizable, _though multi-phase kernel enqueue required due to hierarchical structure of Binary Merkle Tree_.

In following implementation I heavily use SYCL2020's USM, which allows me to work with much familiar pointer arithmetics. I also use SYCL's vector intrinsics ( i.e. 4 -element array of type `sycl::uint4` ) for representing/ operating on hash state of BLAKE3.

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
// Find full example https://github.com/itzmeanjan/blake3/blob/095e80f/test/src/main.cpp

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
    blake3::hash(q, in_d, i_size, chunk_count, wg_size, out_d, nullptr);
    // see https://github.com/itzmeanjan/blake3/blob/095e80f/test/src/main.cpp#L40-L43

    // deallocate heap memory

    return 0;
}
```

## Test

For executing accompanying test cases run

```bash
make
```

which prepares random input of 1MB; then applies BLAKE3 using [Rust implementation](https://docs.rs/blake3/1.2.0/blake3) and my [SYCL implementation](https://github.com/itzmeanjan/blake3/blob/095e80ff25436e18d0f80936eb20178d7a852a1f/include/blake3.hpp). Finally both of these 32 -bytes digests are asserted. ✅

## Benchmark

Following tables denote what was only kernel execution time ( on accelerator ) when computing BLAKE3 hash using SYCL implementation and input was of given size. Input is prepared on host; then explicitly transferred to accelerator because I'm using `sycl::malloc_host` and `sycl::malloc_device` for heap allocation; finally computed BLAKE3 digest ( i.e. 32 -bytes ) is transferred back to host. Though none of these data transfer cost are included in following numbers presented. For benchmarking purposes, I enable profiling in SYCL queue and sum of all differences between kernel enqueue event's start and end times are taken.

## On GPU

### Nvidia

Built using `make cuda`

```bash
running on Tesla V100-SXM2-16GB

Benchmarking BLAKE3 SYCL implementation

              input size		  execution time
                   1 MB		         184.767000 us
                   2 MB		         189.505000 us
                   4 MB		         209.632000 us
                   8 MB		         280.928000 us
                  16 MB		         361.565000 us
                  32 MB		         542.240000 us
                  64 MB		         915.430000 us
                 128 MB		           1.858963 ms
                 256 MB		           3.358396 ms
                 512 MB		           6.582276 ms
                1024 MB		          12.488282 ms
```

### Intel

Built using `make aot_gpu`

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Benchmarking BLAKE3 SYCL implementation

              input size		  execution time
                   1 MB		         172.328000 us
                   2 MB		         206.024000 us
                   4 MB		         278.720000 us
                   8 MB		         422.032000 us
                  16 MB		         733.096000 us
                  32 MB		           1.329120 ms
                  64 MB		           2.500264 ms
                 128 MB		           4.880200 ms
                 256 MB		           9.614176 ms
                 512 MB		          19.072976 ms
                1024 MB		          38.018864 ms
```

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

Benchmarking BLAKE3 SYCL implementation

              input size		  execution time
                   1 MB		         465.796000 us
                   2 MB		         451.188000 us
                   4 MB		         871.998000 us
                   8 MB		           1.414154 ms
                  16 MB		           2.523034 ms
                  32 MB		           4.653146 ms
                  64 MB		           8.836512 ms
                 128 MB		          17.294876 ms
                 256 MB		          34.397856 ms
                 512 MB		         100.988756 ms
                1024 MB		         152.490422 ms
```

## On CPU

### Intel

Built using `make aot_cpu`

```bash
# [ CPU(s): 4; used avx2 ]
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Benchmarking BLAKE3 SYCL implementation

              input size                  execution time
                   1 MB                    1.694091 ms
                   2 MB                    2.418789 ms
                   4 MB                    4.609086 ms
                   8 MB                    8.843390 ms
                  16 MB                   17.562423 ms
                  32 MB                   34.939209 ms
                  64 MB                   69.431233 ms
                 128 MB                  139.722805 ms
                 256 MB                  286.154562 ms
                 512 MB                  556.171082 ms
                1024 MB                     1.104227 s
```

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Benchmarking BLAKE3 SYCL implementation

              input size		  execution time
                   1 MB		         636.858000 us
                   2 MB		         557.543000 us
                   4 MB		         923.242000 us
                   8 MB		           1.819586 ms
                  16 MB		           3.668861 ms
                  32 MB		           6.665910 ms
                  64 MB		           9.772532 ms
                 128 MB		          14.093261 ms
                 256 MB		          22.050254 ms
                 512 MB		          40.135892 ms
                1024 MB		          78.760016 ms
```

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Benchmarking BLAKE3 SYCL implementation

              input size		  execution time
                   1 MB		         455.411000 us
                   2 MB		         383.676000 us
                   4 MB		         805.415000 us
                   8 MB		           1.184185 ms
                  16 MB		           2.586108 ms
                  32 MB		           6.362890 ms
                  64 MB		          11.808920 ms
                 128 MB		          15.889609 ms
                 256 MB		          20.456640 ms
                 512 MB		          38.492936 ms
                1024 MB		          76.519211 ms
```

```bash
# [ CPU(s): 128; used avx512 ]
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Benchmarking BLAKE3 SYCL implementation

              input size		  execution time
                   1 MB		           1.542443 ms
                   2 MB		           1.270531 ms
                   4 MB		           1.369875 ms
                   8 MB		           1.536247 ms
                  16 MB		           2.287743 ms
                  32 MB		           3.456803 ms
                  64 MB		           5.669439 ms
                 128 MB		           8.310061 ms
                 256 MB		           9.677808 ms
                 512 MB		          13.976085 ms
                1024 MB		          24.362968 ms
```
