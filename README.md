# blake3
SYCL accelerated BLAKE3 Hash Implementation

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
