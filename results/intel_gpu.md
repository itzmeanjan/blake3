Built using

```bash
BLAKE3_SIMD_LANES={2,4,8,16} make aot_gpu
```

### On `Intel(R) UHD Graphics P630 [0x3e96]`

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time
                   1 MB                  537.051000 us
                   2 MB                  554.108000 us
                   4 MB                  966.410000 us
                   8 MB                    1.559404 ms
                  16 MB                    4.102192 ms
                  32 MB                    5.176751 ms
                  64 MB                    9.727143 ms
                 128 MB                   18.788544 ms
                 256 MB                   36.848597 ms
                 512 MB                   72.139408 ms
                1024 MB                  145.486342 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                  589.756000 us
                   2 MB                  663.419000 us
                   4 MB                    1.204828 ms
                   8 MB                    2.190868 ms
                  16 MB                    3.998027 ms
                  32 MB                    7.314458 ms
                  64 MB                   20.172154 ms
                 128 MB                   27.232466 ms
                 256 MB                   53.998555 ms
                 512 MB                  115.996691 ms
                1024 MB                  213.364157 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                   14.836167 ms
                   2 MB                   10.452646 ms
                   4 MB                   11.759108 ms
                   8 MB                   38.073220 ms
                  16 MB                   59.713727 ms
                  32 MB                  112.923492 ms
                  64 MB                  146.802141 ms
                 128 MB                  178.022010 ms
                 256 MB                  316.222571 ms
                 512 MB                  640.830176 ms
                1024 MB                     1.264373 s
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                   15.560342 ms
                   2 MB                   11.355520 ms
                   4 MB                   12.293711 ms
                   8 MB                   15.675878 ms
                  16 MB                   35.710916 ms
                  32 MB                   77.741867 ms
                  64 MB                  160.861677 ms
                 128 MB                  281.493836 ms
                 256 MB                  555.355241 ms
                 512 MB                     1.122544 s
                1024 MB                     2.277012 s
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                   23.753230 ms
                   2 MB                   28.056946 ms
                   4 MB                   30.302262 ms
                   8 MB                   33.091104 ms
                  16 MB                   51.108702 ms
                  32 MB                  109.853861 ms
                  64 MB                  245.284339 ms
                 128 MB                  438.284737 ms
                 256 MB                  850.511997 ms
                 512 MB                     1.684447 s
                1024 MB                     3.315495 s
```
