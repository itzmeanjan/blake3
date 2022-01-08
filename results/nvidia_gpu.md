Built using

```bash
BLAKE3_SIMD_LANES={2,4,8,16} make cuda
```

### On `Tesla V100-SXM2-16GB`

```bash
running on Tesla V100-SXM2-16GB

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time
                   1 MB                  158.310000 us
                   2 MB                  170.454000 us
                   4 MB                  194.167000 us
                   8 MB                  253.067000 us
                  16 MB                  317.383000 us
                  32 MB                  500.367000 us
                  64 MB                    1.036608 ms
                 128 MB                    1.742003 ms
                 256 MB                    3.257110 ms
                 512 MB                    6.030942 ms
                1024 MB                   10.450562 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                  193.848000 us
                   2 MB                  193.847000 us
                   4 MB                  204.101000 us
                   8 MB                  254.150000 us
                  16 MB                  331.543000 us
                  32 MB                  476.318000 us
                  64 MB                  875.978000 us
                 128 MB                    1.810791 ms
                 256 MB                    3.352050 ms
                 512 MB                    6.414551 ms
                1024 MB                   12.277342 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                  328.124000 us
                   2 MB                  302.979000 us
                   4 MB                  320.800000 us
                   8 MB                  338.134000 us
                  16 MB                  429.443000 us
                  32 MB                  592.529000 us
                  64 MB                  967.529000 us
                 128 MB                    1.975097 ms
                 256 MB                    3.493408 ms
                 512 MB                    6.756837 ms
                1024 MB                   12.599365 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    1.956297 ms
                   2 MB                  677.489000 us
                   4 MB                  688.965000 us
                   8 MB                  710.692000 us
                  16 MB                  737.060000 us
                  32 MB                    1.019774 ms
                  64 MB                    1.720215 ms
                 128 MB                    4.105224 ms
                 256 MB                   10.151125 ms
                 512 MB                   19.336424 ms
                1024 MB                   37.365477 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    2.573731 ms
                   2 MB                    1.459473 ms
                   4 MB                    1.479737 ms
                   8 MB                    1.520751 ms
                  16 MB                    1.534669 ms
                  32 MB                    1.673339 ms
                  64 MB                    2.642822 ms
                 128 MB                    5.867675 ms
                 256 MB                   13.152344 ms
                 512 MB                   27.395751 ms
                1024 MB                   55.669432 ms
```
