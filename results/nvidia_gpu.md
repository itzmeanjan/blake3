Built using

```bash
BLAKE3_SIMD_LANES={2,4,8,16} make cuda
```

### On `Tesla V100-SXM2-16GB`

```bash
running on Tesla V100-SXM2-16GB

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  154.362875 us                   108.415750 us                     5.561750 us
                   2 MB                  162.331625 us                   205.455875 us                     4.825500 us
                   4 MB                  190.280500 us                   401.607625 us                     5.623000 us
                   8 MB                  243.529875 us                   792.728250 us                     8.998750 us
                  16 MB                  320.465750 us                     1.573578 ms                     6.149375 us
                  32 MB                  482.268250 us                     3.118729 ms                     7.126000 us
                  64 MB                  844.598250 us                     6.166145 ms                     6.973250 us
                 128 MB                    1.800964 ms                    12.269974 ms                     7.080000 us
                 256 MB                    3.267731 ms                    24.462952 ms                     6.805500 us
                 512 MB                    5.998047 ms                    48.833740 ms                     6.713750 us
                1024 MB                   11.915527 ms                    97.573730 ms                     8.423000 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  189.209125 us                   467.285250 us                     5.370875 us
                   2 MB                  191.650125 us                   205.322500 us                     4.882875 us
                   4 MB                  204.345750 us                   401.611375 us                     5.859500 us
                   8 MB                  248.046125 us                   792.968625 us                     5.859250 us
                  16 MB                  341.796500 us                     1.574707 ms                     5.127250 us
                  32 MB                  496.093000 us                     3.122803 ms                     7.080250 us
                  64 MB                    1.016358 ms                     6.172363 ms                     7.568375 us
                 128 MB                    1.838379 ms                    12.268555 ms                     7.568375 us
                 256 MB                    3.539550 ms                    24.455078 ms                     7.080250 us
                 512 MB                    6.729248 ms                    48.828125 ms                     7.079750 us
                1024 MB                   11.715087 ms                    97.482910 ms                     9.765625 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  324.707250 us                   397.949125 us                     5.127250 us
                   2 MB                  318.847625 us                   205.566625 us                     5.126750 us
                   4 MB                  334.716500 us                   401.122875 us                     5.615500 us
                   8 MB                  348.632875 us                   792.724625 us                     6.347250 us
                  16 MB                  427.735250 us                     1.571533 ms                     5.615375 us
                  32 MB                  622.559000 us                     3.118164 ms                     7.324125 us
                  64 MB                  923.828375 us                     6.168457 ms                     7.323875 us
                 128 MB                    2.014648 ms                    12.263916 ms                     7.568375 us
                 256 MB                    4.190674 ms                    24.442871 ms                     7.080000 us
                 512 MB                    7.277343 ms                    48.827637 ms                     6.836000 us
                1024 MB                   12.184326 ms                    97.552734 ms                     8.300750 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.199463 ms                   468.505875 us                     5.615250 us
                   2 MB                  650.634750 us                   205.566500 us                     4.882750 us
                   4 MB                  678.466625 us                   401.367375 us                     5.859125 us
                   8 MB                  695.312125 us                   792.968750 us                     5.859125 us
                  16 MB                  716.064500 us                     1.573974 ms                     6.103250 us
                  32 MB                  928.221500 us                     3.122070 ms                     7.324375 us
                  64 MB                    1.318848 ms                     6.168945 ms                     7.812625 us
                 128 MB                    2.197510 ms                    12.266113 ms                     6.591625 us
                 256 MB                    5.203370 ms                    24.459961 ms                     7.568250 us
                 512 MB                   10.184327 ms                    48.817139 ms                     8.056625 us
                1024 MB                   18.732911 ms                    97.577148 ms                     7.812250 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    2.113281 ms                   437.011750 us                     5.859250 us
                   2 MB                    1.431641 ms                   205.322125 us                     5.127000 us
                   4 MB                    1.480224 ms                   401.367250 us                     6.103625 us
                   8 MB                    1.505371 ms                   792.236375 us                     6.103750 us
                  16 MB                    1.516845 ms                     1.573974 ms                     5.615250 us
                  32 MB                    1.594970 ms                     3.113770 ms                     7.324375 us
                  64 MB                    2.055176 ms                     6.176270 ms                    10.254000 us
                 128 MB                    5.214600 ms                    12.262207 ms                     7.323875 us
                 256 MB                   13.925293 ms                    24.453369 ms                     7.568375 us
                 512 MB                   27.458252 ms                    48.818115 ms                     7.812375 us
                1024 MB                   52.898436 ms                    97.524170 ms                     8.056625 us
```
