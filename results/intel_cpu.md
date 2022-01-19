Built using

```bash
BLAKE3_SIMD_LANES={2,4,8,16} make aot_cpu
```

### On `Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz`

```bash
# [ CPU(s): 4; used avx2 ]
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.411340 ms                   302.379750 us                     3.128375 us
                   2 MB                    2.382188 ms                   502.974125 us                     2.244875 us
                   4 MB                    4.551627 ms                   978.917750 us                     1.813500 us
                   8 MB                    8.884377 ms                     1.928709 ms                     2.123000 us
                  16 MB                   17.718145 ms                     4.421330 ms                     2.502000 us
                  32 MB                   35.164200 ms                     8.941067 ms                     3.316250 us
                  64 MB                   73.808522 ms                    18.136102 ms                     2.870250 us
                 128 MB                  149.423070 ms                    37.800974 ms                     3.243750 us
                 256 MB                  278.700030 ms                    73.604903 ms                     2.825250 us
                 512 MB                  556.898126 ms                   148.596441 ms                     2.950250 us
                1024 MB                     1.126861 s                   293.848222 ms                     3.435000 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    2.458254 ms                   290.517125 us                     2.240000 us
                   2 MB                    4.403440 ms                   516.484625 us                     2.325625 us
                   4 MB                    8.571496 ms                     1.013186 ms                     2.281625 us
                   8 MB                   17.256722 ms                     2.134683 ms                     2.414000 us
                  16 MB                   33.931112 ms                     4.449573 ms                     2.584750 us
                  32 MB                   67.583474 ms                     9.160075 ms                     3.510875 us
                  64 MB                  134.817304 ms                    17.928418 ms                     3.086500 us
                 128 MB                  269.274363 ms                    35.857533 ms                     3.475250 us
                 256 MB                  536.382903 ms                    74.012070 ms                     2.823750 us
                 512 MB                     1.107967 s                   149.691718 ms                     3.079375 us
                1024 MB                     2.264229 s                   300.741099 ms                     2.879375 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.457193 ms                   297.706000 us                     2.182750 us
                   2 MB                    2.597208 ms                   535.761625 us                     2.190375 us
                   4 MB                    4.948043 ms                   984.825125 us                     1.835000 us
                   8 MB                    9.656003 ms                     1.903438 ms                     2.037875 us
                  16 MB                   19.059954 ms                     4.102028 ms                     2.157000 us
                  32 MB                   37.820617 ms                     9.093352 ms                     2.961375 us
                  64 MB                   75.408616 ms                    17.937515 ms                     2.687875 us
                 128 MB                  150.304798 ms                    35.845284 ms                     2.909750 us
                 256 MB                  299.651055 ms                    74.402049 ms                     2.493250 us
                 512 MB                  599.306090 ms                   148.221302 ms                     2.565250 us
                1024 MB                     1.214943 s                   299.679108 ms                     3.162625 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.454525 ms                   300.303500 us                     2.049000 us
                   2 MB                    2.490826 ms                   532.865750 us                     2.462125 us
                   4 MB                    4.858460 ms                     1.058994 ms                     2.142125 us
                   8 MB                    9.548375 ms                     2.168333 ms                     2.666750 us
                  16 MB                   18.724246 ms                     4.372800 ms                     2.873750 us
                  32 MB                   36.474041 ms                     9.020951 ms                     2.978500 us
                  64 MB                   72.999172 ms                    17.845079 ms                     2.894000 us
                 128 MB                  145.707888 ms                    36.177141 ms                     2.912500 us
                 256 MB                  296.205519 ms                    75.066909 ms                     2.762125 us
                 512 MB                  578.887824 ms                   145.805123 ms                     2.769000 us
                1024 MB                     1.153393 s                   289.286738 ms                     2.724625 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    5.484708 ms                   293.350750 us                     1.923000 us
                   2 MB                    7.038828 ms                   526.952375 us                     2.219125 us
                   4 MB                   14.028125 ms                     1.044917 ms                     2.349875 us
                   8 MB                   28.113723 ms                     2.153057 ms                     2.522125 us
                  16 MB                   55.646375 ms                     4.334506 ms                     2.868500 us
                  32 MB                  107.368298 ms                     9.117504 ms                     3.173625 us
                  64 MB                  211.528465 ms                    18.809700 ms                     2.976500 us
                 128 MB                  422.311642 ms                    35.872375 ms                     3.016750 us
                 256 MB                  839.858409 ms                    73.837866 ms                     2.992875 us
                 512 MB                     1.679492 s                   147.312903 ms                     2.831250 us
                1024 MB                     3.354590 s                   294.967816 ms                     2.814375 us
```

---

### On `Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz`

```bash
# [ CPU(s): 128; used avx512 ]
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.035622 ms                   941.733125 us                    12.349500 us
                   2 MB                  949.878750 us                     1.528471 ms                     8.627375 us
                   4 MB                    1.224290 ms                     1.763114 ms                     5.351500 us
                   8 MB                    1.713689 ms                     2.125699 ms                     5.597750 us
                  16 MB                    2.568303 ms                     3.061066 ms                     5.793125 us
                  32 MB                    4.168138 ms                     5.230914 ms                     6.145125 us
                  64 MB                    6.239875 ms                     9.797500 ms                     2.525625 us
                 128 MB                    8.187520 ms                    20.664062 ms                     1.242000 us
                 256 MB                    8.853032 ms                    32.455801 ms                     1.047125 us
                 512 MB                   14.807205 ms                    48.242437 ms                     1.063000 us
                1024 MB                   22.864140 ms                    79.047688 ms                     1.088500 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.435731 ms                   829.346250 us                     2.538125 us
                   2 MB                    1.235621 ms                     1.160755 ms                     4.529125 us
                   4 MB                    1.467002 ms                     1.388256 ms                     4.071250 us
                   8 MB                    1.676693 ms                     1.940926 ms                     5.912250 us
                  16 MB                    2.752172 ms                     3.576378 ms                     3.839875 us
                  32 MB                    4.738109 ms                     6.948434 ms                     4.638500 us
                  64 MB                    7.193866 ms                    13.689200 ms                     4.531500 us
                 128 MB                    8.897695 ms                    23.638901 ms                   987.625000 ns
                 256 MB                   10.928828 ms                    33.915237 ms                   967.625000 ns
                 512 MB                   18.045038 ms                    48.268317 ms                     1.031375 us
                1024 MB                   35.084335 ms                    76.414034 ms                   903.625000 ns
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.332648 ms                   778.364750 us                     4.491625 us
                   2 MB                    1.517697 ms                     1.216136 ms                     5.699500 us
                   4 MB                    1.878738 ms                     1.405623 ms                     6.585000 us
                   8 MB                    1.955160 ms                     1.890794 ms                     6.283625 us
                  16 MB                    2.295545 ms                     3.461322 ms                     4.179250 us
                  32 MB                    3.848329 ms                     6.653417 ms                     4.561875 us
                  64 MB                    6.739462 ms                    14.008103 ms                     2.967625 us
                 128 MB                    7.512532 ms                    21.777197 ms                   857.000000 ns
                 256 MB                    8.854166 ms                    32.901272 ms                   976.500000 ns
                 512 MB                   13.710111 ms                    47.194467 ms                   965.875000 ns
                1024 MB                   25.805447 ms                    79.800968 ms                     1.052625 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.957192 ms                   715.214875 us                     5.877000 us
                   2 MB                    1.925397 ms                     1.117507 ms                     3.798000 us
                   4 MB                    3.097539 ms                     1.451939 ms                    13.072625 us
                   8 MB                    3.446283 ms                     1.918505 ms                     6.290750 us
                  16 MB                    3.690009 ms                     3.503353 ms                     6.937875 us
                  32 MB                    4.282732 ms                     6.779036 ms                     5.052125 us
                  64 MB                    7.261953 ms                    14.829467 ms                     2.978000 us
                 128 MB                    9.568175 ms                    22.821287 ms                   878.000000 ns
                 256 MB                   10.290331 ms                    33.643110 ms                     1.030125 us
                 512 MB                   15.784226 ms                    47.931818 ms                   863.375000 ns
                1024 MB                   28.765474 ms                    80.076494 ms                     1.017875 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    6.539631 ms                   949.643375 us                     6.476625 us
                   2 MB                    7.431457 ms                     1.504223 ms                     6.843000 us
                   4 MB                    7.110438 ms                     1.818267 ms                     5.692250 us
                   8 MB                   10.708573 ms                     3.247708 ms                     5.178500 us
                  16 MB                   13.190738 ms                     3.451771 ms                     3.659125 us
                  32 MB                   13.757709 ms                     5.561875 ms                     3.393625 us
                  64 MB                   11.546031 ms                    13.229008 ms                     1.385125 us
                 128 MB                   14.950347 ms                    22.513792 ms                     1.014875 us
                 256 MB                   18.790299 ms                    33.722702 ms                   966.125000 ns
                 512 MB                   33.867609 ms                    47.791360 ms                     1.216500 us
                1024 MB                   61.698307 ms                    75.633475 ms                     1.107500 us
```
---

### On `Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz`

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  357.017500 us                   308.464000 us                     1.038625 us
                   2 MB                  536.530125 us                   551.713750 us                     2.091125 us
                   4 MB                    1.139873 ms                   811.976750 us                     2.913250 us
                   8 MB                    1.910413 ms                     1.322750 ms                     2.355250 us
                  16 MB                    3.611798 ms                     2.494118 ms                   971.125000 ns
                  32 MB                    6.082726 ms                     4.624181 ms                   921.625000 ns
                  64 MB                   10.105840 ms                     8.526520 ms                   897.000000 ns
                 128 MB                   14.036209 ms                    15.155051 ms                     1.050000 us
                 256 MB                   20.757668 ms                    27.175328 ms                     1.198875 us
                 512 MB                   39.001923 ms                    44.536675 ms                     1.223750 us
                1024 MB                   77.331343 ms                    91.330096 ms                     1.203125 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  725.956250 us                   321.646625 us                     1.812625 us
                   2 MB                    1.063598 ms                   574.803250 us                     2.297000 us
                   4 MB                    1.853857 ms                   811.289625 us                     2.623625 us
                   8 MB                    3.265703 ms                     1.355988 ms                     1.887375 us
                  16 MB                    5.695174 ms                     2.375724 ms                   669.875000 ns
                  32 MB                   10.207269 ms                     5.329347 ms                   849.000000 ns
                  64 MB                   14.268392 ms                     7.847511 ms                   967.375000 ns
                 128 MB                   20.377278 ms                    14.777122 ms                     1.126875 us
                 256 MB                   33.007169 ms                    25.580627 ms                     1.522875 us
                 512 MB                   62.540990 ms                    47.311655 ms                     1.087875 us
                1024 MB                  124.047302 ms                    83.641604 ms                     1.418875 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  850.803500 us                   240.196625 us                     1.433625 us
                   2 MB                    1.028460 ms                   613.574500 us                     2.795375 us
                   4 MB                    1.686378 ms                   819.039000 us                     2.232500 us
                   8 MB                    2.583981 ms                     1.327759 ms                     1.756625 us
                  16 MB                    4.535992 ms                     2.399812 ms                   739.750000 ns
                  32 MB                    7.715149 ms                     4.566527 ms                   868.625000 ns
                  64 MB                   12.397610 ms                     8.777051 ms                   987.500000 ns
                 128 MB                   16.035606 ms                    15.090863 ms                     1.112125 us
                 256 MB                   23.686623 ms                    28.505377 ms                     1.196625 us
                 512 MB                   44.435378 ms                    44.838008 ms                     1.637875 us
                1024 MB                   87.095254 ms                    86.813492 ms                     1.280750 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.434828 ms                   341.291500 us                     2.925000 us
                   2 MB                    2.028067 ms                   691.156625 us                     2.299750 us
                   4 MB                    2.169944 ms                   841.397000 us                     2.468625 us
                   8 MB                    3.577898 ms                     1.354724 ms                     2.677625 us
                  16 MB                    5.726758 ms                     2.392596 ms                   585.125000 ns
                  32 MB                    9.302333 ms                     4.546784 ms                   669.750000 ns
                  64 MB                   14.719235 ms                     8.442529 ms                     1.213750 us
                 128 MB                   17.816075 ms                    14.553390 ms                   897.250000 ns
                 256 MB                   27.092123 ms                    25.995475 ms                     1.458500 us
                 512 MB                   51.581485 ms                    48.473726 ms                     1.213000 us
                1024 MB                  101.127829 ms                    85.415163 ms                   981.500000 ns
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    4.988956 ms                   345.659875 us                     2.517500 us
                   2 MB                    6.923827 ms                   721.461250 us                     1.908625 us
                   4 MB                    8.343197 ms                     1.045322 ms                     1.605875 us
                   8 MB                    9.554885 ms                     1.576714 ms                   928.750000 ns
                  16 MB                   14.489025 ms                     2.544420 ms                   813.875000 ns
                  32 MB                   17.014290 ms                     4.073476 ms                     1.041375 us
                  64 MB                   25.710434 ms                     7.544063 ms                     1.052000 us
                 128 MB                   36.536837 ms                    16.001267 ms                     1.483125 us
                 256 MB                   63.283484 ms                    26.332025 ms                     1.233125 us
                 512 MB                  121.902091 ms                    45.809442 ms                     1.067750 us
                1024 MB                  245.176609 ms                    83.243050 ms                     1.076375 us
```

---

### On `Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz`

```bash
# [ CPU(s): 12; used avx2 ]
running on Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  327.447125 us                   164.797750 us                     3.334250 us
                   2 MB                  888.252500 us                   208.945000 us                   600.375000 ns
                   4 MB                    1.049447 ms                   297.244750 us                     2.543375 us
                   8 MB                    2.007394 ms                   561.511250 us                     2.600250 us
                  16 MB                    3.868704 ms                     1.221595 ms                   820.250000 ns
                  32 MB                    7.259625 ms                     2.429787 ms                   677.000000 ns
                  64 MB                   13.978607 ms                     4.754956 ms                   736.250000 ns
                 128 MB                   27.780513 ms                     9.330577 ms                   749.500000 ns
                 256 MB                   55.302504 ms                    18.498851 ms                   875.750000 ns
                 512 MB                  110.761080 ms                    36.818487 ms                   809.000000 ns
                1024 MB                  220.764015 ms                    73.411591 ms                   849.000000 ns
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  551.211125 us                   148.376625 us                   616.875000 ns
                   2 MB                  796.079000 us                   196.313375 us                   505.250000 ns
                   4 MB                    1.401199 ms                   286.559000 us                   705.500000 ns
                   8 MB                    2.544254 ms                   535.589000 us                   550.000000 ns
                  16 MB                    4.819938 ms                     1.201561 ms                   566.750000 ns
                  32 MB                    9.369013 ms                     2.404684 ms                   592.625000 ns
                  64 MB                   18.504175 ms                     4.758154 ms                   673.000000 ns
                 128 MB                   36.716331 ms                     9.319614 ms                   613.750000 ns
                 256 MB                   73.104492 ms                    18.491373 ms                     2.659625 us
                 512 MB                  145.770998 ms                    36.832426 ms                   780.250000 ns
                1024 MB                  291.536176 ms                    73.392269 ms                   795.375000 ns
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  477.501875 us                   128.105000 us                   617.250000 ns
                   2 MB                  819.885750 us                   212.460875 us                     2.089375 us
                   4 MB                    1.207255 ms                   312.599375 us                   488.250000 ns
                   8 MB                    2.296374 ms                   603.953250 us                   690.875000 ns
                  16 MB                    4.145224 ms                     1.188992 ms                   784.875000 ns
                  32 MB                    7.876318 ms                     2.414666 ms                   616.250000 ns
                  64 MB                   15.409500 ms                     4.763488 ms                   733.750000 ns
                 128 MB                   30.456902 ms                     9.318929 ms                   903.000000 ns
                 256 MB                   60.333662 ms                    18.490137 ms                   786.500000 ns
                 512 MB                  125.270049 ms                    36.855571 ms                     2.960625 us
                1024 MB                  246.221199 ms                    73.462219 ms                   746.875000 ns
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  734.042875 us                   149.736375 us                   808.000000 ns
                   2 MB                  805.438250 us                   218.295750 us                     2.174000 us
                   4 MB                    1.437623 ms                   316.812250 us                   623.375000 ns
                   8 MB                    2.154406 ms                   545.291375 us                   578.750000 ns
                  16 MB                    3.991284 ms                     1.201524 ms                   794.500000 ns
                  32 MB                    7.543487 ms                     2.410248 ms                   702.125000 ns
                  64 MB                   14.637162 ms                     4.762942 ms                   654.750000 ns
                 128 MB                   28.733478 ms                     9.324469 ms                   756.750000 ns
                 256 MB                   56.476447 ms                    18.507013 ms                   754.375000 ns
                 512 MB                  114.561923 ms                    36.832606 ms                   812.875000 ns
                1024 MB                  223.173748 ms                    73.414237 ms                   805.375000 ns
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    1.953116 ms                   153.074750 us                   695.500000 ns
                   2 MB                    2.619823 ms                   220.599125 us                   450.875000 ns
                   4 MB                    3.475796 ms                   307.221125 us                   566.375000 ns
                   8 MB                    6.571765 ms                   538.039000 us                   624.750000 ns
                  16 MB                   10.114083 ms                     1.205012 ms                   623.750000 ns
                  32 MB                   19.093997 ms                     2.413813 ms                   670.625000 ns
                  64 MB                   36.166322 ms                     4.747795 ms                   622.375000 ns
                 128 MB                   70.424445 ms                     9.322505 ms                   771.875000 ns
                 256 MB                  146.945313 ms                    18.511632 ms                   726.625000 ns
                 512 MB                  277.160686 ms                    36.837060 ms                   696.500000 ns
                1024 MB                  555.891941 ms                    73.445136 ms                   647.875000 ns
```

---

### On `Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz`

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  287.689500 us                   189.727375 us                   614.875000 ns
                   2 MB                  406.320375 us                   260.700125 us                     1.006625 us
                   4 MB                  955.492625 us                   528.503750 us                     1.154000 us
                   8 MB                    1.851749 ms                     1.076688 ms                     1.454250 us
                  16 MB                    3.337930 ms                     2.010356 ms                     1.128250 us
                  32 MB                    6.305565 ms                     3.795659 ms                   739.250000 ns
                  64 MB                   10.437824 ms                     6.534568 ms                   901.375000 ns
                 128 MB                   12.251488 ms                    11.281635 ms                     1.639750 us
                 256 MB                   20.167881 ms                    21.406765 ms                     3.363500 us
                 512 MB                   38.131119 ms                    42.965097 ms                     2.042500 us
                1024 MB                   75.931118 ms                    83.779978 ms                     1.040875 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  524.511750 us                   207.601500 us                     1.250125 us
                   2 MB                  759.372875 us                   281.304875 us                     1.258625 us
                   4 MB                    1.499872 ms                   552.455250 us                   969.125000 ns
                   8 MB                    3.127564 ms                     1.076804 ms                     1.603250 us
                  16 MB                    5.524879 ms                     1.899661 ms                   648.000000 ns
                  32 MB                    9.836432 ms                     3.396039 ms                   730.750000 ns
                  64 MB                   15.650938 ms                     6.069018 ms                     1.148250 us
                 128 MB                   18.723169 ms                    10.971188 ms                     1.871750 us
                 256 MB                   32.749660 ms                    21.564225 ms                     3.174125 us
                 512 MB                   61.956078 ms                    42.942366 ms                     2.200125 us
                1024 MB                  123.413433 ms                    83.530199 ms                     1.155125 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  506.508000 us                   203.766750 us                     1.356250 us
                   2 MB                  837.958125 us                   278.578625 us                     1.731750 us
                   4 MB                    1.546308 ms                   576.977875 us                     1.151375 us
                   8 MB                    2.524301 ms                     1.061731 ms                     1.648500 us
                  16 MB                    4.285972 ms                     1.953383 ms                   718.250000 ns
                  32 MB                    7.319340 ms                     3.144723 ms                   785.500000 ns
                  64 MB                   12.537320 ms                     6.331435 ms                   981.875000 ns
                 128 MB                   14.435122 ms                    11.271371 ms                     1.870750 us
                 256 MB                   23.267783 ms                    21.478793 ms                     2.205375 us
                 512 MB                   44.550463 ms                    42.958929 ms                     2.074250 us
                1024 MB                   87.935820 ms                    83.861408 ms                     1.289500 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  870.483125 us                   187.190625 us                     1.251125 us
                   2 MB                    1.508372 ms                   316.221125 us                     1.891500 us
                   4 MB                    2.113352 ms                   682.485125 us                     1.983875 us
                   8 MB                    3.727835 ms                     1.104546 ms                     1.586875 us
                  16 MB                    5.643806 ms                     1.971286 ms                     1.438000 us
                  32 MB                    9.390062 ms                     3.753546 ms                     1.002125 us
                  64 MB                   13.937508 ms                     6.169532 ms                     1.561625 us
                 128 MB                   16.372279 ms                    11.216718 ms                     1.925375 us
                 256 MB                   27.431491 ms                    21.480846 ms                     2.435125 us
                 512 MB                   52.565492 ms                    43.762668 ms                     1.989000 us
                1024 MB                  104.741443 ms                    83.498288 ms                     1.402000 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    3.329761 ms                   240.807375 us                   859.125000 ns
                   2 MB                    5.913015 ms                   550.045250 us                     1.404000 us
                   4 MB                    7.884556 ms                   835.023375 us                     2.029750 us
                   8 MB                    9.328399 ms                     1.370686 ms                     1.551750 us
                  16 MB                   11.553297 ms                     1.869081 ms                   825.250000 ns
                  32 MB                   16.870859 ms                     2.848457 ms                     1.110250 us
                  64 MB                   20.134988 ms                     5.629939 ms                     1.897625 us
                 128 MB                   33.664087 ms                    10.947977 ms                     2.285125 us
                 256 MB                   62.258975 ms                    21.575464 ms                     2.707000 us
                 512 MB                  121.081411 ms                    42.869413 ms                     2.246250 us
                1024 MB                  241.314097 ms                    83.418688 ms                   973.000000 ns
```
