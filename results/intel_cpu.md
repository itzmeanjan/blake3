Built using

```bash
BLAKE3_SIMD_LANES={2,4,8,16} make aot_cpu
```

### On `Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz`

```bash
# [ CPU(s): 4; used avx2 ]
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time
                   1 MB                    1.535053 ms
                   2 MB                    2.324566 ms
                   4 MB                    4.421891 ms
                   8 MB                    8.682190 ms
                  16 MB                   17.326016 ms
                  32 MB                   34.776514 ms
                  64 MB                   68.288191 ms
                 128 MB                  141.008220 ms
                 256 MB                  271.600802 ms
                 512 MB                  541.594876 ms
                1024 MB                     1.086021 s
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    2.640065 ms
                   2 MB                    4.589143 ms
                   4 MB                    8.974861 ms
                   8 MB                   17.664158 ms
                  16 MB                   35.176459 ms
                  32 MB                   70.473792 ms
                  64 MB                  140.738687 ms
                 128 MB                  280.730679 ms
                 256 MB                  558.651098 ms
                 512 MB                     1.119648 s
                1024 MB                     2.231913 s
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    2.410590 ms
                   2 MB                    4.483136 ms
                   4 MB                    9.490823 ms
                   8 MB                   17.348939 ms
                  16 MB                   27.069447 ms
                  32 MB                   53.444846 ms
                  64 MB                  107.797906 ms
                 128 MB                  211.315984 ms
                 256 MB                  421.151311 ms
                 512 MB                  851.662736 ms
                1024 MB                     1.704885 s
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    3.255826 ms
                   2 MB                    5.385626 ms
                   4 MB                   10.517980 ms
                   8 MB                   20.980429 ms
                  16 MB                   41.627366 ms
                  32 MB                   81.449933 ms
                  64 MB                  164.480672 ms
                 128 MB                  321.240538 ms
                 256 MB                  644.155882 ms
                 512 MB                     1.280831 s
                1024 MB                     2.557385 s
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    9.663129 ms
                   2 MB                   12.756929 ms
                   4 MB                   25.793409 ms
                   8 MB                   50.483286 ms
                  16 MB                  102.267502 ms
                  32 MB                  202.261208 ms
                  64 MB                  395.185795 ms
                 128 MB                  791.928735 ms
                 256 MB                     1.572042 s
                 512 MB                     3.143899 s
                1024 MB                     6.256395 s
```

---

### On `Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz`

```bash
# [ CPU(s): 128; used avx512 ]
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time
                   1 MB                    1.120702 ms
                   2 MB                    1.202172 ms
                   4 MB                    1.410422 ms
                   8 MB                    2.018987 ms
                  16 MB                    2.500873 ms
                  32 MB                    4.527778 ms
                  64 MB                    6.451395 ms
                 128 MB                    8.719392 ms
                 256 MB                    9.835248 ms
                 512 MB                   13.150252 ms
                1024 MB                   22.911430 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    1.428717 ms
                   2 MB                    1.642099 ms
                   4 MB                    1.635514 ms
                   8 MB                    1.794220 ms
                  16 MB                    2.877134 ms
                  32 MB                    4.975101 ms
                  64 MB                    7.723535 ms
                 128 MB                    9.583591 ms
                 256 MB                   11.267384 ms
                 512 MB                   19.438637 ms
                1024 MB                   36.229899 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    1.917579 ms
                   2 MB                    2.283876 ms
                   4 MB                    2.204972 ms
                   8 MB                    2.605916 ms
                  16 MB                    2.589251 ms
                  32 MB                    4.494377 ms
                  64 MB                    6.678979 ms
                 128 MB                    8.999207 ms
                 256 MB                   10.714447 ms
                 512 MB                   23.924218 ms
                1024 MB                   30.603181 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    5.885517 ms
                   2 MB                    4.910260 ms
                   4 MB                    6.543840 ms
                   8 MB                    6.718442 ms
                  16 MB                    7.986787 ms
                  32 MB                    7.846674 ms
                  64 MB                   11.835337 ms
                 128 MB                   15.333313 ms
                 256 MB                   20.527009 ms
                 512 MB                   34.983701 ms
                1024 MB                   66.248736 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                   10.410064 ms
                   2 MB                   10.397995 ms
                   4 MB                   10.909045 ms
                   8 MB                   16.981154 ms
                  16 MB                   17.341112 ms
                  32 MB                   17.184630 ms
                  64 MB                   14.893981 ms
                 128 MB                   18.590236 ms
                 256 MB                   25.271265 ms
                 512 MB                   49.042535 ms
                1024 MB                   91.502870 ms
```
---

### On `Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz`

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time
                   1 MB                  487.521000 us
                   2 MB                  762.363000 us
                   4 MB                    1.586235 ms
                   8 MB                    1.867494 ms
                  16 MB                    4.839855 ms
                  32 MB                    6.741284 ms
                  64 MB                   11.746437 ms
                 128 MB                   15.007609 ms
                 256 MB                   22.384890 ms
                 512 MB                   42.334697 ms
                1024 MB                   84.325357 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                  853.001000 us
                   2 MB                    1.210407 ms
                   4 MB                    1.973600 ms
                   8 MB                    3.354756 ms
                  16 MB                    6.106087 ms
                  32 MB                   12.527090 ms
                  64 MB                   15.471883 ms
                 128 MB                   21.568805 ms
                 256 MB                   35.273785 ms
                 512 MB                   68.225128 ms
                1024 MB                  135.332831 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    1.243179 ms
                   2 MB                    1.280228 ms
                   4 MB                    2.007901 ms
                   8 MB                    3.308409 ms
                  16 MB                    5.598625 ms
                  32 MB                    9.700557 ms
                  64 MB                   13.496357 ms
                 128 MB                   18.964755 ms
                 256 MB                   29.735171 ms
                 512 MB                   56.552652 ms
                1024 MB                  112.375748 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    3.402214 ms
                   2 MB                    3.643949 ms
                   4 MB                    4.609026 ms
                   8 MB                    8.684951 ms
                  16 MB                   13.997617 ms
                  32 MB                   17.660809 ms
                  64 MB                   24.978581 ms
                 128 MB                   36.671089 ms
                 256 MB                   63.932004 ms
                 512 MB                  122.961880 ms
                1024 MB                  249.416206 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    7.460626 ms
                   2 MB                    7.486521 ms
                   4 MB                   11.438846 ms
                   8 MB                   13.760393 ms
                  16 MB                   17.528906 ms
                  32 MB                   24.643870 ms
                  64 MB                   34.102165 ms
                 128 MB                   53.097703 ms
                 256 MB                  108.135065 ms
                 512 MB                  184.124070 ms
                1024 MB                  369.643404 ms
```

---

### On `Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz`

```bash
# [ CPU(s): 12; used avx2 ]
running on Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time
                   1 MB                  564.571000 us
                   2 MB                  936.153000 us
                   4 MB                    1.410566 ms
                   8 MB                    2.505761 ms
                  16 MB                    4.465773 ms
                  32 MB                    8.561860 ms
                  64 MB                   17.522937 ms
                 128 MB                   31.613256 ms
                 256 MB                   62.566890 ms
                 512 MB                  124.029188 ms
                1024 MB                  250.726431 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                  868.955000 us
                   2 MB                    1.116056 ms
                   4 MB                    1.941763 ms
                   8 MB                    3.510570 ms
                  16 MB                    6.327400 ms
                  32 MB                   11.889369 ms
                  64 MB                   23.159883 ms
                 128 MB                   45.566010 ms
                 256 MB                   90.169047 ms
                 512 MB                  187.951293 ms
                1024 MB                  358.965746 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                  887.412000 us
                   2 MB                    1.355160 ms
                   4 MB                    1.966733 ms
                   8 MB                    3.624923 ms
                  16 MB                    6.220828 ms
                  32 MB                   11.620348 ms
                  64 MB                   22.279148 ms
                 128 MB                   43.852904 ms
                 256 MB                   86.801124 ms
                 512 MB                  172.498208 ms
                1024 MB                  347.183344 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    1.570597 ms
                   2 MB                    1.850503 ms
                   4 MB                    3.342631 ms
                   8 MB                    4.997062 ms
                  16 MB                    9.387379 ms
                  32 MB                   17.081930 ms
                  64 MB                   32.851853 ms
                 128 MB                   64.640995 ms
                 256 MB                  127.836171 ms
                 512 MB                  254.326319 ms
                1024 MB                  510.912368 ms
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time
                   1 MB                    3.490090 ms
                   2 MB                    4.617491 ms
                   4 MB                    6.602516 ms
                   8 MB                   12.637082 ms
                  16 MB                   19.329369 ms
                  32 MB                   37.111232 ms
                  64 MB                   70.162086 ms
                 128 MB                  135.925034 ms
                 256 MB                  269.854776 ms
                 512 MB                  531.697001 ms
                1024 MB                     1.062264 s
```
