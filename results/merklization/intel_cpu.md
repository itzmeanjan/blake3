Built using

```bash
make aot_cpu
```

### On `Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz`

```bash
# [ CPU(s): 128; used avx512 ]
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                     2.584194 ms                     6.707623 ms                     6.248120 ms
        2 ^ 21                     3.701423 ms                    13.165455 ms                     7.915633 ms
        2 ^ 22                     4.610586 ms                    22.517784 ms                     8.780380 ms
        2 ^ 23                     6.394468 ms                    33.124851 ms                    11.021414 ms
        2 ^ 24                    11.133987 ms                    45.021938 ms                    20.972211 ms
        2 ^ 25                    37.840892 ms                    65.168096 ms                    41.135593 ms
```

---

### On `Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz`

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                     5.009278 ms                     3.263662 ms                     3.298174 ms
        2 ^ 21                     9.161365 ms                     6.833179 ms                     5.589549 ms
        2 ^ 22                    16.001656 ms                    11.343515 ms                    10.693312 ms
        2 ^ 23                    32.221961 ms                    22.518283 ms                    21.226782 ms
        2 ^ 24                    64.794958 ms                    43.620266 ms                    42.600062 ms
        2 ^ 25                   130.010189 ms                    83.767304 ms                    82.963254 ms
```

---

### On `Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz`

```bash
# [ CPU(s): 24; used avx512 ]
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                     5.551719 ms                     4.590406 ms                     3.706361 ms
        2 ^ 21                     8.721216 ms                     8.527807 ms                     5.709464 ms
        2 ^ 22                    13.354889 ms                    17.448928 ms                     8.763114 ms
        2 ^ 23                    20.580858 ms                    26.836197 ms                    15.679715 ms
        2 ^ 24                    41.294036 ms                    42.733808 ms                    31.073722 ms
        2 ^ 25                    86.344991 ms                    74.129915 ms                    63.705284 ms
```

---

### On `Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz`

```bash
# [ CPU(s): 12; used avx2 ]
running on Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                    11.084651 ms                     2.477921 ms                     2.404031 ms
        2 ^ 21                    30.814870 ms                     4.801900 ms                     4.723098 ms
        2 ^ 22                    41.885615 ms                     9.376421 ms                     9.240906 ms
        2 ^ 23                    83.008743 ms                    18.497232 ms                    18.774122 ms
        2 ^ 24                   168.063260 ms                    36.848257 ms                    36.746164 ms
        2 ^ 25                   329.425297 ms                    73.505884 ms                    73.295890 ms
```

---

### On `Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz`

```bash
# [ CPU(s): 4; used avx2 ]
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                    52.397037 ms                     8.867935 ms                     9.157633 ms
        2 ^ 21                   103.264167 ms                    17.781408 ms                    18.116898 ms
        2 ^ 22                   210.549346 ms                    35.255670 ms                    36.297967 ms
        2 ^ 23                   410.769689 ms                    73.672694 ms                    75.193556 ms
        2 ^ 24                   819.721430 ms                   146.009166 ms                   146.013156 ms
        2 ^ 25                      1.636137 s                   291.046629 ms                   293.293869 ms
```
