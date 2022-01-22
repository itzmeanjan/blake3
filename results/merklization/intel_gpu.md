Built using

```bash
make aot_gpu
```

### On `Intel(R) Iris(R) Xe MAX Graphics [0x4905]`

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                     2.157610 ms                     8.871336 ms                     5.907590 ms
        2 ^ 21                     4.263610 ms                    17.726937 ms                    11.813484 ms
        2 ^ 22                     8.504262 ms                    35.450772 ms                    23.627377 ms
        2 ^ 23                    16.990376 ms                    70.889182 ms                    47.227765 ms
        2 ^ 24                    33.977788 ms                   141.732019 ms                    94.446241 ms
        2 ^ 25                    67.982122 ms                   283.346856 ms                   188.886958 ms
```

---

### On `Intel(R) UHD Graphics P630 [0x3e96]`

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                     7.023128 ms                     3.601432 ms                     3.587156 ms
        2 ^ 21                    16.578150 ms                     7.059918 ms                     7.063964 ms
        2 ^ 22                    30.324506 ms                    14.185405 ms                    14.163888 ms
        2 ^ 23                    54.047400 ms                    28.143246 ms                    28.182754 ms
        2 ^ 24                   107.691172 ms                    54.678097 ms                    54.542184 ms
        2 ^ 25                   214.863594 ms                    97.307644 ms                    96.219285 ms
```
