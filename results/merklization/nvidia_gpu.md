Built using

```bash
make cuda
```

### On `Tesla V100-SXM2-16GB`

```bash
running on Tesla V100-SXM2-16GB

Benchmarking Binary Merklization using BLAKE3

      leaf count                  execution time                host-to-device tx time          device-to-host tx time
        2 ^ 20                   360.351500 us                     3.114746 ms                     2.682128 ms
        2 ^ 21                   591.308250 us                     6.153320 ms                     5.361816 ms
        2 ^ 22                     1.044922 ms                    12.252441 ms                    10.715332 ms
        2 ^ 23                     1.942382 ms                    24.408203 ms                    21.438965 ms
        2 ^ 24                     3.741699 ms                    48.793457 ms                    42.858398 ms
        2 ^ 25                     7.312989 ms                    97.534180 ms                    85.627441 ms
```
