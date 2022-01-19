Built using

```bash
BLAKE3_SIMD_LANES={2,4,8,16} make aot_gpu
```

### On `Intel(R) Iris(R) Xe MAX Graphics [0x4905]`

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  206.908000 us                   279.064500 us                     1.358500 us
                   2 MB                  269.451000 us                   556.263500 us                     1.306500 us
                   4 MB                  416.468000 us                     1.112813 ms                     1.319500 us
                   8 MB                  726.752000 us                     2.230943 ms                     1.306500 us
                  16 MB                    1.345435 ms                     4.456166 ms                     1.326000 us
                  32 MB                    2.553941 ms                     8.902653 ms                     1.352000 us
                  64 MB                    4.974242 ms                    17.749401 ms                     1.319500 us
                 128 MB                    9.812348 ms                    35.475108 ms                     1.319500 us
                 256 MB                   19.465823 ms                    70.886068 ms                     1.293500 us
                 512 MB                   39.271700 ms                   141.716997 ms                     1.313000 us
                1024 MB                   77.556440 ms                   283.341799 ms                     1.534000 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  227.552000 us                   278.291000 us                     1.287000 us
                   2 MB                  264.277000 us                   554.963500 us                     1.332500 us
                   4 MB                  333.541000 us                     1.111337 ms                     1.300000 us
                   8 MB                  507.754000 us                     2.226458 ms                     1.287000 us
                  16 MB                  883.519000 us                     4.462686 ms                     1.319500 us
                  32 MB                    1.641120 ms                     8.904168 ms                     1.365000 us
                  64 MB                    3.106389 ms                    17.748458 ms                     1.365000 us
                 128 MB                    6.047769 ms                    35.462187 ms                     1.306500 us
                 256 MB                   11.941254 ms                    70.892191 ms                     1.326000 us
                 512 MB                   24.103391 ms                   141.712876 ms                     1.306500 us
                1024 MB                   47.515533 ms                   283.342299 ms                     1.319500 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    5.102851 ms                   279.233500 us                     1.397500 us
                   2 MB                    5.229380 ms                   555.600500 us                     1.365000 us
                   4 MB                    5.272852 ms                     1.116499 ms                     1.339000 us
                   8 MB                    5.443074 ms                     2.232724 ms                     1.274000 us
                  16 MB                    7.803627 ms                     4.467320 ms                     1.365000 us
                  32 MB                   14.718652 ms                     8.905110 ms                     1.397500 us
                  64 MB                   28.628951 ms                    17.749823 ms                     1.332500 us
                 128 MB                   55.370055 ms                    35.462908 ms                     1.313000 us
                 256 MB                  110.007846 ms                    70.894857 ms                     1.339000 us
                 512 MB                  217.368879 ms                   141.709763 ms                     1.326000 us
                1024 MB                  431.324205 ms                   283.344665 ms                     1.306500 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    9.324380 ms                   278.460000 us                     1.371500 us
                   2 MB                    9.350029 ms                   554.996000 us                     1.300000 us
                   4 MB                    9.575748 ms                     1.115653 ms                     1.365000 us
                   8 MB                    9.677213 ms                     2.231476 ms                     1.293500 us
                  16 MB                   11.115767 ms                     4.458499 ms                     1.332500 us
                  32 MB                   26.391196 ms                     8.902127 ms                     1.352000 us
                  64 MB                   56.188691 ms                    17.748861 ms                     1.326000 us
                 128 MB                  128.767808 ms                    35.488212 ms                     1.339000 us
                 256 MB                  245.655891 ms                    70.883748 ms                     1.391000 us
                 512 MB                  476.839155 ms                   141.710504 ms                     1.326000 us
                1024 MB                  938.669251 ms                   283.345946 ms                     2.164500 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                   16.163628 ms                   278.291000 us                     1.378000 us
                   2 MB                   16.559127 ms                   554.983000 us                     1.332500 us
                   4 MB                   16.669354 ms                     1.114230 ms                     1.358500 us
                   8 MB                   18.026554 ms                     2.226952 ms                     1.378000 us
                  16 MB                   21.859695 ms                     4.447020 ms                     1.319500 us
                  32 MB                   41.686437 ms                     8.904811 ms                     1.326000 us
                  64 MB                  105.559818 ms                    17.749823 ms                     1.365000 us
                 128 MB                  216.700575 ms                    35.476174 ms                     1.287000 us
                 256 MB                  475.246200 ms                    70.886621 ms                     1.332500 us
                 512 MB                  932.357764 ms                   141.704667 ms                     1.371500 us
                1024 MB                     1.843786 s                   283.342267 ms                     2.164500 us
```

---

### On `Intel(R) UHD Graphics P630 [0x3e96]`

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

Benchmarking BLAKE3 SYCL implementation (v1)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  403.566750 us                    86.320000 us                     5.395000 us
                   2 MB                  702.138500 us                   117.984500 us                     4.357500 us
                   4 MB                    1.217153 ms                   191.750750 us                     4.502750 us
                   8 MB                    2.193814 ms                   621.213500 us                    12.968750 us
                  16 MB                    3.999957 ms                     1.343189 ms                     7.739750 us
                  32 MB                    7.448918 ms                     3.384491 ms                     4.523500 us
                  64 MB                   14.436294 ms                     5.935828 ms                     5.976000 us
                 128 MB                   28.348629 ms                    12.623117 ms                     5.602500 us
                 256 MB                   56.290641 ms                    21.704043 ms                     5.498750 us
                 512 MB                  111.882983 ms                    42.580390 ms                     6.328750 us
                1024 MB                  223.132739 ms                    60.189857 ms                     8.611250 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=2
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                  413.091000 us                    79.431000 us                    12.844250 us
                   2 MB                  483.910750 us                   120.204750 us                    12.802750 us
                   4 MB                  947.943000 us                   186.687750 us                    12.885750 us
                   8 MB                    1.681455 ms                   573.509250 us                    13.010250 us
                  16 MB                    3.051184 ms                     1.336570 ms                    13.072500 us
                  32 MB                    5.650578 ms                     2.657017 ms                    12.968750 us
                  64 MB                   10.648153 ms                     5.967140 ms                    12.948000 us
                 128 MB                   20.820882 ms                    10.938300 ms                    10.893750 us
                 256 MB                   41.282789 ms                    19.979739 ms                     5.498750 us
                 512 MB                   82.030684 ms                    30.104619 ms                     8.341500 us
                1024 MB                  161.822216 ms                    59.488652 ms                     5.540250 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=4
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    4.147448 ms                    70.176500 us                     6.474000 us
                   2 MB                    4.511714 ms                    86.340750 us                     5.166750 us
                   4 MB                    5.211674 ms                   185.069250 us                     5.353500 us
                   8 MB                   10.751135 ms                   575.542750 us                     4.565000 us
                  16 MB                   20.591345 ms                     1.490701 ms                     4.689500 us
                  32 MB                   37.061741 ms                     3.040622 ms                     6.515500 us
                  64 MB                   70.038409 ms                     5.961600 ms                     6.702250 us
                 128 MB                  135.915177 ms                     7.712111 ms                     6.577750 us
                 256 MB                  268.948137 ms                    15.139988 ms                     6.681500 us
                 512 MB                  535.521976 ms                    36.160486 ms                     7.822750 us
                1024 MB                     1.066891 s                    59.627657 ms                     8.798000 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=8
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                    8.870915 ms                    70.010500 us                     5.498750 us
                   2 MB                    9.908063 ms                   101.820250 us                     5.104500 us
                   4 MB                   10.823345 ms                   193.659750 us                     6.225000 us
                   8 MB                   13.922835 ms                   578.323250 us                     7.241750 us
                  16 MB                   30.877660 ms                     1.238277 ms                     6.079750 us
                  32 MB                   66.345884 ms                     2.846236 ms                     9.856250 us
                  64 MB                  123.892606 ms                     4.746355 ms                     8.818750 us
                 128 MB                  239.482701 ms                     7.742862 ms                     9.711000 us
                 256 MB                  471.554270 ms                    15.200101 ms                     6.328750 us
                 512 MB                  951.620439 ms                    36.128115 ms                    10.790000 us
                1024 MB                     1.897564 s                    59.495666 ms                    13.010250 us
```

```bash
# compiled using BLAKE3_SIMD_LANES=16
Benchmarking BLAKE3 SYCL implementation (v2)

              input size                  execution time                host-to-device tx time          device-to-host tx time
                   1 MB                   21.197121 ms                    84.473250 us                    11.661500 us
                   2 MB                   25.279206 ms                   121.014000 us                    12.865000 us
                   4 MB                   27.317707 ms                   193.369250 us                    11.827500 us
                   8 MB                   30.231816 ms                   582.369500 us                    12.989500 us
                  16 MB                   48.974295 ms                     1.260895 ms                    12.968750 us
                  32 MB                  103.661314 ms                     2.298457 ms                    13.031000 us
                  64 MB                  223.461170 ms                     3.978377 ms                    13.653500 us
                 128 MB                  427.311265 ms                     7.710804 ms                    13.612000 us
                 256 MB                  834.896523 ms                    16.717694 ms                    12.989500 us
                 512 MB                     1.646072 s                    29.985472 ms                    12.906500 us
                1024 MB                     3.256520 s                    59.557854 ms                    12.802750 us
```
