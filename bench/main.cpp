#include "bench_blake3.hpp"
#include <iomanip>
#include <iostream>

enum BLAKE3_VARIANT
{
  V1,
  V2
};

sycl::cl_ulong
avg_kernel_exec_tm(sycl::queue& q,
                   size_t chunk_count,
                   size_t wg_size,
                   size_t itr_cnt,
                   BLAKE3_VARIANT variant);

std::string
to_readable_timespan(sycl::cl_ulong ts);

int
main(int argc, char** argv)
{
  sycl::device d{ sycl::default_selector{} };
  // enabling profiling in queue is required when benchmarking blake3
  // implementation
  sycl::queue q{ d, sycl::property::queue::enable_profiling() };

  std::cout << "running on " << d.get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  const size_t wg_size = 1 << 5;
  const size_t itr_cnt = 1 << 2;

  std::cout << "Benchmarking BLAKE3 SYCL implementation (v1)" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    std::string ts = to_readable_timespan(
      avg_kernel_exec_tm(q, i, wg_size, itr_cnt, BLAKE3_VARIANT::V1));

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right << ts << std::endl;
  }

  std::cout << "\nBenchmarking BLAKE3 SYCL implementation (v2)" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    std::string ts = to_readable_timespan(
      avg_kernel_exec_tm(q, i, wg_size, itr_cnt, BLAKE3_VARIANT::V2));

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right << ts << std::endl;
  }

  return 0;
}

sycl::cl_ulong
avg_kernel_exec_tm(sycl::queue& q,
                   size_t chunk_count,
                   size_t wg_size,
                   size_t itr_cnt,
                   BLAKE3_VARIANT variant)
{
  sycl::cl_ulong ts = 0;

  for (size_t i = 0; i < itr_cnt; i++) {
    if (variant == BLAKE3_VARIANT::V1) {
      ts += benchmark_blake3_v1(q, chunk_count, wg_size);
    } else if (variant == BLAKE3_VARIANT::V2) {
      ts += benchmark_blake3_v2(q, chunk_count, wg_size);
    } else {
      throw "can't benchmark unsupported BLAKE3 variant";
    }
  }

  return ts / static_cast<sycl::cl_ulong>(itr_cnt);
}

std::string
to_readable_timespan(sycl::cl_ulong ts)
{
  return ts >= 1e9 ? std::to_string(ts * 1e-9) + " s"
                   : ts >= 1e6 ? std::to_string(ts * 1e-6) + " ms"
                               : ts >= 1e3 ? std::to_string(ts * 1e-3) + " us"
                                           : std::to_string(ts) + " ns";
}
