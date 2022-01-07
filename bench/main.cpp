#include "bench_blake3.hpp"
#include <iomanip>
#include <iostream>

std::string
to_readable_timespan(sycl::ulong ts);

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

  const size_t wg_size = 1 << 6;

  std::cout << "Benchmarking BLAKE3 SYCL implementation (v1)" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    sycl::cl_ulong ts = benchmark_blake3_v1(q, i, wg_size);

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(ts) << std::endl;
  }

  std::cout << "\nBenchmarking BLAKE3 SYCL implementation (v2)" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    sycl::cl_ulong ts = benchmark_blake3_v2(q, i, wg_size);

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(ts) << std::endl;
  }

  return 0;
}

std::string
to_readable_timespan(sycl::ulong ts)
{
  return ts >= 1e9 ? std::to_string(ts * 1e-9) + " s"
                   : ts >= 1e6 ? std::to_string(ts * 1e-6) + " ms"
                               : ts >= 1e3 ? std::to_string(ts * 1e-3) + " us"
                                           : std::to_string(ts) + " ns";
}
