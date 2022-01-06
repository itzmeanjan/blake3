#include "bench_blake3.hpp"
#include <iomanip>
#include <iostream>

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

  std::cout << "Benchmarking BLAKE3 SYCL implementation" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(20) << std::right << "execution time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 13; i <<= 1) {
    sycl::cl_ulong ts = benchmark_blake3(q, i, wg_size);

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right << ts * 1e-3 << " us"
              << std::endl;
  }

  return 0;
}
