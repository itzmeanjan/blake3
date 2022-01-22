#include "bench_blake3.hpp"
#include "bench_merklize.hpp"
#include <iomanip>
#include <iostream>

enum BENCH_VARIANT
{
  V1,
  V2,
  MERKLIZATION
};

void
avg_kernel_exec_tm(sycl::queue& q,
                   size_t chunk_count,
                   size_t wg_size,
                   size_t itr_cnt,
                   BENCH_VARIANT variant,
                   double* const ts);

std::string
to_readable_timespan(double ts);

int
main(int argc, char** argv)
{
  sycl::default_selector sel{};
  sycl::device d{ sel };
  // using explicit context, instead of relying on default context created by
  // sycl::queue
  sycl::context ctx{ d };
  // enabling profiling in queue is required when benchmarking blake3
  // implementation
  sycl::queue q{ ctx, d, sycl::property::queue::enable_profiling() };

  std::cout << "running on " << d.get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  const size_t wg_size = 1 << 5;
  const size_t itr_cnt = 1 << 3;

  double* ts = static_cast<double*>(std::malloc(sizeof(double) * 3));

  std::cout << "Benchmarking BLAKE3 SYCL implementation (v1)" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << "\t\t" << std::setw(16) << std::right << "host-to-device tx time"
            << "\t\t" << std::setw(16) << std::right << "device-to-host tx time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    avg_kernel_exec_tm(q, i, wg_size, itr_cnt, BENCH_VARIANT::V1, ts);

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(*(ts + 1)) << "\t\t" << std::setw(22)
              << std::right << to_readable_timespan(*(ts + 0)) << "\t\t"
              << std::setw(22) << std::right << to_readable_timespan(*(ts + 2))
              << std::endl;
  }

  std::cout << "\nBenchmarking BLAKE3 SYCL implementation (v2)" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << "\t\t" << std::setw(16) << std::right << "host-to-device tx time"
            << "\t\t" << std::setw(16) << std::right << "device-to-host tx time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    avg_kernel_exec_tm(q, i, wg_size, itr_cnt, BENCH_VARIANT::V2, ts);

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(*(ts + 1)) << "\t\t" << std::setw(22)
              << std::right << to_readable_timespan(*(ts + 0)) << "\t\t"
              << std::setw(22) << std::right << to_readable_timespan(*(ts + 2))
              << std::endl;
  }

  std::cout << "\nBenchmarking Binary Merklization using BLAKE3" << std::endl
            << std::endl;
  std::cout << std::setw(16) << std::right << "leaf count"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << "\t\t" << std::setw(16) << std::right << "host-to-device tx time"
            << "\t\t" << std::setw(16) << std::right << "device-to-host tx time"
            << std::endl;

  for (size_t i = 20; i <= 25; i++) {
    const size_t leaf_cnt = 1 << i;

    avg_kernel_exec_tm(
      q, leaf_cnt, wg_size, itr_cnt, BENCH_VARIANT::MERKLIZATION, ts);

    std::cout << std::setw(12) << std::right << "2 ^ " << i << "\t\t"
              << std::setw(22) << std::right << to_readable_timespan(*(ts + 1))
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(*(ts + 0)) << "\t\t" << std::setw(22)
              << std::right << to_readable_timespan(*(ts + 2)) << std::endl;
  }

  std::free(ts);

  return 0;
}

void
avg_kernel_exec_tm(
  sycl::queue& q,
  // when bench variant is either of v1/ v2, it's considered to be chunk count
  //
  // while the variant is merklization, it's considered to be leaf count of
  // merkle tree
  size_t chunk_or_leaf_count,
  size_t wg_size,
  size_t itr_cnt,
  BENCH_VARIANT variant,
  double* const ts)
{
  // allocate memory on host
  sycl::cl_ulong* ts_sum =
    static_cast<sycl::cl_ulong*>(std::malloc(sizeof(sycl::cl_ulong) * 3));
  sycl::cl_ulong* ts_rnd =
    static_cast<sycl::cl_ulong*>(std::malloc(sizeof(sycl::cl_ulong) * 3));

  // so that average execution/ data transfer time can be safely computed !
  std::memset(ts_sum, 0, sizeof(sycl::cl_ulong) * 3);

  for (size_t i = 0; i < itr_cnt; i++) {
    if (variant == BENCH_VARIANT::V1) {
      benchmark_blake3_v1(q, chunk_or_leaf_count, wg_size, ts_rnd);
    } else if (variant == BENCH_VARIANT::V2) {
      benchmark_blake3_v2(q, chunk_or_leaf_count, wg_size, ts_rnd);
    } else if (variant == BENCH_VARIANT::MERKLIZATION) {
      benchmark_merklize(q, chunk_or_leaf_count, wg_size, ts_rnd);
    } else {
      throw "can't benchmark unknown variant";
    }

    for (size_t j = 0; j < 3; j++) {
      *(ts_sum + j) += *(ts_rnd + j);
    }
  }

  for (size_t i = 0; i < 3; i++) {
    *(ts + i) = (double)*(ts_sum + i) / (double)itr_cnt;
  }

  // deallocate resources
  std::free(ts_sum);
  std::free(ts_rnd);
}

std::string
to_readable_timespan(double ts)
{
  return ts >= 1e9 ? std::to_string(ts * 1e-9) + " s"
                   : ts >= 1e6 ? std::to_string(ts * 1e-6) + " ms"
                               : ts >= 1e3 ? std::to_string(ts * 1e-3) + " us"
                                           : std::to_string(ts) + " ns";
}
