#pragma once
#include "blake3.hpp"
#include <random>

sycl::cl_ulong
benchmark_blake3(sycl::queue& q, size_t chunk_count, size_t wg_size)
{
  // current implementation of blake3 only works
  // with power of 2 -number of chunks
  assert((chunk_count & (chunk_count - 1)) == 0);
  assert(wg_size <= chunk_count);

  const size_t i_size = chunk_count * blake3::CHUNK_LEN; // in bytes
  const size_t o_size = blake3::OUT_LEN;                 // in bytes

  sycl::uchar* i_h = static_cast<sycl::uchar*>(sycl::malloc_host(i_size, q));
  sycl::uchar* i_d = static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));

  sycl::uchar* o_h = static_cast<sycl::uchar*>(sycl::malloc_host(o_size, q));
  sycl::uchar* o_d = static_cast<sycl::uchar*>(sycl::malloc_device(o_size, q));

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    memset(i_h, dis(gen), i_size); // prepare (random) input
  }

  sycl::cl_ulong ts = 0; // timing info, ensure queue has profiling enabled

  q.memcpy(i_d, i_h, i_size).wait();
  blake3::hash(q, i_d, i_size, chunk_count, wg_size, o_d, &ts);
  q.memcpy(o_h, o_d, o_size).wait();

  sycl::free(i_h, q);
  sycl::free(i_d, q);
  sycl::free(o_h, q);
  sycl::free(o_d, q);

  return ts;
}
