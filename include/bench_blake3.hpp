#pragma once
#include "blake3.hpp"
#include <random>

void
benchmark_blake3_v1(sycl::queue& q,
                    size_t chunk_count,
                    size_t wg_size,
                    sycl::cl_ulong* const ts)
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

  sycl::cl_ulong ts_0, ts_1, ts_2;

  // host to device data transfer
  sycl::event evt_0 = q.memcpy(i_d, i_h, i_size);
  evt_0.wait();
  ts_0 = time_event(evt_0);

  // compute hash on device
  blake3::v1::hash(q, i_d, i_size, chunk_count, wg_size, o_d, &ts_1);

  // device to host data transfer i.e. get 32 -bytes digest back
  sycl::event evt_1 = q.memcpy(o_h, o_d, o_size);
  evt_1.wait();
  ts_2 = time_event(evt_1);

  sycl::free(i_h, q);
  sycl::free(i_d, q);
  sycl::free(o_h, q);
  sycl::free(o_d, q);

  *(ts + 0) = ts_0; // host to device data transfer cost
  *(ts + 1) = ts_1; // kernel execution cost
  *(ts + 2) = ts_2; // device to host data transfer cost
}

void
benchmark_blake3_v2(sycl::queue& q,
                    size_t chunk_count,
                    size_t wg_size,
                    sycl::cl_ulong* const ts)
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

  sycl::cl_ulong ts_0, ts_1, ts_2;

  // host to device data transfer
  sycl::event evt_0 = q.memcpy(i_d, i_h, i_size);
  evt_0.wait();
  ts_0 = time_event(evt_0);

  // compute hash on device
  blake3::v2::hash(q, i_d, i_size, chunk_count, wg_size, o_d, &ts_1);

  // device to host data transfer i.e. get 32 -bytes digest back
  sycl::event evt_1 = q.memcpy(o_h, o_d, o_size);
  evt_1.wait();
  ts_2 = time_event(evt_1);

  sycl::free(i_h, q);
  sycl::free(i_d, q);
  sycl::free(o_h, q);
  sycl::free(o_d, q);

  *(ts + 0) = ts_0; // host to device data transfer cost
  *(ts + 1) = ts_1; // kernel execution cost
  *(ts + 2) = ts_2; // device to host data transfer cost
}
