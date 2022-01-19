#include "blake3.hpp"
#include <cassert>
#include <iostream>
#include <random>

extern "C" uint8_t*
rust_blake3(uint8_t* input, uint64_t i_size);

int
main()
{
  sycl::default_selector sel{};
  sycl::device d{ sel };
  // using explicit context, instead of relying on default context created by
  // sycl::queue
  sycl::context ctx{ d };
  sycl::queue q{ ctx, d };

  const size_t chunk_count = 1 << 10;
  const size_t wg_size = 1 << 6;
  const size_t i_size = chunk_count * blake3::CHUNK_LEN;

  uint8_t* in = static_cast<uint8_t*>(malloc(i_size));

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    // setting up input to hash function
    memset(in, dis(gen), i_size);
  }

  sycl::uchar* in_d_v1 =
    static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));
  sycl::uchar* in_d_v2 =
    static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));
  sycl::uchar* out_h_v1 =
    static_cast<sycl::uchar*>(sycl::malloc_host(blake3::OUT_LEN, q));
  sycl::uchar* out_h_v2 =
    static_cast<sycl::uchar*>(sycl::malloc_host(blake3::OUT_LEN, q));
  sycl::uchar* out_d_v1 =
    static_cast<sycl::uchar*>(sycl::malloc_device(blake3::OUT_LEN, q));
  sycl::uchar* out_d_v2 =
    static_cast<sycl::uchar*>(sycl::malloc_device(blake3::OUT_LEN, q));

  // copy input to device; for testing both blake3 implementations
  // wait until completed !
  q.memcpy(in_d_v1, in, i_size).wait();
  q.memcpy(in_d_v2, in, i_size).wait();

  uint8_t* digest = rust_blake3(in, i_size);
  blake3::v1::hash(q, in_d_v1, i_size, chunk_count, wg_size, out_d_v1, nullptr);
  blake3::v2::hash(q, in_d_v2, i_size, chunk_count, wg_size, out_d_v2, nullptr);

  // copy both output digests to host; wait until completed !
  q.memcpy(out_h_v1, out_d_v1, blake3::OUT_LEN).wait();
  q.memcpy(out_h_v2, out_d_v2, blake3::OUT_LEN).wait();

  for (size_t i = 0; i < blake3::OUT_LEN; i++) {
    assert(*(digest + i) == *(out_h_v1 + i));
    assert(*(digest + i) == *(out_h_v2 + i));
  }

  std::cout << "✅ passed blake3 tests !" << std::endl;

  sycl::free(in_d_v1, q);
  sycl::free(in_d_v2, q);
  sycl::free(out_h_v1, q);
  sycl::free(out_h_v2, q);
  sycl::free(out_d_v1, q);
  sycl::free(out_d_v2, q);

  return 0;
}
