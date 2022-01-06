#include "blake3.hpp"
#include <cassert>
#include <iostream>
#include <random>

extern "C" uint8_t*
rust_blake3(uint8_t* input, uint64_t i_size);

int
main()
{
  sycl::device d{ sycl::default_selector{} };
  sycl::queue q{ d };

  const size_t chunk_count = 1 << 10;
  const size_t i_size = chunk_count * blake3::CHUNK_LEN;

  uint8_t* in = static_cast<uint8_t*>(malloc(i_size));

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    // setting up input to hash function
    memset(in, dis(gen), i_size);
  }

  sycl::uchar* in_d = static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));
  sycl::uchar* out_h =
    static_cast<sycl::uchar*>(sycl::malloc_host(blake3::OUT_LEN, q));
  sycl::uchar* out_d =
    static_cast<sycl::uchar*>(sycl::malloc_device(blake3::OUT_LEN, q));

  // copy input to device & wait until completed !
  q.memcpy(in_d, in, i_size).wait();

  uint8_t* digest = rust_blake3(in, i_size);
  blake3::hash(q, in_d, i_size, chunk_count, chunk_count, out_d);

  // copy output digest to host and wait until completed !
  q.memcpy(out_h, out_d, blake3::OUT_LEN).wait();

  for (size_t i = 0; i < blake3::OUT_LEN; i++) {
    assert(*(digest + i) == *(out_h + i));
  }

  std::cout << "✅ passed blake3 tests !" << std::endl;

  sycl::free(in_d, q);
  sycl::free(out_h, q);
  sycl::free(out_d, q);

  return 0;
}
