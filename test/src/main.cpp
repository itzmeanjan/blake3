#include "utils.hpp"
#include <cassert>
#include <iostream>

extern "C" uint8_t*
rust_blake3(uint8_t* input, uint64_t i_size, uint8_t* digest);

int
main()
{
  sycl::device d{ sycl::default_selector{} };
  sycl::queue q{ d };

  const size_t chunk_count = 2;
  const size_t i_size = chunk_count * blake3::CHUNK_LEN;

  uint8_t* in = static_cast<uint8_t*>(malloc(i_size));
  uint8_t* out = static_cast<uint8_t*>(malloc(blake3::OUT_LEN));
  prepare_blake3_input(chunk_count, in);

  sycl::uchar* in_d = static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));
  sycl::uchar* out_h =
    static_cast<sycl::uchar*>(sycl::malloc_host(blake3::OUT_LEN, q));
  sycl::uchar* out_d =
    static_cast<sycl::uchar*>(sycl::malloc_device(blake3::OUT_LEN, q));

  // copy input to device & wait until completed !
  q.memcpy(in_d, in, i_size).wait();

  uint8_t* digest = rust_blake3(in, i_size, out);

  blake3::hash(q, in_d, i_size, chunk_count, chunk_count, out_d);
  // copy output digest into host and wait
  q.memcpy(out_h, out_d, blake3::OUT_LEN).wait();

  for (size_t i = 0; i < blake3::OUT_LEN; i++) {
    assert(*(digest + i) == *(out_h + i));
  }

  std::cout << "passed blake3 tests !" << std::endl;

  sycl::free(in_d, q);
  sycl::free(out_h, q);
  sycl::free(out_d, q);

  return 0;
}
