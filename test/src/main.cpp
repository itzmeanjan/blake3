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

  // input size for testing normal blake3::hash variant
  const size_t i_size = chunk_count * blake3::CHUNK_LEN;
  // input size for testing 2-to-1 hashing blake3::merge
  const size_t merge_i_size = blake3::OUT_LEN << 1;

  // allocating enough memory for generating random input bytes
  // for testing normal blake3::hash
  uint8_t* in = static_cast<uint8_t*>(malloc(i_size));

  // allocating memory so that 64 random bytes can be generated
  // which will be used for testing blake3::merge
  uint8_t* merge_in_0 = static_cast<uint8_t*>(malloc(merge_i_size));
  // randomly generated 64 little endian bytes to be interpreted
  // as sixteen 32 -bit unsigned integers
  uint32_t* merge_in_1 = static_cast<uint32_t*>(malloc(merge_i_size));

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    // setting up random input bytes for hash function
    memset(in, dis(gen), i_size);

    // prepare 64 random bytes for testing blake3::merge
    // function, which applies 2-to-1 hashing on input
    // byte array
    memset(merge_in_0, dis(gen), merge_i_size);
    // from randomly generated little endian input byte array preparing
    // sixteen 32 -bit unsigned integers, which will be passed to
    // blake3::merge function for computing 2-to-1 hash
    words_from_le_bytes(static_cast<sycl::uchar*>(merge_in_0),
                        merge_i_size,
                        static_cast<sycl::uint*>(merge_in_1),
                        merge_i_size >> 2);
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

  // allocate memory host/ device memory to be managed by sycl
  // runtime, which will be used for storing input words on device memory
  // or output words/ bytes on device/ host memory
  sycl::uint* merge_in_d =
    static_cast<sycl::uint*>(sycl::malloc_device(merge_i_size, q));
  sycl::uint* merge_out_h_0 =
    static_cast<sycl::uint*>(sycl::malloc_host(blake3::OUT_LEN, q));
  sycl::uchar* merge_out_h_1 =
    static_cast<sycl::uchar*>(sycl::malloc_host(blake3::OUT_LEN, q));
  sycl::uint* merge_out_d =
    static_cast<sycl::uint*>(sycl::malloc_device(blake3::OUT_LEN, q));

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

  // copy 16 BLAKE3 words to device, which will be used for
  // computing blake3 2-to-1 hash on device
  q.memcpy(merge_in_d, merge_in_1, merge_i_size).wait();

  // compute 2-to-1 hash using RUST blake3 implementation
  uint8_t* merge_digest = rust_blake3(merge_in_0, merge_i_size);
  // enqueue kernel for computing blake3::merge on accelerator
  q.single_task<class kernelTestBLAKE3Merge>([=]() {
     sycl::uint msg_words[16];

#pragma unroll 16
     for (size_t i = 0; i < 16; i++) {
       // I'm copying input because merge function
       // permutes input message words !
       msg_words[i] = *(merge_in_d + i);
     }

     blake3::v1::merge(merge_in_d, merge_out_d);
   })
    .wait();

  // copy blake3 merge computed on device back to host
  q.memcpy(merge_out_h_0, merge_out_d, blake3::OUT_LEN).wait();
  // now convert blake3 digest words into little endian bytes
  words_to_le_bytes(
    merge_out_h_0, blake3::OUT_LEN >> 2, merge_out_h_1, blake3::OUT_LEN);

  for (size_t i = 0; i < blake3::OUT_LEN; i++) {
    assert(*(digest + i) == *(out_h_v1 + i));
    assert(*(digest + i) == *(out_h_v2 + i));
    // assertion for testing whether blake3::merge works
    // as expected or not !
    assert(*(merge_digest + i) == *(merge_out_h_1 + i));
  }

  std::cout << "✅ passed blake3 tests !" << std::endl;

  sycl::free(in_d_v1, q);
  sycl::free(in_d_v2, q);
  sycl::free(out_h_v1, q);
  sycl::free(out_h_v2, q);
  sycl::free(out_d_v1, q);
  sycl::free(out_d_v2, q);
  sycl::free(merge_in_d, q);
  sycl::free(merge_out_h_0, q);
  sycl::free(merge_out_h_1, q);
  sycl::free(merge_out_d, q);

  std::free(merge_in_1);

  return 0;
}
