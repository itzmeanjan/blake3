#pragma once
#include <CL/sycl.hpp>
#include <blake3.hpp>

void
prepare_blake3_input(size_t chunk_count, sycl::uchar* const in)
{
  for (size_t i = 0; i < chunk_count; i++) {
    for (size_t j = 0; j < blake3::CHUNK_LEN; j++) {
      *(in + i * blake3::CHUNK_LEN + j) = static_cast<sycl::uchar>(j % 0xff);
    }
  }
}
