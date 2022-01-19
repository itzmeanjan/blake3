#pragma once
#include "blake3.hpp"
#include <CL/sycl.hpp>

void
prepare_blake3_input(size_t chunk_count, sycl::uchar* const in)
{
  for (size_t i = 0; i < chunk_count; i++) {
    for (size_t j = 0; j < blake3::CHUNK_LEN; j++) {
      *(in + i * blake3::CHUNK_LEN + j) = static_cast<sycl::uchar>(j % 0xff);
    }
  }
}

// Ensure that SYCL queue has profiling enabled !
//
// Returns actual execution time of command, submission of which
// resulted into this SYCL event
inline sycl::cl_ulong
time_event(sycl::event& evt)
{
  const sycl::cl_ulong start =
    evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  const sycl::cl_ulong end =
    evt.get_profiling_info<sycl::info::event_profiling::command_end>();

  return (end - start);
}
