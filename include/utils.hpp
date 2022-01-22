#pragma once
#include "blake3_consts.hpp"
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

void
words_from_le_bytes(const sycl::uchar* input,
                    size_t i_size,
                    sycl::uint* const msg_words,
                    size_t m_cnt)
{
  // because each message word of BLAKE3 is of 4 -bytes
  assert(i_size == m_cnt << 2);

#pragma unroll 8 // partial unrolling !
  for (size_t i = 0; i < m_cnt; i++) {
    const sycl::uchar* i_start = (input + i * 4);

    *(msg_words + i) = (static_cast<sycl::uint>(*(i_start + 3)) << 24) |
                       (static_cast<sycl::uint>(*(i_start + 2)) << 16) |
                       (static_cast<sycl::uint>(*(i_start + 1)) << 8) |
                       (static_cast<sycl::uint>(*(i_start + 0)) << 0);
  }
}

void
words_to_le_bytes(const sycl::uint* msg_words,
                  size_t m_cnt,
                  sycl::uchar* const output,
                  size_t o_size)
{
  // because each message word of BLAKE3 is of 4 -bytes
  assert(o_size == m_cnt << 2);

#pragma unroll 8 // partial unrolling !
  for (size_t i = 0; i < m_cnt; i++) {
    const sycl::uint num = *(msg_words + i);
    sycl::uchar* out = output + (i << 2);

    *(out + 0) = static_cast<sycl::uchar>((num >> 0) & 0xff);
    *(out + 1) = static_cast<sycl::uchar>((num >> 8) & 0xff);
    *(out + 2) = static_cast<sycl::uchar>((num >> 16) & 0xff);
    *(out + 3) = static_cast<sycl::uchar>((num >> 24) & 0xff);
  }
}
