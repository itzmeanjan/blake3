#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>
#include <cassert>

#ifndef BLAKE3_SIMD_LANES
#define BLAKE3_SIMD_LANES 4
#else
#if BLAKE3_SIMD_LANES == 4 || BLAKE3_SIMD_LANES == 8 || BLAKE3_SIMD_LANES == 16
#else
#error Unsupported many SIMD lanes requested; supports only {4, 8, 16}
#endif
#endif

// Just to make sure user knows it's possible to choose SIMD lane count 
// from {4, 8, 16}
#define STR2(x) #x
#define STR(x) STR2(x)
#define PREP_MSG(x) "Compressing " STR(x) " chunks in parallel; see https://github.com/itzmeanjan/blake3/pull/1"
#pragma message PREP_MSG(BLAKE3_SIMD_LANES)

namespace blake3 {
constexpr size_t MSG_PERMUTATION[16] = { 2, 6,  3,  10, 7, 0,  4,  13,
                                         1, 11, 12, 5,  9, 14, 15, 8 };

constexpr sycl::uint IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

constexpr size_t CHUNK_LEN = 1024;
constexpr size_t OUT_LEN = 32;

constexpr size_t ROUNDS = 7;

constexpr sycl::uint BLOCK_LEN = 64;
constexpr sycl::uint CHUNK_START = 1 << 0;
constexpr sycl::uint CHUNK_END = 1 << 1;
constexpr sycl::uint PARENT = 1 << 2;
constexpr sycl::uint ROOT = 1 << 3;

void
permute(sycl::uint* const msg);

void
words_from_le_bytes(const sycl::uchar* input, sycl::uint* const block_words);

void
words_to_le_bytes(const sycl::uint* input, sycl::uchar* const output);

void
parent_cv(const sycl::uint* left_cv,
          const sycl::uint* right_cv,
          const sycl::uint* key_words,
          sycl::uint flags,
          sycl::uint* const out_cv);

void
root_cv(const sycl::uint* left_cv,
        const sycl::uint* right_cv,
        const sycl::uint* key_words,
        sycl::uint* const out_cv);

namespace v1 {
void
round(sycl::uint4* const state, const sycl::uint* msg);

void
compress(const sycl::uint* in_cv,
         sycl::uint* const block_words,
         sycl::ulong counter,
         sycl::uint block_len,
         sycl::uint flags,
         sycl::uint* const out_cv);

void
chunkify(const sycl::uint* key_words,
         sycl::ulong chunk_counter,
         sycl::uint flags,
         const sycl::uchar* input,
         sycl::uint* const out_cv);

void
hash(sycl::queue& q,
     const sycl::uchar* input,
     size_t i_size,
     size_t chunk_count,
     size_t wg_size,
     sycl::uchar* const digest,
     sycl::cl_ulong* const ts);
}

namespace v2 {

void
g(
#if BLAKE3_SIMD_LANES == 4
  sycl::uint4* const state,
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8* const state,
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16* const state,
#endif
  size_t a,
  size_t b,
  size_t c,
  size_t d,
#if BLAKE3_SIMD_LANES == 4
  sycl::uint4 mx,
  sycl::uint4 my
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8 mx,
  sycl::uint8 my
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16 mx,
  sycl::uint16 my
#endif
);

void
round(
#if BLAKE3_SIMD_LANES == 4
  sycl::uint4* const state,
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8* const state,
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16* const state,
#endif
  const sycl::uint* msg);

void
compress(const sycl::uint* in_cv,
         sycl::uint* const block_words,
         sycl::ulong counter,
         sycl::uint block_len,
         sycl::uint flags,
         sycl::uint* const out_cv);

void
chunkify(const sycl::uint* key_words,
         sycl::ulong chunk_counter,
         sycl::uint flags,
         const sycl::uchar* input,
         sycl::uint* const out_cv);

void
hash(sycl::queue& q,
     const sycl::uchar* input,
     size_t i_size,
     size_t chunk_count,
     size_t wg_size,
     sycl::uchar* const digest,
     sycl::cl_ulong* const ts);
}
}

inline void
blake3::v2::g(
#if BLAKE3_SIMD_LANES == 4
  sycl::uint4* const state,
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8* const state,
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16* const state,
#endif
  size_t a,
  size_t b,
  size_t c,
  size_t d,
#if BLAKE3_SIMD_LANES == 4
  sycl::uint4 mx,
  sycl::uint4 my
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8 mx,
  sycl::uint8 my
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16 mx,
  sycl::uint16 my
#endif
)
{

#if BLAKE3_SIMD_LANES == 4
  sycl::uint4 rrot16 = sycl::uint4(16);
  sycl::uint4 rrot12 = sycl::uint4(20);
  sycl::uint4 rrot8 = sycl::uint4(24);
  sycl::uint4 rrot7 = sycl::uint4(25);
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8 rrot16 = sycl::uint8(16);
  sycl::uint8 rrot12 = sycl::uint8(20);
  sycl::uint8 rrot8 = sycl::uint8(24);
  sycl::uint8 rrot7 = sycl::uint8(25);
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16 rrot16 = sycl::uint16(16);
  sycl::uint16 rrot12 = sycl::uint16(20);
  sycl::uint16 rrot8 = sycl::uint16(24);
  sycl::uint16 rrot7 = sycl::uint16(25);
#endif

  *(state + a) = *(state + a) + *(state + b) + mx;
  *(state + d) = sycl::rotate(*(state + d) ^ *(state + a), rrot16);
  *(state + c) = *(state + c) + *(state + d);
  *(state + b) = sycl::rotate(*(state + b) ^ *(state + c), rrot12);
  *(state + a) = *(state + a) + *(state + b) + my;
  *(state + d) = sycl::rotate(*(state + d) ^ *(state + a), rrot8);
  *(state + c) = *(state + c) + *(state + d);
  *(state + b) = sycl::rotate(*(state + b) ^ *(state + c), rrot7);
}

inline void
blake3::v2::round(
#if BLAKE3_SIMD_LANES == 4
  sycl::uint4* const state,
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8* const state,
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16* const state,
#endif
  const sycl::uint* msg)
{
  // column-wise hash state manipulation starts
  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 0),
                                 *(msg + 16 * 1 + 0),
                                 *(msg + 16 * 2 + 0),
                                 *(msg + 16 * 3 + 0));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 1),
                                 *(msg + 16 * 1 + 1),
                                 *(msg + 16 * 2 + 1),
                                 *(msg + 16 * 3 + 1));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 0),
                                 *(msg + 16 * 1 + 0),
                                 *(msg + 16 * 2 + 0),
                                 *(msg + 16 * 3 + 0),
                                 *(msg + 16 * 4 + 0),
                                 *(msg + 16 * 5 + 0),
                                 *(msg + 16 * 6 + 0),
                                 *(msg + 16 * 7 + 0));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 1),
                                 *(msg + 16 * 1 + 1),
                                 *(msg + 16 * 2 + 1),
                                 *(msg + 16 * 3 + 1),
                                 *(msg + 16 * 4 + 1),
                                 *(msg + 16 * 5 + 1),
                                 *(msg + 16 * 6 + 1),
                                 *(msg + 16 * 7 + 1));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 0),
                                   *(msg + 16 * 1 + 0),
                                   *(msg + 16 * 2 + 0),
                                   *(msg + 16 * 3 + 0),
                                   *(msg + 16 * 4 + 0),
                                   *(msg + 16 * 5 + 0),
                                   *(msg + 16 * 6 + 0),
                                   *(msg + 16 * 7 + 0),
                                   *(msg + 16 * 8 + 0),
                                   *(msg + 16 * 9 + 0),
                                   *(msg + 16 * 10 + 0),
                                   *(msg + 16 * 11 + 0),
                                   *(msg + 16 * 12 + 0),
                                   *(msg + 16 * 13 + 0),
                                   *(msg + 16 * 14 + 0),
                                   *(msg + 16 * 15 + 0));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 1),
                                   *(msg + 16 * 1 + 1),
                                   *(msg + 16 * 2 + 1),
                                   *(msg + 16 * 3 + 1),
                                   *(msg + 16 * 4 + 1),
                                   *(msg + 16 * 5 + 1),
                                   *(msg + 16 * 6 + 1),
                                   *(msg + 16 * 7 + 1),
                                   *(msg + 16 * 8 + 1),
                                   *(msg + 16 * 9 + 1),
                                   *(msg + 16 * 10 + 1),
                                   *(msg + 16 * 11 + 1),
                                   *(msg + 16 * 12 + 1),
                                   *(msg + 16 * 13 + 1),
                                   *(msg + 16 * 14 + 1),
                                   *(msg + 16 * 15 + 1));
#endif

    blake3::v2::g(state, 0, 4, 8, 12, mx, my);
  }

  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 2),
                                 *(msg + 16 * 1 + 2),
                                 *(msg + 16 * 2 + 2),
                                 *(msg + 16 * 3 + 2));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 3),
                                 *(msg + 16 * 1 + 3),
                                 *(msg + 16 * 2 + 3),
                                 *(msg + 16 * 3 + 3));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 2),
                                 *(msg + 16 * 1 + 2),
                                 *(msg + 16 * 2 + 2),
                                 *(msg + 16 * 3 + 2),
                                 *(msg + 16 * 4 + 2),
                                 *(msg + 16 * 5 + 2),
                                 *(msg + 16 * 6 + 2),
                                 *(msg + 16 * 7 + 2));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 3),
                                 *(msg + 16 * 1 + 3),
                                 *(msg + 16 * 2 + 3),
                                 *(msg + 16 * 3 + 3),
                                 *(msg + 16 * 4 + 3),
                                 *(msg + 16 * 5 + 3),
                                 *(msg + 16 * 6 + 3),
                                 *(msg + 16 * 7 + 3));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 2),
                                   *(msg + 16 * 1 + 2),
                                   *(msg + 16 * 2 + 2),
                                   *(msg + 16 * 3 + 2),
                                   *(msg + 16 * 4 + 2),
                                   *(msg + 16 * 5 + 2),
                                   *(msg + 16 * 6 + 2),
                                   *(msg + 16 * 7 + 2),
                                   *(msg + 16 * 8 + 2),
                                   *(msg + 16 * 9 + 2),
                                   *(msg + 16 * 10 + 2),
                                   *(msg + 16 * 11 + 2),
                                   *(msg + 16 * 12 + 2),
                                   *(msg + 16 * 13 + 2),
                                   *(msg + 16 * 14 + 2),
                                   *(msg + 16 * 15 + 2));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 3),
                                   *(msg + 16 * 1 + 3),
                                   *(msg + 16 * 2 + 3),
                                   *(msg + 16 * 3 + 3),
                                   *(msg + 16 * 4 + 3),
                                   *(msg + 16 * 5 + 3),
                                   *(msg + 16 * 6 + 3),
                                   *(msg + 16 * 7 + 3),
                                   *(msg + 16 * 8 + 3),
                                   *(msg + 16 * 9 + 3),
                                   *(msg + 16 * 10 + 3),
                                   *(msg + 16 * 11 + 3),
                                   *(msg + 16 * 12 + 3),
                                   *(msg + 16 * 13 + 3),
                                   *(msg + 16 * 14 + 3),
                                   *(msg + 16 * 15 + 3));
#endif

    blake3::v2::g(state, 1, 5, 9, 13, mx, my);
  }

  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 4),
                                 *(msg + 16 * 1 + 4),
                                 *(msg + 16 * 2 + 4),
                                 *(msg + 16 * 3 + 4));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 5),
                                 *(msg + 16 * 1 + 5),
                                 *(msg + 16 * 2 + 5),
                                 *(msg + 16 * 3 + 5));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 4),
                                 *(msg + 16 * 1 + 4),
                                 *(msg + 16 * 2 + 4),
                                 *(msg + 16 * 3 + 4),
                                 *(msg + 16 * 4 + 4),
                                 *(msg + 16 * 5 + 4),
                                 *(msg + 16 * 6 + 4),
                                 *(msg + 16 * 7 + 4));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 5),
                                 *(msg + 16 * 1 + 5),
                                 *(msg + 16 * 2 + 5),
                                 *(msg + 16 * 3 + 5),
                                 *(msg + 16 * 4 + 5),
                                 *(msg + 16 * 5 + 5),
                                 *(msg + 16 * 6 + 5),
                                 *(msg + 16 * 7 + 5));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 4),
                                   *(msg + 16 * 1 + 4),
                                   *(msg + 16 * 2 + 4),
                                   *(msg + 16 * 3 + 4),
                                   *(msg + 16 * 4 + 4),
                                   *(msg + 16 * 5 + 4),
                                   *(msg + 16 * 6 + 4),
                                   *(msg + 16 * 7 + 4),
                                   *(msg + 16 * 8 + 4),
                                   *(msg + 16 * 9 + 4),
                                   *(msg + 16 * 10 + 4),
                                   *(msg + 16 * 11 + 4),
                                   *(msg + 16 * 12 + 4),
                                   *(msg + 16 * 13 + 4),
                                   *(msg + 16 * 14 + 4),
                                   *(msg + 16 * 15 + 4));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 5),
                                   *(msg + 16 * 1 + 5),
                                   *(msg + 16 * 2 + 5),
                                   *(msg + 16 * 3 + 5),
                                   *(msg + 16 * 4 + 5),
                                   *(msg + 16 * 5 + 5),
                                   *(msg + 16 * 6 + 5),
                                   *(msg + 16 * 7 + 5),
                                   *(msg + 16 * 8 + 5),
                                   *(msg + 16 * 9 + 5),
                                   *(msg + 16 * 10 + 5),
                                   *(msg + 16 * 11 + 5),
                                   *(msg + 16 * 12 + 5),
                                   *(msg + 16 * 13 + 5),
                                   *(msg + 16 * 14 + 5),
                                   *(msg + 16 * 15 + 5));
#endif

    blake3::v2::g(state, 2, 6, 10, 14, mx, my);
  }

  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 6),
                                 *(msg + 16 * 1 + 6),
                                 *(msg + 16 * 2 + 6),
                                 *(msg + 16 * 3 + 6));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 7),
                                 *(msg + 16 * 1 + 7),
                                 *(msg + 16 * 2 + 7),
                                 *(msg + 16 * 3 + 7));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 6),
                                 *(msg + 16 * 1 + 6),
                                 *(msg + 16 * 2 + 6),
                                 *(msg + 16 * 3 + 6),
                                 *(msg + 16 * 4 + 6),
                                 *(msg + 16 * 5 + 6),
                                 *(msg + 16 * 6 + 6),
                                 *(msg + 16 * 7 + 6));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 7),
                                 *(msg + 16 * 1 + 7),
                                 *(msg + 16 * 2 + 7),
                                 *(msg + 16 * 3 + 7),
                                 *(msg + 16 * 4 + 7),
                                 *(msg + 16 * 5 + 7),
                                 *(msg + 16 * 6 + 7),
                                 *(msg + 16 * 7 + 7));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 6),
                                   *(msg + 16 * 1 + 6),
                                   *(msg + 16 * 2 + 6),
                                   *(msg + 16 * 3 + 6),
                                   *(msg + 16 * 4 + 6),
                                   *(msg + 16 * 5 + 6),
                                   *(msg + 16 * 6 + 6),
                                   *(msg + 16 * 7 + 6),
                                   *(msg + 16 * 8 + 6),
                                   *(msg + 16 * 9 + 6),
                                   *(msg + 16 * 10 + 6),
                                   *(msg + 16 * 11 + 6),
                                   *(msg + 16 * 12 + 6),
                                   *(msg + 16 * 13 + 6),
                                   *(msg + 16 * 14 + 6),
                                   *(msg + 16 * 15 + 6));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 7),
                                   *(msg + 16 * 1 + 7),
                                   *(msg + 16 * 2 + 7),
                                   *(msg + 16 * 3 + 7),
                                   *(msg + 16 * 4 + 7),
                                   *(msg + 16 * 5 + 7),
                                   *(msg + 16 * 6 + 7),
                                   *(msg + 16 * 7 + 7),
                                   *(msg + 16 * 8 + 7),
                                   *(msg + 16 * 9 + 7),
                                   *(msg + 16 * 10 + 7),
                                   *(msg + 16 * 11 + 7),
                                   *(msg + 16 * 12 + 7),
                                   *(msg + 16 * 13 + 7),
                                   *(msg + 16 * 14 + 7),
                                   *(msg + 16 * 15 + 7));
#endif

    blake3::v2::g(state, 3, 7, 11, 15, mx, my);
  }
  // column-wise hash state manipulation ends

  // diagonal hash state manipulation starts
  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 8),
                                 *(msg + 16 * 1 + 8),
                                 *(msg + 16 * 2 + 8),
                                 *(msg + 16 * 3 + 8));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 9),
                                 *(msg + 16 * 1 + 9),
                                 *(msg + 16 * 2 + 9),
                                 *(msg + 16 * 3 + 9));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 8),
                                 *(msg + 16 * 1 + 8),
                                 *(msg + 16 * 2 + 8),
                                 *(msg + 16 * 3 + 8),
                                 *(msg + 16 * 4 + 8),
                                 *(msg + 16 * 5 + 8),
                                 *(msg + 16 * 6 + 8),
                                 *(msg + 16 * 7 + 8));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 9),
                                 *(msg + 16 * 1 + 9),
                                 *(msg + 16 * 2 + 9),
                                 *(msg + 16 * 3 + 9),
                                 *(msg + 16 * 4 + 9),
                                 *(msg + 16 * 5 + 9),
                                 *(msg + 16 * 6 + 9),
                                 *(msg + 16 * 7 + 9));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 8),
                                   *(msg + 16 * 1 + 8),
                                   *(msg + 16 * 2 + 8),
                                   *(msg + 16 * 3 + 8),
                                   *(msg + 16 * 4 + 8),
                                   *(msg + 16 * 5 + 8),
                                   *(msg + 16 * 6 + 8),
                                   *(msg + 16 * 7 + 8),
                                   *(msg + 16 * 8 + 8),
                                   *(msg + 16 * 9 + 8),
                                   *(msg + 16 * 10 + 8),
                                   *(msg + 16 * 11 + 8),
                                   *(msg + 16 * 12 + 8),
                                   *(msg + 16 * 13 + 8),
                                   *(msg + 16 * 14 + 8),
                                   *(msg + 16 * 15 + 8));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 9),
                                   *(msg + 16 * 1 + 9),
                                   *(msg + 16 * 2 + 9),
                                   *(msg + 16 * 3 + 9),
                                   *(msg + 16 * 4 + 9),
                                   *(msg + 16 * 5 + 9),
                                   *(msg + 16 * 6 + 9),
                                   *(msg + 16 * 7 + 9),
                                   *(msg + 16 * 8 + 9),
                                   *(msg + 16 * 9 + 9),
                                   *(msg + 16 * 10 + 9),
                                   *(msg + 16 * 11 + 9),
                                   *(msg + 16 * 12 + 9),
                                   *(msg + 16 * 13 + 9),
                                   *(msg + 16 * 14 + 9),
                                   *(msg + 16 * 15 + 9));
#endif

    blake3::v2::g(state, 0, 5, 10, 15, mx, my);
  }

  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 10),
                                 *(msg + 16 * 1 + 10),
                                 *(msg + 16 * 2 + 10),
                                 *(msg + 16 * 3 + 10));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 11),
                                 *(msg + 16 * 1 + 11),
                                 *(msg + 16 * 2 + 11),
                                 *(msg + 16 * 3 + 11));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 10),
                                 *(msg + 16 * 1 + 10),
                                 *(msg + 16 * 2 + 10),
                                 *(msg + 16 * 3 + 10),
                                 *(msg + 16 * 4 + 10),
                                 *(msg + 16 * 5 + 10),
                                 *(msg + 16 * 6 + 10),
                                 *(msg + 16 * 7 + 10));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 11),
                                 *(msg + 16 * 1 + 11),
                                 *(msg + 16 * 2 + 11),
                                 *(msg + 16 * 3 + 11),
                                 *(msg + 16 * 4 + 11),
                                 *(msg + 16 * 5 + 11),
                                 *(msg + 16 * 6 + 11),
                                 *(msg + 16 * 7 + 11));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 10),
                                   *(msg + 16 * 1 + 10),
                                   *(msg + 16 * 2 + 10),
                                   *(msg + 16 * 3 + 10),
                                   *(msg + 16 * 4 + 10),
                                   *(msg + 16 * 5 + 10),
                                   *(msg + 16 * 6 + 10),
                                   *(msg + 16 * 7 + 10),
                                   *(msg + 16 * 8 + 10),
                                   *(msg + 16 * 9 + 10),
                                   *(msg + 16 * 10 + 10),
                                   *(msg + 16 * 11 + 10),
                                   *(msg + 16 * 12 + 10),
                                   *(msg + 16 * 13 + 10),
                                   *(msg + 16 * 14 + 10),
                                   *(msg + 16 * 15 + 10));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 11),
                                   *(msg + 16 * 1 + 11),
                                   *(msg + 16 * 2 + 11),
                                   *(msg + 16 * 3 + 11),
                                   *(msg + 16 * 4 + 11),
                                   *(msg + 16 * 5 + 11),
                                   *(msg + 16 * 6 + 11),
                                   *(msg + 16 * 7 + 11),
                                   *(msg + 16 * 8 + 11),
                                   *(msg + 16 * 9 + 11),
                                   *(msg + 16 * 10 + 11),
                                   *(msg + 16 * 11 + 11),
                                   *(msg + 16 * 12 + 11),
                                   *(msg + 16 * 13 + 11),
                                   *(msg + 16 * 14 + 11),
                                   *(msg + 16 * 15 + 11));
#endif

    blake3::v2::g(state, 1, 6, 11, 12, mx, my);
  }

  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 12),
                                 *(msg + 16 * 1 + 12),
                                 *(msg + 16 * 2 + 12),
                                 *(msg + 16 * 3 + 12));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 13),
                                 *(msg + 16 * 1 + 13),
                                 *(msg + 16 * 2 + 13),
                                 *(msg + 16 * 3 + 13));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 12),
                                 *(msg + 16 * 1 + 12),
                                 *(msg + 16 * 2 + 12),
                                 *(msg + 16 * 3 + 12),
                                 *(msg + 16 * 4 + 12),
                                 *(msg + 16 * 5 + 12),
                                 *(msg + 16 * 6 + 12),
                                 *(msg + 16 * 7 + 12));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 13),
                                 *(msg + 16 * 1 + 13),
                                 *(msg + 16 * 2 + 13),
                                 *(msg + 16 * 3 + 13),
                                 *(msg + 16 * 4 + 13),
                                 *(msg + 16 * 5 + 13),
                                 *(msg + 16 * 6 + 13),
                                 *(msg + 16 * 7 + 13));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 12),
                                   *(msg + 16 * 1 + 12),
                                   *(msg + 16 * 2 + 12),
                                   *(msg + 16 * 3 + 12),
                                   *(msg + 16 * 4 + 12),
                                   *(msg + 16 * 5 + 12),
                                   *(msg + 16 * 6 + 12),
                                   *(msg + 16 * 7 + 12),
                                   *(msg + 16 * 8 + 12),
                                   *(msg + 16 * 9 + 12),
                                   *(msg + 16 * 10 + 12),
                                   *(msg + 16 * 11 + 12),
                                   *(msg + 16 * 12 + 12),
                                   *(msg + 16 * 13 + 12),
                                   *(msg + 16 * 14 + 12),
                                   *(msg + 16 * 15 + 12));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 13),
                                   *(msg + 16 * 1 + 13),
                                   *(msg + 16 * 2 + 13),
                                   *(msg + 16 * 3 + 13),
                                   *(msg + 16 * 4 + 13),
                                   *(msg + 16 * 5 + 13),
                                   *(msg + 16 * 6 + 13),
                                   *(msg + 16 * 7 + 13),
                                   *(msg + 16 * 8 + 13),
                                   *(msg + 16 * 9 + 13),
                                   *(msg + 16 * 10 + 13),
                                   *(msg + 16 * 11 + 13),
                                   *(msg + 16 * 12 + 13),
                                   *(msg + 16 * 13 + 13),
                                   *(msg + 16 * 14 + 13),
                                   *(msg + 16 * 15 + 13));
#endif

    blake3::v2::g(state, 2, 7, 8, 13, mx, my);
  }

  {
#if BLAKE3_SIMD_LANES == 4
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 14),
                                 *(msg + 16 * 1 + 14),
                                 *(msg + 16 * 2 + 14),
                                 *(msg + 16 * 3 + 14));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 15),
                                 *(msg + 16 * 1 + 15),
                                 *(msg + 16 * 2 + 15),
                                 *(msg + 16 * 3 + 15));
#elif BLAKE3_SIMD_LANES == 8
    sycl::uint8 mx = sycl::uint8(*(msg + 16 * 0 + 14),
                                 *(msg + 16 * 1 + 14),
                                 *(msg + 16 * 2 + 14),
                                 *(msg + 16 * 3 + 14),
                                 *(msg + 16 * 4 + 14),
                                 *(msg + 16 * 5 + 14),
                                 *(msg + 16 * 6 + 14),
                                 *(msg + 16 * 7 + 14));
    sycl::uint8 my = sycl::uint8(*(msg + 16 * 0 + 15),
                                 *(msg + 16 * 1 + 15),
                                 *(msg + 16 * 2 + 15),
                                 *(msg + 16 * 3 + 15),
                                 *(msg + 16 * 4 + 15),
                                 *(msg + 16 * 5 + 15),
                                 *(msg + 16 * 6 + 15),
                                 *(msg + 16 * 7 + 15));
#elif BLAKE3_SIMD_LANES == 16
    sycl::uint16 mx = sycl::uint16(*(msg + 16 * 0 + 14),
                                   *(msg + 16 * 1 + 14),
                                   *(msg + 16 * 2 + 14),
                                   *(msg + 16 * 3 + 14),
                                   *(msg + 16 * 4 + 14),
                                   *(msg + 16 * 5 + 14),
                                   *(msg + 16 * 6 + 14),
                                   *(msg + 16 * 7 + 14),
                                   *(msg + 16 * 8 + 14),
                                   *(msg + 16 * 9 + 14),
                                   *(msg + 16 * 10 + 14),
                                   *(msg + 16 * 11 + 14),
                                   *(msg + 16 * 12 + 14),
                                   *(msg + 16 * 13 + 14),
                                   *(msg + 16 * 14 + 14),
                                   *(msg + 16 * 15 + 14));
    sycl::uint16 my = sycl::uint16(*(msg + 16 * 0 + 15),
                                   *(msg + 16 * 1 + 15),
                                   *(msg + 16 * 2 + 15),
                                   *(msg + 16 * 3 + 15),
                                   *(msg + 16 * 4 + 15),
                                   *(msg + 16 * 5 + 15),
                                   *(msg + 16 * 6 + 15),
                                   *(msg + 16 * 7 + 15),
                                   *(msg + 16 * 8 + 15),
                                   *(msg + 16 * 9 + 15),
                                   *(msg + 16 * 10 + 15),
                                   *(msg + 16 * 11 + 15),
                                   *(msg + 16 * 12 + 15),
                                   *(msg + 16 * 13 + 15),
                                   *(msg + 16 * 14 + 15),
                                   *(msg + 16 * 15 + 15));
#endif

    blake3::v2::g(state, 3, 4, 9, 14, mx, my);
  }
  // diagonal hash state manipulation ends
}

void
blake3::v2::compress(const sycl::uint* in_cv,
                     sycl::uint* const block_words,
                     sycl::ulong counter,
                     sycl::uint block_len,
                     sycl::uint flags,
                     sycl::uint* const out_cv)
{
  // hash state of 4/ 8/ 16 chunks; to be processed in parallel ( clustered
  // together )
  //
  // See section 5.3 of Blake3 specification for understanding
  // how this SIMD technique can be applied for arbitrary many SIMD lanes

#if BLAKE3_SIMD_LANES == 4
  sycl::uint4 state[16] = {
    sycl::uint4(*(in_cv + 8 * 0 + 0),
                *(in_cv + 8 * 1 + 0),
                *(in_cv + 8 * 2 + 0),
                *(in_cv + 8 * 3 + 0)),
    sycl::uint4(*(in_cv + 8 * 0 + 1),
                *(in_cv + 8 * 1 + 1),
                *(in_cv + 8 * 2 + 1),
                *(in_cv + 8 * 3 + 1)),
    sycl::uint4(*(in_cv + 8 * 0 + 2),
                *(in_cv + 8 * 1 + 2),
                *(in_cv + 8 * 2 + 2),
                *(in_cv + 8 * 3 + 2)),
    sycl::uint4(*(in_cv + 8 * 0 + 3),
                *(in_cv + 8 * 1 + 3),
                *(in_cv + 8 * 2 + 3),
                *(in_cv + 8 * 3 + 3)),
    sycl::uint4(*(in_cv + 8 * 0 + 4),
                *(in_cv + 8 * 1 + 4),
                *(in_cv + 8 * 2 + 4),
                *(in_cv + 8 * 3 + 4)),
    sycl::uint4(*(in_cv + 8 * 0 + 5),
                *(in_cv + 8 * 1 + 5),
                *(in_cv + 8 * 2 + 5),
                *(in_cv + 8 * 3 + 5)),
    sycl::uint4(*(in_cv + 8 * 0 + 6),
                *(in_cv + 8 * 1 + 6),
                *(in_cv + 8 * 2 + 6),
                *(in_cv + 8 * 3 + 6)),
    sycl::uint4(*(in_cv + 8 * 0 + 7),
                *(in_cv + 8 * 1 + 7),
                *(in_cv + 8 * 2 + 7),
                *(in_cv + 8 * 3 + 7)),
    sycl::uint4(IV[0]),
    sycl::uint4(IV[1]),
    sycl::uint4(IV[2]),
    sycl::uint4(IV[3]),
    sycl::uint4(static_cast<sycl::uint>((counter + 0) & 0xffffffff),
                static_cast<sycl::uint>((counter + 1) & 0xffffffff),
                static_cast<sycl::uint>((counter + 2) & 0xffffffff),
                static_cast<sycl::uint>((counter + 3) & 0xffffffff)),
    sycl::uint4(static_cast<sycl::uint>((counter + 0) >> 32),
                static_cast<sycl::uint>((counter + 1) >> 32),
                static_cast<sycl::uint>((counter + 2) >> 32),
                static_cast<sycl::uint>((counter + 3) >> 32)),
    sycl::uint4(block_len),
    sycl::uint4(flags)
  };
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint8 state[16] = {
    sycl::uint8(*(in_cv + 8 * 0 + 0),
                *(in_cv + 8 * 1 + 0),
                *(in_cv + 8 * 2 + 0),
                *(in_cv + 8 * 3 + 0),
                *(in_cv + 8 * 4 + 0),
                *(in_cv + 8 * 5 + 0),
                *(in_cv + 8 * 6 + 0),
                *(in_cv + 8 * 7 + 0)),
    sycl::uint8(*(in_cv + 8 * 0 + 1),
                *(in_cv + 8 * 1 + 1),
                *(in_cv + 8 * 2 + 1),
                *(in_cv + 8 * 3 + 1),
                *(in_cv + 8 * 4 + 1),
                *(in_cv + 8 * 5 + 1),
                *(in_cv + 8 * 6 + 1),
                *(in_cv + 8 * 7 + 1)),
    sycl::uint8(*(in_cv + 8 * 0 + 2),
                *(in_cv + 8 * 1 + 2),
                *(in_cv + 8 * 2 + 2),
                *(in_cv + 8 * 3 + 2),
                *(in_cv + 8 * 4 + 2),
                *(in_cv + 8 * 5 + 2),
                *(in_cv + 8 * 6 + 2),
                *(in_cv + 8 * 7 + 2)),
    sycl::uint8(*(in_cv + 8 * 0 + 3),
                *(in_cv + 8 * 1 + 3),
                *(in_cv + 8 * 2 + 3),
                *(in_cv + 8 * 3 + 3),
                *(in_cv + 8 * 4 + 3),
                *(in_cv + 8 * 5 + 3),
                *(in_cv + 8 * 6 + 3),
                *(in_cv + 8 * 7 + 3)),
    sycl::uint8(*(in_cv + 8 * 0 + 4),
                *(in_cv + 8 * 1 + 4),
                *(in_cv + 8 * 2 + 4),
                *(in_cv + 8 * 3 + 4),
                *(in_cv + 8 * 4 + 4),
                *(in_cv + 8 * 5 + 4),
                *(in_cv + 8 * 6 + 4),
                *(in_cv + 8 * 7 + 4)),
    sycl::uint8(*(in_cv + 8 * 0 + 5),
                *(in_cv + 8 * 1 + 5),
                *(in_cv + 8 * 2 + 5),
                *(in_cv + 8 * 3 + 5),
                *(in_cv + 8 * 4 + 5),
                *(in_cv + 8 * 5 + 5),
                *(in_cv + 8 * 6 + 5),
                *(in_cv + 8 * 7 + 5)),
    sycl::uint8(*(in_cv + 8 * 0 + 6),
                *(in_cv + 8 * 1 + 6),
                *(in_cv + 8 * 2 + 6),
                *(in_cv + 8 * 3 + 6),
                *(in_cv + 8 * 4 + 6),
                *(in_cv + 8 * 5 + 6),
                *(in_cv + 8 * 6 + 6),
                *(in_cv + 8 * 7 + 6)),
    sycl::uint8(*(in_cv + 8 * 0 + 7),
                *(in_cv + 8 * 1 + 7),
                *(in_cv + 8 * 2 + 7),
                *(in_cv + 8 * 3 + 7),
                *(in_cv + 8 * 4 + 7),
                *(in_cv + 8 * 5 + 7),
                *(in_cv + 8 * 6 + 7),
                *(in_cv + 8 * 7 + 7)),
    sycl::uint8(IV[0]),
    sycl::uint8(IV[1]),
    sycl::uint8(IV[2]),
    sycl::uint8(IV[3]),
    sycl::uint8(static_cast<sycl::uint>((counter + 0) & 0xffffffff),
                static_cast<sycl::uint>((counter + 1) & 0xffffffff),
                static_cast<sycl::uint>((counter + 2) & 0xffffffff),
                static_cast<sycl::uint>((counter + 3) & 0xffffffff),
                static_cast<sycl::uint>((counter + 4) & 0xffffffff),
                static_cast<sycl::uint>((counter + 5) & 0xffffffff),
                static_cast<sycl::uint>((counter + 6) & 0xffffffff),
                static_cast<sycl::uint>((counter + 7) & 0xffffffff)),
    sycl::uint8(static_cast<sycl::uint>((counter + 0) >> 32),
                static_cast<sycl::uint>((counter + 1) >> 32),
                static_cast<sycl::uint>((counter + 2) >> 32),
                static_cast<sycl::uint>((counter + 3) >> 32),
                static_cast<sycl::uint>((counter + 4) >> 32),
                static_cast<sycl::uint>((counter + 5) >> 32),
                static_cast<sycl::uint>((counter + 6) >> 32),
                static_cast<sycl::uint>((counter + 7) >> 32)),
    sycl::uint8(block_len),
    sycl::uint8(flags)
  };
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint16 state[16] = {
    sycl::uint16(*(in_cv + 8 * 0 + 0),
                 *(in_cv + 8 * 1 + 0),
                 *(in_cv + 8 * 2 + 0),
                 *(in_cv + 8 * 3 + 0),
                 *(in_cv + 8 * 4 + 0),
                 *(in_cv + 8 * 5 + 0),
                 *(in_cv + 8 * 6 + 0),
                 *(in_cv + 8 * 7 + 0),
                 *(in_cv + 8 * 8 + 0),
                 *(in_cv + 8 * 9 + 0),
                 *(in_cv + 8 * 10 + 0),
                 *(in_cv + 8 * 11 + 0),
                 *(in_cv + 8 * 12 + 0),
                 *(in_cv + 8 * 13 + 0),
                 *(in_cv + 8 * 14 + 0),
                 *(in_cv + 8 * 15 + 0)),
    sycl::uint16(*(in_cv + 8 * 0 + 1),
                 *(in_cv + 8 * 1 + 1),
                 *(in_cv + 8 * 2 + 1),
                 *(in_cv + 8 * 3 + 1),
                 *(in_cv + 8 * 4 + 1),
                 *(in_cv + 8 * 5 + 1),
                 *(in_cv + 8 * 6 + 1),
                 *(in_cv + 8 * 7 + 1),
                 *(in_cv + 8 * 8 + 1),
                 *(in_cv + 8 * 9 + 1),
                 *(in_cv + 8 * 10 + 1),
                 *(in_cv + 8 * 11 + 1),
                 *(in_cv + 8 * 12 + 1),
                 *(in_cv + 8 * 13 + 1),
                 *(in_cv + 8 * 14 + 1),
                 *(in_cv + 8 * 15 + 1)),
    sycl::uint16(*(in_cv + 8 * 0 + 2),
                 *(in_cv + 8 * 1 + 2),
                 *(in_cv + 8 * 2 + 2),
                 *(in_cv + 8 * 3 + 2),
                 *(in_cv + 8 * 4 + 2),
                 *(in_cv + 8 * 5 + 2),
                 *(in_cv + 8 * 6 + 2),
                 *(in_cv + 8 * 7 + 2),
                 *(in_cv + 8 * 8 + 2),
                 *(in_cv + 8 * 9 + 2),
                 *(in_cv + 8 * 10 + 2),
                 *(in_cv + 8 * 11 + 2),
                 *(in_cv + 8 * 12 + 2),
                 *(in_cv + 8 * 13 + 2),
                 *(in_cv + 8 * 14 + 2),
                 *(in_cv + 8 * 15 + 2)),
    sycl::uint16(*(in_cv + 8 * 0 + 3),
                 *(in_cv + 8 * 1 + 3),
                 *(in_cv + 8 * 2 + 3),
                 *(in_cv + 8 * 3 + 3),
                 *(in_cv + 8 * 4 + 3),
                 *(in_cv + 8 * 5 + 3),
                 *(in_cv + 8 * 6 + 3),
                 *(in_cv + 8 * 7 + 3),
                 *(in_cv + 8 * 8 + 3),
                 *(in_cv + 8 * 9 + 3),
                 *(in_cv + 8 * 10 + 3),
                 *(in_cv + 8 * 11 + 3),
                 *(in_cv + 8 * 12 + 3),
                 *(in_cv + 8 * 13 + 3),
                 *(in_cv + 8 * 14 + 3),
                 *(in_cv + 8 * 15 + 3)),
    sycl::uint16(*(in_cv + 8 * 0 + 4),
                 *(in_cv + 8 * 1 + 4),
                 *(in_cv + 8 * 2 + 4),
                 *(in_cv + 8 * 3 + 4),
                 *(in_cv + 8 * 4 + 4),
                 *(in_cv + 8 * 5 + 4),
                 *(in_cv + 8 * 6 + 4),
                 *(in_cv + 8 * 7 + 4),
                 *(in_cv + 8 * 8 + 4),
                 *(in_cv + 8 * 9 + 4),
                 *(in_cv + 8 * 10 + 4),
                 *(in_cv + 8 * 11 + 4),
                 *(in_cv + 8 * 12 + 4),
                 *(in_cv + 8 * 13 + 4),
                 *(in_cv + 8 * 14 + 4),
                 *(in_cv + 8 * 15 + 4)),
    sycl::uint16(*(in_cv + 8 * 0 + 5),
                 *(in_cv + 8 * 1 + 5),
                 *(in_cv + 8 * 2 + 5),
                 *(in_cv + 8 * 3 + 5),
                 *(in_cv + 8 * 4 + 5),
                 *(in_cv + 8 * 5 + 5),
                 *(in_cv + 8 * 6 + 5),
                 *(in_cv + 8 * 7 + 5),
                 *(in_cv + 8 * 8 + 5),
                 *(in_cv + 8 * 9 + 5),
                 *(in_cv + 8 * 10 + 5),
                 *(in_cv + 8 * 11 + 5),
                 *(in_cv + 8 * 12 + 5),
                 *(in_cv + 8 * 13 + 5),
                 *(in_cv + 8 * 14 + 5),
                 *(in_cv + 8 * 15 + 5)),
    sycl::uint16(*(in_cv + 8 * 0 + 6),
                 *(in_cv + 8 * 1 + 6),
                 *(in_cv + 8 * 2 + 6),
                 *(in_cv + 8 * 3 + 6),
                 *(in_cv + 8 * 4 + 6),
                 *(in_cv + 8 * 5 + 6),
                 *(in_cv + 8 * 6 + 6),
                 *(in_cv + 8 * 7 + 6),
                 *(in_cv + 8 * 8 + 6),
                 *(in_cv + 8 * 9 + 6),
                 *(in_cv + 8 * 10 + 6),
                 *(in_cv + 8 * 11 + 6),
                 *(in_cv + 8 * 12 + 6),
                 *(in_cv + 8 * 13 + 6),
                 *(in_cv + 8 * 14 + 6),
                 *(in_cv + 8 * 15 + 6)),
    sycl::uint16(*(in_cv + 8 * 0 + 7),
                 *(in_cv + 8 * 1 + 7),
                 *(in_cv + 8 * 2 + 7),
                 *(in_cv + 8 * 3 + 7),
                 *(in_cv + 8 * 4 + 7),
                 *(in_cv + 8 * 5 + 7),
                 *(in_cv + 8 * 6 + 7),
                 *(in_cv + 8 * 7 + 7),
                 *(in_cv + 8 * 8 + 7),
                 *(in_cv + 8 * 9 + 7),
                 *(in_cv + 8 * 10 + 7),
                 *(in_cv + 8 * 11 + 7),
                 *(in_cv + 8 * 12 + 7),
                 *(in_cv + 8 * 13 + 7),
                 *(in_cv + 8 * 14 + 7),
                 *(in_cv + 8 * 15 + 7)),
    sycl::uint16(IV[0]),
    sycl::uint16(IV[1]),
    sycl::uint16(IV[2]),
    sycl::uint16(IV[3]),
    sycl::uint16(static_cast<sycl::uint>((counter + 0) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 1) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 2) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 3) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 4) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 5) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 6) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 7) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 8) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 9) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 10) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 11) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 12) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 13) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 14) & 0xffffffff),
                 static_cast<sycl::uint>((counter + 15) & 0xffffffff)),
    sycl::uint16(static_cast<sycl::uint>((counter + 0) >> 32),
                 static_cast<sycl::uint>((counter + 1) >> 32),
                 static_cast<sycl::uint>((counter + 2) >> 32),
                 static_cast<sycl::uint>((counter + 3) >> 32),
                 static_cast<sycl::uint>((counter + 4) >> 32),
                 static_cast<sycl::uint>((counter + 5) >> 32),
                 static_cast<sycl::uint>((counter + 6) >> 32),
                 static_cast<sycl::uint>((counter + 7) >> 32),
                 static_cast<sycl::uint>((counter + 8) >> 32),
                 static_cast<sycl::uint>((counter + 9) >> 32),
                 static_cast<sycl::uint>((counter + 10) >> 32),
                 static_cast<sycl::uint>((counter + 11) >> 32),
                 static_cast<sycl::uint>((counter + 12) >> 32),
                 static_cast<sycl::uint>((counter + 13) >> 32),
                 static_cast<sycl::uint>((counter + 14) >> 32),
                 static_cast<sycl::uint>((counter + 15) >> 32)),
    sycl::uint16(block_len),
    sycl::uint16(flags)
  };
#endif

  // apply 7 rounds of mixing
  for (size_t i = 0; i < blake3::ROUNDS; i++) {
    // round i = {0, 1, ... 7}
    blake3::v2::round(state, block_words);

    if (i < 7) {
      for (size_t j = 0; j < BLAKE3_SIMD_LANES; j++) {
        blake3::permute(block_words + 16 * j);
      }
    }
  }

  // prepare output chaining values for 4/ 8/ 16 chunks
  // being compressed in parallel
  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[i + 8];
  }

#if BLAKE3_SIMD_LANES == 4
// writing 32 -bytes output chaining value
// for first chunk in this batch
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 0 + i) = state[i].x();
  }

// output chaining value of second chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 1 + i) = state[i].y();
  }

// output chaining value of third chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 2 + i) = state[i].z();
  }

  // output chaining value of fourth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 3 + i) = state[i].w();
  }
#elif BLAKE3_SIMD_LANES == 8
  // writing 32 -bytes output chaining value
  // for first chunk in this batch
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 0 + i) = state[i].s0();
  }

// output chaining value of second chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 1 + i) = state[i].s1();
  }

  // output chaining value of third chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 2 + i) = state[i].s2();
  }

  // output chaining value of fourth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 3 + i) = state[i].s3();
  }

  // output chaining value of fifth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 4 + i) = state[i].s4();
  }

  // output chaining value of sixth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 5 + i) = state[i].s5();
  }

  // output chaining value of seventh chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 6 + i) = state[i].s6();
  }

  // output chaining value of eighth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 7 + i) = state[i].s7();
  }
#elif BLAKE3_SIMD_LANES == 16
  // writing 32 -bytes output chaining value
  // for first chunk in this batch
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 0 + i) = state[i].s0();
  }

  // output chaining value of second chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 1 + i) = state[i].s1();
  }

  // output chaining value of third chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 2 + i) = state[i].s2();
  }

  // output chaining value of fourth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 3 + i) = state[i].s3();
  }

  // output chaining value of fifth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 4 + i) = state[i].s4();
  }

  // output chaining value of sixth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 5 + i) = state[i].s5();
  }

  // output chaining value of seventh chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 6 + i) = state[i].s6();
  }

  // output chaining value of eighth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 7 + i) = state[i].s7();
  }

  // output chaining value of ninth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 8 + i) = state[i].s8();
  }

  // output chaining value of tenth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 9 + i) = state[i].s9();
  }

  // output chaining value of eleventh chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 10 + i) = state[i].sA();
  }

  // output chaining value of twelveth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 11 + i) = state[i].sB();
  }

  // output chaining value of thirteenth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 12 + i) = state[i].sC();
  }

  // output chaining value of fourteenth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 13 + i) = state[i].sD();
  }

  // output chaining value of fifteenth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 14 + i) = state[i].sE();
  }

// output chaining value of sixteenth chunk
#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    *(out_cv + 8 * 15 + i) = state[i].sF();
  }
#endif
}

void
blake3::v2::chunkify(const sycl::uint* key_words,
                     sycl::ulong chunk_counter,
                     sycl::uint flags,
                     const sycl::uchar* input,
                     sycl::uint* const out_cv)
{
#if BLAKE3_SIMD_LANES == 4
  sycl::uint in_cv[32] = { 0 };
  sycl::uint priv_out_cv[32] = { 0 };
  sycl::uint block_words[64] = { 0 };
#elif BLAKE3_SIMD_LANES == 8
  sycl::uint in_cv[64] = { 0 };
  sycl::uint priv_out_cv[64] = { 0 };
  sycl::uint block_words[128] = { 0 };
#elif BLAKE3_SIMD_LANES == 16
  sycl::uint in_cv[128] = { 0 };
  sycl::uint priv_out_cv[128] = { 0 };
  sycl::uint block_words[256] = { 0 };
#endif

  for (size_t i = 0; i < 8; i++) {
    sycl::uint tmp = *(key_words + i);

#pragma unroll(4)
    for (size_t j = 0; j < BLAKE3_SIMD_LANES; j++) {
      in_cv[i + 8 * j] = tmp;
    }
  }

  for (size_t i = 0; i < 16; i++) {
    // prepare input of N -many chunks for
    // consumption into hash state
    for (size_t j = 0; j < BLAKE3_SIMD_LANES; j++) {
      blake3::words_from_le_bytes(input + blake3::BLOCK_LEN * i +
                                    blake3::CHUNK_LEN * j,
                                  block_words + 16 * j);
    }

    switch (i) {
      case 0:
        blake3::v2::compress(in_cv,
                             block_words,
                             chunk_counter,
                             blake3::BLOCK_LEN,
                             flags | blake3::CHUNK_START,
                             priv_out_cv);
        break;
      case 15:
        blake3::v2::compress(in_cv,
                             block_words,
                             chunk_counter,
                             blake3::BLOCK_LEN,
                             flags | blake3::CHUNK_END,
                             out_cv);
        break;
      default:
        blake3::v2::compress(in_cv,
                             block_words,
                             chunk_counter,
                             blake3::BLOCK_LEN,
                             flags,
                             priv_out_cv);
    }

    if (i < 15) {
#pragma unroll(4)
      for (size_t j = 0; j < 8 * BLAKE3_SIMD_LANES; j++) {
        in_cv[j] = priv_out_cv[j];
      }
    }
  }
}

void
blake3::v2::hash(sycl::queue& q,
                 const sycl::uchar* input,
                 size_t i_size,
                 size_t chunk_count,
                 size_t wg_size,
                 sycl::uchar* const digest,
                 sycl::cl_ulong* const ts)
{
  assert(i_size == chunk_count * blake3::CHUNK_LEN);
  assert(chunk_count >= 1024); // minimum 1MB input required !
  assert((chunk_count & (chunk_count - 1)) == 0); // power of 2 check

#if BLAKE3_SIMD_LANES == 4
  assert(wg_size <= (chunk_count >> 2));
#elif BLAKE3_SIMD_LANES == 8
  assert(wg_size <= (chunk_count >> 3));
#elif BLAKE3_SIMD_LANES == 16
  assert(wg_size <= (chunk_count >> 4));
#endif

  const size_t mem_size = static_cast<size_t>(blake3::BLOCK_LEN) * chunk_count;
  sycl::uint* mem = static_cast<sycl::uint*>(sycl::malloc_device(mem_size, q));
  const size_t mem_offset = (blake3::OUT_LEN >> 2) * chunk_count;

  sycl::event evt_0 = q.parallel_for<class kernelBlake3HashV2ChunkifyLeafNodes>(
    sycl::nd_range<1>{ sycl::range<1>{
#if BLAKE3_SIMD_LANES == 4
                         chunk_count >> 2
#elif BLAKE3_SIMD_LANES == 8
                         chunk_count >> 3
#elif BLAKE3_SIMD_LANES == 16
                         chunk_count >> 4
#endif
                       },
                       sycl::range<1>{ wg_size } },
    [=](sycl::nd_item<1> it) {
  // because 4/ 8/ 16 chunks are clustered together
  // and operated on using single v2::chunkify
  // function invocation

#if BLAKE3_SIMD_LANES == 4
      const size_t idx = it.get_global_linear_id() << 2;
#elif BLAKE3_SIMD_LANES == 8
      const size_t idx = it.get_global_linear_id() << 3;
#elif BLAKE3_SIMD_LANES == 16
      const size_t idx = it.get_global_linear_id() << 4;
#endif

      blake3::v2::chunkify(blake3::IV,
                           static_cast<sycl::ulong>(idx),
                           0,
                           input + idx * blake3::CHUNK_LEN,
                           mem + mem_offset + idx * (blake3::OUT_LEN >> 2));
    });

  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(chunk_count))) - 1;

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      const size_t read_offset = mem_offset >> r;
      const size_t write_offset = read_offset >> 1;
      const size_t glb_work_items = chunk_count >> (r + 1);
      const size_t loc_work_items =
        glb_work_items < wg_size ? glb_work_items : wg_size;

      h.parallel_for<class kernelBlake3HashV2ParentChaining>(
        sycl::nd_range<1>{ sycl::range<1>{ glb_work_items },
                           sycl::range<1>{ loc_work_items } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          blake3::parent_cv(
            mem + read_offset + (idx << 1) * (blake3::OUT_LEN >> 2),
            mem + read_offset + ((idx << 1) + 1) * (blake3::OUT_LEN >> 2),
            blake3::IV,
            0,
            mem + write_offset + idx * (blake3::OUT_LEN >> 2));
        });
    });
    evts.push_back(evt);
  }

  sycl::event evt_1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(rounds - 1));
    h.single_task([=]() {
      blake3::root_cv(
        mem + ((blake3::OUT_LEN >> 2) << 1) + 0 * (blake3::OUT_LEN >> 2),
        mem + ((blake3::OUT_LEN >> 2) << 1) + 1 * (blake3::OUT_LEN >> 2),
        blake3::IV,
        mem + 1 * (blake3::OUT_LEN >> 2));
      blake3::words_to_le_bytes(mem + 1 * (blake3::OUT_LEN >> 2), digest);
    });
  });

  evt_1.wait();
  sycl::free(mem, q);

  // time kernel executions only when explicitly asked to
  //
  // when asked, ensure queue has profiling enabled, otherwise
  // following function invocations will panic out !
  if (ts != nullptr) {
    sycl::cl_ulong ts_ = 0;

    {
      const sycl::cl_ulong start =
        evt_0.get_profiling_info<sycl::info::event_profiling::command_start>();
      const sycl::cl_ulong end =
        evt_0.get_profiling_info<sycl::info::event_profiling::command_end>();

      ts_ += (end - start);
    }

    for (auto evt : evts) {
      const sycl::cl_ulong start =
        evt.get_profiling_info<sycl::info::event_profiling::command_start>();
      const sycl::cl_ulong end =
        evt.get_profiling_info<sycl::info::event_profiling::command_end>();

      ts_ += (end - start);
    }

    {
      const sycl::cl_ulong start =
        evt_1.get_profiling_info<sycl::info::event_profiling::command_start>();
      const sycl::cl_ulong end =
        evt_1.get_profiling_info<sycl::info::event_profiling::command_end>();

      ts_ += (end - start);
    }

    *ts = ts_;
  }
}

inline void
blake3::v1::round(sycl::uint4* const state, const sycl::uint* msg)
{
  sycl::uint4 mx = sycl::uint4(*(msg + 0), *(msg + 2), *(msg + 4), *(msg + 6));
  sycl::uint4 my = sycl::uint4(*(msg + 1), *(msg + 3), *(msg + 5), *(msg + 7));
  sycl::uint4 mz =
    sycl::uint4(*(msg + 8), *(msg + 10), *(msg + 12), *(msg + 14));
  sycl::uint4 mw =
    sycl::uint4(*(msg + 9), *(msg + 11), *(msg + 13), *(msg + 15));

  sycl::uint4 rrot_16 = sycl::uint4(16);
  sycl::uint4 rrot_12 = sycl::uint4(20);
  sycl::uint4 rrot_8 = sycl::uint4(24);
  sycl::uint4 rrot_7 = sycl::uint4(25);

  // columnar state processing
  *(state + 0) = *(state + 0) + *(state + 1) + mx;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_16);

  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_12);

  *(state + 0) = *(state + 0) + *(state + 1) + my;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_8);

  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_7);

  // diagonalize
  *(state + 1) = (*(state + 1)).yzwx();
  *(state + 2) = (*(state + 2)).zwxy();
  *(state + 3) = (*(state + 3)).wxyz();

  // diagonal state processing
  *(state + 0) = *(state + 0) + *(state + 1) + mz;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_16);

  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_12);

  *(state + 0) = *(state + 0) + *(state + 1) + mw;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_8);

  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_7);

  // non-diagonalize
  *(state + 1) = (*(state + 1)).wxyz();
  *(state + 2) = (*(state + 2)).zwxy();
  *(state + 3) = (*(state + 3)).yzwx();
}

inline void
blake3::permute(sycl::uint* const msg)
{
  sycl::uint permuted[16] = { 0 };

  for (size_t i = 0; i < 16; i++) {
    permuted[i] = *(msg + blake3::MSG_PERMUTATION[i]);
  }

  for (size_t i = 0; i < 16; i++) {
    *(msg + i) = permuted[i];
  }
}

inline void
blake3::v1::compress(const sycl::uint* in_cv,
                     sycl::uint* const block_words,
                     sycl::ulong counter,
                     sycl::uint block_len,
                     sycl::uint flags,
                     sycl::uint* const out_cv)
{
  sycl::uint4 cv0 =
    sycl::uint4(*(in_cv + 0), *(in_cv + 1), *(in_cv + 2), *(in_cv + 3));
  sycl::uint4 cv1 =
    sycl::uint4(*(in_cv + 4), *(in_cv + 5), *(in_cv + 6), *(in_cv + 7));

  sycl::uint4 state[4] = { cv0,
                           cv1,
                           sycl::uint4(IV[0], IV[1], IV[2], IV[3]),
                           sycl::uint4(counter & 0xffffffff,
                                       static_cast<sycl::uint>(counter >> 32),
                                       block_len,
                                       flags) };

  // round 1
  blake3::v1::round(state, block_words);
  blake3::permute(block_words);

  // round 2
  blake3::v1::round(state, block_words);
  blake3::permute(block_words);

  // round 3
  blake3::v1::round(state, block_words);
  blake3::permute(block_words);

  // round 4
  blake3::v1::round(state, block_words);
  blake3::permute(block_words);

  // round 5
  blake3::v1::round(state, block_words);
  blake3::permute(block_words);

  // round 6
  blake3::v1::round(state, block_words);
  blake3::permute(block_words);

  // round 7
  blake3::v1::round(state, block_words);

  state[0] ^= state[2];
  state[1] ^= state[3];
  // following two lines don't dictate output chaining value
  // of this block ( or chunk ), so they can be safely commented out !
  // state[2] ^= cv0;
  // state[3] ^= cv1;

  // output chaining value of this block to be used as
  // input chaining value for next block in same chunk
  *(out_cv + 0) = state[0].x();
  *(out_cv + 1) = state[0].y();
  *(out_cv + 2) = state[0].z();
  *(out_cv + 3) = state[0].w();
  *(out_cv + 4) = state[1].x();
  *(out_cv + 5) = state[1].y();
  *(out_cv + 6) = state[1].z();
  *(out_cv + 7) = state[1].w();
}

inline void
blake3::words_from_le_bytes(const sycl::uchar* input,
                            sycl::uint* const block_words)
{
  for (size_t i = 0; i < 16; i++) {
    const sycl::uchar* num = (input + i * 4);
    *(block_words + i) = (static_cast<sycl::uint>(*(num + 3)) << 24) |
                         (static_cast<sycl::uint>(*(num + 2)) << 16) |
                         (static_cast<sycl::uint>(*(num + 1)) << 8) |
                         (static_cast<sycl::uint>(*(num + 0)) << 0);
  }
}

inline void
blake3::words_to_le_bytes(const sycl::uint* input, sycl::uchar* const output)
{
  for (size_t i = 0; i < 8; i++) {
    const sycl::uint num = *(input + i);
    sycl::uchar* out_ = output + i * 4;

    *(out_ + 0) = static_cast<sycl::uchar>(num & 0xff);
    *(out_ + 1) = static_cast<sycl::uchar>((num >> 8) & 0xff);
    *(out_ + 2) = static_cast<sycl::uchar>((num >> 16) & 0xff);
    *(out_ + 3) = static_cast<sycl::uchar>((num >> 24) & 0xff);
  }
}

void
blake3::v1::chunkify(const sycl::uint* key_words,
                     sycl::ulong chunk_counter,
                     sycl::uint flags,
                     const sycl::uchar* input,
                     sycl::uint* const out_cv)
{
  sycl::uint in_cv[8] = { 0 };
  sycl::uint priv_out_cv[8] = { 0 };
  sycl::uint block_words[16] = { 0 };

#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    in_cv[i] = *(key_words + i);
  }

  for (size_t i = 0; i < 16; i++) {
    blake3::words_from_le_bytes(input + i * blake3::BLOCK_LEN, block_words);

    switch (i) {
      case 0:
        blake3::v1::compress(in_cv,
                             block_words,
                             chunk_counter,
                             blake3::BLOCK_LEN,
                             flags | blake3::CHUNK_START,
                             priv_out_cv);
        break;
      case 15:
        blake3::v1::compress(in_cv,
                             block_words,
                             chunk_counter,
                             blake3::BLOCK_LEN,
                             flags | blake3::CHUNK_END,
                             out_cv);
        break;
      default:
        blake3::v1::compress(in_cv,
                             block_words,
                             chunk_counter,
                             blake3::BLOCK_LEN,
                             flags,
                             priv_out_cv);
    }

    if (i < 15) {
#pragma unroll(4)
      for (size_t j = 0; j < 8; j++) {
        in_cv[j] = priv_out_cv[j];
      }
    }
  }
}

inline void
blake3::parent_cv(const sycl::uint* left_cv,
                  const sycl::uint* right_cv,
                  const sycl::uint* key_words,
                  sycl::uint flags,
                  sycl::uint* const out_cv)
{
  sycl::uint block_words[16] = { 0 };

#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    block_words[i] = *(left_cv + i);
    block_words[i + 8] = *(right_cv + i);
  }

  blake3::v1::compress(key_words,
                       block_words,
                       0,
                       blake3::BLOCK_LEN,
                       flags | blake3::PARENT,
                       out_cv);
}

inline void
blake3::root_cv(const sycl::uint* left_cv,
                const sycl::uint* right_cv,
                const sycl::uint* key_words,
                sycl::uint* const out_cv)
{
  blake3::parent_cv(left_cv, right_cv, key_words, blake3::ROOT, out_cv);
}

void
blake3::v1::hash(sycl::queue& q,
                 const sycl::uchar* input,
                 size_t i_size,
                 size_t chunk_count,
                 size_t wg_size,
                 sycl::uchar* const digest,
                 sycl::cl_ulong* const ts)
{
  assert(i_size == chunk_count * blake3::CHUNK_LEN);
  assert(chunk_count >= 2);
  assert((chunk_count & (chunk_count - 1)) == 0);
  assert(wg_size <= chunk_count);

  const size_t mem_size = static_cast<size_t>(blake3::BLOCK_LEN) * chunk_count;
  sycl::uint* mem = static_cast<sycl::uint*>(sycl::malloc_device(mem_size, q));
  const size_t mem_offset = (blake3::OUT_LEN >> 2) * chunk_count;

  sycl::event evt_0 = q.parallel_for<class kernelBlake3HashV1ChunkifyLeafNodes>(
    sycl::nd_range<1>{ sycl::range<1>{ chunk_count },
                       sycl::range<1>{ wg_size } },
    [=](sycl::nd_item<1> it) {
      const size_t idx = it.get_global_linear_id();

      blake3::v1::chunkify(blake3::IV,
                           static_cast<sycl::ulong>(idx),
                           0,
                           input + idx * blake3::CHUNK_LEN,
                           mem + mem_offset + idx * (blake3::OUT_LEN >> 2));
    });

  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(chunk_count))) - 1;

  if (rounds == 0) {
    sycl::event evt_1 = q.submit([&](sycl::handler& h) {
      h.depends_on(evt_0);
      h.single_task([=]() {
        blake3::root_cv(mem + mem_offset + 0 * (blake3::OUT_LEN >> 2),
                        mem + mem_offset + 1 * (blake3::OUT_LEN >> 2),
                        blake3::IV,
                        mem + 1 * (blake3::OUT_LEN >> 2));
        blake3::words_to_le_bytes(mem + 1 * (blake3::OUT_LEN >> 2), digest);
      });
    });

    evt_1.wait();
    sycl::free(mem, q);

    // time kernel executions only when asked to
    //
    // when asked, ensure queue has profiling enabled !
    if (ts != nullptr) {
      sycl::cl_ulong ts_ = 0;

      {
        const sycl::cl_ulong start =
          evt_0
            .get_profiling_info<sycl::info::event_profiling::command_start>();
        const sycl::cl_ulong end =
          evt_0.get_profiling_info<sycl::info::event_profiling::command_end>();

        ts_ += (end - start);
      }

      {
        const sycl::cl_ulong start =
          evt_1
            .get_profiling_info<sycl::info::event_profiling::command_start>();
        const sycl::cl_ulong end =
          evt_1.get_profiling_info<sycl::info::event_profiling::command_end>();

        ts_ += (end - start);
      }

      *ts = ts_;
    }

    return;
  }

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      const size_t read_offset = mem_offset >> r;
      const size_t write_offset = read_offset >> 1;
      const size_t glb_work_items = chunk_count >> (r + 1);
      const size_t loc_work_items =
        glb_work_items < wg_size ? glb_work_items : wg_size;

      h.parallel_for<class kernelBlake3HashV1ParentChaining>(
        sycl::nd_range<1>{ sycl::range<1>{ glb_work_items },
                           sycl::range<1>{ loc_work_items } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          blake3::parent_cv(
            mem + read_offset + (idx << 1) * (blake3::OUT_LEN >> 2),
            mem + read_offset + ((idx << 1) + 1) * (blake3::OUT_LEN >> 2),
            blake3::IV,
            0,
            mem + write_offset + idx * (blake3::OUT_LEN >> 2));
        });
    });
    evts.push_back(evt);
  }

  sycl::event evt_1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(rounds - 1));
    h.single_task([=]() {
      blake3::root_cv(
        mem + ((blake3::OUT_LEN >> 2) << 1) + 0 * (blake3::OUT_LEN >> 2),
        mem + ((blake3::OUT_LEN >> 2) << 1) + 1 * (blake3::OUT_LEN >> 2),
        blake3::IV,
        mem + 1 * (blake3::OUT_LEN >> 2));
      blake3::words_to_le_bytes(mem + 1 * (blake3::OUT_LEN >> 2), digest);
    });
  });

  evt_1.wait();
  sycl::free(mem, q);

  // time kernel executions only when explicitly asked to
  //
  // when asked, ensure queue has profiling enabled, otherwise
  // following function invocations will panic out !
  if (ts != nullptr) {
    sycl::cl_ulong ts_ = 0;

    {
      const sycl::cl_ulong start =
        evt_0.get_profiling_info<sycl::info::event_profiling::command_start>();
      const sycl::cl_ulong end =
        evt_0.get_profiling_info<sycl::info::event_profiling::command_end>();

      ts_ += (end - start);
    }

    for (auto evt : evts) {
      const sycl::cl_ulong start =
        evt.get_profiling_info<sycl::info::event_profiling::command_start>();
      const sycl::cl_ulong end =
        evt.get_profiling_info<sycl::info::event_profiling::command_end>();

      ts_ += (end - start);
    }

    {
      const sycl::cl_ulong start =
        evt_1.get_profiling_info<sycl::info::event_profiling::command_start>();
      const sycl::cl_ulong end =
        evt_1.get_profiling_info<sycl::info::event_profiling::command_end>();

      ts_ += (end - start);
    }

    *ts = ts_;
  }
}
