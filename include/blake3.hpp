#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>

namespace blake3 {
constexpr size_t MSG_PERMUTATION[16] = { 2, 6,  3,  10, 7, 0,  4,  13,
                                         1, 11, 12, 5,  9, 14, 15, 8 };

constexpr sycl::uint IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

constexpr sycl::uint BLOCK_LEN = 64;
constexpr sycl::uint CHUNK_START = 1 << 0;
constexpr sycl::uint CHUNK_END = 1 << 1;
constexpr sycl::uint PARENT = 1 << 2;
constexpr sycl::uint ROOT = 1 << 3;

void
round(sycl::uint4* const state, const sycl::uint* msg);

void
permute(sycl::uint* const msg);

void
compress(const sycl::uint* in_cv,
         sycl::uint* const block_words,
         sycl::ulong counter,
         sycl::uint block_len,
         sycl::uint flags,
         sycl::uint* const out_cv);

void
words_from_le_bytes(const sycl::uchar* input, sycl::uint* const block_words);

void
chunkify(const sycl::uint* key_words,
         sycl::ulong chunk_counter,
         sycl::uint flags,
         const sycl::uchar* input,
         sycl::uint* const out_cv);

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
}

void
blake3::round(sycl::uint4* const state, const sycl::uint* msg)
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

void
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

void
blake3::compress(const sycl::uint* in_cv,
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
  blake3::round(state, block_words);
  blake3::permute(block_words);

  // round 2
  blake3::round(state, block_words);
  blake3::permute(block_words);

  // round 3
  blake3::round(state, block_words);
  blake3::permute(block_words);

  // round 4
  blake3::round(state, block_words);
  blake3::permute(block_words);

  // round 5
  blake3::round(state, block_words);
  blake3::permute(block_words);

  // round 6
  blake3::round(state, block_words);
  blake3::permute(block_words);

  // round 7
  blake3::round(state, block_words);

  state[0] ^= state[2];
  state[1] ^= state[3];
  // following two lines don't dictate output chaining value
  // of this block ( or chunk ), so they can be commented out !
  state[2] ^= cv0;
  state[3] ^= cv1;

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

void
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

void
blake3::chunkify(const sycl::uint* key_words,
                 sycl::ulong chunk_counter,
                 sycl::uint flags,
                 const sycl::uchar* input,
                 sycl::uint* const out_cv)
{
  sycl::uint in_cv[8] = { *(key_words + 0), *(key_words + 1), *(key_words + 2),
                          *(key_words + 3), *(key_words + 4), *(key_words + 5),
                          *(key_words + 6), *(key_words + 7) };
  sycl::uint block_words[16] = { 0 };

  for (size_t i = 0; i < 16; i++) {
    blake3::words_from_le_bytes(input + i * blake3::BLOCK_LEN, block_words);

    switch (i) {
      case 0:
        blake3::compress(in_cv,
                         block_words,
                         chunk_counter,
                         blake3::BLOCK_LEN,
                         flags | blake3::CHUNK_START,
                         out_cv);
        break;
      case 15:
        blake3::compress(in_cv,
                         block_words,
                         chunk_counter,
                         blake3::BLOCK_LEN,
                         flags | blake3::CHUNK_END,
                         out_cv);
        break;
      default:
        blake3::compress(
          in_cv, block_words, chunk_counter, blake3::BLOCK_LEN, flags, out_cv);
    }

    if (i < 15) {
      in_cv[0] = out_cv[0];
      in_cv[1] = out_cv[1];
      in_cv[2] = out_cv[2];
      in_cv[3] = out_cv[3];
      in_cv[4] = out_cv[4];
      in_cv[5] = out_cv[5];
      in_cv[6] = out_cv[6];
      in_cv[7] = out_cv[7];
    }
  }
}

void
blake3::parent_cv(const sycl::uint* left_cv,
                  const sycl::uint* right_cv,
                  const sycl::uint* key_words,
                  sycl::uint flags,
                  sycl::uint* const out_cv)
{
  sycl::uint block_words[16] = {
    *(left_cv + 0),  *(left_cv + 1),  *(left_cv + 2),  *(left_cv + 3),
    *(left_cv + 4),  *(left_cv + 5),  *(left_cv + 6),  *(left_cv + 7),
    *(right_cv + 0), *(right_cv + 1), *(right_cv + 2), *(right_cv + 3),
    *(right_cv + 4), *(right_cv + 5), *(right_cv + 6), *(right_cv + 7)
  };

  blake3::compress(key_words,
                   block_words,
                   0,
                   blake3::BLOCK_LEN,
                   flags | blake3::PARENT,
                   out_cv);
}

void
blake3::root_cv(const sycl::uint* left_cv,
                const sycl::uint* right_cv,
                const sycl::uint* key_words,
                sycl::uint* const out_cv)
{
  blake3::parent_cv(left_cv, right_cv, key_words, blake3::ROOT, out_cv);
}
