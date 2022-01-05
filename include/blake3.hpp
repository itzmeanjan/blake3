#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>

namespace blake3 {
constexpr size_t MSG_PERMUTATION[16] = { 2, 6,  3,  10, 7, 0,  4,  13,
                                         1, 11, 12, 5,  9, 14, 15, 8 };

constexpr sycl::uint IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

void
round(sycl::uint4* const state, const sycl::uint* msg);

void
permute(sycl::uint* const msg);

void
compress(const sycl::uint* chaining_value,
         sycl::uint* const block_words,
         sycl::ulong counter,
         sycl::uint block_len,
         sycl::uint flags);
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
blake3::compress(const sycl::uint* chaining_value,
                 sycl::uint* const block_words,
                 sycl::ulong counter,
                 sycl::uint block_len,
                 sycl::uint flags)
{
  sycl::uint4 cv0 = sycl::uint4(*(chaining_value + 0),
                                *(chaining_value + 1),
                                *(chaining_value + 2),
                                *(chaining_value + 3));
  sycl::uint4 cv1 = sycl::uint4(*(chaining_value + 4),
                                *(chaining_value + 5),
                                *(chaining_value + 6),
                                *(chaining_value + 7));

  sycl::uint4 state[4] = { cv0,
                           cv1,
                           sycl::uint4(IV[0], IV[1], IV[2], IV[3]),
                           sycl::uint4(counter & 0xffffffff,
                                       static_cast<sycl::uint>(counter >> 32),
                                       block_len,
                                       flags) };

  blake3::round(state, block_words);
  blake3::permute(block_words);

  blake3::round(state, block_words);
  blake3::permute(block_words);

  blake3::round(state, block_words);
  blake3::permute(block_words);

  blake3::round(state, block_words);
  blake3::permute(block_words);

  blake3::round(state, block_words);
  blake3::permute(block_words);

  blake3::round(state, block_words);
  blake3::permute(block_words);

  blake3::round(state, block_words);
  blake3::permute(block_words);

  state[0] ^= state[2];
  state[1] ^= state[3];
  state[2] ^= cv0;
  state[3] ^= cv1;
}
