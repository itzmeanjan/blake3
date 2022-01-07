#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>
#include <cassert>

namespace blake3 {
constexpr size_t MSG_PERMUTATION[16] = { 2, 6,  3,  10, 7, 0,  4,  13,
                                         1, 11, 12, 5,  9, 14, 15, 8 };

constexpr sycl::uint IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

constexpr size_t CHUNK_LEN = 1024;
constexpr size_t OUT_LEN = 32;

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
g(sycl::uint4* const state,
  size_t a,
  size_t b,
  size_t c,
  size_t d,
  sycl::uint4 mx,
  sycl::uint4 my);

void
round(sycl::uint4* const state, const sycl::uint* msg);

void
compress(const sycl::uint* in_cv,
         sycl::uint* const block_words,
         const sycl::ulong* counter,
         const sycl::uint* block_len,
         const sycl::uint* flags,
         sycl::uint* const out_cv);

void
chunkify(const sycl::uint* key_words,
         const sycl::ulong* chunk_counter,
         const sycl::uint* flags,
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
blake3::v2::g(sycl::uint4* const state,
              size_t a,
              size_t b,
              size_t c,
              size_t d,
              sycl::uint4 mx,
              sycl::uint4 my)
{
  *(state + a) = *(state + a) + *(state + b) + mx;
  *(state + d) = sycl::rotate(*(state + d) ^ *(state + a), sycl::uint4(16));
  *(state + c) = *(state + c) + *(state + d);
  *(state + b) = sycl::rotate(*(state + b) ^ *(state + c), sycl::uint4(20));
  *(state + a) = *(state + a) + *(state + b) + my;
  *(state + d) = sycl::rotate(*(state + d) ^ *(state + a), sycl::uint4(24));
  *(state + c) = *(state + c) + *(state + d);
  *(state + b) = sycl::rotate(*(state + b) ^ *(state + c), sycl::uint4(25));
}

inline void
blake3::v2::round(sycl::uint4* const state, const sycl::uint* msg)
{
  // column-wise hash state manipulation starts
  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 0),
                                 *(msg + 16 * 1 + 0),
                                 *(msg + 16 * 2 + 0),
                                 *(msg + 16 * 3 + 0));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 1),
                                 *(msg + 16 * 1 + 1),
                                 *(msg + 16 * 2 + 1),
                                 *(msg + 16 * 3 + 1));
    blake3::v2::g(state, 0, 4, 8, 12, mx, my);
  }

  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 2),
                                 *(msg + 16 * 1 + 2),
                                 *(msg + 16 * 2 + 2),
                                 *(msg + 16 * 3 + 2));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 3),
                                 *(msg + 16 * 1 + 3),
                                 *(msg + 16 * 2 + 3),
                                 *(msg + 16 * 3 + 3));
    blake3::v2::g(state, 1, 5, 9, 13, mx, my);
  }

  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 4),
                                 *(msg + 16 * 1 + 4),
                                 *(msg + 16 * 2 + 4),
                                 *(msg + 16 * 3 + 4));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 5),
                                 *(msg + 16 * 1 + 5),
                                 *(msg + 16 * 2 + 5),
                                 *(msg + 16 * 3 + 5));
    blake3::v2::g(state, 2, 6, 10, 14, mx, my);
  }

  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 6),
                                 *(msg + 16 * 1 + 6),
                                 *(msg + 16 * 2 + 6),
                                 *(msg + 16 * 3 + 6));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 7),
                                 *(msg + 16 * 1 + 7),
                                 *(msg + 16 * 2 + 7),
                                 *(msg + 16 * 3 + 7));
    blake3::v2::g(state, 3, 7, 11, 15, mx, my);
  }
  // column-wise hash state manipulation ends

  // diagonal hash state manipulation starts
  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 8),
                                 *(msg + 16 * 1 + 8),
                                 *(msg + 16 * 2 + 8),
                                 *(msg + 16 * 3 + 8));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 9),
                                 *(msg + 16 * 1 + 9),
                                 *(msg + 16 * 2 + 9),
                                 *(msg + 16 * 3 + 9));
    blake3::v2::g(state, 0, 5, 10, 15, mx, my);
  }

  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 10),
                                 *(msg + 16 * 1 + 10),
                                 *(msg + 16 * 2 + 10),
                                 *(msg + 16 * 3 + 10));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 11),
                                 *(msg + 16 * 1 + 11),
                                 *(msg + 16 * 2 + 11),
                                 *(msg + 16 * 3 + 11));
    blake3::v2::g(state, 1, 6, 11, 12, mx, my);
  }

  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 12),
                                 *(msg + 16 * 1 + 12),
                                 *(msg + 16 * 2 + 12),
                                 *(msg + 16 * 3 + 12));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 13),
                                 *(msg + 16 * 1 + 13),
                                 *(msg + 16 * 2 + 13),
                                 *(msg + 16 * 3 + 13));
    blake3::v2::g(state, 2, 7, 8, 13, mx, my);
  }

  {
    sycl::uint4 mx = sycl::uint4(*(msg + 16 * 0 + 14),
                                 *(msg + 16 * 1 + 14),
                                 *(msg + 16 * 2 + 14),
                                 *(msg + 16 * 3 + 14));
    sycl::uint4 my = sycl::uint4(*(msg + 16 * 0 + 15),
                                 *(msg + 16 * 1 + 15),
                                 *(msg + 16 * 2 + 15),
                                 *(msg + 16 * 3 + 15));
    blake3::v2::g(state, 3, 4, 9, 14, mx, my);
  }
  // diagonal hash state manipulation ends
}

inline void
blake3::v2::compress(const sycl::uint* in_cv,
                     sycl::uint* const block_words,
                     const sycl::ulong* counter,
                     const sycl::uint* block_len,
                     const sycl::uint* flags,
                     sycl::uint* const out_cv)
{
  // hash state of 4 chunks; to be processed in parallel
  //
  // See section 5.3 of Blake3 specification for understanding
  // how this SIMD technique can be applied for wider hash state
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
    sycl::uint4(static_cast<sycl::uint>(*(counter + 0) & 0xffffffff),
                static_cast<sycl::uint>(*(counter + 1) & 0xffffffff),
                static_cast<sycl::uint>(*(counter + 2) & 0xffffffff),
                static_cast<sycl::uint>(*(counter + 3) & 0xffffffff)),
    sycl::uint4(static_cast<sycl::uint>(*(counter + 0) >> 32),
                static_cast<sycl::uint>(*(counter + 1) >> 32),
                static_cast<sycl::uint>(*(counter + 2) >> 32),
                static_cast<sycl::uint>(*(counter + 3) >> 32)),
    sycl::uint4(
      *(block_len + 0), *(block_len + 1), *(block_len + 2), *(block_len + 3)),
    sycl::uint4(*(flags + 0), *(flags + 1), *(flags + 2), *(flags + 3))
  };

  // round 1
  blake3::v2::round(state, block_words);
  blake3::permute(block_words + 16 * 0);
  blake3::permute(block_words + 16 * 1);
  blake3::permute(block_words + 16 * 2);
  blake3::permute(block_words + 16 * 3);

  // round 2
  blake3::v2::round(state, block_words);
  blake3::permute(block_words + 16 * 0);
  blake3::permute(block_words + 16 * 1);
  blake3::permute(block_words + 16 * 2);
  blake3::permute(block_words + 16 * 3);

  // round 3
  blake3::v2::round(state, block_words);
  blake3::permute(block_words + 16 * 0);
  blake3::permute(block_words + 16 * 1);
  blake3::permute(block_words + 16 * 2);
  blake3::permute(block_words + 16 * 3);

  // round 4
  blake3::v2::round(state, block_words);
  blake3::permute(block_words + 16 * 0);
  blake3::permute(block_words + 16 * 1);
  blake3::permute(block_words + 16 * 2);
  blake3::permute(block_words + 16 * 3);

  // round 5
  blake3::v2::round(state, block_words);
  blake3::permute(block_words + 16 * 0);
  blake3::permute(block_words + 16 * 1);
  blake3::permute(block_words + 16 * 2);
  blake3::permute(block_words + 16 * 3);

  // round 6
  blake3::v2::round(state, block_words);
  blake3::permute(block_words + 16 * 0);
  blake3::permute(block_words + 16 * 1);
  blake3::permute(block_words + 16 * 2);
  blake3::permute(block_words + 16 * 3);

  // round 7
  blake3::v2::round(state, block_words);
  // message words don't need to be permuted anymore !

  // prepare output chaining values for 4 chunks
  // being compressed in parallel
  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[i + 8];
  }

  // writing 32 -bytes output chaining value
  // for first chunk in this batch
  *(out_cv + 8 * 0 + 0) = state[0].x();
  *(out_cv + 8 * 0 + 1) = state[1].x();
  *(out_cv + 8 * 0 + 2) = state[2].x();
  *(out_cv + 8 * 0 + 3) = state[3].x();
  *(out_cv + 8 * 0 + 4) = state[4].x();
  *(out_cv + 8 * 0 + 5) = state[5].x();
  *(out_cv + 8 * 0 + 6) = state[6].x();
  *(out_cv + 8 * 0 + 7) = state[7].x();

  // this is output chaining value of second chunk
  // in this batch
  *(out_cv + 8 * 1 + 0) = state[0].y();
  *(out_cv + 8 * 1 + 1) = state[1].y();
  *(out_cv + 8 * 1 + 2) = state[2].y();
  *(out_cv + 8 * 1 + 3) = state[3].y();
  *(out_cv + 8 * 1 + 4) = state[4].y();
  *(out_cv + 8 * 1 + 5) = state[5].y();
  *(out_cv + 8 * 1 + 6) = state[6].y();
  *(out_cv + 8 * 1 + 7) = state[7].y();

  // this is output chaining value of third chunk
  // in this batch
  *(out_cv + 8 * 2 + 0) = state[0].z();
  *(out_cv + 8 * 2 + 1) = state[1].z();
  *(out_cv + 8 * 2 + 2) = state[2].z();
  *(out_cv + 8 * 2 + 3) = state[3].z();
  *(out_cv + 8 * 2 + 4) = state[4].z();
  *(out_cv + 8 * 2 + 5) = state[5].z();
  *(out_cv + 8 * 2 + 6) = state[6].z();
  *(out_cv + 8 * 2 + 7) = state[7].z();

  // finally last output chaining value of last chunk
  // in this batch
  *(out_cv + 8 * 3 + 0) = state[0].w();
  *(out_cv + 8 * 3 + 1) = state[1].w();
  *(out_cv + 8 * 3 + 2) = state[2].w();
  *(out_cv + 8 * 3 + 3) = state[3].w();
  *(out_cv + 8 * 3 + 4) = state[4].w();
  *(out_cv + 8 * 3 + 5) = state[5].w();
  *(out_cv + 8 * 3 + 6) = state[6].w();
  *(out_cv + 8 * 3 + 7) = state[7].w();
}

void
blake3::v2::chunkify(const sycl::uint* key_words,
                     const sycl::ulong* chunk_counter,
                     const sycl::uint* flags,
                     const sycl::uchar* input,
                     sycl::uint* const out_cv)
{
  sycl::uint in_cv[32] = { 0 };
  sycl::uint priv_out_cv[32] = { 0 };
  sycl::uint block_words[64] = { 0 };
  sycl::uint block_len[4] = {
    blake3::BLOCK_LEN, blake3::BLOCK_LEN, blake3::BLOCK_LEN, blake3::BLOCK_LEN
  };
  sycl::uint flags_[4] = { 0 };

#pragma unroll(4)
  for (size_t i = 0; i < 8; i++) {
    sycl::uint tmp = *(key_words + i);

    in_cv[i + 8 * 0] = tmp;
    in_cv[i + 8 * 1] = tmp;
    in_cv[i + 8 * 2] = tmp;
    in_cv[i + 8 * 3] = tmp;
  }

  for (size_t i = 0; i < 16; i++) {
    blake3::words_from_le_bytes(input + blake3::BLOCK_LEN * i +
                                  blake3::CHUNK_LEN * 0,
                                block_words + 16 * 0);
    blake3::words_from_le_bytes(input + blake3::BLOCK_LEN * i +
                                  blake3::CHUNK_LEN * 1,
                                block_words + 16 * 1);
    blake3::words_from_le_bytes(input + blake3::BLOCK_LEN * i +
                                  blake3::CHUNK_LEN * 2,
                                block_words + 16 * 2);
    blake3::words_from_le_bytes(input + blake3::BLOCK_LEN * i +
                                  blake3::CHUNK_LEN * 3,
                                block_words + 16 * 3);

    switch (i) {
      case 0:
        flags_[0] = *(flags + 0) | blake3::CHUNK_START;
        flags_[1] = *(flags + 1) | blake3::CHUNK_START;
        flags_[2] = *(flags + 2) | blake3::CHUNK_START;
        flags_[3] = *(flags + 3) | blake3::CHUNK_START;

        blake3::v2::compress(
          in_cv, block_words, chunk_counter, block_len, flags_, priv_out_cv);
        break;
      case 15:
        flags_[0] = *(flags + 0) | blake3::CHUNK_END;
        flags_[1] = *(flags + 1) | blake3::CHUNK_END;
        flags_[2] = *(flags + 2) | blake3::CHUNK_END;
        flags_[3] = *(flags + 3) | blake3::CHUNK_END;

        blake3::v2::compress(
          in_cv, block_words, chunk_counter, block_len, flags_, out_cv);
        break;
      default:
        blake3::v2::compress(
          in_cv, block_words, chunk_counter, block_len, flags, priv_out_cv);
    }

    if (i < 15) {
#pragma unroll(4)
      for (size_t j = 0; j < 32; j++) {
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
  assert(chunk_count >= 1024);
  assert((chunk_count & (chunk_count - 1)) == 0);
  assert(wg_size <= (chunk_count >> 2));

  const size_t mem_size = static_cast<size_t>(blake3::BLOCK_LEN) * chunk_count;
  sycl::uint* mem = static_cast<sycl::uint*>(sycl::malloc_device(mem_size, q));
  const size_t mem_offset = (blake3::OUT_LEN >> 2) * chunk_count;

  sycl::event evt_0 = q.parallel_for<class kernelBlake3HashV2ChunkifyLeafNodes>(
    sycl::nd_range<1>{ sycl::range<1>{ chunk_count >> 2 },
                       sycl::range<1>{ wg_size } },
    [=](sycl::nd_item<1> it) {
      const size_t idx = it.get_global_linear_id();
      const sycl::ulong chunk_counter[4] = {
        static_cast<sycl::ulong>(idx << 2) + 0,
        static_cast<sycl::ulong>(idx << 2) + 1,
        static_cast<sycl::ulong>(idx << 2) + 2,
        static_cast<sycl::ulong>(idx << 2) + 3
      };
      const sycl::uint flags[4] = { 0 };

      blake3::v2::chunkify(blake3::IV,
                           chunk_counter,
                           flags,
                           input + (idx << 2) * blake3::CHUNK_LEN,
                           mem + mem_offset +
                             (idx << 2) * (blake3::OUT_LEN >> 2));
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
