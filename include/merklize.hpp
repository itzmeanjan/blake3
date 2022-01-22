#pragma once
#include "blake3.hpp"

sycl::cl_ulong
merklize(sycl::queue& q,
         sycl::uint* const __restrict leaf_nodes,
         size_t i_size, // leaf nodes size in bytes
         size_t leaf_cnt,
         sycl::uint* const __restrict intermediates,
         size_t o_size, // intermediate nodes size in bytes
         size_t itmd_cnt,
         size_t wg_size)
{
  // A binary merkle tree with N -many leaf
  // nodes should have (N - 1) -many intermediates
  //
  // Note N = power of 2
  assert(leaf_cnt == itmd_cnt + 1);

  // because each leaf node is 32 -bytes wide
  assert(i_size == (leaf_cnt << 5));
  // if a binary tree has N -many leaf nodes, where N = 2 ^ i | i = {1, 2, ...}
  // then # -of intermediate nodes in that binary merkle tree should be =
  // leaf_count - 1
  assert(o_size == ((itmd_cnt + 1) << 5));
  // both input and output allocation has same size
  assert(i_size == o_size);

  // only tree with power of 2 many leaf nodes
  // can be merklized by this implementation
  assert((leaf_cnt & (leaf_cnt - 1)) == 0);

  // At first N -many leaf nodes to be merged into
  // N/ 2 -many intermediate nodes, which are living
  // just above leaf nodes
  const size_t work_item_cnt = leaf_cnt >> 1;

  // validate whether this work group size can be used for next kernel dispatch
  assert(wg_size <= work_item_cnt);
  // ensure all dispatched work groups have equal many
  // active work-items
  assert(work_item_cnt % wg_size == 0);

  // # -of 32 -bit unsigned integers, which can be contiguously placed
  // on output memory allocation
  const size_t elm_cnt = o_size >> 2;
  const size_t i_offset = 0;
  const size_t o_offset = elm_cnt >> 1;

  // computes all intermediate nodes which are living just above leaf nodes of
  // binary merkle tree
  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelBinaryMerklizationUsingBLAKE3Phase0>(
      sycl::nd_range<1>{ sycl::range<1>{ work_item_cnt },
                         sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        // copy message words to array allocated on private memory
        // because these message words are permuted, when `blake3::merge`
        // function is invoked
        sycl::uint msg_words[16];

#pragma unroll 16 // attempt to fully parallelize the loop !
        for (size_t i = 0; i < 16; i++) {
          msg_words[i] = *(leaf_nodes + i_offset + (idx << 4) + i);
        }

        blake3::v1::merge(msg_words, intermediates + o_offset + (idx << 3));
      });
  });

  // these many kernel dispatch rounds still remaining
  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(work_item_cnt)));

  std::vector<sycl::event> evts_0;
  // reserve enough space in vector so that all events obtained as result
  // of enqueuing kernel execution commands, can be accomodated
  evts_0.reserve(rounds + 1);

  // storing event obtained as result of enqueuing first phase of kernel
  // execution command, where leaf nodes are paired to generate all intermediate
  // nodes living immediately above them
  evts_0.push_back(evt_0);

  for (size_t r = 0; r < rounds; r++) {
    // multiple rounds of kernel dispatches, where intermediate nodes are being
    // computed from already computed (in previous dispatch round) intermediate
    // nodes
    sycl::event evt_1 = q.submit([&](sycl::handler& h) {
      // note dependency chain being built !
      //
      // each dispatch round depends on previously enqueued dispatch round
      h.depends_on(evts_0.at(r));

      const size_t work_item_cnt_ = work_item_cnt >> (r + 1);
      const size_t wg_size_ =
        wg_size <= work_item_cnt_ ? wg_size : work_item_cnt_;

      const size_t i_offset_ = o_offset >> r;
      const size_t o_offset_ = i_offset_ >> 1;

      h.parallel_for<class kernelBinaryMerklizationUsingBLAKE3Phase1>(
        sycl::nd_range<1>{ sycl::range<1>{ work_item_cnt_ },
                           sycl::range<1>{ wg_size_ } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          // copy message words to array allocated on private memory
          // because these message words are permuted, when `blake3::merge`
          // function is invoked
          sycl::uint msg_words[16];

#pragma unroll 16 // attempt to fully parallelize the loop !
          for (size_t i = 0; i < 16; i++) {
            msg_words[i] = *(intermediates + i_offset_ + (idx << 4) + i);
          }

          blake3::v1::merge(msg_words, intermediates + o_offset_ + (idx << 3));
        });
    });
    evts_0.push_back(evt_1);
  }

  // wait for last kernel dispatch round, where root of binary merkle tree is
  // computed !
  evts_0.at(rounds).wait();

  // time execution of all enqueued kernels with nanosecond level granularity
  sycl::cl_ulong ts = 0;
  for (size_t r = 0; r < rounds + 1; r++) {
    ts += time_event(evts_0.at(r));
  }

  // return total kernel execution cost, in terms of nanosecond
  return ts;
}
