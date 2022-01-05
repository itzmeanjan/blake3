#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>

namespace blake3 {

void
round(sycl::uint4* const state, const sycl::uint* msg)
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

}
