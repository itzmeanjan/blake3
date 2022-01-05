#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>

// Mixes blake3 hash state both column-wise and diagonally
void
round(sycl::uint4* const state, const sycl::uint* msg);
