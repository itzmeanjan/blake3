#!/bin/bash

# Use this script to run test on all possible variants
# of BLAKE3 implementation
#
# By variation, I mean proprocessor tricks employed to
# decide which sources are finally compiled

make clean

BLAKE3_SIMD_LANES=2 make; make clean
BLAKE3_SIMD_LANES=4 make; make clean
BLAKE3_SIMD_LANES=8 make; make clean
BLAKE3_SIMD_LANES=16 make; make clean
