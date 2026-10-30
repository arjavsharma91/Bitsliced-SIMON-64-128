// THIS FILE EXISTS TO BEGIN THE MAPPING OF THE TRANSPOSITION LOGIC FROM THE NSA SUPERCOP IMPLEMENTATION

#ifndef SIMON_BITSLICE_MEMORY_HPP
#define SIMON_BITSLICE_MEMORY_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>

#if !defined(__AVX2__)
#error "SIMON Bitsliced Engine requires AVX2 support. Compile with flags: -O3 -mavx2"
#endif

static inline void transpose_from_linear(u256 M[]) noexcept {
    const __m256i maskone = _mm256_set1_epi32(0x55555555);
    const __m256i maskfour = _mm256_set1_epi32(0x0f0f0f0f);
    const __m256i masktwo = _mm256_set1_epi32(0x33333333);
