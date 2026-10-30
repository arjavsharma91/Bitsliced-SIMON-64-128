// THIS FILE EXISTS TO BEGIN THE MAPPING OF THE TRANSPOSITION LOGIC FROM THE NSA SUPERCOP IMPLEMENTATION

#ifndef SIMON_BITSLICE_MEMORY_HPP
#define SIMON_BITSLICE_MEMORY_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>

#if !defined(__AVX2__)
#error "SIMON Bitsliced Engine requires AVX2 support. Compile with flags: -O3 -mavx2"
#endif

static inline void transpose_from_linear(u256 m[]) noexcept {
    const __m256i maskone = _mm256_set1_epi32(0x55555555);
    const __m256i maskfour = _mm256_set1_epi32(0x0f0f0f0f);
    const __m256i masktwo = _mm256_set1_epi32(0x33333333);

    u256 q[4];

    q[0] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_si256(m[0], 4), m[4]), maskfour);
    m[4] = _mm256_xor_si256(m[4], q[0]);
    q[0] = _mm256_slli_si256(q[0], 4);
    m[0] = _mm256_xor_si256(m[0], q[0]);

    q[1] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_si256(m[1], 4), m[5]), maskfour);
    m[5] = _mm256_xor_si256(m[5], q[1]);
    q[1] = _mm256_slli_si256(q[1], 4);
    m[1] = _mm256_xor_si256(m[1], q[1]);

    q[2] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_si256(m[2], 4), m[6]), maskfour);
    m[6] = _mm256_xor_si256(m[6], q[2]);
    q[2] = _mm256_slli_si256(q[2], 4);
    m[2] = _mm256_xor_si256(m[2], q[2]);

    q[3] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_si256(m[3], 4), m[7]), maskfour);
    m[7] = _mm256_xor_si256(m[7], q[3]);
    q[3] = _mm256_slli_si256(q[3], 4);
    m[3] = _mm256_xor_si256(m[3], q[3]);
