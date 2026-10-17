#ifndef SIMON_BITSLICE_MEMORY_HPP
#define SIMON_BITSLICE_MEMORY_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>

#if !defined(__AVX2__)
#error "SIMON Bitsliced Engine requires AVX2 support. Compile with flags: -O3 -mavx2"
#endif

namespace simon {

constexpr size_t kAlignment = 32;
constexpr size_t kBlockSizeWords = 2;
constexpr size_t kParallelBlocks = 64;
constexpr size_t kBatchBytes = kParallelBlocks * (kBlockSizeWords * sizeof(uint32_t));

struct alignas(kAlignment) SimonBitsliceBatch {
    __m256i regs[16];

    constexpr SimonBitsliceBatch() noexcept {
        for (size_t i = 0; i < 16; ++i) {
            regs[i] = _mm256_setzero_si256();
        }
    }

    // 100% Pure-Vector Transpose: Zero scalar loops, zero GPR bounces, 
    // running entirely inside YMM vector execution units.
    inline void transpose_from_linear(const uint8_t* src) noexcept {
        const __m256i* src_vec = reinterpret_cast<const __m256i*>(src);
        __m256i t[16];

        // Load 512 bytes (16 YMM registers) directly into vector registers
        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            t[i] = _mm256_loadu_si256(src_vec + i);
        }

        // --- STAGE 1: 32-bit word interleaving ---
        __m256i u0  = _mm256_unpacklo_epi32(t[0], t[1]);   __m256i u1  = _mm256_unpackhi_epi32(t[0], t[1]);
        __m256i u2  = _mm256_unpacklo_epi32(t[2], t[3]);   __m256i u3  = _mm256_unpackhi_epi32(t[2], t[3]);
        __m256i u4  = _mm256_unpacklo_epi32(t[4], t[5]);   __m256i u5  = _mm256_unpackhi_epi32(t[4], t[5]);
        __m256i u6  = _mm256_unpacklo_epi32(t[6], t[7]);   __m256i u7  = _mm256_unpackhi_epi32(t[6], t[7]);
        __m256i u8  = _mm256_unpacklo_epi32(t[8], t[9]);   __m256i u9  = _mm256_unpackhi_epi32(t[8], t[9]);
        __m256i u10 = _mm256_unpacklo_epi32(t[10], t[11]); __m256i u11 = _mm256_unpackhi_epi32(t[10], t[11]);
        __m256i u12 = _mm256_unpacklo_epi32(t[12], t[13]); __m256i u13 = _mm256_unpackhi_epi32(t[12], t[13]);
        __m256i u14 = _mm256_unpacklo_epi32(t[14], t[15]); __m256i u15 = _mm256_unpackhi_epi32(t[14], t[15]);

        // --- STAGE 2: 64-bit block butterfly ---
        __m256i v0  = _mm256_unpacklo_epi64(u0, u2);   __m256i v1  = _mm256_unpackhi_epi64(u0, u2);
        __m256i v2  = _mm256_unpacklo_epi64(u1, u3);   __m256i v3  = _mm256_unpackhi_epi64(u1, u3);
        __m256i v4  = _mm256_unpacklo_epi64(u4, u6);   __m256i v5  = _mm256_unpackhi_epi64(u4, u6);
        __m256i v6  = _mm256_unpacklo_epi64(u5, u7);   __m256i v7  = _mm256_unpackhi_epi64(u5, u7);
        __m256i v8  = _mm256_unpacklo_epi64(u8, u10);  __m256i v9  = _mm256_unpackhi_epi64(u8, u10);
        __m256i v10 = _mm256_unpacklo_epi64(u9, u11);  __m256i v11 = _mm256_unpackhi_epi64(u9, u11);
        __m256i v12 = _mm256_unpacklo_epi64(u12, u14); __m256i v13 = _mm256_unpackhi_epi64(u12, u14);
        __m256i v14 = _mm256_unpacklo_epi64(u13, u15); __m256i v15 = _mm256_unpackhi_epi64(u13, u15);

        // --- STAGE 3: 128-bit cross-lane layout ---
        __m256i w[16];
        w[0]  = _mm256_permute2x128_si256(v0, v4, 0x20);   w[1]  = _mm256_permute2x128_si256(v1, v5, 0x20);
        w[2]  = _mm256_permute2x128_si256(v2, v6, 0x20);   w[3]  = _mm256_permute2x128_si256(v3, v7, 0x20);
        w[4]  = _mm256_permute2x128_si256(v0, v4, 0x31);   w[5]  = _mm256_permute2x128_si256(v1, v5, 0x31);
        w[6]  = _mm256_permute2x128_si256(v2, v6, 0x31);   w[7]  = _mm256_permute2x128_si256(v3, v7, 0x31);
        w[8]  = _mm256_permute2x128_si256(v8, v12, 0x20);  w[9]  = _mm256_permute2x128_si256(v9, v13, 0x20);
        w[10] = _mm256_permute2x128_si256(v10, v14, 0x20); w[11] = _mm256_permute2x128_si256(v11, v15, 0x20);
        w[12] = _mm256_permute2x128_si256(v8, v12, 0x31);  w[13] = _mm256_permute2x128_si256(v9, v13, 0x31);
        w[14] = _mm256_permute2x128_si256(v10, v14, 0x31); w[15] = _mm256_permute2x128_si256(v11, v15, 0x31);

        // --- STAGE 4: Intra-word Bit-Level SWAR Transposition (Pure Vector) ---
        const __m256i m1  = _mm256_set1_epi32(0x55555555);
        const __m256i m2  = _mm256_set1_epi32(0x33333333);
        const __m256i m4  = _mm256_set1_epi32(0x0F0F0F0F);
        const __m256i m8  = _mm256_set1_epi32(0x00FF00FF);
        const __m256i m16 = _mm256_set1_epi32(0x0000FFFF);

        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            __m256i x = w[i];

            __m256i t1 = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 1)), m1);
            x = _mm256_xor_si256(x, _mm256_xor_si256(t1, _mm256_slli_epi32(t1, 1)));

            __m256i t2 = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 2)), m2);
            x = _mm256_xor_si256(x, _mm256_xor_si256(t2, _mm256_slli_epi32(t2, 2)));

            __m256i t4 = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 4)), m4);
            x = _mm256_xor_si256(x, _mm256_xor_si256(t4, _mm256_slli_epi32(t4, 4)));

            __m256i t8 = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 8)), m8);
            x = _mm256_xor_si256(x, _mm256_xor_si256(t8, _mm256_slli_epi32(t8, 8)));

            __m256i t16 = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 16)), m16);
            x = _mm256_xor_si256(x, _mm256_xor_si256(t16, _mm256_slli_epi32(t16, 16)));

            regs[i] = x;
        }
    }
