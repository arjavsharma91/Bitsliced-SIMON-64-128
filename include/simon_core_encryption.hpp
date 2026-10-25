#ifndef SIMON_ENGINE_HPP
#define SIMON_ENGINE_HPP

#include "simon_bitslice_memory.hpp"
#include "simon_key_expansion.hpp"
#include <immintrin.h>
#include <cstdint>

namespace simon {

class alignas(32) Simon64128Engine {
private:
    SimonKeyExpansion key_expansion_;

public:
    explicit Simon64128Engine(const uint8_t master_key[16]) noexcept
        : key_expansion_(master_key) {}

    void encrypt_64_blocks(const uint8_t* plaintext, uint8_t* ciphertext) const noexcept {
        SimonBitsliceBatch batch;
        batch.transpose_from_linear(plaintext);

        const __m256i mask_p0 = _mm256_set1_epi32(0x11111111); // Plane 0 bit slot
        const __m256i mask_p1 = _mm256_set1_epi32(0x22222222); // Plane 1 bit slot
        const __m256i mask_p2 = _mm256_set1_epi32(0x44444444); // Plane 2 bit slot
        const __m256i mask_p3 = _mm256_set1_epi32(0x88888888); // Plane 3 bit slot

        for (int round = 0; round < 44; ++round) {
            __m256i rk = key_expansion_.get_round_key(round);

            __m256i l[8], r[8];
            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                l[i] = batch.regs[i];
                r[i] = batch.regs[8 + i];
            }

            #define COMPUTE_S1(curr, prev, out) \
                { \
                    __m256i p0 = _mm256_and_si256(prev, mask_p3); \
                    __m256i p1 = _mm256_and_si256(curr, mask_p0); \
                    __m256i p2 = _mm256_and_si256(curr, mask_p1); \
                    __m256i p3 = _mm256_and_si256(curr, mask_p2); \
                    out = _mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(p0, 3), _mm256_slli_epi32(p1, 1)), \
                                          _mm256_or_si256(_mm256_slli_epi32(p2, 1), _mm256_slli_epi32(p3, 1))); \
                }

            #define COMPUTE_S2(curr, prev, out) \
                { \
                    __m256i p0 = _mm256_and_si256(prev, mask_p2); \
                    __m256i p1 = _mm256_and_si256(prev, mask_p3); \
                    __m256i p2 = _mm256_and_si256(curr, mask_p0); \
                    __m256i p3 = _mm256_and_si256(curr, mask_p1); \
                    out = _mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(p0, 2), _mm256_srli_epi32(p1, 2)), \
                                          _mm256_or_si256(_mm256_slli_epi32(p2, 2), _mm256_slli_epi32(p3, 2))); \
                }

            __m256i s1[8], s2[8], s8[8];

            COMPUTE_S1(l[0], l[7], s1[0]);
            COMPUTE_S1(l[1], l[0], s1[1]);
            COMPUTE_S1(l[2], l[1], s1[2]);
            COMPUTE_S1(l[3], l[2], s1[3]);
            COMPUTE_S1(l[4], l[3], s1[4]);
            COMPUTE_S1(l[5], l[4], s1[5]);
            COMPUTE_S1(l[6], l[5], s1[6]);
            COMPUTE_S1(l[7], l[6], s1[7]);

            COMPUTE_S2(l[0], l[7], s2[0]);
            COMPUTE_S2(l[1], l[0], s2[1]);
            COMPUTE_S2(l[2], l[1], s2[2]);
            COMPUTE_S2(l[3], l[2], s2[3]);
            COMPUTE_S2(l[4], l[3], s2[4]);
            COMPUTE_S2(l[5], l[4], s2[5]);
            COMPUTE_S2(l[6], l[5], s2[6]);
            COMPUTE_S2(l[7], l[6], s2[7]);

            s8[0] = l[2]; s8[1] = l[3]; s8[2] = l[4]; s8[3] = l[5];
            s8[4] = l[6]; s8[5] = l[7]; s8[6] = l[0]; s8[7] = l[1];

            #undef COMPUTE_S1
            #undef COMPUTE_S2

            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                __m256i f_res = _mm256_xor_si256(_mm256_and_si256(s1[i], s8[i]), s2[i]);
                
                batch.regs[i]     = _mm256_xor_si256(_mm256_xor_si256(r[i], f_res), rk);
                batch.regs[8 + i] = l[i];
            }
        }

        batch.transpose_to_linear(ciphertext);
    }
};

} // namespace simon

#endif // SIMON_ENGINE_HPP
