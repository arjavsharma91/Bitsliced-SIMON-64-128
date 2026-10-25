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

        // Bitmasks corresponding to the 4 bit-planes packed into every 4-bit dword envelope
        const __m256i mask_p0 = _mm256_set1_epi32(0x11111111); // Plane 0 bit slot
        const __m256i mask_p1 = _mm256_set1_epi32(0x22222222); // Plane 1 bit slot
        const __m256i mask_p2 = _mm256_set1_epi32(0x44444444); // Plane 2 bit slot
        const __m256i mask_p3 = _mm256_set1_epi32(0x88888888); // Plane 3 bit slot

        for (int round = 0; round < 44; ++round) {
            __m256i rk = key_expansion_.get_round_key(round);

            // Bind the active 16 state slots to physical registers
            __m256i l0 = batch.regs[0], l1 = batch.regs[1], l2 = batch.regs[2], l3 = batch.regs[3];
            __m256i l4 = batch.regs[4], l5 = batch.regs[5], l6 = batch.regs[6], l7 = batch.regs[7];
            
            __m256i r0 = batch.regs[8], r1 = batch.regs[9], r2 = batch.regs[10], r3 = batch.regs[11];
            __m256i r4 = batch.regs[12], r5 = batch.regs[13], r6 = batch.regs[14], r7 = batch.regs[15];

            // Macro-optimized True Left-Circular Rotations matching your Layer 1 layout
            // S^1 (Shift up 1 bit-plane position; wraps from plane 3 of prev register to plane 0 of current)
            #define COMPUTE_S1(curr, prev, out) \
                { \
                    __m256i p0 = _mm256_and_si256(prev, mask_p3); \
                    __m256i p1 = _mm256_and_si256(curr, mask_p0); \
                    __m256i p2 = _mm256_and_si256(curr, mask_p1); \
                    __m256i p3 = _mm256_and_si256(curr, mask_p2); \
                    out = _mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(p0, 3), _mm256_slli_epi32(p1, 1)), \
                                          _mm256_or_si256(_mm256_slli_epi32(p2, 1), _mm256_slli_epi32(p3, 1))); \
                }

            // S^2 (Shift up 2 bit-plane positions; wraps planes 2 and 3 of prev register)
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

            // Execute S^1 across the register chain
            COMPUTE_S1(l0, l7, s1[0]); COMPUTE_S1(l1, l0, s1[1]); COMPUTE_S1(l2, l1, s1[2]); COMPUTE_S1(l3, l2, s1[3]);
            COMPUTE_S1(l4, l3, s1[4]); COMPUTE_S1(l5, l4, s1[5]); COMPUTE_S1(l6, l5, s1[6]); COMPUTE_S1(l7, l6, s1[7]);

            // Execute S^2 across the register chain
            COMPUTE_S2(l0, l6, s2[0]); COMPUTE_S2(l1, l7, s2[1]); COMPUTE_S2(l2, l0, s2[2]); COMPUTE_S2(l3, l1, s2[3]);
            COMPUTE_S2(l4, l2, s2[4]); COMPUTE_S2(l5, l3, s2[5]); COMPUTE_S2(l6, l4, s2[6]); COMPUTE_S2(l7, l5, s2[7]);

            // Execute S^8 (Pure, zero-overhead register re-indexing: 8 bits / 4 planes = 2 steps)
            s8[0] = l2; s8[1] = l3; s8[2] = l4; s8[3] = l5;
            s8[4] = l6; s8[5] = l7; s8[6] = l0; s8[7] = l1;

            #undef COMPUTE_S1
            #undef COMPUTE_S2

            // 3. Feistel update layer evaluating f(L) and updating the state in-place
            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                __m256i f_res = _mm256_xor_si256(_mm256_and_si256(s1[i], s8[i]), s2[i]);
                
                // L_new = R_old ^ f(L) ^ rk (Pre-broadcasted key word automatically maps bit-to-plane)
                batch.regs[i] = _mm256_xor_si256(_mm256_xor_si256(r[i], f_res), rk);
                
                // R_new = L_old
                batch.regs[8 + i] = l[i];
            }
        }

        batch.transpose_to_linear(ciphertext);
    }
};

} // namespace simon

#endif // SIMON_ENGINE_HPP
