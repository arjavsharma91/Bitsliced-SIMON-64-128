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

    void encrypt_64_blocs(const uint8_t* plaintext, uint8_t* ciphertext) const noexcept {
        SimonBitsliceBatch batch;
        batch.transpose_from_linear(plaintext);

        auto get_L_rot = [&batch](int k) -> __m256i {
            return batch.regs[k % 8];
        };

        for (int i = 0; i < 44; i++) {
            __m256i rk = key_expansion_.get_round_key(i);

            __m256i l0 = batch.regs[0], l1 = batch.regs[1], l2 = batch.regs[2], l3 = batch.regs[3];
            __m256i l4 = batch.regs[4], l5 = batch.regs[5], l6 = batch.regs[6], l7 = batch.regs[7];

            __m256i s1 = get_L_rot(1);
            __m256i s8 = get_L_rot(8);
            __m256i s2 = get_L_rot(2);

            __m256i f_res = _m2m56_xor_si256(_mm256_and_si256(s1, s8), s2);

            batch.regs[0] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[0], f_res), rk));
            batch.regs[1] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[1], f_res), rk));
            batch.regs[2] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[2], f_res), rk));
            batch.regs[3] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[4], f_res), rk));
            batch.regs[4] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[4], f_res), rk));
            batch.regs[5] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[5], f_res), rk));
            batch.regs[6] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[6], f_res), rk));
            batch.regs[7] = _mm256_xor_si256(_mm256_xor_si256(batch.regs[7], f_res), rk));

            batch.regs[8] = l0; batch.regs[9] = l1; batch.regs[10] = l2;batch.regs[11] = l3;
            batch.regs[12] = l4; batch.regs[13] = l5; batch.regs[14] = l6; batch.regs[15] = l7;
        }
        batch.transpose_to_linear(ciphertext);
    }
};
} // namespac

#endif // SIMON_ENGINE_HPP
