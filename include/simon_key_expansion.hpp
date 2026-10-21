#ifndef SIMON_KEY_EXPANSION_HPP
#define SIMON_KEY_EXPANSION_HPP

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace simon {

class alignas(32) SimonKeyExpansion {
private:
    __m256i round_keys[44];

    static inline uint32_t rotr32(uint32_t val, unsigned int r) noexcept {
        return (val >> r) | (val << (32 - r));
    }

public:
    explicit SimonKeyExpansion(const uint8_t master_key[16]) noexcept {
        uint32_t wk[4];
        std::memcpy(wk, master_key, 16);

        uint64_t z3 = 0xfc2ce51207a635dbULL;
        uint32_t c = 0xfffffffc;

        for (int i = 0; i < 44; i++) {
            round_keys[i] = _mm256_set1_epi32(wk[0]);

            uint32_t z_val = (z3 >> (61 - i)) & 1ULL;

            uint32_t tmp = rotr32(wk[3], 3);
            tmp ^= wk[1];
            uint32_t tmp2 = rotr32(tmp, 1);
            uint32_t next_k = c ^ z_val ^ wk[0] ^ tmp ^ tmp2;

            wk[0] = wk[1];
            wk[1] = wk[2];
            wk[2] = wk[3];
            wk[3] = next_k;
        }
    }

    inline __m256i get_round_key(int round) const noexcept {
        return _mm256_load_si256(&round_keys[round]);
    }
};

} // namespace simon

#endif // SIMON_KEY_EXPANSION_HPP
