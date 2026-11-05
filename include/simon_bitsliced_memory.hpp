// this uses hackers delight chapter 7 bit matrix transposition

#ifndef SIMON_BITSLICE_MEMORY_HPP
#define SIMON_BITSLICE_MEMORY_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>

#define rotr8(x) (_mm256_shuffle_epi8(x, _mm256_set1_epi32(0x0c0f0e0d)))
#define rotl8(x) (_mm256_shuffle_epi8(x, _mm256_set1_epi32(0x0e0d0c0f)))

#if !defined(__AVX2__)
#error "SIMON Bitsliced Engine requires AVX2 support. Compile with flags: -O3 -mavx2"
#endif

namespace simon {

struct alignas(32) Simon64128Engine {
    __m256i round_keys[44][8];

    private:

        static inline void transpose_from_linear(__m256i m[]) noexcept {
            const __m256i maskfour = _mm256_set1_epi32(0x0f0f0f0f);
            __m256i q[4];

            // Step one

            q[0] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[0], 4), m[4]), maskfour);
            m[4] = _mm256_xor_si256(m[4], q[0]);
            q[0] = _mm256_slli_epi32(q[0], 4);
            m[0] = _mm256_xor_si256(m[0], q[0]);

            q[1] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[1], 4), m[5]), maskfour);
            m[5] = _mm256_xor_si256(m[5], q[1]);
            q[1] = _mm256_slli_epi32(q[1], 4);
            m[1] = _mm256_xor_si256(m[1], q[1]);

            q[2] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[2], 4), m[6]), maskfour);
            m[6] = _mm256_xor_si256(m[6], q[2]);
            q[2] = _mm256_slli_epi32(q[2], 4);
            m[2] = _mm256_xor_si256(m[2], q[2]);

            q[3] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[3], 4), m[7]), maskfour);
            m[7] = _mm256_xor_si256(m[7], q[3]);
            q[3] = _mm256_slli_epi32(q[3], 4);
            m[3] = _mm256_xor_si256(m[3], q[3]);

            // Step two
            const __m256i masktwo = _mm256_set1_epi32(0x33333333);
    
            q[0] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[0], 2), m[2]), masktwo);
            m[2] = _mm256_xor_si256(m[2], q[0]);
            q[0] = _mm256_slli_epi32(q[0], 2);
            m[0] = _mm256_xor_si256(m[0], q[0]);

            q[1] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[1], 2), m[3]), masktwo);
            m[3] = _mm256_xor_si256(m[3], q[1]);
            q[1] = _mm256_slli_epi32(q[1], 2);
            m[1] = _mm256_xor_si256(m[1], q[1]);

            q[2] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[4], 2), m[6]), masktwo);
            m[6] = _mm256_xor_si256(m[6], q[2]);
            q[2] = _mm256_slli_epi32(q[2], 2);
            m[4] = _mm256_xor_si256(m[4], q[2]);

            q[3] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[5], 2), m[7]), masktwo);
            m[7] = _mm256_xor_si256(m[7], q[3]);
            q[3] = _mm256_slli_epi32(q[3], 2);
            m[5] = _mm256_xor_si256(m[5], q[3]);

            // Step three
            const __m256i maskone = _mm256_set1_epi32(0x55555555);
    
            q[0] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[0], 1), m[1]), maskone);
            m[1] = _mm256_xor_si256(m[1], q[0]);
            q[0] = _mm256_slli_epi32(q[0], 1);
            m[0] = _mm256_xor_si256(q[0], m[0]);

            q[1] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[2], 1), m[3]), maskone);
            m[3] = _mm256_xor_si256(m[3], q[1]);
            q[1] = _mm256_slli_epi32(q[1], 1);
            m[2] = _mm256_xor_si256(m[2], q[1]);

            q[2] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[4], 1), m[5]), maskone);
            m[5] = _mm256_xor_si256(q[2], m[5]);
            q[2] = _mm256_slli_epi32(q[2], 1);
            m[4] = _mm256_xor_si256(m[4], q[2]);

            q[3] = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(m[6], 1), m[7]), maskone);
            m[7] = _mm256_xor_si256(q[3], m[7]);
            q[3] = _mm256_slli_epi32(q[3], 1);
            m[6] = _mm256_xor_si256(m[6], q[3]);
        }

        inline void roundkey(__m256i rk[44][8], int r, __m256i v, __m256i c, __m256i d) noexcept {
            rk[r][7] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rotr8(rk[r-3][0]), rotr8(rk[r-1][3])), rotr8(rk[r-1][2])), rk[r-4][7]), rk[r-3][7]), d);
            rk[r][6] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rotr8(rk[r-1][1]), rotr8(rk[r-1][2])), rk[r-4][6]), rk[r-3][6]), rk[r-3][7]), d);
            rk[r][5] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rotr8(rk[r-1][0]), rotr8(rk[r-1][1])), rk[r-4][5]), rk[r-3][5]), rk[r-3][6]), d);
            rk[r][4] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rk[r-4][4], rotr8(rk[r-1][0])), rk[r-1][7]), rk[r-3][4]), rk[r-3][5]), d);
            rk[r][3] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rk[r-4][3], rk[r-1][6]), rk[r-3][3]), rk[r-1][7]), rk[r-3][4]), d);
            rk[r][2] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rk[r-4][2], rk[r-1][5]), rk[r-3][2]), rk[r-1][6]), rk[r-3][3]), d);
            rk[r][1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rk[r-4][1], rk[r-1][4]), rk[r-3][1]), rk[r-1][5]), rk[r-3][2]), c);
            rk[r][0] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(rk[r-4][0], rk[r-1][3]), rk[r-3][0]), rk[r-1][4]), rk[r-3][1]), v);
        }

        inline void expandkey(__m256i rk[44][8], const uint32_t f[4]) noexcept {
            for (int i=0 ; i < 4 ; i++) {
                __m256i broadcast = _mm256_set1_epi32(f[i]);
                for (int j=0 ; j < 8 ; j++) {
                    rk[i][j] = broadcast;
                }
                transpose_from_linear(rk[i]);
            }

            static const uint8_t v_flags[40] = {
                1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
                0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 1, 0, 0, 0
            };

            const __m256i c = _mm256_set1_epi32(0xFFFFFF00);
            const __m256i d = _mm256_set1_epi32(0xFFFFFFFF);

            #pragma GCC unroll 40
            for (int i = 4; i < 44; ++i) {
                __m256i V = v_flags[i-4] ? d : c;
                roundkey(rk, i, V, c, d);
            }
        }

    public:
        inline void core_encryption(__m256i a[8], __m256i b[8]) const noexcept {
            #pragma GCC unroll 22
            for (int i=0; i < 44; i += 2) {
                b[0] = _mm256_xor_si256(_mm256_xor_si256(rotl8(_mm256_xor_si256(_mm256_and_si256(a[0], a[7]), a[6])), rk[i][0]), b[0]);
                b[1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(a[0], rotl8(a[1])), rk[i][1]), rotl(a[7])), b[1]);
                #pragma GCC unroll 6
                for (int j=2; j < 8; ++i) {
                    b[j] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(a[j-1], rotl8(a[j])), rk[i][j]), a[j-2]), b[j]);
};

} // namespace simon

#endif // SIMON_BITSLICE_MEMORY_HPP
