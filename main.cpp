#include <iostream>
#include <iomanip>
#include <array>
#include <cstdint>
#include <cstring>

#define private public
#include "simon_ctr_encryption.hpp"
#undef private

namespace {

uint32_t extract_lane0_word(const __m256i slices[8]) noexcept {
    __m256i tmp[8];
    for (int i = 0; i < 8; ++i) {
        tmp[i] = slices[i];
    }
    simon::Simon64128Engine::transpose_from_linear(tmp);

    alignas(32) uint32_t words[8] = {0};
    _mm256_store_si256(reinterpret_cast<__m256i*>(words), tmp[0]);
    return words[0];
}

void pack_lane0_word(uint32_t word, __m256i out_slices[8]) noexcept {
    __m256i v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, static_cast<int32_t>(word));
    out_slices[0] = v;
    for (int i = 1; i < 8; ++i) {
        out_slices[i] = _mm256_setzero_si256();
    }
    simon::Simon64128Engine::transpose_from_linear(out_slices);
}

void step_two_rounds(__m256i a[8], __m256i b[8], const __m256i rk[44][8], int r) noexcept {
    b[0] = _mm256_xor_si256(_mm256_xor_si256(rotl8(_mm256_xor_si256(_mm256_and_si256(a[0], a[7]), a[6])), rk[r][0]), b[0]);
    b[1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(a[0], rotl8(a[1])), rk[r][1]), rotl8(a[7])), b[1]);
    for (int j = 2; j < 8; ++j) {
        b[j] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(a[j - 1], rotl8(a[j])), rk[r][j]), a[j - 2]), b[j]);
    }

    for (int j = 2; j < 8; ++j) {
        a[j] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(b[j - 1], rotl8(b[j])), rk[r + 1][j]), b[j - 2]), a[j]);
    }
    a[0] = _mm256_xor_si256(_mm256_xor_si256(rotl8(_mm256_xor_si256(_mm256_and_si256(b[0], b[7]), b[6])), rk[r + 1][0]), a[0]);
    a[1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(b[0], rotl8(b[1])), rk[r + 1][1]), rotl8(b[7])), a[1]);
}
