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
