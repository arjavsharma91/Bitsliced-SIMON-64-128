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
