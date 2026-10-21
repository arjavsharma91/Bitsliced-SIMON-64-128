#ifndef SIMON_KEY_EXPANSION_HPP
#define SIMON_KEY_EXPANSION_HPP

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace simon {

class alignas(32) SimonKeyExpansion {
private:
    __m256i round_keys_[44];

    static inline uint32_t rotr32(uint32_t val, unsigned int r) noexcept {
        return (val >> r) | (val << (32 - r));
    }
