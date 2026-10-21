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

public:
    explicit SimonKeyExpansion(const uint8_t master_key[16]) noexcept {
        uint32_t wk[4];
        std::memcpy(wk, master_key, 16);

        uint64_t z3 = 0xfc2ce51207a635dbULL;
        uint32_t c = 0xfffffffc;
