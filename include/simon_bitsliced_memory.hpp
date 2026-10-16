#ifndef SIMON_BITSLICE_MEMORY_HPP
#define SIMON_BITSLICE_MEMORY_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

#if !(defined (__AVX2__))
#error "SIMON Bitsliced Engine requires AVX2 support. Compile with flags: -O3 -mavx2"
#endif


namespace simon {

constexpr size_t kAlignment = 32;
constexpr size_t kBlockSizeWords = 2;
constexpr size_t kParallelBlocks = 64;
constexpr size_t kBatchBytes = kParallelBlocks * (kBlockSizeWords * sizeof(uint32_t));

struct alignas(kAlignment) SimonBitsliceMatch {
    __m256i regs[16];

    constexpr SimonBitsliceMatch() noexcept {
        for (size_t i = 0; i < 16; ++i) {
            regs[i] = _mm256_setzero_si256();
        }
    }
