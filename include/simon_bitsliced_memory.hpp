#ifndef SIMON_BITSLICE_MEMORY_HPP
#define SIMON_BITSLICE_MEMORY_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

#if !defined(__AVX2__)
#error "SIMON Bitsliced Engine requires AVX2 support. Compile with flags: -O3 -mavx2"
#endif

namespace simon {

constexpr size_t kAlignment = 32;
constexpr size_t kBlockSizeWords = 2;
constexpr size_t kParallelBlocks = 64;
constexpr size_t kBatchBytes = kParallelBlocks * (kBlockSizeWords * sizeof(uint32_t));

struct alignas(kAlignment) SimonBitsliceBatch {
    __m256i regs[16];

    constexpr SimonBitsliceBatch() noexcept {
        for (size_t i = 0; i < 16; ++i) {
            regs[i] = _mm256_setzero_si256();
        }
    }

    inline void transpose_from_linear(const uint8_t* src) noexcept {
        uint32_t blocks_x[64];
        uint32_t blocks_y[64];
        
        #pragma GCC unroll 64
        for (int i = 0; i < 64; ++i) {
            std::memcpy(&blocks_x[i], src + (i * 8), 4);
            std::memcpy(&blocks_y[i], src + (i * 8) + 4, 4);
        }

        uint64_t planes[64];
        
        #pragma GCC unroll 64
        for (int bit = 0; bit < 64; ++bit) {
            uint64_t plane_bits = 0;
            #pragma GCC unroll 64
            for (int block = 0; block < 64; ++block) {
                uint32_t val = (bit < 32) ? blocks_x[block] : blocks_y[block];
                int shift = bit % 32;
                if ((val >> shift) & 1U) {
                    plane_bits |= (1ULL << block);
                }
            }
            planes[bit] = plane_bits;
        }

        #pragma GCC unroll 16
        for (int reg_idx = 0; reg_idx < 16; ++reg_idx) {
            uint64_t p0 = planes[reg_idx * 4 + 0];
            uint64_t p1 = planes[reg_idx * 4 + 1];
            uint64_t p2 = planes[reg_idx * 4 + 2];
            uint64_t p3 = planes[reg_idx * 4 + 3];

            regs[reg_idx] = _mm256_set_epi64x(
                static_cast<int64_t>(p3),
                static_cast<int64_t>(p2),
                static_cast<int64_t>(p1),
                static_cast<int64_t>(p0)
            );
        }
    }

    inline void transpose_to_linear(uint8_t* dest) const noexcept {
        uint64_t planes[64];

        #pragma GCC unroll 16
        for (int reg_idx = 0; reg_idx < 16; ++reg_idx) {
            alignas(32) uint64_t lanes[4];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), regs[reg_idx]);
            planes[reg_idx * 4 + 0] = lanes[0];
            planes[reg_idx * 4 + 1] = lanes[1];
            planes[reg_idx * 4 + 2] = lanes[2];
            planes[reg_idx * 4 + 3] = lanes[3];
        }

        #pragma GCC unroll 64
        for (int block = 0; block < 64; ++block) {
            uint32_t x = 0;
            uint32_t y = 0;

            #pragma GCC unroll 32
            for (int bit = 0; bit < 32; ++bit) {
                if ((planes[bit] >> block) & 1ULL) {
                    x |= (1U << bit);
                }
                if ((planes[bit + 32] >> block) & 1ULL) {
                    y |= (1U << bit);
                }
            }

            std::memcpy(dest + (block * 8), &x, 4);
            std::memcpy(dest + (block * 8) + 4, &y, 4);
        }
    }
};

} // namespace simon

#endif
