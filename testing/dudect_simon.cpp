#include <cstdint>
#include <cstring>
#include <random>

#define DUDECT_IMPLEMENTATION
#include "dudect.h"
#include "simon_ctr_encryption.hpp"

namespace {

constexpr size_t kChunkBytes = 512;

const uint32_t kFixedKey[4] = {
    0x03020100U, 0x0b0a0908U, 0x13121110U, 0x1b1a1918U
};
const simon::Simon64128Engine g_engine(kFixedKey);
alignas(32) uint8_t g_output[kChunkBytes];
volatile uint8_t g_sink = 0;

std::mt19937_64& rng() {
    static std::mt19937_64 gen(0xC0FFEE123456789ULL);
    return gen;
}
