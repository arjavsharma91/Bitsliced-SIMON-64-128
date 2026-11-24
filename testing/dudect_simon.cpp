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

void fill_random_bytes(uint8_t* dst, size_t len) {
    auto& gen = rng();
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t r = gen();
        std::memcpy(dst + i, &r, sizeof(r));
        i += 8;
    }
    if (i < len) {
        uint64_t r = gen();
        std::memcpy(dst + i, &r, len - i);
    }
}

} // namespace

extern "C" {

void prepare_inputs(dudect_config_t* c, uint8_t* input_data, uint8_t* classes) {
    auto& gen = rng();
    for (size_t i = 0; i < c->number_measurements; ++i) {
        uint8_t cls = static_cast<uint8_t>(gen() & 1ULL);
        classes[i] = cls;
        uint8_t* slot = input_data + i * c->chunk_size;
        if (cls == 0) {
            std::memset(slot, 0x00, c->chunk_size);
        } else {
            fill_random_bytes(slot, c->chunk_size);
        }
    }
}

uint8_t do_one_computation(uint8_t* data) {
    g_engine.encrypt(data, g_output, kChunkBytes, 0ULL, 1U);
    g_sink ^= g_output[0] ^ g_output[kChunkBytes - 1];
    return g_sink;
}

} // extern "C"

int main() {
    dudect_config_t config = {
        .chunk_size = kChunkBytes,
        .number_measurements = 10000,
    }

    dudect_ctx_t ctx;
    dudect_init(&ctx, &config);

    dudect_state_t state = DUDECT_NO_LEAKAGE_EVIDENCE_YET;
    while (state == DUDECT_NO_LEAKAGE_EVIDENCE_YET) {
        state = dudect_main(&ctx);
    }

    dudect_free(&ctx);
    return static_cast<int>(state);
}
