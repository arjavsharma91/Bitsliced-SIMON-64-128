#ifndef SIMON_ENGINE_HPP
#define SIMON_ENGINE_HPP

#include "simon_bitslice_memory.hpp"
#include "simon_key_expansion.hpp"
#include <immintrin.h>
#include <cstdint>

namespace simon {

class alignas(32) Simon64128Engine {
private:
    SimonKeyExpansion key_expansion_;

public:
    explicit Simon64128Engine(const uint8_t master_key[16]) noexcept
        : key_expansion_(master_key) {}

    void encrypt_64_blocs(const uint8_t* plaintext, uint8_t* ciphertext) const noexcept {
        SimonBitsliceBatch batch;
        batch.transpose_from_linear(plaintext);

        auto get_L_rot = [&batch](int k) -> __m256i {
            return batch.regs[k % 8];
        };
