#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include "simon_ctr_encryption.hpp"

namespace {

inline uint32_t rotl32(uint32_t x, int r) noexcept {
    return (x << r) | (x >> (32 - r));
}

// Standalone Simon 64/128 block cipher (m = 4 key words, T = 44 rounds, z_2 sequence)
void simon64_128_encrypt_block(const uint8_t key_bytes[16],
                               const uint8_t pt_bytes[8],
                               uint8_t ct_bytes[8]) noexcept {
    // NSA guide: BytesToWords32 converts little-endian byte arrays to uint32_t words
    uint32_t k[4] = {0};
    std::memcpy(k, key_bytes, 16);

    // Simon 64/128 uses z_2 (62-bit sequence repeated over 44 rounds)
    constexpr uint64_t z2 = 0x7369f885192c0ef5ULL;

    uint32_t rk[44] = {0};
    rk[0] = k[0];
    rk[1] = k[1];
    rk[2] = k[2];
    rk[3] = k[3];

    constexpr uint32_t c = 0xfffffffcU; // 2^32 - 4
    for (int i = 4; i < 44; ++i) {
        uint32_t tmp = rotl32(rk[i - 1], -3) ^ rk[i - 3];
        tmp = tmp ^ rotl32(tmp, -1);
        uint8_t z_bit = static_cast<uint8_t>((z2 >> ((i - 4) % 62)) & 1ULL);
        rk[i] = static_cast<uint32_t>(~rk[i - 4]) ^ tmp ^ static_cast<uint32_t>(z_bit) ^ 3U;
    }
    (void)c;

    // NSA guide: Pt = (Pt[1], Pt[0]) where Pt[0] is the lower/first word and Pt[1] is the upper/second word
    uint32_t pt_words[2] = {0};
    std::memcpy(pt_words, pt_bytes, 8);
    uint32_t x = pt_words[1];
    uint32_t y = pt_words[0];

    for (int i = 0; i < 44; ++i) {
        uint32_t f = (rotl32(x, 1) & rotl32(x, 8)) ^ rotl32(x, 2);
        uint32_t next_x = y ^ f ^ rk[i];
        y = x;
        x = next_x;
    }

    // Ct = (Ct[1], Ct[0]) -> WordsToBytes32 writes Ct[0] first, then Ct[1]
    uint32_t ct_words[2] = {y, x};
    std::memcpy(ct_bytes, ct_words, 8);
}

void print_hex(const char* label, const uint8_t* data, size_t n) {
    std::cout << label;
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(data[i]) << (i + 1 < n ? " " : "");
    }
    std::cout << std::dec << "\n";
}

} // namespace

int main() {
    const uint8_t master_key[16] = {
        0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0a, 0x0b,
        0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1a, 0x1b
    };

    const uint8_t single_plaintext[8] = {
        0x75, 0x6e, 0x64, 0x20, 0x6c, 0x69, 0x6b, 0x65
    };

    const uint8_t expected_ciphertext[8] = {
        0x7a, 0xa0, 0xdf, 0xb9, 0x20, 0xfc, 0xc8, 0x44
    };

    uint8_t actual_ciphertext[8] = {0};
    simon64_128_encrypt_block(master_key, single_plaintext, actual_ciphertext);

    print_hex("Plaintext : ", single_plaintext, 8);
    print_hex("Expected  : ", expected_ciphertext, 8);
    print_hex("Actual    : ", actual_ciphertext, 8);

    if (std::memcmp(actual_ciphertext, expected_ciphertext, 8) == 0) {
        std::cout << "[+] PASS: Matches official NSA SIMON 64/128 test vector!\n";
        return 0;
    }

    std::cout << "[-] FAIL: Mismatch against official NSA SIMON 64/128 test vector.\n";
    return 1;
}
