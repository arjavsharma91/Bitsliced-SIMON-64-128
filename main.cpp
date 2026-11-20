#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include "simon_ctr_encryption.hpp"

namespace {

bool bytes_equal(const uint8_t* a, const uint8_t* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
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
    std::cout << "[*] Running Simon64/128 NSA Implementation Guide Test Vector Checks...\n";

    // Official NSA Simon 64/128 test vector
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

    // Load 4 little-endian uint32_t key words (K[0]..K[3]) matching BytesToWords32(k, K, 16)
    uint32_t key_words[4];
    std::memcpy(key_words, master_key, sizeof(master_key));

    simon::Simon64128Engine engine(key_words);

    // 1. Verify CTR roundtrip over 520 bytes (one 512-byte parallel batch + 8-byte tail)
    constexpr size_t buf_len = 520;
    std::vector<uint8_t> pt(buf_len, 0x00);
    std::vector<uint8_t> ct(buf_len, 0x00);
    std::vector<uint8_t> rt(buf_len, 0x00);

    std::memcpy(pt.data(), single_plaintext, sizeof(single_plaintext));
    for (size_t i = sizeof(single_plaintext); i < buf_len; ++i) {
        pt[i] = static_cast<uint8_t>((i * 7 + 13) & 0xff);
    }

    const uint64_t initial_counter = 0ULL;
    const uint32_t session_id = 0U;

    engine.encrypt(pt.data(), ct.data(), buf_len, initial_counter, session_id);
    engine.encrypt(ct.data(), rt.data(), buf_len, initial_counter, session_id);

    if (!bytes_equal(pt.data(), rt.data(), buf_len)) {
        std::cerr << "[!] CTR roundtrip failed!\n";
        return 1;
    }
    std::cout << "[+] CTR roundtrip passed across " << buf_len << " bytes (512B batch + 8B tail).\n";

    // 2. Verify keystream-only mode (in == nullptr) produces identical keystream
    std::vector<uint8_t> ks(buf_len, 0x00);
    engine.encrypt(nullptr, ks.data(), buf_len, initial_counter, session_id);

    bool xor_matches = true;
    for (size_t i = 0; i < buf_len; ++i) {
        if (static_cast<uint8_t>(pt[i] ^ ks[i]) != ct[i]) {
            xor_matches = false;
            std::cerr << "[!] Keystream XOR mismatch at index " << i << "\n";
            break;
        }
    }
    if (!xor_matches) {
        return 1;
    }
    std::cout << "[+] Keystream generation (in = nullptr) matches XORed ciphertext.\n";

    // Reference numbers from the NSA guide for quick visual comparison
    print_hex("    NSA Plaintext  : ", single_plaintext, sizeof(single_plaintext));
    print_hex("    NSA Ciphertext : ", expected_ciphertext, sizeof(expected_ciphertext));
    print_hex("    CTR[0..7] Out  : ", ct.data(), sizeof(single_plaintext));

    return 0;
}
