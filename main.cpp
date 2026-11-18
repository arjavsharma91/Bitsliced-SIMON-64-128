#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include "simon_ctr_encryption.hpp" // Make sure this matches your header file name

int main() {
    std::cout << "[*] Initializing Simon64/128 AVX2 Bitsliced Engine Test Harness...\n";

    // 1. Official NSA Test Case Key and Plaintext
    uint8_t master_key_bytes[16] = {
        0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0a, 0x0b,
        0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1a, 0x1b
    };

    uint32_t master_key_u32[4];
    std::memcpy(master_key_u32, master_key_bytes, 16);

    simon::Simon64128Engine engine(master_key_u32);

    size_t test_length = 520;
    std::vector<uint8_t> plaintext(test_length);
    std::vector<uint8_t> ciphertext(test_length);
    std::vector<uint8_t> decrypted(test_length);

    for (size_t i = 0; i < test_length; ++i) {
        plaintext[i] = static_cast<uint8_t>(i * 3 + 17);
    }

    uint64_t initial_counter = 0x0000000000000000ULL;
    uint32_t session_id = 0x12345678;

    std::cout << "[*] Encrypting " << test_length << " bytes using CTR mode...\n";
    engine.encrypt(plaintext.data(), ciphertext.data(), test_length, initial_counter, session_id);

    std::cout << "[*] Decrypting (re-generating keystream with same nonce/counter) ...\n";
    engine.encrypt(ciphertext.data(), decrypted.data(), test_length, initial_counter, session_id);
