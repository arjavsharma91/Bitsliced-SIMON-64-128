#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include "simon_engine.hpp"

void print_block(const char* label, const uint8_t* block, size_t len) {
    std::cout << label << ": ";
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(block[i]) << " ";
    }
    std::cout << std::dec << "\n";
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "  SIMON-64/128 AVX2 Bitsliced Engine Test Harness\n";
    std::cout << "=================================================\n\n";
    // official NSA test case btw
    uint8_t master_key[16] = {
        0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0a, 0x0b,
        0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1a, 0x1b
    };

    uint8_t single_plaintext[8] = {
        0x75, 0x6e, 0x64, 0x20, 0x6c, 0x69, 0x6b, 0x65
    };

    uint8_t expected_ciphertext[8] = {
        0x7a, 0xa0, 0xdf, 0xb9, 0x20, 0xfc, 0xc8, 0x44
    };

    constexpr int batch_blocks = 64;
    constexpr size_t batch_bytes = batch_blocks * 8;

    std::vector<uint8_t> input_batch(batch_bytes);
    std::vector<uint8_t> output_batch(batch_bytes, 0);

    for (int i = 0; i < batch_blocks; ++i) {
        std::memcpy(input_batch.data() + (i * 8), single_plaintext, 8);
    }

    print_block("Input Plaintext (Lane 0)", input_batch.data(), 8);
    print_block("Master Key              ", master_key, 16);
    print_block("Expected Ciphertext     ", expected_ciphertext, 8);
    std::cout << "-------------------------------------------------\n";

    try {
        simon::Simon64128Engine engine(master_key);

        std::cout << "Executing 44-round bitsliced batch encryption across 64 parallel blocks...\n";
        
        engine.encrypt_64_blocks(input_batch.data(), output_batch.data());

        std::cout << "Encryption completed successfully!\n\n";

        bool all_lanes_match = true;
        for (int i = 1; i < batch_blocks; ++i) {
            if (std::memcmp(output_batch.data(), output_batch.data() + (i * 8), 8) != 0) {
                all_lanes_match = false;
                std::cerr << "[!] Lane " << i << " output mismatch!\n";
            }
        }

        print_block("Computed Ciphertext (Lane 0)", output_batch.data(), 8);
