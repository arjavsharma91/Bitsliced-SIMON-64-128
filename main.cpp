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
