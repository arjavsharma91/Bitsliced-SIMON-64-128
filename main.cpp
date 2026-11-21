#include <iostream>
#include <iomanip>
#include <array>
#include <cstdint>
#include <cstring>

#define private public
#include "simon_ctr_encryption.hpp"
#undef private

namespace {

uint32_t extract_lane0_word(const __m256i slices[8]) noexcept {
    __m256i tmp[8];
    for (int i = 0; i < 8; ++i) {
        tmp[i] = slices[i];
    }
    simon::Simon64128Engine::transpose_from_linear(tmp);

    alignas(32) uint32_t words[8] = {0};
    _mm256_store_si256(reinterpret_cast<__m256i*>(words), tmp[0]);
    return words[0];
}

void pack_lane0_word(uint32_t word, __m256i out_slices[8]) noexcept {
    __m256i v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, static_cast<int32_t>(word));
    out_slices[0] = v;
    for (int i = 1; i < 8; ++i) {
        out_slices[i] = _mm256_setzero_si256();
    }
    simon::Simon64128Engine::transpose_from_linear(out_slices);
}

void step_two_rounds(__m256i a[8], __m256i b[8], const __m256i rk[44][8], int r) noexcept {
    b[0] = _mm256_xor_si256(_mm256_xor_si256(rotl8(_mm256_xor_si256(_mm256_and_si256(a[0], a[7]), a[6])), rk[r][0]), b[0]);
    b[1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(a[0], rotl8(a[1])), rk[r][1]), rotl8(a[7])), b[1]);
    for (int j = 2; j < 8; ++j) {
        b[j] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(a[j - 1], rotl8(a[j])), rk[r][j]), a[j - 2]), b[j]);
    }

    for (int j = 2; j < 8; ++j) {
        a[j] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(b[j - 1], rotl8(b[j])), rk[r + 1][j]), b[j - 2]), a[j]);
    }
    a[0] = _mm256_xor_si256(_mm256_xor_si256(rotl8(_mm256_xor_si256(_mm256_and_si256(b[0], b[7]), b[6])), rk[r + 1][0]), a[0]);
    a[1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(b[0], rotl8(b[1])), rk[r + 1][1]), rotl8(b[7])), a[1]);
}

void print_bytes(const char* label, const uint8_t* data, size_t len) {
    std::cout << label;
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(data[i]) << (i + 1 < len ? " " : "");
    }
    std::cout << std::dec << "\n";
}

} // namespace

int main() {
    // 1. Official NSA Test Vector Inputs
    const uint8_t pt_bytes[8] = {0x75, 0x6e, 0x64, 0x20, 0x6c, 0x69, 0x6b, 0x65};
    const uint8_t k_bytes[16] = {
        0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0a, 0x0b,
        0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1a, 0x1b
    };
    const uint8_t expected_ct_bytes[8] = {0x7a, 0xa0, 0xdf, 0xb9, 0x20, 0xfc, 0xc8, 0x44};

    // Official NSA rk[0..43]
    const std::array<uint32_t, 44> expected_rk = {{
        0x03020100U, 0x0b0a0908U, 0x13121110U, 0x1b1a1918U,
        0x70a011c3U, 0xb770ec49U, 0x57e3e835U, 0xd397bc42U,
        0x94dcf81fU, 0xbf4b5f18U, 0x8e5dabb9U, 0xdbf4a863U,
        0xcd0c28fcU, 0x5cb69911U, 0x79f112a5U, 0x77205863U,
        0x99880c12U, 0x1ce97c58U, 0xc8ed2145U, 0xb800dbb8U,
        0xe86a2756U, 0x7c06d4ddU, 0xab52df0aU, 0x247f66a8U,
        0x53587ca6U, 0xd25c13f1U, 0x4583b64bU, 0x7d9c960dU,
        0xefbfc2f3U, 0x89ed8513U, 0x308dfc4eU, 0xbf1a2a36U,
        0xe1499d70U, 0x4ce4d2ffU, 0x32b7ebefU, 0xc47505c1U,
        0xd0e929e8U, 0x8fe484b9U, 0x42054beeU, 0xaf77bae2U,
        0x18199c02U, 0x719e3f1cU, 0x0c1cf793U, 0x15df4696U
    }};

    // Official NSA Pt_1..Pt_44 pairs: {Pt_i[1], Pt_i[0]}
    const std::array<std::pair<uint32_t, uint32_t>, 44> expected_pt_rounds = {{
        {0xfc8b8a84U, 0x656b696cU}, {0x154d4e7fU, 0xfc8b8a84U},
        {0xb2a6be7cU, 0x154d4e7fU}, {0xe0c1d225U, 0xb2a6be7cU},
        {0x8083c368U, 0xe0c1d225U}, {0x54bd334eU, 0x8083c368U},
        {0x2ca6a070U, 0x54bd334eU}, {0x35b04eecU, 0x2ca6a070U},
        {0x4efbefcfU, 0x35b04eecU}, {0x28f361c7U, 0x4efbefcfU},
        {0x320b0062U, 0x28f361c7U}, {0x3b2bc82cU, 0x320b0062U},
        {0x31e80836U, 0x3b2bc82cU}, {0xc03d61c5U, 0x31e80836U},
        {0x488c5c04U, 0xc03d61c5U}, {0x153449bfU, 0x488c5c04U},
        {0xa59de5feU, 0x153449bfU}, {0x968b68b8U, 0xa59de5feU},
        {0x3e5df649U, 0x968b68b8U}, {0x8b4e2236U, 0x3e5df649U},
        {0xfd0f5dccU, 0x8b4e2236U}, {0x09690941U, 0xfd0f5dccU},
        {0x73f9a7c2U, 0x09690941U}, {0x0353b2e0U, 0x73f9a7c2U},
        {0x2f4d70e4U, 0x0353b2e0U}, {0x202a8289U, 0x2f4d70e4U},
        {0xea64cd8bU, 0x202a8289U}, {0xb0eca9a9U, 0xea64cd8bU},
        {0xa6e0a8ceU, 0xb0eca9a9U}, {0xe203cf04U, 0xa6e0a8ceU},
        {0x1e656c93U, 0xe203cf04U}, {0x00c4c678U, 0x1e656c93U},
        {0xfcbfe003U, 0x00c4c678U}, {0x07bf948cU, 0xfcbfe003U},
        {0xdfe251dcU, 0x07bf948cU}, {0x1e0356a7U, 0xdfe251dcU},
        {0x770087a6U, 0x1e0356a7U}, {0x4de4cac3U, 0x770087a6U},
        {0x825e6641U, 0x4de4cac3U}, {0xefcea9a5U, 0x825e6641U},
        {0xebf45d9fU, 0xefcea9a5U}, {0xe5c97bedU, 0xebf45d9fU},
        {0xb9dfa07aU, 0xe5c97bedU}, {0x44c8fc20U, 0xb9dfa07aU}
    }};


