# Status
Passes official NSA test vector, awaiting further constant time testing and assembly inline fencing as well as further optimization, also hardening the CTR side of things. However as of now is doing well on constant time, as dudect verification has shown no evidence of side channel leakage (t < 2.5) for any GCC Optimization Flag (-O0, -O1, -O2, -O3, -Ofast)

# Bitsliced-SIMON-64-128
A bitsliced implementation of the NSAs SIMON 64/128 Feistel Block Cipher

# Objectives:
- Constant time across MSVC, G++, Clang, with the most aggressive optimization flags
- Greater Throughput than Crypto++ Implementation
