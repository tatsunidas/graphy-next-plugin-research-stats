"""xoshiro128** の参照系列を、TypeScript 側とは別に（公開されているアルゴリズムの定義どおり）書いて出す。
usage: python3 tools/gen_rng_fixture.py > test/fixtures/rng.json
シードは splitmix32 で 4 語に広げる（src/core/rng.ts と同じ約束）。"""
import json

M = 0xFFFFFFFF


def splitmix32_seq(seed, n):
    a = seed & M
    out = []
    for _ in range(n):
        a = (a + 0x9E3779B9) & M
        t = a ^ (a >> 16)
        t = (t * 0x21F0AAAD) & M
        t ^= t >> 15
        t = (t * 0x735A2D97) & M
        t ^= t >> 15
        out.append(t & M)
    return out


def rotl(x, k):
    return ((x << k) | (x >> (32 - k))) & M


def xoshiro128ss(seed, n):
    s = splitmix32_seq(seed, 4)
    out = []
    for _ in range(n):
        result = (rotl((s[1] * 5) & M, 7) * 9) & M
        t = (s[1] << 9) & M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 11)
        out.append(result)
    return out


print(json.dumps({str(seed): xoshiro128ss(seed, 12) for seed in (0, 1, 12345, 4294967295)}, indent=1))
