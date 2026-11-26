// SPDX-License-Identifier: Apache-2.0
//
// CPU transforms for lossless KV cache checkpoint packets:
// - Bit-exact 64-pivot low-rank float32 predictor for 128x128 recurrent state matrices
//   (portable scalar reference, AVX2+FMA+BMI2+LZCNT runtime dispatch on x86-64, NEON on aarch64)
// - 4-lane aligned-u32 branchless residual low-bit pack/unpack
// - Byte-lane shuffle (width 1, 2, 4, 20, 32) and compound page value/scale split-shuffle

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KV_X86_AVX2 1
#define KV_AVX2_FMA __attribute__((target("avx2,fma,bmi2,lzcnt")))
#elif defined(__aarch64__)
#include <arm_neon.h>
#define KV_ARM_NEON 1
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

using U = uint32_t;
using B = uint8_t;

constexpr int MAT_DIM = 128;
constexpr int MAT_ELEMS = MAT_DIM * MAT_DIM; // 16384
constexpr int PIVOTS = 64;
constexpr int REM_COLS = MAT_DIM - PIVOTS; // 64
constexpr U MAX_LANE_U32S = 3968;          // ceil(4096 * 31 / 32)

static inline U f2u(float f) {
    U u;
    std::memcpy(&u, &f, 4);
    return u;
}
static inline float u2f(U u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static inline U load32(const B *p) {
    U u;
    std::memcpy(&u, p, 4);
    return u;
}
static inline void store32(B *p, U u) { std::memcpy(p, &u, 4); }
static inline uint64_t load64(const B *p) {
    uint64_t u;
    std::memcpy(&u, p, 8);
    return u;
}

static inline unsigned bit_width32(U v) {
    if (v == 0) return 0u;
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanReverse(&idx, v);
    return static_cast<unsigned>(idx) + 1u;
#else
    return 32u - static_cast<unsigned>(__builtin_clz(v));
#endif
}

static inline U bounded(U w) {
    U e = (w >> 23) & 255u;
    return (e >= 77u && e <= 177u) ? w : 0u;
}
static inline U order_word(U w) { return w ^ ((0u - (w >> 31)) | 0x80000000u); }
static inline U residual_word(U w, U p) {
    U d = order_word(w) - order_word(p);
    return (d << 1) ^ (0u - (d >> 31));
}
static inline U restore_word(U z, U p) {
    U b = order_word(p) + ((z >> 1) ^ (0u - (z & 1u)));
    return b ^ (((b >> 31) - 1u) | 0x80000000u);
}

// Enforce round-to-nearest-even and disable FTZ/DAZ for exact cross-ISA IEEE-754 parity.
// Fast path checks the hardware FP control register directly; under the standard ABI
// (RNE, FTZ=0, DAZ=0) this executes one register read and no serializing writes.
struct ScopedFpEnv {
#if defined(KV_X86_AVX2)
    unsigned int saved = 0;
    bool restore = false;
    ScopedFpEnv() {
        unsigned int mxcsr = _mm_getcsr();
        // Bits 14:13 = rounding mode (00 = RNE), bit 15 = FTZ, bit 6 = DAZ.
        constexpr unsigned int mask = (3u << 13) | (1u << 15) | (1u << 6);
        if ((mxcsr & mask) != 0u) {
            saved = mxcsr;
            _mm_setcsr(mxcsr & ~mask);
            restore = true;
        }
    }
    ~ScopedFpEnv() {
        if (restore) _mm_setcsr(saved);
    }
#elif defined(KV_ARM_NEON)
    uint64_t saved = 0;
    bool restore = false;
    ScopedFpEnv() {
        uint64_t fpcr = 0;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        // Bits 23:22 = RMode (00 = RN), bit 24 = FZ, bit 19 = FZ16.
        constexpr uint64_t mask = (uint64_t(3) << 22) | (uint64_t(1) << 24);
        if ((fpcr & mask) != 0u) {
            saved = fpcr;
            asm volatile("msr fpcr, %0" : : "r"(fpcr & ~mask));
            restore = true;
        }
    }
    ~ScopedFpEnv() {
        if (restore) asm volatile("msr fpcr, %0" : : "r"(saved));
    }
#else
    std::fenv_t saved;
    bool active = false;
    ScopedFpEnv() {
        if (std::fegetround() != FE_TONEAREST && std::feholdexcept(&saved) == 0) {
            std::fesetround(FE_TONEAREST);
            active = true;
        }
    }
    ~ScopedFpEnv() {
        if (active) std::fesetenv(&saved);
    }
#endif
};

// =========================================================================
// Portable scalar reference predictor (normative specification)
// =========================================================================
template <bool Decode>
static void predict_scalar(B *words, U *z) {
    float p[MAT_DIM][MAT_DIM] = {};
    float u[MAT_DIM];
    float v[MAT_DIM];
    bool done[MAT_DIM] = {};

    for (int k = 0; k < PIVOTS; ++k) {
        int c = -1;
        for (int j = 0; j < MAT_DIM; ++j) {
            if (done[j]) continue;
            U pred = bounded(f2u(p[k][j]));
            B *addr = words + 4 * (j * MAT_DIM + k);
            U w;
            if constexpr (Decode) {
                w = restore_word(z[k * MAT_DIM + j], pred);
                store32(addr, w);
            } else {
                w = load32(addr);
                z[k * MAT_DIM + j] = residual_word(w, pred);
            }
            U obs = bounded(w);
            v[j] = u2f(bounded(f2u(u2f(obs) - u2f(pred))));
            if (c < 0 || (f2u(v[j]) & 0x7fffffffu) > (f2u(v[c]) & 0x7fffffffu)) {
                c = j;
            }
        }
        U pivot_bits = f2u(v[c]);
        float pivot = v[c];
        done[c] = true;
        for (int i = k + 1; i < MAT_DIM; ++i) {
            U pred = bounded(f2u(p[i][c]));
            B *addr = words + 4 * (c * MAT_DIM + i);
            U w;
            if constexpr (Decode) {
                w = restore_word(z[i * MAT_DIM + c], pred);
                store32(addr, w);
            } else {
                w = load32(addr);
                z[i * MAT_DIM + c] = residual_word(w, pred);
            }
            U obs = bounded(w);
            U delta = bounded(f2u(u2f(obs) - u2f(pred)));
            u[i] = pivot_bits ? u2f(bounded(f2u(u2f(delta) / pivot))) : 0.0f;
            for (int j = 0; j < MAT_DIM; ++j) {
                if (!done[j]) {
                    p[i][j] = std::fmaf(u[i], v[j], p[i][j]);
                }
            }
        }
    }
    for (int i = PIVOTS; i < MAT_DIM; ++i) {
        for (int j = 0; j < MAT_DIM; ++j) {
            if (done[j]) continue;
            U pred = bounded(f2u(p[i][j]));
            B *addr = words + 4 * (j * MAT_DIM + i);
            if constexpr (Decode) {
                U w = restore_word(z[i * MAT_DIM + j], pred);
                store32(addr, w);
            } else {
                U w = load32(addr);
                z[i * MAT_DIM + j] = residual_word(w, pred);
            }
        }
    }
}

// =========================================================================
// 4-lane aligned-u32 branchless low-bit pack & bounds-checked unpack
// =========================================================================
// GCC does not inline non-target-attributed helpers into KV_AVX2_FMA functions,
// so hot scalar helpers inside KV_PACK_4LANE_BODY / KV_UNPACK_4LANE_BODY are
// parameterized by the caller's target-matched inline helpers.
#define KV_PACK_4LANE_BODY(BW32)                                                                  \
    U streams[4][MAX_LANE_U32S + 2];                                                              \
    U *p0 = streams[0], *p1 = streams[1], *p2 = streams[2], *p3 = streams[3];                     \
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;                                                      \
    unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0;                                                      \
    for (int i = 0; i < MAT_ELEMS; i += 4) {                                                      \
        U v0 = words[i + 0], v1 = words[i + 1], v2 = words[i + 2], v3 = words[i + 3];             \
        unsigned w0 = BW32(v0), w1 = BW32(v1);                                                    \
        unsigned w2 = BW32(v2), w3 = BW32(v3);                                                    \
        widths[i + 0] = B(w0); widths[i + 1] = B(w1);                                             \
        widths[i + 2] = B(w2); widths[i + 3] = B(w3);                                             \
        unsigned l0 = w0 - (w0 != 0u), l1 = w1 - (w1 != 0u);                                      \
        unsigned l2 = w2 - (w2 != 0u), l3 = w3 - (w3 != 0u);                                      \
        a0 |= uint64_t(v0 & ((1u << l0) - 1u)) << b0; b0 += l0;                                   \
        *p0 = U(a0); unsigned f0 = b0 >> 5; p0 += f0; a0 >>= (f0 << 5); b0 &= 31u;                \
        a1 |= uint64_t(v1 & ((1u << l1) - 1u)) << b1; b1 += l1;                                   \
        *p1 = U(a1); unsigned f1 = b1 >> 5; p1 += f1; a1 >>= (f1 << 5); b1 &= 31u;                \
        a2 |= uint64_t(v2 & ((1u << l2) - 1u)) << b2; b2 += l2;                                   \
        *p2 = U(a2); unsigned f2 = b2 >> 5; p2 += f2; a2 >>= (f2 << 5); b2 &= 31u;                \
        a3 |= uint64_t(v3 & ((1u << l3) - 1u)) << b3; b3 += l3;                                   \
        *p3 = U(a3); unsigned f3 = b3 >> 5; p3 += f3; a3 >>= (f3 << 5); b3 &= 31u;                \
    }                                                                                             \
    if (b0) *p0++ = U(a0);                                                                        \
    if (b1) *p1++ = U(a1);                                                                        \
    if (b2) *p2++ = U(a2);                                                                        \
    if (b3) *p3++ = U(a3);                                                                        \
    U *ends[4] = {p0, p1, p2, p3};                                                                \
    size_t pos = 0;                                                                               \
    for (int l = 0; l < 4; ++l) {                                                                 \
        U nw = U(ends[l] - streams[l]);                                                           \
        lane_u32s[l] = static_cast<uint16_t>(nw);                                                 \
        std::memcpy(low_out + pos, streams[l], size_t(nw) * 4);                                   \
        pos += size_t(nw) * 4;                                                                    \
    }                                                                                             \
    return pos;

#define KV_UNPACK_4LANE_BODY(LD32, LD64, SAFE_ELEM)                                               \
    size_t nw[4];                                                                                 \
    size_t total_bytes = 0;                                                                       \
    for (int l = 0; l < 4; ++l) {                                                                 \
        nw[l] = lane_u32s[l];                                                                     \
        if (nw[l] > MAX_LANE_U32S) return -1;                                                     \
        total_bytes += nw[l] * 4;                                                                 \
    }                                                                                             \
    if (total_bytes != low_len) return -1;                                                        \
    const B *p0 = low_in;                                                                         \
    const B *p1 = p0 + nw[0] * 4;                                                                 \
    const B *p2 = p1 + nw[1] * 4;                                                                 \
    const B *p3 = p2 + nw[2] * 4;                                                                 \
    const size_t safe0 = nw[0] >= 2 ? (nw[0] - 1) * 32 : 0;                                       \
    const size_t safe1 = nw[1] >= 2 ? (nw[1] - 1) * 32 : 0;                                       \
    const size_t safe2 = nw[2] >= 2 ? (nw[2] - 1) * 32 : 0;                                       \
    const size_t safe3 = nw[3] >= 2 ? (nw[3] - 1) * 32 : 0;                                       \
    size_t bit0 = 0, bit1 = 0, bit2 = 0, bit3 = 0;                                                \
    int i = 0;                                                                                    \
    unsigned bad = 0;                                                                             \
    for (; i < MAT_ELEMS; i += 4) {                                                               \
        if (!((bit0 < safe0) & (bit1 < safe1) & (bit2 < safe2) & (bit3 < safe3))) break;          \
        unsigned w0 = widths[i + 0], w1 = widths[i + 1], w2 = widths[i + 2], w3 = widths[i + 3];  \
        bad |= (w0 > 32u) | (w1 > 32u) | (w2 > 32u) | (w3 > 32u);                                 \
        unsigned nz0 = (w0 != 0u), nz1 = (w1 != 0u), nz2 = (w2 != 0u), nz3 = (w3 != 0u);          \
        unsigned l0 = (w0 - nz0) & 31u, l1 = (w1 - nz1) & 31u;                                    \
        unsigned l2 = (w2 - nz2) & 31u, l3 = (w3 - nz3) & 31u;                                    \
        U hi0 = nz0 << l0;                                                                        \
        U r0 = U(LD64(p0 + ((bit0 >> 5) << 2)) >> (bit0 & 31u));                                  \
        words[i + 0] = hi0 | (r0 & (hi0 - nz0));                                                  \
        bit0 += l0;                                                                               \
        U hi1 = nz1 << l1;                                                                        \
        U r1 = U(LD64(p1 + ((bit1 >> 5) << 2)) >> (bit1 & 31u));                                  \
        words[i + 1] = hi1 | (r1 & (hi1 - nz1));                                                  \
        bit1 += l1;                                                                               \
        U hi2 = nz2 << l2;                                                                        \
        U r2 = U(LD64(p2 + ((bit2 >> 5) << 2)) >> (bit2 & 31u));                                  \
        words[i + 2] = hi2 | (r2 & (hi2 - nz2));                                                  \
        bit2 += l2;                                                                               \
        U hi3 = nz3 << l3;                                                                        \
        U r3 = U(LD64(p3 + ((bit3 >> 5) << 2)) >> (bit3 & 31u));                                  \
        words[i + 3] = hi3 | (r3 & (hi3 - nz3));                                                  \
        bit3 += l3;                                                                               \
    }                                                                                             \
    if (bad) return -1;                                                                           \
    for (; i < MAT_ELEMS; i += 4) {                                                               \
        if (SAFE_ELEM(p0, nw[0], bit0, widths[i + 0], words[i + 0]) < 0) return -1;               \
        if (SAFE_ELEM(p1, nw[1], bit1, widths[i + 1], words[i + 1]) < 0) return -1;               \
        if (SAFE_ELEM(p2, nw[2], bit2, widths[i + 2], words[i + 2]) < 0) return -1;               \
        if (SAFE_ELEM(p3, nw[3], bit3, widths[i + 3], words[i + 3]) < 0) return -1;               \
    }                                                                                             \
    const B *ps[4] = {p0, p1, p2, p3};                                                            \
    const size_t bits[4] = {bit0, bit1, bit2, bit3};                                              \
    for (int l = 0; l < 4; ++l) {                                                                 \
        if (((bits[l] + 31u) >> 5) != nw[l]) return -1;                                           \
        unsigned rem = static_cast<unsigned>(bits[l] & 31u);                                      \
        if (rem != 0u) {                                                                          \
            U last = LD32(ps[l] + (nw[l] - 1) * 4);                                               \
            if ((last >> rem) != 0u) return -1;                                                   \
        }                                                                                         \
    }                                                                                             \
    return 0;

static size_t pack_4lane_impl(const U *words, B *widths, B *low_out, uint16_t lane_u32s[4]) {
    KV_PACK_4LANE_BODY(bit_width32)
}

static inline int unpack_4lane_safe_elem(const B *p, size_t nwords, size_t &bit, unsigned w, U &out) {
    if (w > 32u) return -1;
    if (w == 0u) {
        out = 0u;
        return 0;
    }
    unsigned low = w - 1u;
    size_t total_bits = nwords * 32u;
    if (bit + low > total_bits) return -1;
    size_t wi = bit >> 5;
    uint64_t q = (wi < nwords) ? uint64_t(load32(p + wi * 4)) : 0u;
    if (wi + 1 < nwords) q |= uint64_t(load32(p + (wi + 1) * 4)) << 32;
    U hi = 1u << low;
    out = hi | (U(q >> (bit & 31u)) & (hi - 1u));
    bit += low;
    return 0;
}

static int unpack_4lane_impl(U *words, const B *widths, const B *low_in, size_t low_len,
                             const uint16_t lane_u32s[4]) {
    KV_UNPACK_4LANE_BODY(load32, load64, unpack_4lane_safe_elem)
}

#if defined(KV_X86_AVX2)
KV_AVX2_FMA static inline U f2u_avx2(float f) {
    U u;
    std::memcpy(&u, &f, 4);
    return u;
}
KV_AVX2_FMA static inline float u2f_avx2(U u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
KV_AVX2_FMA static inline U load32_avx2(const B *p) {
    U u;
    std::memcpy(&u, p, 4);
    return u;
}
KV_AVX2_FMA static inline uint64_t load64_avx2(const B *p) {
    uint64_t u;
    std::memcpy(&u, p, 8);
    return u;
}
KV_AVX2_FMA static inline unsigned bit_width32_avx2(U v) {
    return 32u - static_cast<unsigned>(_lzcnt_u32(v));
}
KV_AVX2_FMA static inline U bounded_avx2(U w) {
    U e = (w >> 23) & 255u;
    return (e >= 77u && e <= 177u) ? w : 0u;
}
KV_AVX2_FMA static inline U order_word_avx2(U w) { return w ^ ((0u - (w >> 31)) | 0x80000000u); }
KV_AVX2_FMA static inline U residual_word_avx2(U w, U p) {
    U d = order_word_avx2(w) - order_word_avx2(p);
    return (d << 1) ^ (0u - (d >> 31));
}
KV_AVX2_FMA static inline U restore_word_avx2(U z, U p) {
    U b = order_word_avx2(p) + ((z >> 1) ^ (0u - (z & 1u)));
    return b ^ (((b >> 31) - 1u) | 0x80000000u);
}
KV_AVX2_FMA static inline int unpack_4lane_safe_elem_avx2(const B *p, size_t nwords, size_t &bit,
                                                          unsigned w, U &out) {
    if (w > 32u) return -1;
    if (w == 0u) {
        out = 0u;
        return 0;
    }
    unsigned low = w - 1u;
    size_t total_bits = nwords * 32u;
    if (bit + low > total_bits) return -1;
    size_t wi = bit >> 5;
    uint64_t q = (wi < nwords) ? uint64_t(load32_avx2(p + wi * 4)) : 0u;
    if (wi + 1 < nwords) q |= uint64_t(load32_avx2(p + (wi + 1) * 4)) << 32;
    U hi = 1u << low;
    out = hi | (U(q >> (bit & 31u)) & (hi - 1u));
    bit += low;
    return 0;
}
KV_AVX2_FMA static size_t pack_4lane_avx2(const U *words, B *widths, B *low_out, uint16_t lane_u32s[4]) {
    KV_PACK_4LANE_BODY(bit_width32_avx2)
}
KV_AVX2_FMA static int unpack_4lane_avx2(U *words, const B *widths, const B *low_in, size_t low_len,
                                         const uint16_t lane_u32s[4]) {
    KV_UNPACK_4LANE_BODY(load32_avx2, load64_avx2, unpack_4lane_safe_elem_avx2)
}

KV_AVX2_FMA static inline __m256i bounded_i(__m256i w) {
    __m256i biased = _mm256_add_epi32(_mm256_slli_epi32(w, 1), _mm256_set1_epi32(int(0x80000000u - (77u << 24))));
    __m256i keep = _mm256_cmpgt_epi32(_mm256_set1_epi32(int(0x80000000u + (101u << 24))), biased);
    return _mm256_and_si256(w, keep);
}
KV_AVX2_FMA static inline __m256i order_i(__m256i w) {
    __m256i sign = _mm256_srai_epi32(w, 31);
    return _mm256_xor_si256(w, _mm256_or_si256(sign, _mm256_set1_epi32(int(0x80000000u))));
}
KV_AVX2_FMA static inline __m256i residual_i(__m256i w, __m256i p) {
    __m256i d = _mm256_sub_epi32(order_i(w), order_i(p));
    return _mm256_xor_si256(_mm256_slli_epi32(d, 1), _mm256_srai_epi32(d, 31));
}
KV_AVX2_FMA static inline __m256i restore_i(__m256i z, __m256i p) {
    __m256i neg = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_and_si256(z, _mm256_set1_epi32(1)));
    __m256i d = _mm256_xor_si256(_mm256_srli_epi32(z, 1), neg);
    __m256i b = _mm256_add_epi32(order_i(p), d);
    __m256i top = _mm256_sub_epi32(_mm256_srli_epi32(b, 31), _mm256_set1_epi32(1));
    return _mm256_xor_si256(b, _mm256_or_si256(top, _mm256_set1_epi32(int(0x80000000u))));
}
KV_AVX2_FMA static inline __m128i bounded_128(__m128i w) {
    __m128i biased = _mm_add_epi32(_mm_slli_epi32(w, 1), _mm_set1_epi32(int(0x80000000u - (77u << 24))));
    __m128i keep = _mm_cmpgt_epi32(_mm_set1_epi32(int(0x80000000u + (101u << 24))), biased);
    return _mm_and_si128(w, keep);
}
KV_AVX2_FMA static inline __m128i order_128(__m128i w) {
    __m128i sign = _mm_srai_epi32(w, 31);
    return _mm_xor_si128(w, _mm_or_si128(sign, _mm_set1_epi32(int(0x80000000u))));
}
KV_AVX2_FMA static inline __m128i residual_128(__m128i w, __m128i p) {
    __m128i d = _mm_sub_epi32(order_128(w), order_128(p));
    return _mm_xor_si128(_mm_slli_epi32(d, 1), _mm_srai_epi32(d, 31));
}
KV_AVX2_FMA static inline __m128i restore_128(__m128i z, __m128i p) {
    __m128i neg = _mm_sub_epi32(_mm_setzero_si128(), _mm_and_si128(z, _mm_set1_epi32(1)));
    __m128i d = _mm_xor_si128(_mm_srli_epi32(z, 1), neg);
    __m128i b = _mm_add_epi32(order_128(p), d);
    __m128i top = _mm_sub_epi32(_mm_srli_epi32(b, 31), _mm_set1_epi32(1));
    return _mm_xor_si128(b, _mm_or_si128(top, _mm_set1_epi32(int(0x80000000u))));
}

KV_AVX2_FMA static void transpose_128x128_avx2(const U *src, U *dst) {
    for (int r = 0; r < 128; r += 8) {
        for (int c = 0; c < 128; c += 8) {
            __m256 r0 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 0) * 128 + c));
            __m256 r1 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 1) * 128 + c));
            __m256 r2 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 2) * 128 + c));
            __m256 r3 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 3) * 128 + c));
            __m256 r4 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 4) * 128 + c));
            __m256 r5 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 5) * 128 + c));
            __m256 r6 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 6) * 128 + c));
            __m256 r7 = _mm256_loadu_ps(reinterpret_cast<const float *>(src + (r + 7) * 128 + c));
            __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
            __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
            __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 0) * 128 + r), _mm256_permute2f128_ps(s0, s4, 0x20));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 1) * 128 + r), _mm256_permute2f128_ps(s1, s5, 0x20));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 2) * 128 + r), _mm256_permute2f128_ps(s2, s6, 0x20));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 3) * 128 + r), _mm256_permute2f128_ps(s3, s7, 0x20));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 4) * 128 + r), _mm256_permute2f128_ps(s0, s4, 0x31));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 5) * 128 + r), _mm256_permute2f128_ps(s1, s5, 0x31));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 6) * 128 + r), _mm256_permute2f128_ps(s2, s6, 0x31));
            _mm256_storeu_ps(reinterpret_cast<float *>(dst + (c + 7) * 128 + r), _mm256_permute2f128_ps(s3, s7, 0x31));
        }
    }
}

KV_AVX2_FMA static inline void gemm_4xN_avx2(float *pred, const float (*U_col)[4], const float *V, int K) {
    for (int j = 0; j < 128; j += 16) {
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        const float *vp = V + j;
        for (int k = 0; k < K; ++k) {
            __m256 v0 = _mm256_load_ps(vp), v1 = _mm256_load_ps(vp + 8);
            vp += 128;
            __m256 u0 = _mm256_set1_ps(U_col[k][0]);
            a00 = _mm256_fmadd_ps(u0, v0, a00);
            a01 = _mm256_fmadd_ps(u0, v1, a01);
            __m256 u1 = _mm256_set1_ps(U_col[k][1]);
            a10 = _mm256_fmadd_ps(u1, v0, a10);
            a11 = _mm256_fmadd_ps(u1, v1, a11);
            __m256 u2 = _mm256_set1_ps(U_col[k][2]);
            a20 = _mm256_fmadd_ps(u2, v0, a20);
            a21 = _mm256_fmadd_ps(u2, v1, a21);
            __m256 u3 = _mm256_set1_ps(U_col[k][3]);
            a30 = _mm256_fmadd_ps(u3, v0, a30);
            a31 = _mm256_fmadd_ps(u3, v1, a31);
        }
        _mm256_store_ps(pred + 0 * 128 + j, a00);
        _mm256_store_ps(pred + 0 * 128 + j + 8, a01);
        _mm256_store_ps(pred + 1 * 128 + j, a10);
        _mm256_store_ps(pred + 1 * 128 + j + 8, a11);
        _mm256_store_ps(pred + 2 * 128 + j, a20);
        _mm256_store_ps(pred + 2 * 128 + j + 8, a21);
        _mm256_store_ps(pred + 3 * 128 + j, a30);
        _mm256_store_ps(pred + 3 * 128 + j + 8, a31);
    }
}

template <bool Decode>
KV_AVX2_FMA static inline void apply_rem_2_avx2(U *dst, __m256 p0, __m256 p1) {
    __m256i pr0 = bounded_i(_mm256_castps_si256(p0));
    __m256i pr1 = bounded_i(_mm256_castps_si256(p1));
    __m256i in0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst));
    __m256i in1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + 8));
    __m256i out0 = Decode ? restore_i(in0, pr0) : residual_i(in0, pr0);
    __m256i out1 = Decode ? restore_i(in1, pr1) : residual_i(in1, pr1);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), out0);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 8), out1);
}

template <bool Decode>
KV_AVX2_FMA static inline void gemm_8x64_fused_avx2(U *rem_io, const float (*U_col)[8], const float *V) {
    for (int half = 0; half < 8; half += 4) {
        for (int j = 0; j < REM_COLS; j += 16) {
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
            __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
            __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
            for (int k = 0; k < PIVOTS; ++k) {
                const float *vp = V + k * REM_COLS + j;
                __m256 v0 = _mm256_load_ps(vp), v1 = _mm256_load_ps(vp + 8);
                __m256 u0 = _mm256_set1_ps(U_col[k][half + 0]);
                a00 = _mm256_fmadd_ps(u0, v0, a00);
                a01 = _mm256_fmadd_ps(u0, v1, a01);
                __m256 u1 = _mm256_set1_ps(U_col[k][half + 1]);
                a10 = _mm256_fmadd_ps(u1, v0, a10);
                a11 = _mm256_fmadd_ps(u1, v1, a11);
                __m256 u2 = _mm256_set1_ps(U_col[k][half + 2]);
                a20 = _mm256_fmadd_ps(u2, v0, a20);
                a21 = _mm256_fmadd_ps(u2, v1, a21);
                __m256 u3 = _mm256_set1_ps(U_col[k][half + 3]);
                a30 = _mm256_fmadd_ps(u3, v0, a30);
                a31 = _mm256_fmadd_ps(u3, v1, a31);
            }
            apply_rem_2_avx2<Decode>(rem_io + (half + 0) * REM_COLS + j, a00, a01);
            apply_rem_2_avx2<Decode>(rem_io + (half + 1) * REM_COLS + j, a10, a11);
            apply_rem_2_avx2<Decode>(rem_io + (half + 2) * REM_COLS + j, a20, a21);
            apply_rem_2_avx2<Decode>(rem_io + (half + 3) * REM_COLS + j, a30, a31);
        }
    }
}

template <bool Decode>
KV_AVX2_FMA static inline int pivot_row_avx2(U *row, const float *pred, const U *active_mask, float *vk) {
    __m256i abs_mask = _mm256_set1_epi32(0x7fffffff);
    __m256i best_mag = _mm256_set1_epi32(-1), best_idx = _mm256_setzero_si256();
    __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i eight = _mm256_set1_epi32(8), neg1 = _mm256_set1_epi32(-1);
    for (int j = 0; j < 128; j += 8) {
        __m256i act = _mm256_load_si256(reinterpret_cast<const __m256i *>(active_mask + j));
        __m256i pr = bounded_i(_mm256_castps_si256(_mm256_load_ps(pred + j)));
        __m256i cur = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + j));
        __m256i obs, out;
        if constexpr (Decode) {
            obs = restore_i(cur, pr);
            out = _mm256_blendv_epi8(cur, obs, act);
        } else {
            obs = cur;
            out = _mm256_blendv_epi8(cur, residual_i(cur, pr), act);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(row + j), out);
        __m256 diff = _mm256_sub_ps(_mm256_castsi256_ps(bounded_i(obs)), _mm256_castsi256_ps(pr));
        __m256i dv = _mm256_and_si256(bounded_i(_mm256_castps_si256(diff)), act);
        _mm256_store_ps(vk + j, _mm256_castsi256_ps(dv));
        __m256i mag = _mm256_blendv_epi8(neg1, _mm256_and_si256(dv, abs_mask), act);
        __m256i gt = _mm256_cmpgt_epi32(mag, best_mag);
        best_mag = _mm256_blendv_epi8(best_mag, mag, gt);
        best_idx = _mm256_blendv_epi8(best_idx, idx, gt);
        idx = _mm256_add_epi32(idx, eight);
    }
    alignas(32) int32_t mags[8], idxs[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(mags), best_mag);
    _mm256_store_si256(reinterpret_cast<__m256i *>(idxs), best_idx);
    int best_c = -1;
    int32_t best_m = -1;
    for (int l = 0; l < 8; ++l) {
        if (mags[l] > best_m || (mags[l] == best_m && idxs[l] < best_c)) {
            best_m = mags[l];
            best_c = idxs[l];
        }
    }
    return best_c;
}

template <bool Decode>
KV_AVX2_FMA static inline void trsm_4row_avx2(U *z, int base, int limit, const int *cols,
                                              const float *pivots, const float (*V_piv)[64],
                                              float (*U_col)[4]) {
    alignas(16) float pred_piv[64][4] = {};
    for (int k = 0; k < limit; ++k) {
        int col = cols[k];
        U *p = z + base * 128 + col;
        __m128i cur = _mm_setr_epi32(int(p[0 * 128]), int(p[1 * 128]), int(p[2 * 128]), int(p[3 * 128]));
        __m128i pr = bounded_128(_mm_castps_si128(_mm_load_ps(pred_piv[k])));
        __m128i obs;
        alignas(16) U out[4];
        if constexpr (Decode) {
            obs = restore_128(cur, pr);
            _mm_store_si128(reinterpret_cast<__m128i *>(out), obs);
        } else {
            obs = cur;
            _mm_store_si128(reinterpret_cast<__m128i *>(out), residual_128(cur, pr));
        }
        p[0 * 128] = out[0];
        p[1 * 128] = out[1];
        p[2 * 128] = out[2];
        p[3 * 128] = out[3];
        float piv = pivots[k];
        __m128 uk = _mm_setzero_ps();
        if (f2u_avx2(piv)) {
            __m128 diff = _mm_sub_ps(_mm_castsi128_ps(bounded_128(obs)), _mm_castsi128_ps(pr));
            __m128 delta = _mm_castsi128_ps(bounded_128(_mm_castps_si128(diff)));
            uk = _mm_castsi128_ps(bounded_128(_mm_castps_si128(_mm_div_ps(delta, _mm_set1_ps(piv)))));
        }
        _mm_store_ps(U_col[k], uk);
        const float *vk = V_piv[k];
        int m = k + 1;
        for (; m + 3 < limit; m += 4) {
            _mm_store_ps(pred_piv[m + 0], _mm_fmadd_ps(uk, _mm_set1_ps(vk[m + 0]), _mm_load_ps(pred_piv[m + 0])));
            _mm_store_ps(pred_piv[m + 1], _mm_fmadd_ps(uk, _mm_set1_ps(vk[m + 1]), _mm_load_ps(pred_piv[m + 1])));
            _mm_store_ps(pred_piv[m + 2], _mm_fmadd_ps(uk, _mm_set1_ps(vk[m + 2]), _mm_load_ps(pred_piv[m + 2])));
            _mm_store_ps(pred_piv[m + 3], _mm_fmadd_ps(uk, _mm_set1_ps(vk[m + 3]), _mm_load_ps(pred_piv[m + 3])));
        }
        for (; m < limit; ++m) {
            _mm_store_ps(pred_piv[m], _mm_fmadd_ps(uk, _mm_set1_ps(vk[m]), _mm_load_ps(pred_piv[m])));
        }
    }
}

template <bool Decode>
KV_AVX2_FMA static inline void trsm_8row_avx2(U *z, int base, const int *cols, const float *pivots,
                                              const float (*V_piv)[64], float (*U_col)[8]) {
    alignas(32) float pred_piv[64][8] = {};
    for (int k = 0; k < PIVOTS; ++k) {
        int col = cols[k];
        U *p = z + base * 128 + col;
        __m256i cur = _mm256_setr_epi32(
            int(p[0 * 128]), int(p[1 * 128]), int(p[2 * 128]), int(p[3 * 128]),
            int(p[4 * 128]), int(p[5 * 128]), int(p[6 * 128]), int(p[7 * 128]));
        __m256i pr = bounded_i(_mm256_castps_si256(_mm256_load_ps(pred_piv[k])));
        __m256i obs;
        alignas(32) U out[8];
        if constexpr (Decode) {
            obs = restore_i(cur, pr);
            _mm256_store_si256(reinterpret_cast<__m256i *>(out), obs);
        } else {
            obs = cur;
            _mm256_store_si256(reinterpret_cast<__m256i *>(out), residual_i(cur, pr));
        }
        for (int r = 0; r < 8; ++r) p[r * 128] = out[r];
        float piv = pivots[k];
        __m256 uk = _mm256_setzero_ps();
        if (f2u_avx2(piv)) {
            __m256 diff = _mm256_sub_ps(_mm256_castsi256_ps(bounded_i(obs)), _mm256_castsi256_ps(pr));
            __m256 delta = _mm256_castsi256_ps(bounded_i(_mm256_castps_si256(diff)));
            uk = _mm256_castsi256_ps(bounded_i(_mm256_castps_si256(_mm256_div_ps(delta, _mm256_set1_ps(piv)))));
        }
        _mm256_store_ps(U_col[k], uk);
        const float *vk = V_piv[k];
        int m = k + 1;
        for (; m + 3 < PIVOTS; m += 4) {
            _mm256_store_ps(pred_piv[m + 0], _mm256_fmadd_ps(uk, _mm256_set1_ps(vk[m + 0]), _mm256_load_ps(pred_piv[m + 0])));
            _mm256_store_ps(pred_piv[m + 1], _mm256_fmadd_ps(uk, _mm256_set1_ps(vk[m + 1]), _mm256_load_ps(pred_piv[m + 1])));
            _mm256_store_ps(pred_piv[m + 2], _mm256_fmadd_ps(uk, _mm256_set1_ps(vk[m + 2]), _mm256_load_ps(pred_piv[m + 2])));
            _mm256_store_ps(pred_piv[m + 3], _mm256_fmadd_ps(uk, _mm256_set1_ps(vk[m + 3]), _mm256_load_ps(pred_piv[m + 3])));
        }
        for (; m < PIVOTS; ++m) {
            _mm256_store_ps(pred_piv[m], _mm256_fmadd_ps(uk, _mm256_set1_ps(vk[m]), _mm256_load_ps(pred_piv[m])));
        }
    }
}

template <bool Decode>
KV_AVX2_FMA static void predict_avx2(B *words, U *z) {
    if constexpr (!Decode) {
        transpose_128x128_avx2(reinterpret_cast<const U *>(words), z);
    }
    alignas(32) float V[PIVOTS][128], V_piv[PIVOTS][64];
    float pivots[PIVOTS];
    int cols[PIVOTS];
    alignas(32) U active_mask[128];
    for (int j = 0; j < 128; ++j) active_mask[j] = ~0u;

    for (int base = 0; base < PIVOTS; base += 4) {
        alignas(32) float pred[4][128];
        if (base == 0) {
            std::memset(pred, 0, sizeof(pred));
        } else {
            alignas(16) float U_col[64][4];
            trsm_4row_avx2<Decode>(z, base, base, cols, pivots, V_piv, U_col);
            gemm_4xN_avx2(&pred[0][0], U_col, &V[0][0], base);
        }
        for (int r = 0; r < 4; ++r) {
            int k = base + r;
            U *row = z + k * 128;
            float *vk = V[k];
            int c = pivot_row_avx2<Decode>(row, pred[r], active_mask, vk);
            float piv = vk[c];
            U piv_bits = f2u_avx2(piv);
            pivots[k] = piv;
            cols[k] = c;
            active_mask[c] = 0u;
            vk[c] = 0.0f;
            for (int m = 0; m < k; ++m) V_piv[m][k] = V[m][c];
            V_piv[k][k] = 0.0f;
            for (int r2 = r + 1; r2 < 4; ++r2) {
                U pr = bounded_avx2(f2u_avx2(pred[r2][c]));
                U &cell = z[(base + r2) * 128 + c];
                U w;
                if constexpr (Decode) {
                    w = restore_word_avx2(cell, pr);
                    cell = w;
                } else {
                    w = cell;
                    cell = residual_word_avx2(w, pr);
                }
                U delta = bounded_avx2(f2u_avx2(u2f_avx2(bounded_avx2(w)) - u2f_avx2(pr)));
                float s = piv_bits ? u2f_avx2(bounded_avx2(f2u_avx2(u2f_avx2(delta) / piv))) : 0.0f;
                if (f2u_avx2(s)) {
                    __m256 sv = _mm256_set1_ps(s);
                    for (int j = 0; j < 128; j += 8) {
                        _mm256_store_ps(pred[r2] + j, _mm256_fmadd_ps(sv, _mm256_load_ps(vk + j), _mm256_load_ps(pred[r2] + j)));
                    }
                }
            }
        }
    }

    int rem_cols[REM_COLS];
    int nr = 0;
    for (int j = 0; j < 128; ++j) {
        if (active_mask[j]) rem_cols[nr++] = j;
    }
    alignas(32) float V_rem[PIVOTS][REM_COLS];
    for (int k = 0; k < PIVOTS; ++k) {
        for (int jj = 0; jj < REM_COLS; ++jj) V_rem[k][jj] = V[k][rem_cols[jj]];
    }

    for (int base = PIVOTS; base < 128; base += 8) {
        alignas(32) float U_col[64][8];
        trsm_8row_avx2<Decode>(z, base, cols, pivots, V_piv, U_col);
        alignas(32) U rem_io[8][REM_COLS];
        for (int r = 0; r < 8; ++r) {
            const U *row = z + (base + r) * 128;
            for (int jj = 0; jj < REM_COLS; ++jj) rem_io[r][jj] = row[rem_cols[jj]];
        }
        gemm_8x64_fused_avx2<Decode>(&rem_io[0][0], U_col, &V_rem[0][0]);
        for (int r = 0; r < 8; ++r) {
            U *row = z + (base + r) * 128;
            for (int jj = 0; jj < REM_COLS; ++jj) row[rem_cols[jj]] = rem_io[r][jj];
        }
    }

    if constexpr (Decode) {
        transpose_128x128_avx2(z, reinterpret_cast<U *>(words));
    }
}

KV_AVX2_FMA static void shuffle_bytes_4_avx2(B *dst, const B *src, size_t elems, bool dec) {
    if (!dec) {
        B *p0 = dst, *p1 = dst + elems, *p2 = dst + 2 * elems, *p3 = dst + 3 * elems;
        size_t i = 0;
        __m256i sh = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                      0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        for (; i + 8 <= elems; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + 4 * i));
            __m256i s = _mm256_shuffle_epi8(v, sh);
            __m128i lo = _mm256_castsi256_si128(s), hi = _mm256_extracti128_si256(s, 1);
            __m128i a = _mm_unpacklo_epi32(lo, hi), b = _mm_unpackhi_epi32(lo, hi);
            _mm_storel_epi64(reinterpret_cast<__m128i *>(p0 + i), a);
            _mm_storeh_pi(reinterpret_cast<__m64 *>(p1 + i), _mm_castsi128_ps(a));
            _mm_storel_epi64(reinterpret_cast<__m128i *>(p2 + i), b);
            _mm_storeh_pi(reinterpret_cast<__m64 *>(p3 + i), _mm_castsi128_ps(b));
        }
        for (; i < elems; ++i) {
            p0[i] = src[4 * i];
            p1[i] = src[4 * i + 1];
            p2[i] = src[4 * i + 2];
            p3[i] = src[4 * i + 3];
        }
    } else {
        const B *p0 = src, *p1 = src + elems, *p2 = src + 2 * elems, *p3 = src + 3 * elems;
        size_t i = 0;
        for (; i + 16 <= elems; i += 16) {
            __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p0 + i));
            __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p1 + i));
            __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p2 + i));
            __m128i b3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p3 + i));
            __m128i lo01 = _mm_unpacklo_epi8(b0, b1), hi01 = _mm_unpackhi_epi8(b0, b1);
            __m128i lo23 = _mm_unpacklo_epi8(b2, b3), hi23 = _mm_unpackhi_epi8(b2, b3);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4 * i + 0), _mm_unpacklo_epi16(lo01, lo23));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4 * i + 16), _mm_unpackhi_epi16(lo01, lo23));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4 * i + 32), _mm_unpacklo_epi16(hi01, hi23));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4 * i + 48), _mm_unpackhi_epi16(hi01, hi23));
        }
        for (; i < elems; ++i) {
            dst[4 * i] = p0[i];
            dst[4 * i + 1] = p1[i];
            dst[4 * i + 2] = p2[i];
            dst[4 * i + 3] = p3[i];
        }
    }
}

KV_AVX2_FMA static void shuffle_bytes_2_avx2(B *dst, const B *src, size_t elems, bool dec) {
    if (!dec) {
        B *p0 = dst, *p1 = dst + elems;
        size_t i = 0;
        __m256i sh = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        for (; i + 16 <= elems; i += 16) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + 2 * i));
            __m256i s = _mm256_shuffle_epi8(v, sh);
            __m128i lo = _mm256_castsi256_si128(s), hi = _mm256_extracti128_si256(s, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(p0 + i), _mm_unpacklo_epi64(lo, hi));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(p1 + i), _mm_unpackhi_epi64(lo, hi));
        }
        for (; i < elems; ++i) {
            p0[i] = src[2 * i];
            p1[i] = src[2 * i + 1];
        }
    } else {
        const B *p0 = src, *p1 = src + elems;
        size_t i = 0;
        for (; i + 16 <= elems; i += 16) {
            __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p0 + i));
            __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p1 + i));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 2 * i + 0), _mm_unpacklo_epi8(b0, b1));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 2 * i + 16), _mm_unpackhi_epi8(b0, b1));
        }
        for (; i < elems; ++i) {
            dst[2 * i] = p0[i];
            dst[2 * i + 1] = p1[i];
        }
    }
}

static inline bool has_avx2_fma() {
    static const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                                  __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("lzcnt");
    return supported;
}

#elif defined(KV_ARM_NEON)
static inline uint32x4_t bounded_i(uint32x4_t w) {
    uint32x4_t biased = vsubq_u32(vshlq_n_u32(w, 1), vdupq_n_u32(77u << 24));
    return vandq_u32(w, vcltq_u32(biased, vdupq_n_u32(101u << 24)));
}
static inline uint32x4_t order_i(uint32x4_t w) {
    uint32x4_t sign = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(w), 31));
    return veorq_u32(w, vorrq_u32(sign, vdupq_n_u32(0x80000000u)));
}
static inline uint32x4_t residual_i(uint32x4_t w, uint32x4_t p) {
    uint32x4_t d = vsubq_u32(order_i(w), order_i(p));
    uint32x4_t s = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(d), 31));
    return veorq_u32(vshlq_n_u32(d, 1), s);
}
static inline uint32x4_t restore_i(uint32x4_t z, uint32x4_t p) {
    uint32x4_t neg = vsubq_u32(vdupq_n_u32(0), vandq_u32(z, vdupq_n_u32(1)));
    uint32x4_t d = veorq_u32(vshrq_n_u32(z, 1), neg);
    uint32x4_t b = vaddq_u32(order_i(p), d);
    uint32x4_t top = vsubq_u32(vshrq_n_u32(b, 31), vdupq_n_u32(1));
    return veorq_u32(b, vorrq_u32(top, vdupq_n_u32(0x80000000u)));
}

static void transpose_128x128_neon(const U *src, U *dst) {
    for (int r = 0; r < 128; r += 4) {
        for (int c = 0; c < 128; c += 4) {
            uint32x4_t l0 = vld1q_u32(src + (r + 0) * 128 + c);
            uint32x4_t l1 = vld1q_u32(src + (r + 1) * 128 + c);
            uint32x4_t l2 = vld1q_u32(src + (r + 2) * 128 + c);
            uint32x4_t l3 = vld1q_u32(src + (r + 3) * 128 + c);
            uint32x4x2_t t01 = vtrnq_u32(l0, l1), t23 = vtrnq_u32(l2, l3);
            uint64x2_t a0 = vreinterpretq_u64_u32(t01.val[0]), b0 = vreinterpretq_u64_u32(t23.val[0]);
            uint64x2_t a1 = vreinterpretq_u64_u32(t01.val[1]), b1 = vreinterpretq_u64_u32(t23.val[1]);
            vst1q_u32(dst + (c + 0) * 128 + r, vreinterpretq_u32_u64(vzip1q_u64(a0, b0)));
            vst1q_u32(dst + (c + 1) * 128 + r, vreinterpretq_u32_u64(vzip1q_u64(a1, b1)));
            vst1q_u32(dst + (c + 2) * 128 + r, vreinterpretq_u32_u64(vzip2q_u64(a0, b0)));
            vst1q_u32(dst + (c + 3) * 128 + r, vreinterpretq_u32_u64(vzip2q_u64(a1, b1)));
        }
    }
}

static inline void gemm_4xN_neon(float *pred, const float (*U_col)[4], const float *V, int K) {
    for (int j = 0; j < 128; j += 16) {
        float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0), a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
        float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0), a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
        float32x4_t a20 = vdupq_n_f32(0), a21 = vdupq_n_f32(0), a22 = vdupq_n_f32(0), a23 = vdupq_n_f32(0);
        float32x4_t a30 = vdupq_n_f32(0), a31 = vdupq_n_f32(0), a32 = vdupq_n_f32(0), a33 = vdupq_n_f32(0);
        const float *vp = V + j;
        for (int k = 0; k < K; ++k) {
            float32x4_t v0 = vld1q_f32(vp + 0), v1 = vld1q_f32(vp + 4);
            float32x4_t v2 = vld1q_f32(vp + 8), v3 = vld1q_f32(vp + 12);
            vp += 128;
            float32x4_t uk = vld1q_f32(U_col[k]);
            a00 = vfmaq_laneq_f32(a00, v0, uk, 0); a01 = vfmaq_laneq_f32(a01, v1, uk, 0);
            a02 = vfmaq_laneq_f32(a02, v2, uk, 0); a03 = vfmaq_laneq_f32(a03, v3, uk, 0);
            a10 = vfmaq_laneq_f32(a10, v0, uk, 1); a11 = vfmaq_laneq_f32(a11, v1, uk, 1);
            a12 = vfmaq_laneq_f32(a12, v2, uk, 1); a13 = vfmaq_laneq_f32(a13, v3, uk, 1);
            a20 = vfmaq_laneq_f32(a20, v0, uk, 2); a21 = vfmaq_laneq_f32(a21, v1, uk, 2);
            a22 = vfmaq_laneq_f32(a22, v2, uk, 2); a23 = vfmaq_laneq_f32(a23, v3, uk, 2);
            a30 = vfmaq_laneq_f32(a30, v0, uk, 3); a31 = vfmaq_laneq_f32(a31, v1, uk, 3);
            a32 = vfmaq_laneq_f32(a32, v2, uk, 3); a33 = vfmaq_laneq_f32(a33, v3, uk, 3);
        }
        vst1q_f32(pred + 0 * 128 + j + 0, a00); vst1q_f32(pred + 0 * 128 + j + 4, a01);
        vst1q_f32(pred + 0 * 128 + j + 8, a02); vst1q_f32(pred + 0 * 128 + j + 12, a03);
        vst1q_f32(pred + 1 * 128 + j + 0, a10); vst1q_f32(pred + 1 * 128 + j + 4, a11);
        vst1q_f32(pred + 1 * 128 + j + 8, a12); vst1q_f32(pred + 1 * 128 + j + 12, a13);
        vst1q_f32(pred + 2 * 128 + j + 0, a20); vst1q_f32(pred + 2 * 128 + j + 4, a21);
        vst1q_f32(pred + 2 * 128 + j + 8, a22); vst1q_f32(pred + 2 * 128 + j + 12, a23);
        vst1q_f32(pred + 3 * 128 + j + 0, a30); vst1q_f32(pred + 3 * 128 + j + 4, a31);
        vst1q_f32(pred + 3 * 128 + j + 8, a32); vst1q_f32(pred + 3 * 128 + j + 12, a33);
    }
}

template <bool Decode>
static inline void gemm_8x64_fused_neon(U *rem_io, const float (*U_col)[8], const float *V) {
    for (int half = 0; half < 8; half += 4) {
        for (int j = 0; j < REM_COLS; j += 16) {
            float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0), a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
            float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0), a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
            float32x4_t a20 = vdupq_n_f32(0), a21 = vdupq_n_f32(0), a22 = vdupq_n_f32(0), a23 = vdupq_n_f32(0);
            float32x4_t a30 = vdupq_n_f32(0), a31 = vdupq_n_f32(0), a32 = vdupq_n_f32(0), a33 = vdupq_n_f32(0);
            for (int k = 0; k < PIVOTS; ++k) {
                const float *vp = V + k * REM_COLS + j;
                float32x4_t v0 = vld1q_f32(vp + 0), v1 = vld1q_f32(vp + 4);
                float32x4_t v2 = vld1q_f32(vp + 8), v3 = vld1q_f32(vp + 12);
                float32x4_t uk = vld1q_f32(U_col[k] + half);
                a00 = vfmaq_laneq_f32(a00, v0, uk, 0); a01 = vfmaq_laneq_f32(a01, v1, uk, 0);
                a02 = vfmaq_laneq_f32(a02, v2, uk, 0); a03 = vfmaq_laneq_f32(a03, v3, uk, 0);
                a10 = vfmaq_laneq_f32(a10, v0, uk, 1); a11 = vfmaq_laneq_f32(a11, v1, uk, 1);
                a12 = vfmaq_laneq_f32(a12, v2, uk, 1); a13 = vfmaq_laneq_f32(a13, v3, uk, 1);
                a20 = vfmaq_laneq_f32(a20, v0, uk, 2); a21 = vfmaq_laneq_f32(a21, v1, uk, 2);
                a22 = vfmaq_laneq_f32(a22, v2, uk, 2); a23 = vfmaq_laneq_f32(a23, v3, uk, 2);
                a30 = vfmaq_laneq_f32(a30, v0, uk, 3); a31 = vfmaq_laneq_f32(a31, v1, uk, 3);
                a32 = vfmaq_laneq_f32(a32, v2, uk, 3); a33 = vfmaq_laneq_f32(a33, v3, uk, 3);
            }
            auto apply = [&](int r, float32x4_t p0, float32x4_t p1, float32x4_t p2, float32x4_t p3) {
                U *dst = rem_io + (half + r) * REM_COLS + j;
                float32x4_t ps[4] = {p0, p1, p2, p3};
                for (int q = 0; q < 4; ++q) {
                    uint32x4_t pr = bounded_i(vreinterpretq_u32_f32(ps[q]));
                    uint32x4_t in = vld1q_u32(dst + 4 * q);
                    vst1q_u32(dst + 4 * q, Decode ? restore_i(in, pr) : residual_i(in, pr));
                }
            };
            apply(0, a00, a01, a02, a03);
            apply(1, a10, a11, a12, a13);
            apply(2, a20, a21, a22, a23);
            apply(3, a30, a31, a32, a33);
        }
    }
}

template <bool Decode>
static inline int pivot_row_neon(U *row, const float *pred, const U *active_mask, float *vk) {
    uint32x4_t abs_mask = vdupq_n_u32(0x7fffffffu);
    int32x4_t best_mag = vdupq_n_s32(-1), best_idx = vdupq_n_s32(0);
    const int32_t init_i[4] = {0, 1, 2, 3};
    int32x4_t idx = vld1q_s32(init_i), four = vdupq_n_s32(4), neg1 = vdupq_n_s32(-1);
    for (int j = 0; j < 128; j += 4) {
        uint32x4_t act = vld1q_u32(active_mask + j);
        uint32x4_t pr = bounded_i(vreinterpretq_u32_f32(vld1q_f32(pred + j)));
        uint32x4_t cur = vld1q_u32(row + j), obs, out;
        if constexpr (Decode) {
            obs = restore_i(cur, pr);
            out = vbslq_u32(act, obs, cur);
        } else {
            obs = cur;
            out = vbslq_u32(act, residual_i(cur, pr), cur);
        }
        vst1q_u32(row + j, out);
        float32x4_t diff = vsubq_f32(vreinterpretq_f32_u32(bounded_i(obs)), vreinterpretq_f32_u32(pr));
        uint32x4_t dv = vandq_u32(bounded_i(vreinterpretq_u32_f32(diff)), act);
        vst1q_f32(vk + j, vreinterpretq_f32_u32(dv));
        int32x4_t mag = vbslq_s32(act, vreinterpretq_s32_u32(vandq_u32(dv, abs_mask)), neg1);
        uint32x4_t gt = vcgtq_s32(mag, best_mag);
        best_mag = vbslq_s32(gt, mag, best_mag);
        best_idx = vbslq_s32(gt, idx, best_idx);
        idx = vaddq_s32(idx, four);
    }
    int32_t mags[4], idxs[4];
    vst1q_s32(mags, best_mag);
    vst1q_s32(idxs, best_idx);
    int best_c = -1;
    int32_t best_m = -1;
    for (int l = 0; l < 4; ++l) {
        if (mags[l] > best_m || (mags[l] == best_m && idxs[l] < best_c)) {
            best_m = mags[l];
            best_c = idxs[l];
        }
    }
    return best_c;
}

template <bool Decode>
static inline void trsm_4row_neon(U *z, int base, int limit, const int *cols,
                                  const float *pivots, const float (*V_piv)[64],
                                  float (*U_col)[4]) {
    alignas(16) float pred_piv[64][4] = {};
    for (int k = 0; k < limit; ++k) {
        int col = cols[k];
        U *p = z + base * 128 + col;
        U cv[4] = {p[0 * 128], p[1 * 128], p[2 * 128], p[3 * 128]};
        uint32x4_t cur = vld1q_u32(cv);
        uint32x4_t pr = bounded_i(vreinterpretq_u32_f32(vld1q_f32(pred_piv[k])));
        uint32x4_t obs;
        U out[4];
        if constexpr (Decode) {
            obs = restore_i(cur, pr);
            vst1q_u32(out, obs);
        } else {
            obs = cur;
            vst1q_u32(out, residual_i(cur, pr));
        }
        p[0 * 128] = out[0]; p[1 * 128] = out[1]; p[2 * 128] = out[2]; p[3 * 128] = out[3];
        float piv = pivots[k];
        float32x4_t uk = vdupq_n_f32(0.0f);
        if (f2u(piv)) {
            float32x4_t diff = vsubq_f32(vreinterpretq_f32_u32(bounded_i(obs)), vreinterpretq_f32_u32(pr));
            float32x4_t delta = vreinterpretq_f32_u32(bounded_i(vreinterpretq_u32_f32(diff)));
            uk = vreinterpretq_f32_u32(bounded_i(vreinterpretq_u32_f32(vdivq_f32(delta, vdupq_n_f32(piv)))));
        }
        vst1q_f32(U_col[k], uk);
        const float *vk = V_piv[k];
        int m = k + 1;
        for (; m + 3 < limit; m += 4) {
            float32x4_t vv = vld1q_f32(vk + m);
            vst1q_f32(pred_piv[m + 0], vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 0]), uk, vv, 0));
            vst1q_f32(pred_piv[m + 1], vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 1]), uk, vv, 1));
            vst1q_f32(pred_piv[m + 2], vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 2]), uk, vv, 2));
            vst1q_f32(pred_piv[m + 3], vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 3]), uk, vv, 3));
        }
        for (; m < limit; ++m) {
            vst1q_f32(pred_piv[m], vfmaq_n_f32(vld1q_f32(pred_piv[m]), uk, vk[m]));
        }
    }
}

template <bool Decode>
static inline void trsm_8row_neon(U *z, int base, const int *cols, const float *pivots,
                                  const float (*V_piv)[64], float (*U_col)[8]) {
    alignas(16) float pred_piv[64][8] = {};
    for (int k = 0; k < PIVOTS; ++k) {
        int col = cols[k];
        U *p = z + base * 128 + col;
        U cv0[4] = {p[0 * 128], p[1 * 128], p[2 * 128], p[3 * 128]};
        U cv1[4] = {p[4 * 128], p[5 * 128], p[6 * 128], p[7 * 128]};
        uint32x4_t cur0 = vld1q_u32(cv0), cur1 = vld1q_u32(cv1);
        uint32x4_t pr0 = bounded_i(vreinterpretq_u32_f32(vld1q_f32(pred_piv[k] + 0)));
        uint32x4_t pr1 = bounded_i(vreinterpretq_u32_f32(vld1q_f32(pred_piv[k] + 4)));
        uint32x4_t obs0, obs1;
        U o0[4], o1[4];
        if constexpr (Decode) {
            obs0 = restore_i(cur0, pr0); obs1 = restore_i(cur1, pr1);
            vst1q_u32(o0, obs0); vst1q_u32(o1, obs1);
        } else {
            obs0 = cur0; obs1 = cur1;
            vst1q_u32(o0, residual_i(cur0, pr0)); vst1q_u32(o1, residual_i(cur1, pr1));
        }
        p[0 * 128] = o0[0]; p[1 * 128] = o0[1]; p[2 * 128] = o0[2]; p[3 * 128] = o0[3];
        p[4 * 128] = o1[0]; p[5 * 128] = o1[1]; p[6 * 128] = o1[2]; p[7 * 128] = o1[3];
        float piv = pivots[k];
        float32x4_t uk0 = vdupq_n_f32(0), uk1 = vdupq_n_f32(0);
        if (f2u(piv)) {
            float32x4_t pv = vdupq_n_f32(piv);
            float32x4_t d0 = vreinterpretq_f32_u32(bounded_i(vreinterpretq_u32_f32(
                vsubq_f32(vreinterpretq_f32_u32(bounded_i(obs0)), vreinterpretq_f32_u32(pr0)))));
            float32x4_t d1 = vreinterpretq_f32_u32(bounded_i(vreinterpretq_u32_f32(
                vsubq_f32(vreinterpretq_f32_u32(bounded_i(obs1)), vreinterpretq_f32_u32(pr1)))));
            uk0 = vreinterpretq_f32_u32(bounded_i(vreinterpretq_u32_f32(vdivq_f32(d0, pv))));
            uk1 = vreinterpretq_f32_u32(bounded_i(vreinterpretq_u32_f32(vdivq_f32(d1, pv))));
        }
        vst1q_f32(U_col[k] + 0, uk0);
        vst1q_f32(U_col[k] + 4, uk1);
        const float *vk = V_piv[k];
        int m = k + 1;
        for (; m + 3 < PIVOTS; m += 4) {
            float32x4_t vv = vld1q_f32(vk + m);
            vst1q_f32(pred_piv[m + 0] + 0, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 0] + 0), uk0, vv, 0));
            vst1q_f32(pred_piv[m + 0] + 4, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 0] + 4), uk1, vv, 0));
            vst1q_f32(pred_piv[m + 1] + 0, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 1] + 0), uk0, vv, 1));
            vst1q_f32(pred_piv[m + 1] + 4, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 1] + 4), uk1, vv, 1));
            vst1q_f32(pred_piv[m + 2] + 0, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 2] + 0), uk0, vv, 2));
            vst1q_f32(pred_piv[m + 2] + 4, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 2] + 4), uk1, vv, 2));
            vst1q_f32(pred_piv[m + 3] + 0, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 3] + 0), uk0, vv, 3));
            vst1q_f32(pred_piv[m + 3] + 4, vfmaq_laneq_f32(vld1q_f32(pred_piv[m + 3] + 4), uk1, vv, 3));
        }
        for (; m < PIVOTS; ++m) {
            float s = vk[m];
            vst1q_f32(pred_piv[m] + 0, vfmaq_n_f32(vld1q_f32(pred_piv[m] + 0), uk0, s));
            vst1q_f32(pred_piv[m] + 4, vfmaq_n_f32(vld1q_f32(pred_piv[m] + 4), uk1, s));
        }
    }
}

template <bool Decode>
static void predict_neon(B *words, U *z) {
    if constexpr (!Decode) {
        transpose_128x128_neon(reinterpret_cast<const U *>(words), z);
    }
    alignas(16) float V[PIVOTS][128], V_piv[PIVOTS][64];
    float pivots[PIVOTS];
    int cols[PIVOTS];
    alignas(16) U active_mask[128];
    for (int j = 0; j < 128; ++j) active_mask[j] = ~0u;

    for (int base = 0; base < PIVOTS; base += 4) {
        alignas(16) float pred[4][128];
        if (base == 0) {
            std::memset(pred, 0, sizeof(pred));
        } else {
            alignas(16) float U_col[64][4];
            trsm_4row_neon<Decode>(z, base, base, cols, pivots, V_piv, U_col);
            gemm_4xN_neon(&pred[0][0], U_col, &V[0][0], base);
        }
        for (int r = 0; r < 4; ++r) {
            int k = base + r;
            U *row = z + k * 128;
            float *vk = V[k];
            int c = pivot_row_neon<Decode>(row, pred[r], active_mask, vk);
            float piv = vk[c];
            U piv_bits = f2u(piv);
            pivots[k] = piv;
            cols[k] = c;
            active_mask[c] = 0u;
            vk[c] = 0.0f;
            for (int m = 0; m < k; ++m) V_piv[m][k] = V[m][c];
            V_piv[k][k] = 0.0f;
            for (int r2 = r + 1; r2 < 4; ++r2) {
                U pr = bounded(f2u(pred[r2][c]));
                U &cell = z[(base + r2) * 128 + c];
                U w;
                if constexpr (Decode) {
                    w = restore_word(cell, pr);
                    cell = w;
                } else {
                    w = cell;
                    cell = residual_word(w, pr);
                }
                U delta = bounded(f2u(u2f(bounded(w)) - u2f(pr)));
                float s = piv_bits ? u2f(bounded(f2u(u2f(delta) / piv))) : 0.0f;
                if (f2u(s)) {
                    for (int j = 0; j < 128; j += 4) {
                        vst1q_f32(pred[r2] + j, vfmaq_n_f32(vld1q_f32(pred[r2] + j), vld1q_f32(vk + j), s));
                    }
                }
            }
        }
    }

    int rem_cols[REM_COLS];
    int nr = 0;
    for (int j = 0; j < 128; ++j) {
        if (active_mask[j]) rem_cols[nr++] = j;
    }
    alignas(16) float V_rem[PIVOTS][REM_COLS];
    for (int k = 0; k < PIVOTS; ++k) {
        for (int jj = 0; jj < REM_COLS; ++jj) V_rem[k][jj] = V[k][rem_cols[jj]];
    }

    for (int base = PIVOTS; base < 128; base += 8) {
        alignas(16) float U_col[64][8];
        trsm_8row_neon<Decode>(z, base, cols, pivots, V_piv, U_col);
        alignas(16) U rem_io[8][REM_COLS];
        for (int r = 0; r < 8; ++r) {
            const U *row = z + (base + r) * 128;
            for (int jj = 0; jj < REM_COLS; ++jj) rem_io[r][jj] = row[rem_cols[jj]];
        }
        gemm_8x64_fused_neon<Decode>(&rem_io[0][0], U_col, &V_rem[0][0]);
        for (int r = 0; r < 8; ++r) {
            U *row = z + (base + r) * 128;
            for (int jj = 0; jj < REM_COLS; ++jj) row[rem_cols[jj]] = rem_io[r][jj];
        }
    }

    if constexpr (Decode) {
        transpose_128x128_neon(z, reinterpret_cast<U *>(words));
    }
}
#endif

static void shuffle_bytes_impl(B *dst, const B *src, size_t n, uint32_t width, bool dec) {
    if (width <= 1 || n == 0) {
        if (n) std::memcpy(dst, src, n);
        return;
    }
    size_t elems = n / width;
    if (width == 4) {
#if defined(KV_X86_AVX2)
        if (has_avx2_fma()) {
            shuffle_bytes_4_avx2(dst, src, elems, dec);
            return;
        }
#elif defined(KV_ARM_NEON)
        if (!dec) {
            B *p0 = dst, *p1 = dst + elems, *p2 = dst + 2 * elems, *p3 = dst + 3 * elems;
            size_t i = 0;
            for (; i + 16 <= elems; i += 16) {
                uint8x16x4_t v = vld4q_u8(src + 4 * i);
                vst1q_u8(p0 + i, v.val[0]); vst1q_u8(p1 + i, v.val[1]);
                vst1q_u8(p2 + i, v.val[2]); vst1q_u8(p3 + i, v.val[3]);
            }
            for (; i < elems; ++i) {
                p0[i] = src[4 * i]; p1[i] = src[4 * i + 1];
                p2[i] = src[4 * i + 2]; p3[i] = src[4 * i + 3];
            }
        } else {
            const B *p0 = src, *p1 = src + elems, *p2 = src + 2 * elems, *p3 = src + 3 * elems;
            size_t i = 0;
            for (; i + 16 <= elems; i += 16) {
                uint8x16x4_t v = {{vld1q_u8(p0 + i), vld1q_u8(p1 + i), vld1q_u8(p2 + i), vld1q_u8(p3 + i)}};
                vst4q_u8(dst + 4 * i, v);
            }
            for (; i < elems; ++i) {
                dst[4 * i] = p0[i]; dst[4 * i + 1] = p1[i];
                dst[4 * i + 2] = p2[i]; dst[4 * i + 3] = p3[i];
            }
        }
        return;
#endif
    }
    if (width == 2) {
#if defined(KV_X86_AVX2)
        if (has_avx2_fma()) {
            shuffle_bytes_2_avx2(dst, src, elems, dec);
            return;
        }
#elif defined(KV_ARM_NEON)
        if (!dec) {
            B *p0 = dst, *p1 = dst + elems;
            size_t i = 0;
            for (; i + 16 <= elems; i += 16) {
                uint8x16x2_t v = vld2q_u8(src + 2 * i);
                vst1q_u8(p0 + i, v.val[0]); vst1q_u8(p1 + i, v.val[1]);
            }
            for (; i < elems; ++i) {
                p0[i] = src[2 * i]; p1[i] = src[2 * i + 1];
            }
        } else {
            const B *p0 = src, *p1 = src + elems;
            size_t i = 0;
            for (; i + 16 <= elems; i += 16) {
                uint8x16x2_t v = {{vld1q_u8(p0 + i), vld1q_u8(p1 + i)}};
                vst2q_u8(dst + 2 * i, v);
            }
            for (; i < elems; ++i) {
                dst[2 * i] = p0[i]; dst[2 * i + 1] = p1[i];
            }
        }
        return;
#endif
    }
    for (size_t lane = 0; lane < width; ++lane) {
        for (size_t i = 0; i < elems; ++i) {
            if (dec) {
                dst[i * width + lane] = src[lane * elems + i];
            } else {
                dst[lane * elems + i] = src[i * width + lane];
            }
        }
    }
}

struct PageLayout {
    size_t block, rows, cols, stride;
    uint32_t scale_w;
    bool planar;
};

static bool page_layout(uint32_t kind, PageLayout &out) {
    switch (kind) {
        case 40: out = {135168, 256, 512, 528, 4, false}; return true;
        case 41: out = {8448, 64, 128, 128, 4, true}; return true;
        case 42: out = {32768, 64, 512, 512, 1, false}; return true;
        case 45: out = {33792, 64, 512, 528, 1, false}; return true;
        case 47: out = {18432, 64, 256, 288, 1, false}; return true;
        case 48: out = {73728, 128, 512, 576, 1, false}; return true;
        default: return false;
    }
}

} // namespace

extern "C" {

// Predict one 128x128 float32 matrix and pack residuals into widths[16384] + 4-lane low bits.
// Returns the number of bytes written to low_out (always <= 63488).
size_t kv_predict_pack_f32_128(const uint8_t *raw_mat, uint8_t *widths_out, uint8_t *low_out,
                               uint16_t lane_u32s[4]) {
    ScopedFpEnv fp;
    alignas(32) U words[MAT_ELEMS];
#if defined(KV_X86_AVX2)
    if (has_avx2_fma()) {
        predict_avx2<false>(const_cast<B *>(raw_mat), words);
        return pack_4lane_avx2(words, widths_out, low_out, lane_u32s);
    }
#elif defined(KV_ARM_NEON)
    predict_neon<false>(const_cast<B *>(raw_mat), words);
    return pack_4lane_impl(words, widths_out, low_out, lane_u32s);
#endif
    predict_scalar<false>(const_cast<B *>(raw_mat), words);
    return pack_4lane_impl(words, widths_out, low_out, lane_u32s);
}

// Unpack residuals and invert the 128x128 float32 predictor into raw_mat[65536].
// Returns 0 on success, -1 on invalid widths or malformed lane lengths.
int kv_unpack_restore_f32_128(uint8_t *raw_mat, const uint8_t *widths_in, const uint8_t *low_in,
                              size_t low_len, const uint16_t lane_u32s[4]) {
    ScopedFpEnv fp;
    alignas(32) U words[MAT_ELEMS];
#if defined(KV_X86_AVX2)
    if (has_avx2_fma()) {
        if (unpack_4lane_avx2(words, widths_in, low_in, low_len, lane_u32s) < 0) return -1;
        predict_avx2<true>(raw_mat, words);
        return 0;
    }
#elif defined(KV_ARM_NEON)
    if (unpack_4lane_impl(words, widths_in, low_in, low_len, lane_u32s) < 0) return -1;
    predict_neon<true>(raw_mat, words);
    return 0;
#endif
    if (unpack_4lane_impl(words, widths_in, low_in, low_len, lane_u32s) < 0) return -1;
    predict_scalar<true>(raw_mat, words);
    return 0;
}

// Scalar reference entry points for cross-ISA bit-exactness verification in tests.
size_t kv_predict_pack_f32_128_scalar(const uint8_t *raw_mat, uint8_t *widths_out, uint8_t *low_out,
                                      uint16_t lane_u32s[4]) {
    ScopedFpEnv fp;
    alignas(32) U words[MAT_ELEMS];
    predict_scalar<false>(const_cast<B *>(raw_mat), words);
    return pack_4lane_impl(words, widths_out, low_out, lane_u32s);
}

int kv_unpack_restore_f32_128_scalar(uint8_t *raw_mat, const uint8_t *widths_in,
                                     const uint8_t *low_in, size_t low_len,
                                     const uint16_t lane_u32s[4]) {
    ScopedFpEnv fp;
    alignas(32) U words[MAT_ELEMS];
    if (unpack_4lane_impl(words, widths_in, low_in, low_len, lane_u32s) < 0) return -1;
    predict_scalar<true>(raw_mat, words);
    return 0;
}

void kv_byte_shuffle(uint8_t *dst, const uint8_t *src, size_t n, uint32_t width, bool decode) {
    shuffle_bytes_impl(dst, src, n, width, decode);
}

int kv_page_split_shuffle(uint8_t *dst, const uint8_t *src, size_t n, uint32_t kind, bool decode,
                          uint8_t *tmp) {
    PageLayout p;
    if (!page_layout(kind, p) || n % p.block != 0) return -1;
    size_t nval = p.rows * p.cols;
    size_t sc_row = (p.block - nval) / p.rows;
    size_t ns = p.block - nval;
    for (size_t off = 0; off < n; off += p.block) {
        const B *s = src + off;
        B *d = dst + off;
        if (!decode) {
            if (p.planar || sc_row == 0) {
                std::memcpy(d, s, nval);
                if (ns) shuffle_bytes_impl(d + nval, s + nval, ns, p.scale_w, false);
            } else if (p.scale_w <= 1) {
                for (size_t r = 0; r < p.rows; ++r) {
                    std::memcpy(d + r * p.cols, s + r * p.stride, p.cols);
                    std::memcpy(d + nval + r * sc_row, s + r * p.stride + p.cols, sc_row);
                }
            } else {
                for (size_t r = 0; r < p.rows; ++r) {
                    std::memcpy(d + r * p.cols, s + r * p.stride, p.cols);
                    std::memcpy(tmp + r * sc_row, s + r * p.stride + p.cols, sc_row);
                }
                shuffle_bytes_impl(d + nval, tmp, ns, p.scale_w, false);
            }
        } else {
            if (p.planar || sc_row == 0) {
                std::memcpy(d, s, nval);
                if (ns) shuffle_bytes_impl(d + nval, s + nval, ns, p.scale_w, true);
            } else if (p.scale_w <= 1) {
                for (size_t r = 0; r < p.rows; ++r) {
                    std::memcpy(d + r * p.stride, s + r * p.cols, p.cols);
                    std::memcpy(d + r * p.stride + p.cols, s + nval + r * sc_row, sc_row);
                }
            } else {
                shuffle_bytes_impl(tmp, s + nval, ns, p.scale_w, true);
                for (size_t r = 0; r < p.rows; ++r) {
                    std::memcpy(d + r * p.stride, s + r * p.cols, p.cols);
                    std::memcpy(d + r * p.stride + p.cols, tmp + r * sc_row, sc_row);
                }
            }
        }
    }
    return 0;
}

} // extern "C"
