// SPDX-License-Identifier: Apache-2.0
//
// Pure-CUDA GPU kernels for lossless GKV1/v2 KV-cache checkpoint packets (VRAM <-> DRAM):
// - Pure CUDA driver-API cubin with zero external runtime libraries beyond libcuda.so.1.
// - Bit-exact 64-pivot low-rank float32 predictor for 128x128 recurrent state matrices
//   (256-thread swizzled 64 KiB shared memory, 1 CTA per matrix, IEEE-754 parity with cpu.cpp)
// - 4-lane aligned-u32 branchless residual low-bit pack/unpack with bounds & zero-pad validation
// - Byte-lane shuffle (width 2, 4, 20, 32) and compound page value/scale split-shuffle
// - 32-way warp-interleaved 12-bit static rANS entropy coder (64 KiB chunks, RLE + sparse/dense tables)
// - Parallel GF(2)-combined IEEE 802.3 CRC32 (bit-exact with crc32fast)
// - On-GPU packet header/directory assembly and chunk/lane compaction (eliminates intermediate host syncs)
//
// Compiled by Ignition at model core export via `nvcc --cubin` for the target SM architecture,
// embedded as a standalone cubin module in the exported core, and loaded lazily via `cuModuleLoadData`
// only when `--lossless-kv-compression gpu` is enabled (honors `CUDA_DISABLE_JIT=1`).

#include <stdint.h>
#include <stddef.h>

using U = uint32_t;
using B = uint8_t;

static constexpr int MAT_ELEMS = 16384;
static constexpr int PIVOTS = 64;
static constexpr int MAX_LANE_U32S = 3968;
static constexpr int LANE_STRIDE = MAX_LANE_U32S + 2;

static constexpr U GKV1_MAGIC = 0x31564B47u; // "GKV1" LE
static constexpr U GKV1_VERSION = 2u;
static constexpr U RAW_FLAG = 1u << 31;
static constexpr int HEADER_BYTES = 24;

static constexpr int RANS_LANES = 32;
static constexpr int RANS_PROB_BITS = 12;
static constexpr int RANS_PROB_SCALE = 1 << RANS_PROB_BITS; // 4096
static constexpr U RANS_L = 1u << 16;                       // 65536
static constexpr int RANS_HDR_RESERVE = 640;
static constexpr int RANS_LANE_CAP_U16 = 1152;              // 2304 B/lane (>112.5% of 2 KiB lane)
static constexpr int RANS_LANE_CAP_BYTES = RANS_LANE_CAP_U16 * 2;
static constexpr int RANS_SLOT_STRIDE = RANS_HDR_RESERVE + RANS_LANES * RANS_LANE_CAP_BYTES; // 74368 B

static constexpr int CRC_BLOCKS = 256;

__device__ __forceinline__ int swiz(int i, int j) {
    return i * 128 + (j ^ (i & 31));
}
__device__ __forceinline__ U bounded_u(U w) {
    U e = (w >> 23) & 255u;
    return (e >= 77u && e <= 177u) ? w : 0u;
}
__device__ __forceinline__ U order_u(U w) {
    return w ^ ((0u - (w >> 31)) | 0x80000000u);
}
__device__ __forceinline__ U residual_u(U w, U p) {
    U d = order_u(w) - order_u(p);
    return (d << 1) ^ (0u - (d >> 31));
}
__device__ __forceinline__ U restore_u(U z, U p) {
    U b = order_u(p) + ((z >> 1) ^ (0u - (z & 1u)));
    return b ^ (((b >> 31) - 1u) | 0x80000000u);
}

static __device__ __forceinline__ uint16_t load_u16_le(const B *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static __device__ __forceinline__ void store_u16_le(B *p, uint16_t v) {
    p[0] = (B)(v & 0xFFu);
    p[1] = (B)(v >> 8);
}
static __device__ __forceinline__ U load_u32_le(const B *p) {
    return (U)p[0] | ((U)p[1] << 8) | ((U)p[2] << 16) | ((U)p[3] << 24);
}
static __device__ __forceinline__ void store_u32_le(B *p, U v) {
    p[0] = (B)(v & 0xFFu);
    p[1] = (B)((v >> 8) & 0xFFu);
    p[2] = (B)((v >> 16) & 0xFFu);
    p[3] = (B)((v >> 24) & 0xFFu);
}

// =========================================================================
// 1. 64-pivot low-rank float32 128x128 matrix predictor & 4-lane bit-packer
// =========================================================================

template <bool Decode>
__device__ __forceinline__ void predict_f32_128_impl(
    B *raw_base,
    const uint64_t *mat_offs,
    U *z_mats,
    int num_mats
) {
    int m = blockIdx.x;
    if (m >= num_mats) return;
    int tid = threadIdx.x; // 0..255
    int j = tid & 127;
    int half = tid >> 7;

    extern __shared__ float p[]; // 128 * 128 floats = 65536 B
    __shared__ float v_sh[128];
    __shared__ float u_sh[128];
    __shared__ uint8_t done_sh[128];
    __shared__ int32_t warp_mag[4];
    __shared__ int32_t warp_idx[4];
    __shared__ int piv_col;
    __shared__ float piv_val;
    __shared__ U piv_bits;

    U *raw = reinterpret_cast<U *>(raw_base + mat_offs[m]);
    U *z = z_mats + (size_t)m * MAT_ELEMS;

    for (int i = half; i < 128; i += 2) p[swiz(i, j)] = 0.0f;
    if (half == 0) done_sh[j] = 0;
    __syncthreads();

    for (int k = 0; k < PIVOTS; ++k) {
        if (half == 0) {
            int32_t my_mag = -1;
            if (!done_sh[j]) {
                U pred = bounded_u(__float_as_uint(p[swiz(k, j)]));
                U w;
                if constexpr (Decode) {
                    w = restore_u(z[k * 128 + j], pred);
                    raw[j * 128 + k] = w;
                } else {
                    w = raw[j * 128 + k];
                    z[k * 128 + j] = residual_u(w, pred);
                }
                U obs = bounded_u(w);
                float diff = __fsub_rn(__uint_as_float(obs), __uint_as_float(pred));
                U dv = bounded_u(__float_as_uint(diff));
                v_sh[j] = __uint_as_float(dv);
                my_mag = (int32_t)(dv & 0x7fffffffu);
            } else {
                v_sh[j] = 0.0f;
            }
            int32_t best_m = my_mag, best_i = j;
            for (int off = 16; off > 0; off >>= 1) {
                int32_t om = __shfl_down_sync(0xffffffffu, best_m, off);
                int32_t oi = __shfl_down_sync(0xffffffffu, best_i, off);
                if (om > best_m || (om == best_m && oi < best_i)) {
                    best_m = om;
                    best_i = oi;
                }
            }
            if ((j & 31) == 0) {
                warp_mag[j >> 5] = best_m;
                warp_idx[j >> 5] = best_i;
            }
        }
        __syncthreads();
        if (tid == 0) {
            int32_t bm = warp_mag[0], bi = warp_idx[0];
            for (int w = 1; w < 4; ++w) {
                if (warp_mag[w] > bm || (warp_mag[w] == bm && warp_idx[w] < bi)) {
                    bm = warp_mag[w];
                    bi = warp_idx[w];
                }
            }
            piv_col = bi;
            piv_val = v_sh[bi];
            piv_bits = __float_as_uint(piv_val);
            done_sh[bi] = 1;
        }
        __syncthreads();

        int c = piv_col;
        if (half == 0 && j > k) {
            U pred = bounded_u(__float_as_uint(p[swiz(j, c)]));
            U w;
            if constexpr (Decode) {
                w = restore_u(z[j * 128 + c], pred);
                raw[c * 128 + j] = w;
            } else {
                w = raw[c * 128 + j];
                z[j * 128 + c] = residual_u(w, pred);
            }
            U obs = bounded_u(w);
            float diff = __fsub_rn(__uint_as_float(obs), __uint_as_float(pred));
            U delta = bounded_u(__float_as_uint(diff));
            u_sh[j] = piv_bits
                ? __uint_as_float(bounded_u(__float_as_uint(__fdiv_rn(__uint_as_float(delta), piv_val))))
                : 0.0f;
        }
        __syncthreads();

        if (!done_sh[j]) {
            float vj = v_sh[j];
            #pragma unroll 4
            for (int i = k + 1 + half; i < 128; i += 2) {
                int idx = swiz(i, j);
                p[idx] = __fmaf_rn(u_sh[i], vj, p[idx]);
            }
        }
        __syncthreads();
    }

    if (!done_sh[j]) {
        for (int i = PIVOTS + half; i < 128; i += 2) {
            U pred = bounded_u(__float_as_uint(p[swiz(i, j)]));
            if constexpr (Decode) {
                raw[j * 128 + i] = restore_u(z[i * 128 + j], pred);
            } else {
                z[i * 128 + j] = residual_u(raw[j * 128 + i], pred);
            }
        }
    }
}

// =========================================================================
// 2. Parallel IEEE 802.3 CRC32 helpers (GF(2) combine, bit-exact)
// =========================================================================

// Compute IEEE 802.3 CRC32 lookup entries and GF(2) byte-shift matrices M(2^k)
// in shared memory so the standalone cubin declares zero ELF global/constant symbols.
static __device__ __forceinline__ U crc32_byte_entry(U b) {
    U c = b;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        U mask = 0u - (c & 1u);
        // Prevent NVVM/LLVM CRC table synthesis (CUDA 13+) from emitting
        // .nv.global.init `_$_crctable` symbols into this standalone cubin.
        asm("" : "+r"(mask));
        c = (c >> 1) ^ (0xEDB88320u & mask);
    }
    return c;
}

static __device__ __forceinline__ U gf2_matrix_times(const U *mat, U vec) {
    U sum = 0u;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if ((vec >> i) & 1u) sum ^= mat[i];
    }
    return sum;
}

static __device__ __forceinline__ void init_gf2_pow_sh(U gf2_pow[][32], int levels, int tid) {
    if (tid < 32) {
        gf2_pow[0][tid] = (tid < 8) ? crc32_byte_entry(1u << tid) : (1u << (tid - 8));
        __syncwarp();
        for (int k = 1; k < levels; ++k) {
            gf2_pow[k][tid] = gf2_matrix_times(gf2_pow[k - 1], gf2_pow[k - 1][tid]);
            __syncwarp();
        }
    }
}

static __device__ __forceinline__ U crc32_combine_dev(
    const U gf2_pow[][32], int levels, U crc1, U crc2, uint64_t len2
) {
    int bit = 0;
    while (len2 && bit < levels) {
        if (len2 & 1ull) crc1 = gf2_matrix_times(gf2_pow[bit], crc1);
        len2 >>= 1;
        bit++;
    }
    return crc1 ^ crc2;
}

// Exact Alverson/Granlund-Montgomery unsigned 32-bit division by d in [1, 4095].
static __device__ __forceinline__ void prep_div_u32(U d, U &m_out, U &sh_out) {
    if (d <= 1u) {
        m_out = 0u;
        sh_out = 0u;
        return;
    }
    U l = 32u - (U)__clz(d - 1u); // 2^(l-1) < d <= 2^l, l in [1, 12]
    uint64_t num = ((1ull << l) - (uint64_t)d) << 32;
    m_out = (U)(num / (uint64_t)d) + 1u;
    sh_out = l;
}

static __device__ __forceinline__ U div_u32_fast(U x, U m, U sh) {
    if (sh == 0u) return x;
    U t = __umulhi(x, m);
    return (t + ((x - t) >> 1)) >> (sh - 1u);
}

extern "C" {

__global__ void kv_predict_f32_128_enc(
    const B *raw_base,
    const uint64_t *mat_offs,
    U *z_mats,
    int num_mats
) {
    predict_f32_128_impl<false>(const_cast<B *>(raw_base), mat_offs, z_mats, num_mats);
}

__global__ void kv_predict_f32_128_dec(
    B *raw_base,
    const uint64_t *mat_offs,
    const U *z_mats,
    int num_mats,
    const U *err_flag
) {
    if (*err_flag != 0u) return;
    predict_f32_128_impl<true>(raw_base, mat_offs, const_cast<U *>(z_mats), num_mats);
}

__global__ void kv_pack_4lane_gpu(
    const U *z_mats,
    B *widths_out,
    U *low_padded_out,
    uint16_t *lane_u32s_out,
    int num_mats
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = gid >> 2;
    int l = gid & 3;
    if (m >= num_mats) return;

    const U *z = z_mats + (size_t)m * MAT_ELEMS;
    B *w_out = widths_out + (size_t)m * MAT_ELEMS;
    U *dst = low_padded_out + ((size_t)m * 4 + l) * LANE_STRIDE;

    uint64_t acc = 0;
    unsigned bits = 0;
    U *p = dst;
    for (int i = l; i < MAT_ELEMS; i += 4) {
        U v = z[i];
        unsigned w = v ? (32u - __clz(v)) : 0u;
        w_out[i] = (B)w;
        unsigned low = w - (w != 0u);
        acc |= (uint64_t)(v & ((1u << low) - 1u)) << bits;
        bits += low;
        if (bits >= 32u) {
            *p++ = (U)acc;
            acc >>= 32;
            bits -= 32u;
        }
    }
    if (bits) *p++ = (U)acc;
    lane_u32s_out[m * 4 + l] = (uint16_t)(p - dst);
}

// Bounds-checked 4-lane unpack directly from the compacted low-bit byte buffer.
// Sets *err_flag = 1 if any width > 32, bit count disagrees with lane_u32s, or trailing pad bits are non-zero.
__global__ void kv_unpack_4lane_gpu(
    U *z_mats,
    const B *widths_in,
    const B *low_compact_in,
    const U *lane_byte_offsets,
    const uint16_t *lane_u32s_in,
    int num_mats,
    U *err_flag
) {
    if (*err_flag != 0u) return;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = gid >> 2;
    int l = gid & 3;
    if (m >= num_mats) return;

    int lane_id = m * 4 + l;
    size_t nwords = lane_u32s_in[lane_id];
    if (nwords > (size_t)MAX_LANE_U32S) {
        atomicOr(err_flag, 1u);
        return;
    }
    const B *src_b = low_compact_in + lane_byte_offsets[lane_id];
    bool aligned = (((uintptr_t)src_b & 3u) == 0u);
    const U *src = aligned ? reinterpret_cast<const U *>(src_b) : nullptr;
    auto read_word = [&](size_t wi) -> U {
        return aligned ? src[wi] : load_u32_le(src_b + wi * 4u);
    };
    U *z = z_mats + (size_t)m * MAT_ELEMS;
    const B *w_in = widths_in + (size_t)m * MAT_ELEMS;

    size_t total_bits = nwords * 32u;
    size_t bit = 0;
    for (int i = l; i < MAT_ELEMS; i += 4) {
        unsigned w = w_in[i];
        if (w > 32u) {
            atomicOr(err_flag, 1u);
            return;
        }
        if (w == 0u) {
            z[i] = 0u;
            continue;
        }
        unsigned low = w - 1u;
        if (bit + low > total_bits) {
            atomicOr(err_flag, 1u);
            return;
        }
        size_t wi = bit >> 5;
        uint64_t q = (wi < nwords) ? (uint64_t)read_word(wi) : 0ull;
        if (wi + 1 < nwords) q |= ((uint64_t)read_word(wi + 1)) << 32;
        U hi = 1u << low;
        z[i] = hi | ((U)(q >> (bit & 31u)) & (hi - 1u));
        bit += low;
    }
    if (((bit + 31u) >> 5) != nwords) {
        atomicOr(err_flag, 1u);
        return;
    }
    unsigned rem = (unsigned)(bit & 31u);
    if (rem != 0u && nwords > 0u) {
        if ((read_word(nwords - 1) >> rem) != 0u) {
            atomicOr(err_flag, 1u);
        }
    }
}

static __device__ __forceinline__ void byte_shuffle_elem(
    B *dst, const B *src, size_t idx, size_t elems, U width, int decode
) {
    if (!decode) {
        size_t i = idx / width, lane = idx % width;
        dst[lane * elems + i] = src[idx];
    } else {
        size_t lane = idx / elems, i = idx % elems;
        dst[i * width + lane] = src[idx];
    }
}

static __device__ __forceinline__ void page_split_elem(
    B *dst,
    const B *src,
    size_t idx,
    size_t block,
    size_t rows,
    size_t cols,
    size_t stride,
    U scale_w,
    int planar,
    int decode
) {
    size_t b = idx / block;
    size_t off = idx % block;
    size_t nval = rows * cols;
    size_t sc_row = (block - nval) / rows;
    size_t ns = block - nval;
    size_t s_elems = scale_w > 0 ? (ns / scale_w) : 0;
    const B *sb = src + b * block;
    B *db = dst + b * block;
    if (!decode) {
        if (planar || sc_row == 0) {
            if (off < nval) {
                db[off] = sb[off];
            } else if (scale_w > 1) {
                size_t si = off - nval;
                size_t i = si / scale_w, lane = si % scale_w;
                db[nval + lane * s_elems + i] = sb[off];
            } else {
                db[off] = sb[off];
            }
        } else {
            size_t r = off / stride, c = off % stride;
            if (c < cols) {
                db[r * cols + c] = sb[off];
            } else {
                size_t si = r * sc_row + (c - cols);
                if (scale_w > 1) {
                    size_t i = si / scale_w, lane = si % scale_w;
                    db[nval + lane * s_elems + i] = sb[off];
                } else {
                    db[nval + si] = sb[off];
                }
            }
        }
    } else {
        if (planar || sc_row == 0) {
            if (off < nval) {
                db[off] = sb[off];
            } else if (scale_w > 1) {
                size_t si = off - nval;
                size_t lane = si / s_elems, i = si % s_elems;
                db[nval + i * scale_w + lane] = sb[off];
            } else {
                db[off] = sb[off];
            }
        } else {
            if (off < nval) {
                size_t r = off / cols, c = off % cols;
                db[r * stride + c] = sb[off];
            } else {
                size_t si = off - nval;
                size_t flat_s = si;
                if (scale_w > 1) {
                    size_t lane = si / s_elems, i = si % s_elems;
                    flat_s = i * scale_w + lane;
                }
                size_t r = flat_s / sc_row, sc = flat_s % sc_row;
                db[r * stride + cols + sc] = sb[off];
            }
        }
    }
}

__global__ void kv_byte_shuffle_gpu(B *dst, const B *src, size_t n, U width, int decode) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    byte_shuffle_elem(dst, src, idx, n / width, width, decode);
}

__global__ void kv_page_split_gpu(
    B *dst,
    const B *src,
    size_t n,
    size_t block,
    size_t rows,
    size_t cols,
    size_t stride,
    U scale_w,
    int planar,
    int decode
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    page_split_elem(dst, src, idx, block, rows, cols, stride, scale_w, planar, decode);
}

struct KvTransformDesc {
    uint64_t offset;
    uint64_t length;
    U kind; // 0 = Bytes, 1 = ByteShuffle(width=2), 2 = PageSplit
    U block;
    U rows;
    U cols;
    U stride;
    U scale_w;
    U planar;
    U pad;
};

static __device__ __forceinline__ void copy_span_cta(B *dst, const B *src, size_t len) {
    if ((((uintptr_t)dst | (uintptr_t)src) & 3u) == 0u) {
        size_t nw = len >> 2;
        U *dw = reinterpret_cast<U *>(dst);
        const U *sw = reinterpret_cast<const U *>(src);
        for (size_t i = threadIdx.x; i < nw; i += blockDim.x) {
            dw[i] = sw[i];
        }
        size_t tail = nw << 2;
        for (size_t i = tail + threadIdx.x; i < len; i += blockDim.x) {
            dst[i] = src[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < len; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

// Batched device-to-device copy across arbitrary absolute device addresses (multi-plane span gather/scatter).
__global__ void kv_copy_batch_gpu(
    const uint64_t *dst_ptrs,
    const uint64_t *src_ptrs,
    const uint64_t *lens,
    int n
) {
    int idx = blockIdx.x;
    if (idx >= n) return;
    copy_span_cta(
        reinterpret_cast<B *>(dst_ptrs[idx]),
        reinterpret_cast<const B *>(src_ptrs[idx]),
        (size_t)lens[idx]
    );
}

// Batched device-to-device copy using pre-uploaded static offsets relative to `dst_base` and `src_base`.
__global__ void kv_copy_batch_rel_gpu(
    B *dst_base,
    const B *src_base,
    const uint64_t *dst_offs,
    const uint64_t *src_offs,
    const uint64_t *lens,
    int n
) {
    int idx = blockIdx.x;
    if (idx >= n) return;
    copy_span_cta(dst_base + dst_offs[idx], src_base + src_offs[idx], (size_t)lens[idx]);
}

// Multi-page strided relative batch copy (`grid = n_ops * batch`, `block = 128`).
__global__ void kv_copy_batch_rel_strided_gpu(
    B *dst_base,
    const B *src_base,
    const uint64_t *dst_offs,
    const uint64_t *src_offs,
    const uint64_t *lens,
    int n_ops,
    size_t dst_page_stride,
    size_t src_page_stride,
    int batch
) {
    int bid = blockIdx.x;
    int p = bid / n_ops;
    int op = bid % n_ops;
    if (p >= batch) return;
    copy_span_cta(
        dst_base + (size_t)p * dst_page_stride + dst_offs[op],
        src_base + (size_t)p * src_page_stride + src_offs[op],
        (size_t)lens[op]
    );
}

// Multi-page fused non-matrix transform (`grid = num_descs * batch`, `block = 256`).
__global__ void kv_transform_batch_gpu(
    B *dst_base,
    const B *src_base,
    const KvTransformDesc *descs,
    int num_descs,
    size_t page_stride,
    int batch,
    int decode
) {
    int bid = blockIdx.x;
    int p = bid / num_descs;
    int d = bid % num_descs;
    if (p >= batch) return;
    KvTransformDesc desc = descs[d];
    size_t base_off = (size_t)p * page_stride + (size_t)desc.offset;
    B *dst = dst_base + base_off;
    const B *src = src_base + base_off;
    size_t len = (size_t)desc.length;
    if (desc.kind == 0u) {
        copy_span_cta(dst, src, len);
    } else if (desc.kind == 1u) {
        size_t elems = len >> 1;
        for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
            byte_shuffle_elem(dst, src, idx, elems, 2u, decode);
        }
    } else {
        for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
            page_split_elem(
                dst,
                src,
                idx,
                (size_t)desc.block,
                (size_t)desc.rows,
                (size_t)desc.cols,
                (size_t)desc.stride,
                desc.scale_w,
                (int)desc.planar,
                decode
            );
        }
    }
}

// Single-launch batched IEEE 802.3 CRC32 (`grid = batch`, `block = 256`, 1 CTA per page).
__global__ void kv_crc32_batch_gpu(
    const B *base,
    const uint64_t *offsets,
    const uint64_t *lens,
    uint64_t fixed_stride,
    uint64_t fixed_len,
    uint64_t max_len,
    U *crc_out,
    int batch
) {
    int p = blockIdx.x;
    if (p >= batch) return;
    int tid = threadIdx.x; // 0..255

    const B *data = base + (offsets ? offsets[p] : (uint64_t)p * fixed_stride);
    uint64_t len = lens ? lens[p] : fixed_len;
    if (len == 0ull || (max_len > 0ull && len >= max_len)) {
        if (tid == 0) crc_out[p] = 0u;
        return;
    }

    __shared__ B tile_sh[16384];
    __shared__ U crc_tab[256];
    __shared__ U gf2_pow[15][32];
    __shared__ U sh_crc[256];
    __shared__ uint64_t sh_len[256];

    crc_tab[tid] = crc32_byte_entry((U)tid);
    init_gf2_pow_sh(gf2_pow, 15, tid);
    __syncthreads();

    U blk_crc = 0u;
    uint64_t off = 0ull;
    while (off + 16384ull <= len) {
        const B *src = data + off;
        if ((((uintptr_t)src) & 3u) == 0u) {
            const U *sw = reinterpret_cast<const U *>(src);
            U *dw = reinterpret_cast<U *>(tile_sh);
            for (int i = tid; i < 4096; i += 256) dw[i] = sw[i];
        } else {
            for (int i = tid; i < 16384; i += 256) tile_sh[i] = src[i];
        }
        __syncthreads();

        const B *tp = tile_sh + tid * 64;
        U c = 0xFFFFFFFFu;
        #pragma unroll 4
        for (int i = 0; i < 64; ++i) {
            c = crc_tab[(c ^ tp[i]) & 0xFFu] ^ (c >> 8);
        }
        sh_crc[tid] = c ^ 0xFFFFFFFFu;
        __syncthreads();

        for (int lvl = 0; lvl < 8; ++lvl) {
            int stride = 1 << lvl;
            if ((tid & (2 * stride - 1)) == 0) {
                sh_crc[tid] = gf2_matrix_times(gf2_pow[6 + lvl], sh_crc[tid]) ^ sh_crc[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            blk_crc = gf2_matrix_times(gf2_pow[14], blk_crc) ^ sh_crc[0];
        }
        __syncthreads();
        off += 16384ull;
    }

    uint64_t rem = len - off;
    if (rem > 0ull) {
        const B *src = data + off;
        for (uint64_t i = (uint64_t)tid; i < rem; i += 256ull) tile_sh[i] = src[i];
        __syncthreads();

        uint64_t t_start = (rem * (uint64_t)tid) / 256ull;
        uint64_t t_end = (rem * (uint64_t)(tid + 1)) / 256ull;
        U c = 0xFFFFFFFFu;
        for (uint64_t i = t_start; i < t_end; ++i) {
            c = crc_tab[(c ^ tile_sh[i]) & 0xFFu] ^ (c >> 8);
        }
        sh_crc[tid] = c ^ 0xFFFFFFFFu;
        sh_len[tid] = t_end - t_start;
        __syncthreads();

        for (int stride = 1; stride < 256; stride <<= 1) {
            if ((tid & (2 * stride - 1)) == 0) {
                U c2 = sh_crc[tid + stride];
                uint64_t l2 = sh_len[tid + stride];
                sh_crc[tid] = crc32_combine_dev(gf2_pow, 15, sh_crc[tid], c2, l2);
                sh_len[tid] += l2;
            }
            __syncthreads();
        }
        if (tid == 0) {
            blk_crc = crc32_combine_dev(gf2_pow, 15, blk_crc, sh_crc[0], rem);
        }
    }

    if (tid == 0) {
        crc_out[p] = blk_crc;
    }
}

// =========================================================================
// 3. Pure-CUDA Parallel IEEE 802.3 CRC32 (256 CTAs x 128 threads, coalesced 16 KiB tiles)
// =========================================================================

// If `len_ptr != nullptr`, reads length from `*len_ptr`; when `max_len > 0 && len >= max_len`,
// skips work (used to elide stored_crc when packet fell back to Literal).
__global__ void kv_crc32_stage1_gpu(
    const B *data,
    const uint64_t *len_ptr,
    uint64_t fixed_len,
    uint64_t max_len,
    U *partials,
    uint64_t *part_lens
) {
    int b = blockIdx.x;
    int tid = threadIdx.x; // 0..127
    uint64_t len = len_ptr ? *len_ptr : fixed_len;
    if (len == 0ull || (max_len > 0ull && len >= max_len)) {
        if (tid == 0) {
            partials[b] = 0u;
            part_lens[b] = 0ull;
        }
        return;
    }

    __shared__ B tile_sh[16384];
    __shared__ U crc_tab[256];
    __shared__ U gf2_pow[15][32];
    __shared__ U sh_crc[128];
    __shared__ uint64_t sh_len[128];

    crc_tab[tid] = crc32_byte_entry((U)tid);
    crc_tab[tid + 128] = crc32_byte_entry((U)(tid + 128));
    init_gf2_pow_sh(gf2_pow, 15, tid);
    __syncthreads();

    uint64_t blk_start = (len * (uint64_t)b) / (uint64_t)CRC_BLOCKS;
    uint64_t blk_end = (len * (uint64_t)(b + 1)) / (uint64_t)CRC_BLOCKS;
    uint64_t blk_len = blk_end - blk_start;

    U blk_crc = 0u;
    uint64_t off = blk_start;

    // Fast path: 16 KiB coalesced tiles with shared-memory CRC32 table & GF(2) shift matrices.
    while (off + 16384ull <= blk_end) {
        const B *src = data + off;
        if ((((uintptr_t)src) & 3u) == 0u) {
            const U *sw = reinterpret_cast<const U *>(src);
            U *dw = reinterpret_cast<U *>(tile_sh);
            for (int i = tid; i < 4096; i += 128) dw[i] = sw[i];
        } else {
            for (int i = tid; i < 16384; i += 128) tile_sh[i] = src[i];
        }
        __syncthreads();

        const B *p = tile_sh + tid * 128;
        U c = 0xFFFFFFFFu;
        #pragma unroll 4
        for (int i = 0; i < 128; ++i) {
            c = crc_tab[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
        }
        sh_crc[tid] = c ^ 0xFFFFFFFFu;
        __syncthreads();

        for (int lvl = 0; lvl < 7; ++lvl) {
            int stride = 1 << lvl;
            if ((tid & (2 * stride - 1)) == 0) {
                sh_crc[tid] = gf2_matrix_times(gf2_pow[7 + lvl], sh_crc[tid]) ^ sh_crc[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            blk_crc = gf2_matrix_times(gf2_pow[14], blk_crc) ^ sh_crc[0];
        }
        __syncthreads();
        off += 16384ull;
    }

    // Trailing partial tile (< 16 KiB).
    uint64_t rem = blk_end - off;
    if (rem > 0ull) {
        const B *src = data + off;
        for (uint64_t i = (uint64_t)tid; i < rem; i += 128ull) tile_sh[i] = src[i];
        __syncthreads();

        uint64_t t_start = (rem * (uint64_t)tid) / 128ull;
        uint64_t t_end = (rem * (uint64_t)(tid + 1)) / 128ull;
        U c = 0xFFFFFFFFu;
        for (uint64_t i = t_start; i < t_end; ++i) {
            c = crc_tab[(c ^ tile_sh[i]) & 0xFFu] ^ (c >> 8);
        }
        sh_crc[tid] = c ^ 0xFFFFFFFFu;
        sh_len[tid] = t_end - t_start;
        __syncthreads();

        for (int stride = 1; stride < 128; stride <<= 1) {
            if ((tid & (2 * stride - 1)) == 0) {
                U c2 = sh_crc[tid + stride];
                uint64_t l2 = sh_len[tid + stride];
                sh_crc[tid] = crc32_combine_dev(gf2_pow, 15, sh_crc[tid], c2, l2);
                sh_len[tid] += l2;
            }
            __syncthreads();
        }
        if (tid == 0) {
            blk_crc = crc32_combine_dev(gf2_pow, 15, blk_crc, sh_crc[0], rem);
        }
    }

    if (tid == 0) {
        partials[b] = blk_crc;
        part_lens[b] = blk_len;
    }
}

__global__ void kv_crc32_finish_gpu(
    const U *partials,
    const uint64_t *part_lens,
    U *crc_out
) {
    int tid = threadIdx.x; // 0..255
    __shared__ U gf2_pow[30][32];
    __shared__ U sh_crc[CRC_BLOCKS];
    __shared__ uint64_t sh_len[CRC_BLOCKS];
    init_gf2_pow_sh(gf2_pow, 30, tid);
    sh_crc[tid] = partials[tid];
    sh_len[tid] = part_lens[tid];
    __syncthreads();

    for (int stride = 1; stride < CRC_BLOCKS; stride <<= 1) {
        if ((tid & (2 * stride - 1)) == 0) {
            U c2 = sh_crc[tid + stride];
            uint64_t l2 = sh_len[tid + stride];
            sh_crc[tid] = crc32_combine_dev(gf2_pow, 30, sh_crc[tid], c2, l2);
            sh_len[tid] += l2;
        }
        __syncthreads();
    }
    if (tid == 0) {
        *crc_out = sh_crc[0];
    }
}

// =========================================================================
// 4. Pure-CUDA 32-way Interleaved 12-bit Static rANS (64 KiB chunks)
// =========================================================================

static __device__ __forceinline__ void rans32_encode_chunk_impl(
    const B *src,
    size_t raw_sz,
    B *slot,
    U *chunk_word_out
) {
    int tid = threadIdx.x; // 0..255
    if (raw_sz == 0) {
        if (tid == 0) *chunk_word_out = RAW_FLAG;
        return;
    }

    __shared__ union {
        U hist4[4][256];
        uint16_t wbuf[RANS_LANES][64];
        B lane_tmp[4][RANS_LANE_CAP_BYTES];
    } scratch_sh;
    __shared__ U freq_sh[256];
    __shared__ U cdf_sh[256];
    __shared__ U rcp_m_sh[256];
    __shared__ U rcp_sh_sh[256];
    __shared__ U warp_tmp_sh[8];
    __shared__ U warp_max_h_sh[8];
    __shared__ int warp_max_s_sh[8];
    __shared__ int warp_only_sh[8];
    __shared__ U active_freqs_sh[152];
    __shared__ B hdr_sh[RANS_HDR_RESERVE];
    __shared__ U states_sh[RANS_LANES];
    __shared__ uint16_t lane_words_sh[RANS_LANES];
    __shared__ U lane_dst_offs_sh[RANS_LANES];
    __shared__ int active_sh;
    __shared__ int max_sym_sh;
    __shared__ int mode_sh;      // 0 = raw fallback, 1 = RLE, 2/3 = rANS
    __shared__ int tbl_bytes_sh;
    __shared__ int overflow_sh;
    __shared__ int total_len_sh;

    scratch_sh.hist4[0][tid] = 0u;
    scratch_sh.hist4[1][tid] = 0u;
    scratch_sh.hist4[2][tid] = 0u;
    scratch_sh.hist4[3][tid] = 0u;
    __syncthreads();

    int bank = tid & 3;
    for (size_t i = tid; i < raw_sz; i += 256) {
        atomicAdd(&scratch_sh.hist4[bank][src[i]], 1u);
    }
    __syncthreads();

    U h = scratch_sh.hist4[0][tid] + scratch_sh.hist4[1][tid]
        + scratch_sh.hist4[2][tid] + scratch_sh.hist4[3][tid];

    int wid = tid >> 5;
    int lane = tid & 31;
    unsigned lane_lt_mask = (1u << lane) - 1u;

    // 1. Warp/CTA-parallel active count, only_sym, and deterministic max_sym (lowest symbol wins ties).
    unsigned nz_mask = __ballot_sync(0xFFFFFFFFu, h > 0u);
    U mh = h;
    int ms = tid;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        U oh = __shfl_down_sync(0xFFFFFFFFu, mh, off);
        int os = __shfl_down_sync(0xFFFFFFFFu, ms, off);
        if (oh > mh || (oh == mh && os < ms)) {
            mh = oh;
            ms = os;
        }
    }
    if (lane == 0) {
        warp_tmp_sh[wid] = (U)__popc(nz_mask);
        warp_only_sh[wid] = (nz_mask != 0u) ? (wid * 32 + (31 - __clz(nz_mask))) : -1;
        warp_max_h_sh[wid] = mh;
        warp_max_s_sh[wid] = ms;
    }
    __syncthreads();

    if (tid == 0) {
        int active = 0;
        int only_sym = 0;
        U best_h = 0u;
        int best_s = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            active += (int)warp_tmp_sh[w];
            if (warp_only_sh[w] >= 0) only_sym = warp_only_sh[w];
            if (warp_max_h_sh[w] > best_h || (warp_max_h_sh[w] == best_h && warp_max_s_sh[w] < best_s)) {
                best_h = warp_max_h_sh[w];
                best_s = warp_max_s_sh[w];
            }
        }
        active_sh = active;
        max_sym_sh = best_s;
        if (active == 1 && raw_sz > 2) {
            slot[0] = 1u;
            slot[1] = (B)only_sym;
            *chunk_word_out = 2u;
            mode_sh = 1;
        } else if (active <= 1 || raw_sz <= 256) {
            mode_sh = 0;
        } else {
            mode_sh = (active <= 150) ? 2 : 3;
        }
    }
    __syncthreads();

    if (mode_sh == 1) return;
    if (mode_sh == 0) {
        copy_span_cta(slot, src, raw_sz);
        if (tid == 0) *chunk_word_out = RAW_FLAG | (U)raw_sz;
        return;
    }

    // 2. Parallel largest-remainder (Hamilton) normalization across all 256 symbols simultaneously.
    int active = active_sh;
    U f = 0u;
    U rem = 0u;
    if (h > 0u) {
        uint64_t budget = (uint64_t)(RANS_PROB_SCALE - active);
        uint64_t prod = (uint64_t)h * budget;
        f = 1u + (U)(prod / (uint64_t)raw_sz);
        rem = (U)(prod % (uint64_t)raw_sz);
    }

    // Sum initial `f` across 8 warps.
    U wf = f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        wf += __shfl_down_sync(0xFFFFFFFFu, wf, off);
    }
    if (lane == 0) warp_tmp_sh[wid] = wf;
    __syncthreads();

    U init_sum = warp_tmp_sh[0] + warp_tmp_sh[1] + warp_tmp_sh[2] + warp_tmp_sh[3]
               + warp_tmp_sh[4] + warp_tmp_sh[5] + warp_tmp_sh[6] + warp_tmp_sh[7];
    __syncthreads();
    U left = (U)RANS_PROB_SCALE - init_sum;

    if (left > 0u) {
        bool elig = (h > 0u && f < 4095u);
        U lo = 0u, hi = (U)raw_sz;
        while (lo < hi) {
            U mid = (lo + hi) >> 1;
            unsigned m = __ballot_sync(0xFFFFFFFFu, elig && rem >= mid);
            if (lane == 0) warp_tmp_sh[wid] = (U)__popc(m);
            __syncthreads();
            U cnt = warp_tmp_sh[0] + warp_tmp_sh[1] + warp_tmp_sh[2] + warp_tmp_sh[3]
                  + warp_tmp_sh[4] + warp_tmp_sh[5] + warp_tmp_sh[6] + warp_tmp_sh[7];
            __syncthreads();
            if (cnt > left) lo = mid + 1u;
            else hi = mid;
        }
        // Increment symbols with `rem >= lo` in ascending symbol order (bit-exact with serial loop).
        unsigned ge_m = __ballot_sync(0xFFFFFFFFu, elig && rem >= lo);
        if (lane == 0) warp_tmp_sh[wid] = (U)__popc(ge_m);
        __syncthreads();
        U ge_base = 0u;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < wid) ge_base += warp_tmp_sh[w];
        }
        U ge_total = warp_tmp_sh[0] + warp_tmp_sh[1] + warp_tmp_sh[2] + warp_tmp_sh[3]
                   + warp_tmp_sh[4] + warp_tmp_sh[5] + warp_tmp_sh[6] + warp_tmp_sh[7];
        __syncthreads();
        if (elig && rem >= lo) {
            U rank = ge_base + (U)__popc(ge_m & lane_lt_mask);
            if (rank < left) f++;
        }
        left = (ge_total >= left) ? 0u : (left - ge_total);
        if (left > 0u && lo > 0u) {
            U tie = lo - 1u;
            unsigned eq_m = __ballot_sync(0xFFFFFFFFu, elig && rem == tie);
            if (lane == 0) warp_tmp_sh[wid] = (U)__popc(eq_m);
            __syncthreads();
            U eq_base = 0u;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < wid) eq_base += warp_tmp_sh[w];
            }
            U eq_total = warp_tmp_sh[0] + warp_tmp_sh[1] + warp_tmp_sh[2] + warp_tmp_sh[3]
                       + warp_tmp_sh[4] + warp_tmp_sh[5] + warp_tmp_sh[6] + warp_tmp_sh[7];
            __syncthreads();
            if (elig && rem == tie) {
                U rank = eq_base + (U)__popc(eq_m & lane_lt_mask);
                if (rank < left) f++;
            }
            left = (eq_total >= left) ? 0u : (left - eq_total);
        }
        if (left > 0u && tid == max_sym_sh) {
            f += left;
        }
    }
    freq_sh[tid] = f;

    // 3. Parallel 256-element exclusive prefix sum for `cdf_sh[tid]` + parallel division reciprocals.
    U scan = f;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        U prev = __shfl_up_sync(0xFFFFFFFFu, scan, off);
        if (lane >= off) scan += prev;
    }
    if (lane == 31) warp_tmp_sh[wid] = scan;
    __syncthreads();
    U cdf_base = 0u;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        if (w < wid) cdf_base += warp_tmp_sh[w];
    }
    __syncthreads();
    cdf_sh[tid] = cdf_base + scan - f;

    if (f > 0u) {
        prep_div_u32(f, rcp_m_sh[tid], rcp_sh_sh[tid]);
    } else {
        rcp_m_sh[tid] = 0u;
        rcp_sh_sh[tid] = 0u;
    }

    // 4. Parallel frequency-table header serialization into `hdr_sh`.
    if (mode_sh == 2) {
        unsigned act_m = __ballot_sync(0xFFFFFFFFu, f > 0u);
        if (lane == 0) warp_tmp_sh[wid] = (U)__popc(act_m);
        __syncthreads();
        U act_base = 0u;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < wid) act_base += warp_tmp_sh[w];
        }
        if (f > 0u) {
            U k = act_base + (U)__popc(act_m & lane_lt_mask);
            hdr_sh[2 + k] = (B)tid;
            active_freqs_sh[k] = f;
        }
        __syncthreads();
        int pairs = active >> 1;
        if (tid < pairs) {
            U a = active_freqs_sh[tid * 2];
            U b = active_freqs_sh[tid * 2 + 1];
            int pos = 2 + active + tid * 3;
            hdr_sh[pos] = (B)(a & 0xFFu);
            hdr_sh[pos + 1] = (B)(((a >> 8) & 0x0Fu) | ((b & 0x0Fu) << 4));
            hdr_sh[pos + 2] = (B)((b >> 4) & 0xFFu);
        }
        if (tid == 0) {
            hdr_sh[0] = 2u;
            hdr_sh[1] = (B)(active - 1);
            int pos = 2 + active + pairs * 3;
            if (active & 1) {
                U a = active_freqs_sh[active - 1];
                hdr_sh[pos++] = (B)(a & 0xFFu);
                hdr_sh[pos++] = (B)((a >> 8) & 0x0Fu);
            }
            tbl_bytes_sh = pos;
        }
    } else {
        __syncthreads();
        if (tid < 128) {
            U a = freq_sh[tid * 2];
            U b = freq_sh[tid * 2 + 1];
            int pos = 1 + tid * 3;
            hdr_sh[pos] = (B)(a & 0xFFu);
            hdr_sh[pos + 1] = (B)(((a >> 8) & 0x0Fu) | ((b & 0x0Fu) << 4));
            hdr_sh[pos + 2] = (B)((b >> 4) & 0xFFu);
        }
        if (tid == 0) {
            hdr_sh[0] = 3u;
            tbl_bytes_sh = 385;
        }
    }
    __syncthreads();

    // Warp 0 encodes all 32 interleaved stride-32 lanes in lockstep (coalesced reads + smem burst writes!).
    if (tid < RANS_LANES) {
        int lane = tid;
        int cnt = (raw_sz > (size_t)lane) ? (int)((raw_sz - 1 - (size_t)lane) / 32 + 1) : 0;
        uint16_t *lane_end_u16 = reinterpret_cast<uint16_t *>(
            slot + RANS_HDR_RESERVE + (size_t)(lane + 1) * RANS_LANE_CAP_BYTES
        );
        U x = RANS_L;
        int nwords = 0;
        int bcnt = 0;
        int ovf = 0;
        for (int k = cnt - 1; k >= 0; --k) {
            B s = src[(size_t)k * 32 + (size_t)lane];
            U f = freq_sh[s];
            U c = cdf_sh[s];
            U x_max = f << 20;
            if (x >= x_max) {
                if (nwords >= RANS_LANE_CAP_U16) {
                    ovf = 1;
                    break;
                }
                scratch_sh.wbuf[lane][63 - bcnt] = (uint16_t)(x & 0xFFFFu);
                bcnt++;
                nwords++;
                if (bcnt == 64) {
                    uint16_t *dst_w = lane_end_u16 - nwords;
                    #pragma unroll 8
                    for (int j = 0; j < 64; ++j) dst_w[j] = scratch_sh.wbuf[lane][j];
                    bcnt = 0;
                }
                x >>= 16;
            }
            U q = div_u32_fast(x, rcp_m_sh[s], rcp_sh_sh[s]);
            U r = x - q * f;
            x = (q << RANS_PROB_BITS) + r + c;
        }
        if (!ovf && bcnt > 0) {
            uint16_t *dst_w = lane_end_u16 - nwords;
            for (int j = 0; j < bcnt; ++j) dst_w[j] = scratch_sh.wbuf[lane][64 - bcnt + j];
        }
        states_sh[lane] = x;
        lane_words_sh[lane] = (uint16_t)nwords;
        unsigned any_ovf = __any_sync(0xFFFFFFFFu, ovf != 0);
        if (lane == 0) overflow_sh = any_ovf ? 1 : 0;
    }
    __syncthreads();

    if (tid < RANS_LANES) {
        int l = tid;
        U len_b = (U)lane_words_sh[l] * 2u;
        U scan_b = len_b;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            U prev = __shfl_up_sync(0xFFFFFFFFu, scan_b, off);
            if (l >= off) scan_b += prev;
        }
        lane_dst_offs_sh[l] = scan_b - len_b;
        int hdr_len = tbl_bytes_sh + RANS_LANES * 6;
        int p_st = tbl_bytes_sh + l * 4;
        store_u32_le(hdr_sh + p_st, states_sh[l]);
        int p_nw = tbl_bytes_sh + RANS_LANES * 4 + l * 2;
        store_u16_le(hdr_sh + p_nw, lane_words_sh[l]);
        if (l == 31) {
            int tot = hdr_len + (int)scan_b;
            total_len_sh = (overflow_sh || (size_t)tot >= raw_sz) ? -1 : tot;
        }
    }
    __syncthreads();

    if (total_len_sh < 0) {
        copy_span_cta(slot, src, raw_sz);
        if (tid == 0) *chunk_word_out = RAW_FLAG | (U)raw_sz;
        return;
    }

    int hdr_len = tbl_bytes_sh + RANS_LANES * 6;
    for (int i = tid; i < hdr_len; i += 256) {
        slot[i] = hdr_sh[i];
    }
    // 4-lane-parallel compaction (2 warps = 64 threads per lane, 8 iterations = 16 barriers
    // instead of 64, aliased with hist4/wbuf to keep total CTA shared memory ~15 KiB for high occupancy).
    // Safe because hdr_len <= 577 < RANS_HDR_RESERVE (640), so dst of group g never overlaps src of > g.
    int sub_lane = tid & 63;
    int grp_lane = tid >> 6; // 0..3
    for (int g = 0; g < 8; ++g) {
        int l = g * 4 + grp_lane;
        int len_l = (int)lane_words_sh[l] * 2;
        if (len_l > 0) {
            const B *src_l = slot + RANS_HDR_RESERVE + (size_t)(l + 1) * RANS_LANE_CAP_BYTES - len_l;
            B *tmp = scratch_sh.lane_tmp[grp_lane];
            for (int i = sub_lane; i < len_l; i += 64) tmp[i] = src_l[i];
        }
        __syncthreads();
        if (len_l > 0) {
            B *dst_l = slot + hdr_len + lane_dst_offs_sh[l];
            const B *tmp = scratch_sh.lane_tmp[grp_lane];
            for (int i = sub_lane; i < len_l; i += 64) dst_l[i] = tmp[i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *chunk_word_out = (U)total_len_sh;
    }
}

// Encodes each chunk `cid` from `ans_in + chunk_offs[cid]` (length `chunk_sizes[cid]`) into
// `slot_out + cid * RANS_SLOT_STRIDE`.
__global__ void kv_rans32_encode_gpu(
    const B *ans_in,
    const uint64_t *chunk_offs,
    const uint64_t *chunk_sizes,
    B *slot_out,
    U *chunk_words,
    int num_chunks
) {
    int cid = blockIdx.x;
    if (cid >= num_chunks) return;
    rans32_encode_chunk_impl(
        ans_in + chunk_offs[cid],
        (size_t)chunk_sizes[cid],
        slot_out + (size_t)cid * (size_t)RANS_SLOT_STRIDE,
        chunk_words + cid
    );
}

// Multi-page batched 32-way rANS encode (`grid = num_chunks_per_page * batch`, `block = 256`).
__global__ void kv_rans32_encode_batch_gpu(
    const B *ans_in,
    const uint64_t *chunk_offs,
    const uint64_t *chunk_sizes,
    B *slot_out,
    U *chunk_words,
    int num_chunks_per_page,
    size_t ans_page_stride,
    int batch
) {
    int bid = blockIdx.x;
    int p = bid / num_chunks_per_page;
    int c = bid % num_chunks_per_page;
    if (p >= batch) return;
    rans32_encode_chunk_impl(
        ans_in + (size_t)p * ans_page_stride + chunk_offs[c],
        (size_t)chunk_sizes[c],
        slot_out + (size_t)bid * (size_t)RANS_SLOT_STRIDE,
        chunk_words + bid
    );
}

static __device__ __forceinline__ void rans32_decode_chunk_impl(
    const B *in,
    U word,
    B *dst,
    size_t raw_sz,
    U *err_flag
) {
    int tid = threadIdx.x; // 0..255
    bool is_raw = (word & RAW_FLAG) != 0u;
    size_t pkt_len = (size_t)(word & ~RAW_FLAG);

    if (is_raw) {
        if (pkt_len != raw_sz) {
            if (tid == 0) atomicOr(err_flag, 1u);
            return;
        }
        copy_span_cta(dst, in, raw_sz);
        return;
    }

    __shared__ U freq_sh[256];
    __shared__ U cdf_sh[256];
    __shared__ U lut_sh[RANS_PROB_SCALE]; // 16 KiB: sym | (freq << 8) | (cdf << 20)
    __shared__ U states_sh[RANS_LANES];
    __shared__ uint16_t lane_words_sh[RANS_LANES];
    __shared__ U lane_offs_sh[RANS_LANES];
    __shared__ int dec_mode_sh; // -1 = err, 1 = RLE, 2 = rANS
    __shared__ B rle_sym_sh;
    __shared__ U payload_off_sh;

    if (tid == 0) {
        dec_mode_sh = -1;
        if (pkt_len >= 2) {
            B mode = in[0];
            if (mode == 1u) {
                if (pkt_len == 2) {
                    rle_sym_sh = in[1];
                    dec_mode_sh = 1;
                }
            } else if (mode == 2u || mode == 3u) {
                for (int s = 0; s < 256; ++s) freq_sh[s] = 0u;
                bool ok = true;
                size_t tbl_len = 0;
                int active = 0;
                if (mode == 2u) {
                    int k = (int)in[1] + 1;
                    size_t fbytes = ((size_t)k * 12 + 7) / 8;
                    tbl_len = 2 + (size_t)k + fbytes;
                    if (k < 2 || k > 150 || pkt_len < tbl_len + (size_t)(RANS_LANES * 6)) {
                        ok = false;
                    } else {
                        int prev = -1;
                        const B *syms = in + 2;
                        const B *fp = in + 2 + k;
                        for (int i = 0; i < k; ++i) {
                            int s = (int)syms[i];
                            if (s <= prev) {
                                ok = false;
                                break;
                            }
                            prev = s;
                            U f;
                            size_t bi = ((size_t)(i >> 1)) * 3;
                            if ((i & 1) == 0) {
                                f = (U)fp[bi] | (((U)fp[bi + 1] & 0x0Fu) << 8);
                            } else {
                                f = ((U)fp[bi + 1] >> 4) | ((U)fp[bi + 2] << 4);
                            }
                            if (f == 0u || f > 4095u) {
                                ok = false;
                                break;
                            }
                            freq_sh[s] = f;
                            active++;
                        }
                        if (ok && (k & 1) != 0) {
                            size_t last_bi = ((size_t)(k >> 1)) * 3 + 1;
                            if ((fp[last_bi] & 0xF0u) != 0u) ok = false;
                        }
                    }
                } else {
                    tbl_len = 1 + 384;
                    if (pkt_len < tbl_len + (size_t)(RANS_LANES * 6)) {
                        ok = false;
                    } else {
                        const B *fp = in + 1;
                        for (int s = 0; s < 256; s += 2) {
                            size_t bi = ((size_t)(s >> 1)) * 3;
                            U a = (U)fp[bi] | (((U)fp[bi + 1] & 0x0Fu) << 8);
                            U b = ((U)fp[bi + 1] >> 4) | ((U)fp[bi + 2] << 4);
                            if (a > 4095u || b > 4095u) {
                                ok = false;
                                break;
                            }
                            freq_sh[s] = a;
                            freq_sh[s + 1] = b;
                            if (a > 0u) active++;
                            if (b > 0u) active++;
                        }
                        if (active < 2) ok = false;
                    }
                }
                if (ok) {
                    U sum = 0u;
                    for (int s = 0; s < 256; ++s) {
                        cdf_sh[s] = sum;
                        sum += freq_sh[s];
                    }
                    if (sum != (U)RANS_PROB_SCALE) {
                        ok = false;
                    }
                }
                if (ok) {
                    size_t p = tbl_len;
                    for (int l = 0; l < RANS_LANES; ++l) {
                        U st = load_u32_le(in + p);
                        if (st < RANS_L) ok = false;
                        states_sh[l] = st;
                        p += 4;
                    }
                    U off = 0u;
                    for (int l = 0; l < RANS_LANES; ++l) {
                        uint16_t nw = load_u16_le(in + p);
                        lane_words_sh[l] = nw;
                        lane_offs_sh[l] = off;
                        off += (U)nw * 2u;
                        p += 2;
                    }
                    if (p + (size_t)off != pkt_len) ok = false;
                    payload_off_sh = (U)p;
                }
                if (ok) dec_mode_sh = 2;
            }
        }
        if (dec_mode_sh < 0) atomicOr(err_flag, 1u);
    }
    __syncthreads();

    if (dec_mode_sh < 0) return;
    if (dec_mode_sh == 1) {
        B sym = rle_sym_sh;
        for (size_t i = tid; i < raw_sz; i += 256) dst[i] = sym;
        return;
    }

    // Build 4096-entry LUT in shared memory using all 256 threads (1 symbol per thread).
    {
        U f = freq_sh[tid];
        if (f > 0u) {
            U c = cdf_sh[tid];
            U entry = (U)tid | (f << 8) | (c << 20);
            for (U i = 0; i < f; ++i) {
                lut_sh[c + i] = entry;
            }
        }
    }
    __syncthreads();

    // Warp 0 decodes all 32 interleaved lanes in lockstep (coalesced writes!).
    if (tid < RANS_LANES) {
        int lane = tid;
        int cnt = (raw_sz > (size_t)lane) ? (int)((raw_sz - 1 - (size_t)lane) / 32 + 1) : 0;
        U x = states_sh[lane];
        int nw = (int)lane_words_sh[lane];
        const B *lp = in + payload_off_sh + lane_offs_sh[lane];
        int wpos = 0;
        int bad = 0;
        for (int k = 0; k < cnt; ++k) {
            U slot = x & 4095u;
            U entry = lut_sh[slot];
            dst[(size_t)k * 32 + (size_t)lane] = (B)(entry & 0xFFu);
            U f = (entry >> 8) & 0xFFFu;
            U c = entry >> 20;
            x = f * (x >> RANS_PROB_BITS) + (slot - c);
            if (x < RANS_L) {
                if (wpos >= nw) {
                    bad = 1;
                    break;
                }
                x = (x << 16) | (U)load_u16_le(lp + (size_t)wpos * 2);
                wpos++;
            }
        }
        if (bad || x != RANS_L || wpos != nw) {
            atomicOr(err_flag, 1u);
        }
    }
}

// Decodes all chunks directly from `d_pkt` into `ans_out` (handles RAW_FLAG, RLE, and RANS32).
__global__ void kv_rans32_decode_gpu(
    const B *d_pkt,
    const U *chunk_pkt_offs,
    const U *chunk_words,
    const uint64_t *ans_offs,
    const uint64_t *chunk_raw_sizes,
    B *ans_out,
    int num_chunks,
    U *err_flag
) {
    int cid = blockIdx.x;
    if (cid >= num_chunks) return;
    rans32_decode_chunk_impl(
        d_pkt + (size_t)chunk_pkt_offs[cid],
        chunk_words[cid],
        ans_out + (size_t)ans_offs[cid],
        (size_t)chunk_raw_sizes[cid],
        err_flag
    );
}

// Multi-page batched 32-way rANS decode (`grid = num_chunks_per_page * batch`, `block = 256`).
__global__ void kv_rans32_decode_batch_gpu(
    const B *d_pkt,
    const uint64_t *pkt_base_offs,
    const U *chunk_pkt_offs,
    const U *chunk_words,
    const uint64_t *ans_offs,
    const uint64_t *chunk_raw_sizes,
    B *ans_out,
    int num_chunks_per_page,
    size_t ans_page_stride,
    int batch,
    U *err_flag
) {
    int bid = blockIdx.x;
    int p = bid / num_chunks_per_page;
    int c = bid % num_chunks_per_page;
    if (p >= batch) return;
    rans32_decode_chunk_impl(
        d_pkt + (size_t)pkt_base_offs[p] + (size_t)chunk_pkt_offs[bid],
        chunk_words[bid],
        ans_out + (size_t)p * ans_page_stride + (size_t)ans_offs[c],
        (size_t)chunk_raw_sizes[c],
        err_flag
    );
}

// =========================================================================
// 5. On-GPU GKV1/v2 packet header/directory assembly & compaction
// =========================================================================

// Single-thread plan builder: computes chunk offsets, low-bit lane offsets, total packet size,
// and writes the complete GKV1/v2 header + chunk dir + lane table directly into `d_pkt`.
__global__ void kv_build_packet_plan_gpu(
    B *d_pkt,
    const U *chunk_words,
    U *chunk_pkt_offs,
    const uint16_t *lane_u32s,
    U *lane_byte_offs,
    uint64_t *total_pkt_out,
    U *low_base_out,
    U orig_bytes,
    U num_chunks,
    U num_mats
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    U total_lanes = num_mats * 4u;
    U meta = (U)HEADER_BYTES + num_chunks * 4u + total_lanes * 2u;

    U pos = meta;
    for (U i = 0; i < num_chunks; ++i) {
        U w = chunk_words[i];
        U len = w & ~RAW_FLAG;
        chunk_pkt_offs[i] = pos;
        pos += len;
    }
    U low_base = pos;
    U low_bytes = 0u;
    for (U i = 0; i < total_lanes; ++i) {
        lane_byte_offs[i] = low_bytes;
        low_bytes += (U)lane_u32s[i] * 4u;
    }
    U total_pkt = low_base + low_bytes;
    *total_pkt_out = (uint64_t)total_pkt;
    *low_base_out = low_base;

    if (total_pkt >= orig_bytes) return;

    store_u32_le(d_pkt + 0, GKV1_MAGIC);
    store_u32_le(d_pkt + 4, GKV1_VERSION);
    store_u32_le(d_pkt + 8, orig_bytes);
    store_u32_le(d_pkt + 12, num_chunks);
    store_u32_le(d_pkt + 16, num_mats);
    store_u32_le(d_pkt + 20, low_bytes);

    U dir_end = (U)HEADER_BYTES + num_chunks * 4u;
    for (U i = 0; i < num_chunks; ++i) {
        store_u32_le(d_pkt + HEADER_BYTES + i * 4u, chunk_words[i]);
    }
    for (U i = 0; i < total_lanes; ++i) {
        store_u16_le(d_pkt + dir_end + i * 2u, lane_u32s[i]);
    }
}

// Compacts all chunks (from `slot_out`) and all low-bit lanes (from `low_padded_in`) into
// contiguous `d_pkt` in ONE launch. Skips immediately if `*total_pkt >= orig_bytes`.
__global__ void kv_compact_packet_gpu(
    B *d_pkt,
    const B *slot_out,
    const U *chunk_words,
    const U *chunk_pkt_offs,
    const U *low_padded_in,
    const uint16_t *lane_u32s,
    const U *lane_byte_offs,
    const uint64_t *total_pkt_ptr,
    const U *low_base_ptr,
    U orig_bytes,
    U num_chunks,
    U num_mats
) {
    if (*total_pkt_ptr >= (uint64_t)orig_bytes) return;
    U bid = (U)blockIdx.x;
    if (bid < num_chunks) {
        U len = chunk_words[bid] & ~RAW_FLAG;
        const B *src = slot_out + (size_t)bid * (size_t)RANS_SLOT_STRIDE;
        B *dst = d_pkt + (size_t)chunk_pkt_offs[bid];
        copy_span_cta(dst, src, (size_t)len);
    } else {
        U lane_id = bid - num_chunks;
        if (lane_id >= num_mats * 4u) return;
        U nwords = (U)lane_u32s[lane_id];
        if (nwords == 0u) return;
        const B *src = reinterpret_cast<const B *>(low_padded_in + (size_t)lane_id * LANE_STRIDE);
        B *dst = d_pkt + (size_t)(*low_base_ptr + lane_byte_offs[lane_id]);
        copy_span_cta(dst, src, (size_t)nwords * 4u);
    }
}

// Multi-page batched packet plan builder for non-matrix page batches (`grid = 1`, `block = 1..64`).
// Strides across `blockDim.x` so both `block = 64` (parallel) and `block = 1` (older binaries)
// pack all `batch` (<= 64) packets contiguously into `d_pkt` (leaving `orig_bytes` literal slots
// for incompressible pages), computing per-page directories and writing `GKV1/v2` headers.
__global__ void kv_build_packet_plan_batch_gpu(
    B *d_pkt,
    const U *chunk_words,
    U *chunk_pkt_offs,
    uint64_t *total_pkt_arr,
    uint64_t *pkt_out_offs,
    uint64_t *total_batch_bytes_out,
    U orig_bytes,
    U num_chunks_per_page,
    int batch
) {
    if (blockIdx.x != 0) return;
    __shared__ uint64_t page_used_sh[64];
    __shared__ uint64_t page_off_sh[64];
    __shared__ U page_pos_sh[64];

    U meta = (U)HEADER_BYTES + num_chunks_per_page * 4u;
    for (int p = threadIdx.x; p < batch; p += blockDim.x) {
        U pos = meta;
        U base_c = (U)p * num_chunks_per_page;
        for (U c = 0; c < num_chunks_per_page; ++c) {
            U w = chunk_words[base_c + c];
            U len = w & ~RAW_FLAG;
            chunk_pkt_offs[base_c + c] = pos;
            pos += len;
        }
        total_pkt_arr[p] = (uint64_t)pos;
        page_pos_sh[p] = pos;
        page_used_sh[p] = (pos < orig_bytes) ? (uint64_t)pos : (uint64_t)orig_bytes;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t batch_pos = 0ull;
        for (int i = 0; i < batch; ++i) {
            page_off_sh[i] = batch_pos;
            batch_pos += page_used_sh[i];
        }
        *total_batch_bytes_out = batch_pos;
    }
    __syncthreads();

    for (int p = threadIdx.x; p < batch; p += blockDim.x) {
        uint64_t out_off = page_off_sh[p];
        pkt_out_offs[p] = out_off;
        if (page_pos_sh[p] < orig_bytes) {
            U base_c = (U)p * num_chunks_per_page;
            B *hdr = d_pkt + out_off;
            store_u32_le(hdr + 0, GKV1_MAGIC);
            store_u32_le(hdr + 4, GKV1_VERSION);
            store_u32_le(hdr + 8, orig_bytes);
            store_u32_le(hdr + 12, num_chunks_per_page);
            store_u32_le(hdr + 16, 0u);
            store_u32_le(hdr + 20, 0u);
            for (U c = 0; c < num_chunks_per_page; ++c) {
                store_u32_le(hdr + HEADER_BYTES + c * 4u, chunk_words[base_c + c]);
            }
        }
    }
}

// Compacts all chunks (and raw literals for incompressible pages) across `batch` pages in one launch.
// `grid = num_chunks_per_page * batch`, `block = 128`.
__global__ void kv_compact_packet_batch_gpu(
    B *d_pkt,
    const B *slot_out,
    const B *raw_buf_c,
    const U *chunk_words,
    const U *chunk_pkt_offs,
    const uint64_t *total_pkt_arr,
    const uint64_t *pkt_out_offs,
    U orig_bytes,
    U num_chunks_per_page,
    int batch
) {
    U bid = (U)blockIdx.x;
    U p = bid / num_chunks_per_page;
    U c = bid % num_chunks_per_page;
    if ((int)p >= batch) return;
    uint64_t pkt_len = total_pkt_arr[p];
    uint64_t out_base = pkt_out_offs[p];
    if (pkt_len < (uint64_t)orig_bytes) {
        U len = chunk_words[bid] & ~RAW_FLAG;
        const B *src = slot_out + (size_t)bid * (size_t)RANS_SLOT_STRIDE;
        B *dst = d_pkt + (size_t)out_base + (size_t)chunk_pkt_offs[bid];
        copy_span_cta(dst, src, (size_t)len);
    } else {
        size_t start = ((size_t)orig_bytes * (size_t)c) / (size_t)num_chunks_per_page;
        size_t end = ((size_t)orig_bytes * (size_t)(c + 1u)) / (size_t)num_chunks_per_page;
        if (end > start) {
            const B *src = raw_buf_c + (size_t)p * (size_t)orig_bytes + start;
            B *dst = d_pkt + (size_t)out_base + start;
            copy_span_cta(dst, src, end - start);
        }
    }
}

} // extern "C"
