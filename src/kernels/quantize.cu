#include "kmeans.cuh"
#include "precision.hpp"
#include <cuda_fp6.h>
#include <cmath>
#include <stdexcept>

namespace {
// Each block contains 32 feature values and shares a UE8M0 scale.
__device__ Scale block_scale(float maximum, float representable_max) {
    return Scale(maximum > 0.0f ? maximum / representable_max : 1.0f);
}

template <int Bits>
__global__ void quantize_kernel(int rows, int d, const float* points, const int* pid,
                                unsigned char* values, Scale* scales, float* errors) {
    const int block = blockIdx.x * 4 + threadIdx.y;
    const int blocks_per_row = d / 32;
    const int row = block / blocks_per_row;
    if (row >= rows) return;  // Uniform across the warp.
    const int feature = (block % blocks_per_row) * 32;
    const int lane = threadIdx.x;
    const int source_row = pid ? pid[row] : row;
    const float x = points[size_t(source_row) * d + feature + lane];
    float maximum = fabsf(x);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    constexpr float max_value = Bits == 8 ? 448.0f : (Bits == 6 ? 7.5f : 6.0f);
    const Scale scale = block_scale(maximum, max_value);
    const float scaled = x / float(scale);
    unsigned raw;
    float reconstructed;
    if constexpr (Bits == 8) {
        const MXFP8 q(scaled);
        raw = q.raw();
        reconstructed = float(q) * float(scale);
    } else if constexpr (Bits == 6) {
        const MXFP6 q(scaled);
        raw = q.raw();
        reconstructed = float(q) * float(scale);
    } else {
        const MXFP4 q(scaled);
        raw = q.raw();
        reconstructed = float(q) * float(scale);
    }
    const size_t byte_base = (size_t(row) * d + feature) * Bits / 8;
    if constexpr (Bits == 8) {
        values[byte_base + lane] = static_cast<unsigned char>(raw);
    } else if constexpr (Bits == 4) {
        const unsigned next = __shfl_down_sync(0xffffffff, raw, 1);
        if ((lane & 1) == 0)
            values[byte_base + lane / 2] = (raw & 0xf) | ((next & 0xf) << 4);
    } else {
        // Four 6-bit values occupy three consecutive bytes.
        const unsigned q1 = __shfl_down_sync(0xffffffff, raw, 1);
        const unsigned q2 = __shfl_down_sync(0xffffffff, raw, 2);
        const unsigned q3 = __shfl_down_sync(0xffffffff, raw, 3);
        if ((lane & 3) == 0) {
            const size_t offset = byte_base + lane / 4 * 3;
            values[offset] = (raw & 0x3f) | ((q1 & 3) << 6);
            values[offset + 1] = (q1 >> 2) | ((q2 & 0xf) << 4);
            values[offset + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    if (lane == 0) scales[block] = scale;
    if (errors) {
        float error = (x - reconstructed) * (x - reconstructed);
        for (int offset = 16; offset; offset >>= 1)
            error = fmaxf(error, __shfl_xor_sync(0xffffffff, error, offset));
        // Squared errors are non-negative, so integer ordering is equivalent.
        if (lane == 0) atomicMax(reinterpret_cast<unsigned*>(errors + row), __float_as_uint(error));
    }
}

__global__ void point_bounds_kernel(int n, int d, const float* points,
                                    const float* norms, float* upper) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= n) return;
    float e4 = 0.0f, e6 = 0.0f;
    for (int base = 0; base < d; base += 32) {
        float maximum = 0.0f;
        for (int j = 0; j < 32; ++j)
            maximum = fmaxf(maximum, fabsf(points[size_t(point) * d + base + j]));
        const Scale s4 = block_scale(maximum, 6.0f);
        const Scale s6 = block_scale(maximum, 7.5f);
        // Scan features 1 through 31 for blockwise error estimates.
        for (int j = 1; j < 32; ++j) {
            const float x = points[size_t(point) * d + base + j];
            e4 = fmaxf(e4, fabsf(float(MXFP4(x / float(s4))) * float(s4) - x));
            e6 = fmaxf(e6, fabsf(float(__nv_fp6_e3m2(x / float(s6))) * float(s6) - x));
        }
    }
    // Three 32-element rows per warp: norm, P6, P4.
    const size_t tile = size_t(point / 32) * 96;
    const int lane = point % 32;
    upper[tile + lane] = norms[point];
    upper[tile + 32 + lane] = norms[point] + d * e6 * e6;
    upper[tile + 64 + lane] = norms[point] + d * e4 * e4;
}

__global__ void precision_kernel(int n, int d, const float* upper, const float* gaps,
                                 float lc, float ec, int* precision) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= n) return;
    const size_t tile = size_t(point / 32) * 96;
    const int lane = point % 32;
    precision[point] = select_precision(upper[tile + 64 + lane], upper[tile + 32 + lane],
                                        lc, ec, d, gaps[point]);
}
}  // namespace

void quantize_rows(int bits, int rows, int d, const float* points, const int* pid,
                   void* values, Scale* scales, float* errors) {
    if (rows == 0) return;
    const int blocks = (rows * (d / 32) + 3) / 4;
    auto* bytes = static_cast<unsigned char*>(values);
    switch (bits) {
        case 4: quantize_kernel<4><<<blocks, dim3(32, 4)>>>(rows, d, points, pid, bytes, scales, errors); break;
        case 6: quantize_kernel<6><<<blocks, dim3(32, 4)>>>(rows, d, points, pid, bytes, scales, errors); break;
        case 8: quantize_kernel<8><<<blocks, dim3(32, 4)>>>(rows, d, points, pid, bytes, scales, errors); break;
        default: throw std::invalid_argument("Precision must be 4, 6, or 8");
    }
    CUDA_CHECK(cudaGetLastError());
}

void precompute_point_bounds(int n, int d, const float* points, const float* norms, float* upper) {
    point_bounds_kernel<<<(n + 255) / 256, 256>>>(n, d, points, norms, upper);
    CUDA_CHECK(cudaGetLastError());
}

void assign_precisions(int n, int d, const float* upper, const float* gaps,
                       float lc, float ec, int* precision) {
    precision_kernel<<<(n + 255) / 256, 256>>>(n, d, upper, gaps, lc, ec, precision);
    CUDA_CHECK(cudaGetLastError());
}
