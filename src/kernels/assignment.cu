#include "kmeans.cuh"
#include "precision.hpp"
#include <climits>
#include <math_constants.h>

namespace {
__global__ void norm_kernel(int rows, int d, const float* points, float* norms) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int j = 0; j < d; ++j) {
        const float x = points[size_t(row) * d + j];
        sum += x * x;
    }
    norms[row] = sum;
}

// One warp per point: nearest-centroid labels and two-largest-distance gaps.
__global__ void nearest_kernel(int rows, int k, const float* point_norms,
                               const float* centroid_norms, const float* dots,
                               const int* pid, int* labels, float* gaps) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const int point = pid ? pid[row] : row;
    float first = CUDART_INF_F, second = CUDART_INF_F;
    float largest = pid ? -1e38f : -1.0f;
    float next_largest = largest;
    int best = INT_MAX;
    for (int c = lane; c < k; c += 32) {
        const float distance = fmaf(-2.0f, dots[size_t(row) * k + c],
                                    point_norms[point] + centroid_norms[c]);
        insert_distance(distance, c, first, second, best);
        insert_largest_distance(distance, largest, next_largest);
    }
    for (int offset = 16; offset; offset >>= 1) {
        const float other_first = __shfl_down_sync(0xffffffff, first, offset);
        const float other_second = __shfl_down_sync(0xffffffff, second, offset);
        const int other_best = __shfl_down_sync(0xffffffff, best, offset);
        const float other_largest = __shfl_down_sync(0xffffffff, largest, offset);
        const float other_next = __shfl_down_sync(0xffffffff, next_largest, offset);
        if (lane + offset < 32) {
            const float merged_second = fminf(fmaxf(first, other_first), fminf(second, other_second));
            if (other_first < first || (other_first == first && other_best < best)) {
                first = other_first;
                best = other_best;
            }
            second = merged_second;
            next_largest = fmaxf(fminf(largest, other_largest), fmaxf(next_largest, other_next));
            largest = fmaxf(largest, other_largest);
        }
    }
    if (lane == 0) {
        labels[point] = best;
        gaps[point] = largest - next_largest;
    }
}
}  // namespace

void compute_l2_norm(int rows, int d, const float* points, float* norms) {
    norm_kernel<<<(rows + 255) / 256, 256>>>(rows, d, points, norms);
    CUDA_CHECK(cudaGetLastError());
}

void assign_clusters(int rows, int k, const float* point_norms,
                     const float* centroid_norms, const float* dots, const int* pid,
                     int* labels, float* gaps) {
    if (!rows) return;
    nearest_kernel<<<(rows + 7) / 8, 256>>>(rows, k, point_norms, centroid_norms, dots, pid, labels, gaps);
    CUDA_CHECK(cudaGetLastError());
}
