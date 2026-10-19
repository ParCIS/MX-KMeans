#include "kmeans.cuh"
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
struct Accumulate {
    int d;
    const float* points;
    const int* labels;
    float* centroids;
    int* counts;
    __device__ void operator()(size_t index) const {
        const int row = index / d;
        const int feature = index % d;
        const int cluster = labels[row];
        if (feature == 0) atomicAdd(counts + cluster, 1);
        atomicAdd(centroids + size_t(cluster) * d + feature, points[index]);
    }
};
struct Normalize {
    int d;
    float* centroids;
    const float* old_centroids;
    const int* counts;
    __device__ void operator()(int index) const {
        const int count = counts[index / d];
        // Preserve an empty cluster's previous center instead of moving it to zero.
        centroids[index] = count ? centroids[index] / count : old_centroids[index];
    }
};
__global__ void shift_kernel(int elements, const float* previous, const float* current, float* sum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        const float diff = previous[i] - current[i];
        atomicAdd(sum, diff * diff);
    }
}
}  // namespace

void update_centroids(int n, int d, int k, const float* points, const int* labels,
                      const float* previous, float* centroids, int* counts) {
    CUDA_CHECK(cudaMemset(centroids, 0, size_t(k) * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(counts, 0, size_t(k) * sizeof(int)));
    thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(size_t(n) * d),
                     Accumulate{d, points, labels, centroids, counts});
    thrust::for_each(thrust::device, thrust::make_counting_iterator<int>(0),
                     thrust::make_counting_iterator<int>(k * d),
                     Normalize{d, centroids, previous, counts});
    CUDA_CHECK(cudaGetLastError());
}

float centroid_shift(int elements, const float* previous, const float* current, float* device_sum) {
    CUDA_CHECK(cudaMemset(device_sum, 0, sizeof(float)));
    shift_kernel<<<(elements + 255) / 256, 256>>>(elements, previous, current, device_sum);
    CUDA_CHECK(cudaGetLastError());
    float result;
    CUDA_CHECK(cudaMemcpy(&result, device_sum, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
