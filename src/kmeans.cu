#include "kmeans.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

int kmeans_mixed(int n, int d, int k, const float* points, float* centroids,
                 int* labels, const KMeansOptions& options) {
    DeviceBuffer<float> point_norms(n), centroid_norms(k), previous(size_t(k) * d);
    DeviceBuffer<float> gaps(n), upper(size_t((n + 31) / 32) * 96), sum(1);
    DeviceBuffer<int> counts(k), precision(n), ids(n);
    // Reuse the same output and quantized-point buffers for each precision group.
    DeviceBuffer<float> dots(size_t(n) * k), centroid_errors(k);
    DeviceBuffer<unsigned char> quantized_points(size_t(n) * d);
    DeviceBuffer<MXFP8> quantized_centroids(size_t(k) * d);
    DeviceBuffer<Scale> point_scales(size_t(n) * (d / 32)), centroid_scales(size_t(k) * (d / 32));
    std::vector<float> host_norms(k), host_errors(k);

    compute_l2_norm(n, d, points, point_norms.get());
    precompute_point_bounds(n, d, points, point_norms.get(), upper.get());
    // Initialize distance statistics before the first precision selection.
    gaps.clear();
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        CUDA_CHECK(cudaMemcpy(previous.get(), centroids, size_t(k) * d * sizeof(float), cudaMemcpyDeviceToDevice));
        compute_l2_norm(k, d, centroids, centroid_norms.get());
        if (iteration < options.fp32_iterations) {
            gemm_fp32(n, k, d, points, centroids, dots.get());
            assign_clusters(n, k, point_norms.get(), centroid_norms.get(), dots.get(), nullptr, labels, gaps.get());
            if (options.verbose) std::cerr << "iteration=" << iteration + 1 << " precision=FP32\n";
        } else {
            centroid_errors.clear();
            quantize_rows(8, k, d, centroids, nullptr, quantized_centroids.get(), centroid_scales.get(), centroid_errors.get());
            CUDA_CHECK(cudaMemcpy(host_errors.data(), centroid_errors.get(), size_t(k) * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_norms.data(), centroid_norms.get(), size_t(k) * sizeof(float), cudaMemcpyDeviceToHost));
            const float ec = std::sqrt(*std::max_element(host_errors.begin(), host_errors.end()));
            const float lc = std::sqrt(*std::max_element(host_norms.begin(), host_norms.end()));
            assign_precisions(n, d, upper.get(), gaps.get(), lc, ec, precision.get());
            std::array<int, 3> group_sizes{};
            int group = 0;
            for (int bits : {4, 6, 8}) {
                auto begin = thrust::make_counting_iterator<int>(0);
                auto end = thrust::copy_if(thrust::device, begin, begin + n,
                    thrust::device_pointer_cast(precision.get()), thrust::device_pointer_cast(ids.get()),
                    [bits] __device__(int p) { return p == bits; });
                const int rows = end - thrust::device_pointer_cast(ids.get());
                group_sizes[group++] = rows;
                if (!rows) continue;
                quantize_rows(bits, rows, d, points, ids.get(), quantized_points.get(), point_scales.get());
                gemm_mx(bits, rows, k, d, quantized_points.get(), point_scales.get(),
                         quantized_centroids.get(), centroid_scales.get(), dots.get());
                assign_clusters(rows, k, point_norms.get(), centroid_norms.get(), dots.get(), ids.get(), labels, gaps.get());
            }
            if (options.verbose)
                std::cerr << "iteration=" << iteration + 1 << " fp4=" << group_sizes[0]
                          << " fp6=" << group_sizes[1] << " fp8=" << group_sizes[2] << '\n';
        }
        update_centroids(n, d, k, points, labels, previous.get(), centroids, counts.get());
        const float shift = centroid_shift(k * d, previous.get(), centroids, sum.get());
        if (!std::isfinite(shift)) throw std::runtime_error("Non-finite centroid shift; check the input scale");
        if (shift <= options.tolerance) return iteration + 1;
    }
    return options.max_iterations;
}
