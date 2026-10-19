#include "kmeans.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void upload(DeviceBuffer<T>& target, const std::vector<T>& source) {
    CUDA_CHECK(cudaMemcpy(target.get(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.size());
    CUDA_CHECK(cudaMemcpy(result.data(), source.get(), result.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

float decode(const std::vector<unsigned char>& bytes, const std::vector<Scale>& scales,
             int bits, int row, int d, int feature) {
    const size_t bit_offset = (size_t(row) * d + feature) * bits;
    const size_t byte = bit_offset / 8;
    unsigned raw = bytes[byte];
    if ((bit_offset % 8) + bits > 8) raw |= unsigned(bytes[byte + 1]) << 8;
    raw = (raw >> (bit_offset % 8)) & ((1u << bits) - 1);
    const float q = bits == 8 ? float(MXFP8::bitcast(raw)) :
                    bits == 6 ? float(MXFP6::bitcast(raw)) : float(MXFP4::bitcast(raw));
    return q * float(scales[size_t(row) * (d / 32) + feature / 32]);
}

void check_gemm_and_quantization() {
    // Cross both row and feature tile boundaries, with incomplete final warps.
    constexpr int m = 131, n = 132, d = 256;
    std::vector<float> a(m * d), b(n * d);
    for (int row = 0; row < m; ++row)
        for (int j = 0; j < d; ++j)
            a[row * d + j] = std::sin(float(row * 17 + j)) * std::ldexp(1.0f, (row + j / 32) % 9 - 4);
    for (int row = 0; row < n; ++row)
        for (int j = 0; j < d; ++j)
            b[row * d + j] = std::cos(float(row * 7 + j)) * std::ldexp(1.0f, (row * 2 + j / 32) % 7 - 3);
    DeviceBuffer<float> da(a.size()), db(b.size()), result(m * n), errors(m);
    DeviceBuffer<unsigned char> qa(a.size()), qb(b.size());
    DeviceBuffer<Scale> sa(m * (d / 32)), sb(n * (d / 32));
    upload(da, a); upload(db, b);
    quantize_rows(8, n, d, db.get(), nullptr, qb.get(), sb.get());
    const auto bbytes = download(qb);
    const auto bscales = download(sb);
    DeviceBuffer<int> ids(m);
    std::vector<int> reverse(m);
    for (int i = 0; i < m; ++i) reverse[i] = m - 1 - i;
    upload(ids, reverse);
    for (int bits : {4, 6, 8}) {
        errors.clear();
        quantize_rows(bits, m, d, da.get(), ids.get(), qa.get(), sa.get(), errors.get());
        const auto abytes = download(qa);
        const auto ascales = download(sa);
        const auto actual_errors = download(errors);
        for (int row = 0; row < m; ++row) {
            float max_error = 0.0f;
            for (int j = 0; j < d; ++j) {
                const float x = a[reverse[row] * d + j];
                const float scale = float(ascales[row * (d / 32) + j / 32]);
                const float expected = (bits == 8 ? float(MXFP8(x / scale)) :
                                        bits == 6 ? float(MXFP6(x / scale)) : float(MXFP4(x / scale))) * scale;
                const float actual = decode(abytes, ascales, bits, row, d, j);
                require(expected == actual, "Quantized values or packing mismatch");
                max_error = std::max(max_error, (x - actual) * (x - actual));
            }
            require(std::abs(max_error - actual_errors[row]) <= 1e-5f * (1 + max_error), "Squared error mismatch");
        }
        gemm_mx(bits, m, n, d, qa.get(), sa.get(), reinterpret_cast<MXFP8*>(qb.get()), sb.get(), result.get());
        const auto actual = download(result);
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
            double expected = 0.0, magnitude = 0.0;
            for (int f = 0; f < d; ++f) {
                const double product = double(decode(abytes, ascales, bits, i, d, f)) * decode(bbytes, bscales, 8, j, d, f);
                expected += product;
                magnitude += std::abs(product);
            }
            require(std::isfinite(actual[i * n + j]) && std::abs(expected - actual[i * n + j]) <= 1e-4 * (1 + magnitude),
                    "MX GEMM disagrees with CPU multiplication of reconstructed values");
        }
    }
}

void check_assignment_and_bounds() {
    constexpr int rows = 35, k = 8, d = 128;
    DeviceBuffer<float> norms(rows), cnorms(k), dots(rows * k), gaps(rows), upper(((rows + 31) / 32) * 96);
    DeviceBuffer<int> labels(rows), precision(rows);
    norms.clear(); cnorms.clear();
    std::vector<float> products(rows * k);
    for (int r = 0; r < rows; ++r) {
        const float distances[k] = {100, 4, 9, 200, 1, 40, 30, 20};
        for (int j = 0; j < k; ++j) products[r * k + j] = -0.5f * distances[j];
    }
    upload(dots, products);
    assign_clusters(rows, k, norms.get(), cnorms.get(), dots.get(), nullptr, labels.get(), gaps.get());
    for (int label : download(labels)) require(label == 4, "Wrong nearest centroid");
    for (float gap : download(gaps)) require(gap == 100, "Wrong two-largest gap");
    // Ties between closest centroids do not change the farthest gap.
    for (int r = 0; r < rows; ++r) products[r * k + 1] = -0.5f;
    upload(dots, products);
    assign_clusters(rows, k, norms.get(), cnorms.get(), dots.get(), nullptr, labels.get(), gaps.get());
    for (int label : download(labels)) require(label == 1, "Tie must select lowest centroid index");
    for (float gap : download(gaps)) require(gap == 100, "Nearest tie must not change the farthest gap");

    std::vector<float> input(rows * d, 0.0f);
    // Preserve the omission of feature zero in the error scan.
    for (int r = 0; r < rows; ++r) { input[r * d] = 1.3f; input[r * d + 1] = 6.0f; }
    DeviceBuffer<float> points(input.size()); upload(points, input);
    compute_l2_norm(rows, d, points.get(), norms.get());
    precompute_point_bounds(rows, d, points.get(), norms.get(), upper.get());
    const auto u = download(upper);
    const auto pnorms = download(norms);
    for (int r = 0; r < rows; ++r)
        require(u[(r / 32) * 96 + 64 + r % 32] == pnorms[r], "error scan must start at feature 1");
    // The error estimate uses E3M2, while packed FP6 uses E2M3.
    for (int r = 0; r < rows; ++r) { input[r * d] = 6.0f; input[r * d + 1] = 1.4f; }
    upload(points, input);
    compute_l2_norm(rows, d, points.get(), norms.get());
    precompute_point_bounds(rows, d, points.get(), norms.get(), upper.get());
    const auto estimated_upper = download(upper);
    const auto point_norms = download(norms);
    const float e3m2_error = float(cutlass::float_e3m2_t(1.4f)) - 1.4f;
    for (int r = 0; r < rows; ++r) {
        const float expected = point_norms[r] + d * e3m2_error * e3m2_error;
        require(std::abs(estimated_upper[(r / 32) * 96 + 32 + r % 32] - expected) < 1e-4f,
                "FP6 error estimate must use E3M2");
    }
    upload(gaps, std::vector<float>(rows, 1e6f));
    assign_precisions(rows, d, upper.get(), gaps.get(), 1, 0, precision.get());
    for (int p : download(precision)) require(p == 8, "MXFP8 override must apply even for a large gap");
    gaps.clear();
    assign_precisions(rows, d, upper.get(), gaps.get(), 1, 0, precision.get());
    for (int p : download(precision)) require(p == 8, "MXFP8 override failed");
}

void check_clustering() {
    constexpr int n = 35, d = 128, k = 4;
    std::vector<float> points(n * d, 0), centers(k * d, 0);
    for (int i = 0; i < n; ++i) points[i * d] = float((i % k) * 10);
    for (int c = 0; c < k; ++c) centers[c * d] = float(c * 10) + 1.0f;
    DeviceBuffer<float> x(points.size()), c(centers.size());
    DeviceBuffer<int> labels(n);
    upload(x, points); upload(c, centers);
    KMeansOptions options;
    require(options.fp32_iterations == 0, "default disables FP32 warm-up");
    options.max_iterations = 3;
    options.tolerance = 0;
    const int iterations = kmeans_mixed(n, d, k, x.get(), c.get(), labels.get(), options);
    require(iterations >= 2 && iterations <= 3, "Iteration count incorrect");
    const auto result = download(labels);
    for (int i = 0; i < n; ++i) require(result[i] == i % k, "Separated clusters changed assignment");
    upload(c, centers);
    options.max_iterations = 1;
    require(kmeans_mixed(n, d, k, x.get(), c.get(), labels.get(), options) == 1, "Iteration limit off by one");
    // Exercise an empty cluster independently of initialization.
    std::vector<int> one_cluster(n, 0); upload(labels, one_cluster);
    DeviceBuffer<float> old(centers.size()); upload(old, centers);
    DeviceBuffer<int> counts(k);
    update_centroids(n, d, k, x.get(), labels.get(), old.get(), c.get(), counts.get());
    const auto updated = download(c);
    for (int j = d; j < k * d; ++j) require(updated[j] == centers[j], "Empty cluster center changed");
}

int main() {
    cudaDeviceProp device{};
    const auto status = cudaGetDeviceProperties(&device, 0);
    if (status != cudaSuccess || device.major != 12 || device.minor != 0) {
        std::cout << "SKIP: an available SM120 Blackwell GPU is required\n";
        return 77;
    }
    try {
        check_gemm_and_quantization();
        check_assignment_and_bounds();
        check_clustering();
        std::cout << "CUDA checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
