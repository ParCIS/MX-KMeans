#pragma once

#include "cuda_utils.cuh"
#include "cutlass/numeric_types.h"

using MXFP8 = cutlass::float_e4m3_t;
using MXFP6 = cutlass::float_e2m3_t;
using MXFP4 = cutlass::float_e2m1_t;
using Scale = cutlass::float_ue8m0_t;

struct KMeansOptions {
    int max_iterations = 300;
    int fp32_iterations = 0;
    float tolerance = 1e-4f;
    bool verbose = false;
};

// Points and centroids are row-major FP32 matrices on the device. The caller
// initializes centroids. Returns the number of completed Lloyd iterations.
int kmeans_mixed(int n, int d, int k, const float* points, float* centroids,
                 int* labels, const KMeansOptions& options);

void compute_l2_norm(int rows, int d, const float* points, float* norms);
void update_centroids(int n, int d, int k, const float* points, const int* labels,
                      const float* old_centroids, float* centroids, int* counts);
float centroid_shift(int elements, const float* previous, const float* current,
                     float* device_sum);

// pid maps a partition row back to the original point. nullptr means identity.
// Labels use the nearest centroid; gaps use the two-largest statistic.
void assign_clusters(int rows, int k, const float* point_norms,
                     const float* centroid_norms, const float* dots, const int* pid,
                     int* labels, float* gaps);

void precompute_point_bounds(int n, int d, const float* points,
                             const float* norms, float* upper);
void assign_precisions(int n, int d, const float* upper, const float* gaps,
                       float lc, float ec, int* precision);
// Packed row-major values and one UE8M0 scale per 32 features. max_errors
// optionally receives the maximum squared reconstruction error per row;
// callers zero that buffer before quantization.
void quantize_rows(int bits, int rows, int d, const float* points, const int* pid,
                   void* values, Scale* scales, float* max_errors = nullptr);

void gemm_fp32(int m, int n, int k, const float* a, const float* b, float* out);
void gemm_mx(int bits, int m, int n, int k, const void* a, const Scale* a_scales,
             const MXFP8* b, const Scale* b_scales, float* out);
