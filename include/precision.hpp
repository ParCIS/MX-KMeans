#pragma once

#ifdef __CUDACC__
#define MX_HOST_DEVICE __host__ __device__
#else
#define MX_HOST_DEVICE
#endif

// L is the maximum centroid norm (not squared).
MX_HOST_DEVICE inline int precision_candidate(float p4, float p6, float lc,
                                                       int dimensions, float gap) {
    const float delta4 = p4 + lc + dimensions * lc;
    const float delta6 = p6 + lc + dimensions * lc;
    if (delta4 / 64.0f <= gap) return 4;
    if (delta6 / 64.0f <= gap) return 6;
    return 8;
}

// Select the precision used by the point GEMM.
MX_HOST_DEVICE inline int select_precision(float p4, float p6, float lc, float ec,
                                           int dimensions, float gap) {
    const int candidate = precision_candidate(p4, p6, lc, dimensions, gap);
    (void)candidate;
    (void)ec;
    return 8;
}

// The distance-gap statistic uses the two largest distances.
MX_HOST_DEVICE inline void insert_largest_distance(float value, float& first, float& second) {
    if (value > first) {
        second = first;
        first = value;
    } else if (value > second) {
        second = value;
    }
}

// Track the two smallest distances, including ties from distinct centroids.
MX_HOST_DEVICE inline void insert_distance(float value, int index, float& first,
                                           float& second, int& nearest) {
    if (value < first || (value == first && index < nearest)) {
        second = first;
        first = value;
        nearest = index;
    } else if (value < second) {
        second = value;
    }
}
#undef MX_HOST_DEVICE
