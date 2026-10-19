#include "kmeans.cuh"
#include <cublas_v2.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace {
void check_cutlass(cutlass::Status status) {
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(cutlassGetStatusString(status));
}
void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error: " + std::to_string(int(status)));
}

// The CUTLASS scale layout is indexed by the original feature coordinate.
// A scale covers 32 features; the layout handles the swizzle and tile strides.
template <typename Layout>
__global__ void reorder_scale_kernel(const Scale* source, Scale* destination,
                                     Layout layout, int rows, int blocks_per_row) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = index % rows;
    const int block = index / rows;
    if (row < rows && block < blocks_per_row)
        destination[layout(row, block * 32, 0)] = source[size_t(row) * blocks_per_row + block];
}

void gemm_mxfp4(int m, int n, int k, const MXFP4* A,
                 const Scale* A_scale, const MXFP8* B, const Scale* B_scale, float* D) {
    using ElementA = cutlass::mx_float4_t<MXFP4>;
    using ThreadBlockShape = cute::Shape<cute::_128,cute::_128,cute::_128>;
    using ClusterShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
    using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = float;
    using ElementD = float;
    using ElementAccumulator = float;
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 128;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    int k_tiles = (k + 31) / 32;
    size_t real_scale_elements = cute::cosize(layout_SFA);
    DeviceBuffer<Scale> a_scale_storage(real_scale_elements);
    a_scale_storage.clear();
    Scale* A_scale_real = a_scale_storage.get();
    dim3 block(256);
    dim3 grid((m * k_tiles + 255) / 256);
    reorder_scale_kernel<<<grid, block>>>(
        A_scale,
        A_scale_real,
        layout_SFA,
        m, k_tiles
    );
    CUDA_CHECK(cudaGetLastError());
    size_t real_scale_elements_B = cute::cosize(layout_SFB);
    DeviceBuffer<Scale> b_scale_storage(real_scale_elements_B);
    b_scale_storage.clear();
    Scale* B_scale_real = b_scale_storage.get();
    dim3 block_B(256);
    dim3 grid_B((n * k_tiles + 255) / 256);
    reorder_scale_kernel<<<grid_B, block_B>>>(
        B_scale,
        B_scale_real,
        layout_SFB,
        n, k_tiles
    );
    CUDA_CHECK(cudaGetLastError());
    Gemm gemm;
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {
            A, stride_A,
            B, stride_B,
            A_scale_real, layout_SFA,
            B_scale_real, layout_SFB
        },
        {
            {1.0f, 0.0f},
            nullptr, stride_C,
            D, stride_D
        }
    };
    size_t workspace_bytes = Gemm::get_workspace_size(arguments);
    DeviceBuffer<unsigned char> workspace(workspace_bytes);
    check_cutlass(gemm.can_implement(arguments));
    check_cutlass(gemm.initialize(arguments, workspace.get()));
    check_cutlass(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gemm_mxfp6(int m, int n, int k, const MXFP6* A,
                 const Scale* A_scale, const MXFP8* B, const Scale* B_scale, float* D) {
    using ElementA = cutlass::mx_float6_t<MXFP6>;
    using ThreadBlockShape = cute::Shape<cute::_128,cute::_128,cute::_128>;
    using ClusterShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
    using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = float;
    using ElementD = float;
    using ElementAccumulator = float;
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 128;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    int k_tiles = (k + 31) / 32;
    size_t real_scale_elements = cute::cosize(layout_SFA);
    DeviceBuffer<Scale> a_scale_storage(real_scale_elements);
    a_scale_storage.clear();
    Scale* A_scale_real = a_scale_storage.get();
    dim3 block(256);
    dim3 grid((m * k_tiles + 255) / 256);
    reorder_scale_kernel<<<grid, block>>>(
        A_scale,
        A_scale_real,
        layout_SFA,
        m, k_tiles
    );
    CUDA_CHECK(cudaGetLastError());
    size_t real_scale_elements_B = cute::cosize(layout_SFB);
    DeviceBuffer<Scale> b_scale_storage(real_scale_elements_B);
    b_scale_storage.clear();
    Scale* B_scale_real = b_scale_storage.get();
    dim3 block_B(256);
    dim3 grid_B((n * k_tiles + 255) / 256);
    reorder_scale_kernel<<<grid_B, block_B>>>(
        B_scale,
        B_scale_real,
        layout_SFB,
        n, k_tiles
    );
    CUDA_CHECK(cudaGetLastError());
    Gemm gemm;
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {
            A, stride_A,
            B, stride_B,
            A_scale_real, layout_SFA,
            B_scale_real, layout_SFB
        },
        {
            {1.0f, 0.0f},
            nullptr, stride_C,
            D, stride_D
        }
    };
    size_t workspace_bytes = Gemm::get_workspace_size(arguments);
    DeviceBuffer<unsigned char> workspace(workspace_bytes);
    check_cutlass(gemm.can_implement(arguments));
    check_cutlass(gemm.initialize(arguments, workspace.get()));
    check_cutlass(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gemm_mxfp8(int m, int n, int k, const MXFP8* A,
                 const Scale* A_scale, const MXFP8* B, const Scale* B_scale, float* D) {
    using ElementA = cutlass::mx_float8_t<MXFP8>;
    using ThreadBlockShape = cute::Shape<cute::_128,cute::_128,cute::_128>;
    using ClusterShape = cute::Shape<cute::_1,cute::_1,cute::_1>;
    using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = float;
    using ElementD = float;
    using ElementAccumulator = float;
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 128;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    int k_tiles = (k + 31) / 32;
    size_t real_scale_elements = cute::cosize(layout_SFA);
    DeviceBuffer<Scale> a_scale_storage(real_scale_elements);
    a_scale_storage.clear();
    Scale* A_scale_real = a_scale_storage.get();
    dim3 block(256);
    dim3 grid((m * k_tiles + 255) / 256);
    reorder_scale_kernel<<<grid, block>>>(
        A_scale,
        A_scale_real,
        layout_SFA,
        m, k_tiles
    );
    CUDA_CHECK(cudaGetLastError());
    size_t real_scale_elements_B = cute::cosize(layout_SFB);
    DeviceBuffer<Scale> b_scale_storage(real_scale_elements_B);
    b_scale_storage.clear();
    Scale* B_scale_real = b_scale_storage.get();
    dim3 block_B(256);
    dim3 grid_B((n * k_tiles + 255) / 256);
    reorder_scale_kernel<<<grid_B, block_B>>>(
        B_scale,
        B_scale_real,
        layout_SFB,
        n, k_tiles
    );
    CUDA_CHECK(cudaGetLastError());
    Gemm gemm;
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {
            A, stride_A,
            B, stride_B,
            A_scale_real, layout_SFA,
            B_scale_real, layout_SFB
        },
        {
            {1.0f, 0.0f},
            nullptr, stride_C,
            D, stride_D
        }
    };
    size_t workspace_bytes = Gemm::get_workspace_size(arguments);
    DeviceBuffer<unsigned char> workspace(workspace_bytes);
    check_cutlass(gemm.can_implement(arguments));
    check_cutlass(gemm.initialize(arguments, workspace.get()));
    check_cutlass(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());
}
}  // namespace

void gemm_mx(int bits, int m, int n, int k, const void* a, const Scale* a_scales,
             const MXFP8* b, const Scale* b_scales, float* out) {
    if (!m) return;
    switch (bits) {
        case 4:
            gemm_mxfp4(m, n, k, static_cast<const MXFP4*>(a), a_scales, b, b_scales, out);
            break;
        case 6:
            gemm_mxfp6(m, n, k, static_cast<const MXFP6*>(a), a_scales, b, b_scales, out);
            break;
        case 8:
            gemm_mxfp8(m, n, k, static_cast<const MXFP8*>(a), a_scales, b, b_scales, out);
            break;
        default: throw std::invalid_argument("Precision must be 4, 6, or 8");
    }
}

void gemm_fp32(int m, int n, int k, const float* a, const float* b, float* out) {
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle));
    try {
        // Disable reduced-precision math for the FP32 initialization stage.
        check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major A B^T is column-major B A^T with dimensions n x m.
        check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 n, m, k, &alpha, b, k, a, k, &beta, out, n));
    } catch (...) {
        cublasDestroy(handle);
        throw;
    }
    check_cublas(cublasDestroy(handle));
}
