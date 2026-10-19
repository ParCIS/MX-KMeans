# MX-KMeans

CUDA implementation of **MX-KMeans: Accelerating K-Means Clustering through
Microscaling Quantization** (ICPP 2026).
[Paper](https://doi.org/10.1145/3832810.3832837).

MX-KMeans accelerates Lloyd-style K-Means on NVIDIA Blackwell GPUs through
microscaling quantization, with FP32 accumulation and centroid updates.

## Overview

The implementation includes MXFP4, MXFP6, and MXFP8 quantization and CUTLASS
GEMM kernels. Groups of 32 feature values share a UE8M0 scaling factor.
Centroids use MXFP8 for matrix products and FP32 for updates.

## Requirements

- Linux, a C++17 compiler, and CMake 3.22 or newer.
- CUDA Toolkit 12.8 or newer.
- An SM120 NVIDIA Blackwell GPU, such as the RTX 5090, with a working driver.
  This implementation targets `sm_120a`; it is not a generic build for every
  Blackwell GPU architecture.
- CUTLASS with SM120 block-scaled GEMM support. The build was checked against
  revision `6b3e607b852f1543dc21323155a2ad71473c8642` (version header: 4.3.5).
- Python 3 for the optional sample-data generator; no Python packages required.

CUTLASS is an external dependency and is not bundled in this repository.

## Build

From this repository's root:

```bash
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass checkout 6b3e607b852f1543dc21323155a2ad71473c8642
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
```

To use an existing CUTLASS checkout:

```bash
cmake -S . -B build -DCUTLASS_DIR=/path/to/cutlass
cmake --build build -j 2
```

## Quick start

```bash
python3 scripts/generate_data.py --n-samples 4096 --dimensions 128 \
  --clusters 4 --seed 42 --output data/example.csv

./build/mxkmeans --n-samples 4096 --dimensions 128 --clusters 4 \
  --input-file data/example.csv --out-filename labels.csv \
  --maxiter 100 --seed 42 --verbose
```

Use `./build/mxkmeans --help` for all options. The output file contains a
`clusters` header followed by one zero-based cluster ID per input row.
Verbose precision counts are written to stderr. Stdout contains a CSV header
and one timing record:

```text
n_samples,dimensions,clusters,mean_seconds,iterations
```

Initialization defaults to seeded K-Means++. Use `--init farthest` for
deterministic farthest-first initialization starting from the first input row.

`mean_seconds` is the average clustering time over `--runs` repetitions,
including algorithm workspace allocation and device synchronization. It excludes
CSV I/O, CPU centroid initialization, the initial point transfer, and resetting
the initial centroids between runs. `iterations` is the count from the last run.
All repetitions start from the same initial centers. `--warmup` controls
unmeasured complete runs; `--fp32-iters` controls FP32 iterations inside each run (default: 0).
These two options serve different purposes.

`--tol` is an absolute threshold on the sum of squared changes in all centroid
coordinates. Iterations stop when this sum is at most the threshold or
`--maxiter` is reached. GPU atomic additions can cause small run-to-run
floating-point differences despite a fixed initialization seed.

## Input format and limits

Use an unquoted, dense CSV with a header and exactly `--dimensions` numeric
feature columns. An optional final label column is ignored. For example:

```csv
x,y,label
1.0,2.0,0
1.1,2.2,0
8.0,9.0,1
8.1,9.2,1
```

Pass the number of **feature** columns as `--dimensions`. Features are internally
zero-padded to a multiple of 128 for the MXFP6 GEMM alignment; padding does not
change Euclidean distances. Row counts and finite numeric values are checked.
Missing values, quoted fields, and sparse LIBSVM input are not supported.

The current GEMM output layout requires `K >= 4` and `K` divisible by 4, with
`K <= N`. `N * padded_D` and `N * K` must each fit in a signed 32-bit index.
The full dataset and a dense `N × K` FP32 product buffer must fit in GPU memory.
The three precision groups reuse their packed-value, scale, and product buffers.

## Validation

```bash
ctest --test-dir build --output-on-failure
```

CPU checks cover precision selection, distance statistics, seeded initialization,
and CSV validation. CUDA checks cover all three packed MX
formats, GEMM versus a CPU reference using reconstructed quantized values,
scale tiles crossing row/feature boundaries, partial warps, assignment ties,
empty clusters, and the complete clustering loop. CUDA checks report **Skipped**
when a compatible GPU/driver is unavailable.

Host-only checks can also be built without CUDA or CUTLASS:

```bash
cmake -S . -B build-host -DMXKMEANS_BUILD_CUDA=OFF
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

## License

MX-KMeans is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
External dependencies, including CUTLASS and CUDA, retain their own licenses.

## Citation

If you use MX-KMeans in your research, please cite:

```bibtex
@inproceedings{fu2026mx,
  title={MX-KMeans: Accelerating K-Means Clustering through Microscaling Quantization},
  author={Fu, Rongtian and Lv, Dongbo and Wang, Xueying and Li, Shigang},
  booktitle={Proceedings of the 55th International Conference on Parallel Processing},
  pages={747--757},
  year={2026}
}
```
