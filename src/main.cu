#include "kmeans.cuh"
#include "host_utils.hpp"
#include <chrono>
#include <climits>
#include <filesystem>
#include <getopt.h>
#include <iostream>

namespace {
struct Arguments {
    int n = 0, d = 0, k = 0, runs = 1, warmup = 0, seed = 42;
    std::string input, output = "labels.csv";
    std::string initialization = "kmeans++";
    KMeansOptions algorithm;
};

void usage() {
    std::cout << R"(Usage: mxkmeans -n N -d D -k K -i points.csv [options]

Required:
  -n, --n-samples N       Number of input rows
  -d, --dimensions D      Number of feature columns (padded internally to 128)
  -k, --clusters K        Cluster count, at least 4 and a multiple of 4
  -i, --input-file PATH   Dense CSV with a header; optional final label column

Options:
  -o, --out-filename PATH Label CSV (default: labels.csv)
  -m, --maxiter N         Maximum Lloyd iterations (default: 300)
  -t, --tol FLOAT         Sum of squared centroid shifts (default: 1e-4)
  -s, --seed N            K-Means++ seed (default: 42)
      --init METHOD      kmeans++ or farthest (default: kmeans++)
  -r, --runs N            Measured repetitions from the same centers (default: 1)
      --warmup N         Unmeasured complete runs (default: 0)
      --fp32-iters N     Initial FP32 Lloyd iterations in each run (default: 0)
      --verbose          Print per-iteration precision counts to stderr
  -h, --help             Show this help

Stdout: CSV with n_samples,dimensions,clusters,mean_seconds,iterations.
Timing includes device workspaces and clustering, excluding CSV I/O,
K-Means++ initialization and the initial host-to-device data transfer.
)";
}

Arguments parse(int argc, char** argv) {
    Arguments args;
    const option options[] = {
        {"n-samples", required_argument, nullptr, 'n'}, {"dimensions", required_argument, nullptr, 'd'},
        {"clusters", required_argument, nullptr, 'k'}, {"input-file", required_argument, nullptr, 'i'},
        {"out-filename", required_argument, nullptr, 'o'}, {"maxiter", required_argument, nullptr, 'm'},
        {"tol", required_argument, nullptr, 't'}, {"seed", required_argument, nullptr, 's'},
        {"runs", required_argument, nullptr, 'r'}, {"warmup", required_argument, nullptr, 1000},
        {"fp32-iters", required_argument, nullptr, 1001}, {"verbose", no_argument, nullptr, 1002},
        {"init", required_argument, nullptr, 1003},
        {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "n:d:k:i:o:m:t:s:r:h", options, nullptr)) != -1) {
        switch (opt) {
            case 'n': args.n = parse_int(optarg); break;
            case 'd': args.d = parse_int(optarg); break;
            case 'k': args.k = parse_int(optarg); break;
            case 'i': args.input = optarg; break;
            case 'o': args.output = optarg; break;
            case 'm': args.algorithm.max_iterations = parse_int(optarg); break;
            case 't': args.algorithm.tolerance = parse_float(optarg); break;
            case 's': args.seed = parse_int(optarg); break;
            case 'r': args.runs = parse_int(optarg); break;
            case 1000: args.warmup = parse_int(optarg); break;
            case 1001: args.algorithm.fp32_iterations = parse_int(optarg); break;
            case 1002: args.algorithm.verbose = true; break;
            case 1003: args.initialization = optarg; break;
            case 'h': usage(); std::exit(0);
            default: throw std::invalid_argument("Invalid option; use --help");
        }
    }
    if (optind != argc) throw std::invalid_argument("Unexpected positional argument");
    if (args.initialization != "kmeans++" && args.initialization != "farthest")
        throw std::invalid_argument("--init must be kmeans++ or farthest");
    if (args.n <= 0 || args.d <= 0 || args.k < 4 || args.k > args.n || args.k % 4)
        throw std::invalid_argument("Require N > 0, D > 0, 4 <= K <= N, and K divisible by 4");
    if (args.d > INT_MAX - 127) throw std::invalid_argument("Too many dimensions");
    const int padded_d = (args.d + 127) / 128 * 128;
    if (int64_t(args.n) * padded_d > INT_MAX || int64_t(args.n) * args.k > INT_MAX)
        throw std::invalid_argument("N*padded_D and N*K must fit in a signed 32-bit index");
    if (args.runs <= 0 || args.warmup < 0 || args.seed < 0 || int64_t(args.runs) + args.warmup > INT_MAX)
        throw std::invalid_argument("Require runs > 0, warmup >= 0, seed >= 0, and runs + warmup <= INT_MAX");
    if (args.algorithm.max_iterations <= 0 || args.algorithm.fp32_iterations < 0 || args.algorithm.tolerance < 0)
        throw std::invalid_argument("Require maxiter > 0, fp32-iters >= 0, and tol >= 0");
    if (args.input.empty() || args.output.empty()) throw std::invalid_argument("Input and output paths are required");
    if (args.input == args.output ||
        (std::filesystem::exists(args.input) && std::filesystem::exists(args.output) &&
         std::filesystem::equivalent(args.input, args.output)))
        throw std::invalid_argument("Input and output paths must differ");
    return args;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse(argc, argv);
        const int d = (args.d + 127) / 128 * 128;
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        if (device.major != 12 || device.minor != 0)
            throw std::runtime_error("This build requires an SM120 Blackwell GPU, such as RTX 5090");
        const auto points = read_points(args.input, args.n, args.d, d);
        const auto initial = args.initialization == "farthest"
            ? init_centroids_farthest(points, args.n, d, args.k)
            : init_centroids(points, args.n, d, args.k, args.seed);
        DeviceBuffer<float> device_points(points.size()), centroids(initial.size());
        DeviceBuffer<int> labels(args.n);
        CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice));
        int iterations = 0;
        double total_seconds = 0.0;
        for (int run = 0; run < args.warmup + args.runs; ++run) {
            CUDA_CHECK(cudaMemcpy(centroids.get(), initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto start = std::chrono::steady_clock::now();
            iterations = kmeans_mixed(args.n, d, args.k, device_points.get(), centroids.get(), labels.get(), args.algorithm);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto end = std::chrono::steady_clock::now();
            if (run >= args.warmup) total_seconds += std::chrono::duration<double>(end - start).count();
        }
        std::vector<int> result(args.n);
        CUDA_CHECK(cudaMemcpy(result.data(), labels.get(), result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        write_labels(args.output, result);
        std::cout << "n_samples,dimensions,clusters,mean_seconds,iterations\n"
                  << args.n << ',' << args.d << ',' << args.k << ',' << total_seconds / args.runs << ',' << iterations << '\n';
    } catch (const std::exception& error) {
        std::cerr << "mxkmeans: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
