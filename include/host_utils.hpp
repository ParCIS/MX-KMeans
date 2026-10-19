#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> cells;
    std::istringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) cells.push_back(cell);
    if (!line.empty() && line.back() == ',') cells.emplace_back();
    return cells;
}

inline float parse_float(const std::string& text) {
    size_t used = 0;
    const float value = std::stof(text, &used);
    if (!std::isfinite(value) || text.find_first_not_of(" \t\r", used) != std::string::npos)
        throw std::invalid_argument("Invalid finite number: " + text);
    return value;
}

inline int parse_int(const std::string& text) {
    size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size()) throw std::invalid_argument("Invalid integer: " + text);
    return value;
}

// Dense CSV with a header, d features, and optionally one ignored label column.
// Zero-padding to the GEMM alignment preserves Euclidean distances.
inline std::vector<float> read_points(const std::string& filename, int n, int d, int padded_d) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open input: " + filename);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Input is empty");
    const size_t columns = split_csv(line).size();
    if (columns != size_t(d) && columns != size_t(d + 1))
        throw std::runtime_error("CSV header must contain d features and optionally one label column");
    std::vector<float> points(size_t(n) * padded_d, 0.0f);
    int row = 0;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        if (row >= n) throw std::runtime_error("CSV contains more rows than --n-samples");
        const auto cells = split_csv(line);
        if (cells.size() != columns)
            throw std::runtime_error("Wrong column count at data row " + std::to_string(row + 1));
        for (int j = 0; j < d; ++j) {
            try { points[size_t(row) * padded_d + j] = parse_float(cells[j]); }
            catch (const std::exception&) {
                throw std::runtime_error("Invalid feature at data row " + std::to_string(row + 1) + ", column " + std::to_string(j + 1));
            }
        }
        ++row;
    }
    if (input.bad()) throw std::runtime_error("Error while reading input: " + filename);
    if (row != n) throw std::runtime_error("CSV row count does not match --n-samples");
    return points;
}

inline void write_labels(const std::string& filename, const std::vector<int>& labels) {
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot open output: " + filename);
    output << "clusters\n";
    for (int label : labels) output << label << '\n';
    output.close();
    if (!output) throw std::runtime_error("Failed to write output: " + filename);
}

// Standard D^2-sampling K-Means++: choose the first center uniformly, then
// sample subsequent centers in proportion to their nearest squared distance.
inline std::vector<float> init_centroids(const std::vector<float>& points, int n, int d,
                                        int k, uint32_t seed) {
    std::mt19937 generator(seed);
    std::vector<float> centers(size_t(k) * d);
    std::vector<double> distances(n, std::numeric_limits<double>::infinity());
    std::vector<bool> selected(n, false);
    int next = std::uniform_int_distribution<int>(0, n - 1)(generator);
    for (int c = 0; c < k; ++c) {
        std::copy_n(points.data() + size_t(next) * d, d, centers.data() + size_t(c) * d);
        selected[next] = true;
        double total = 0.0;
        for (int p = 0; p < n; ++p) {
            double distance = 0.0;
            for (int j = 0; j < d; ++j) {
                const double diff = double(points[size_t(p) * d + j]) - centers[size_t(c) * d + j];
                distance += diff * diff;
            }
            distances[p] = selected[p] ? 0.0 : std::min(distances[p], distance);
            total += distances[p];
        }
        if (c + 1 == k) break;
        next = -1;
        if (total > 0.0) {
            const double target = std::uniform_real_distribution<double>(0.0, total)(generator);
            double cumulative = 0.0;
            for (int p = 0; p < n; ++p) {
                cumulative += distances[p];
                if (!selected[p] && distances[p] > 0.0 && cumulative > target) {
                    next = p;
                    break;
                }
            }
        }
        if (next < 0) {
            // All remaining values may be duplicates, or the sum rounded at its endpoint.
            for (int p = 0; p < n; ++p) if (!selected[p]) { next = p; break; }
        }
    }
    return centers;
}

// Deterministic farthest-first initialization for repeatable benchmark inputs.
inline std::vector<float> init_centroids_farthest(const std::vector<float>& points,
                                                 int n, int d, int k) {
    std::vector<float> centers(size_t(k) * d);
    std::vector<float> distances(n, std::numeric_limits<float>::max());
    int next = 0;
    for (int c = 0; c < k; ++c) {
        std::copy_n(points.data() + size_t(next) * d, d, centers.data() + size_t(c) * d);
        for (int p = 0; p < n; ++p) {
            float distance = 0.0f;
            for (int j = 0; j < d; ++j) {
                const float diff = points[size_t(p) * d + j] - points[size_t(next) * d + j];
                distance += diff * diff;
            }
            distances[p] = std::min(distances[p], distance);
        }
        distances[next] = -1.0f;
        next = std::max_element(distances.begin(), distances.end()) - distances.begin();
    }
    return centers;
}
