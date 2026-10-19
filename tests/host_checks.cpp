#include "host_utils.hpp"
#include "precision.hpp"
#include <filesystem>
#include <iostream>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename F>
void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    require(failed, "Expected input rejection");
}

int main() {
    const auto path = std::filesystem::temp_directory_path() / "mxkmeans-host-check.csv";
    try {
        // Match the /64 candidate thresholds and final MXFP8 override.
        require(precision_candidate(60, 28, 4, 4, 1.25f) == 4, "FP4 candidate");
        require(precision_candidate(60, 28, 4, 4, 0.75f) == 6, "FP6 candidate");
        require(precision_candidate(60, 28, 4, 4, 0.74f) == 8, "FP8 candidate");
        for (float gap : {0.0f, 0.75f, 1.25f, 1e6f})
            for (float ec : {0.0f, 1.0f, 1e6f})
                require(select_precision(60, 28, 4, ec, 4, gap) == 8, "forced-MXFP8 output");
        float largest = -1.0f, next_largest = -1.0f;
        for (float value : {100.0f, 4.0f, 9.0f, 200.0f, 1.0f})
            insert_largest_distance(value, largest, next_largest);
        require(largest - next_largest == 100, "two-largest distance gap");
        insert_largest_distance(200, largest, next_largest);
        require(largest == next_largest, "Tied largest distances have zero gap");
        float first = INFINITY, second = INFINITY;
        int nearest = 999;
        const float values[] = {100, 4, 9, 200, 1};
        for (int i = 0; i < 5; ++i) insert_distance(values[i], i, first, second, nearest);
        require(nearest == 4 && second - first == 3, "Nearest-centroid helper must preserve its minimum candidates");
        insert_distance(1, 0, first, second, nearest);
        require(nearest == 0 && second == first, "Tied nearest centroids must have zero gap");

        const std::vector<float> points{0, 0, 2, 2, 8, 8, 10, 10};
        require(init_centroids_farthest(points, 4, 2, 4) == std::vector<float>({0, 0, 10, 10, 2, 2, 8, 8}),
                "Farthest-first initialization and stable tie-breaking");
        const auto centers = init_centroids(points, 4, 2, 4, 42);
        require(centers == init_centroids(points, 4, 2, 4, 42), "Seeded initialization is reproducible");
        std::vector<float> unique;
        for (int c = 0; c < 4; ++c) unique.push_back(centers[c * 2]);
        std::sort(unique.begin(), unique.end());
        require(unique == std::vector<float>({0, 2, 8, 10}), "Initialization selects distinct rows");
        require(init_centroids(std::vector<float>(8, 3), 4, 2, 4, 1) == std::vector<float>(8, 3), "Duplicate points");

        auto put = [&](const std::string& contents) { std::ofstream out(path); out << contents; };
        put("x,y,label\r\n1,2,a\r\n3,4,b");
        auto loaded = read_points(path.string(), 2, 2, 128);
        require(loaded.size() == 256 && loaded[0] == 1 && loaded[129] == 4, "CSV values and optional labels");
        require(loaded[2] == 0 && loaded[255] == 0, "Feature padding must be zero");
        rejects([&] { read_points(path.string(), 1, 2, 128); });
        rejects([&] { read_points(path.string(), 3, 2, 128); });
        put("x,y\n1,2\n3,4\n");
        require(read_points(path.string(), 2, 2, 128)[129] == 4, "Feature-only CSV");
        put("x,y\n1,nan\n");
        rejects([&] { read_points(path.string(), 1, 2, 128); });
        put("x,y\n1,2,3\n");
        rejects([&] { read_points(path.string(), 1, 2, 128); });
        put("x,y\n1,2oops\n");
        rejects([&] { read_points(path.string(), 1, 2, 128); });
        rejects([] { parse_int("3oops"); });
        rejects([] { parse_float("inf"); });
        std::filesystem::remove(path);
        std::cout << "Host checks passed\n";
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
