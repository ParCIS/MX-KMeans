#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(call) check_cuda((call), #call)

// Own device allocations so early exits and exceptions also release memory.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    size_t size() const { return count_; }
    void clear() { CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(T))); }
private:
    T* data_ = nullptr;
    size_t count_;
};
