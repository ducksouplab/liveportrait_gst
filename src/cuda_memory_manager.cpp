#include "cuda_memory_manager.h"
#include <iostream>
#include <stdexcept>

CudaMemoryManager::CudaMemoryManager() {}

CudaMemoryManager::~CudaMemoryManager() {
    cleanup();
}

void* CudaMemoryManager::allocatePinned(size_t size, const std::string& name) {
    if (cpu_buffers.count(name)) {
        return cpu_buffers[name];
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pinned CPU memory: " + std::string(cudaGetErrorString(err)));
    }

    cpu_buffers[name] = ptr;
    return ptr;
}

void* CudaMemoryManager::allocateDevice(size_t size, const std::string& name) {
    if (gpu_buffers.count(name)) {
        return gpu_buffers[name];
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU device memory: " + std::string(cudaGetErrorString(err)));
    }

    gpu_buffers[name] = ptr;
    return ptr;
}

void* CudaMemoryManager::getBuffer(const std::string& name) {
    if (cpu_buffers.count(name)) return cpu_buffers[name];
    if (gpu_buffers.count(name)) return gpu_buffers[name];
    return nullptr;
}

void CudaMemoryManager::cleanup() {
    for (auto const& [name, ptr] : cpu_buffers) {
        cudaFreeHost(ptr);
    }
    cpu_buffers.clear();

    for (auto const& [name, ptr] : gpu_buffers) {
        cudaFree(ptr);
    }
    gpu_buffers.clear();
}
