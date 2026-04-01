#include "cuda_memory_manager.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

int main() {
    CudaMemoryManager manager;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t size = 1024 * 1024; // 1KB
    
    // Allocate
    uint8_t* h_in = (uint8_t*)manager.allocatePinned(size, "cpu_in");
    uint8_t* d_mem = (uint8_t*)manager.allocateDevice(size, "gpu_mem");
    uint8_t* h_out = (uint8_t*)manager.allocatePinned(size, "cpu_out");

    // Initialize source
    for (size_t i = 0; i < size; ++i) {
        h_in[i] = (uint8_t)(i % 256);
    }

    // Async Copy H2D
    cudaMemcpyAsync(d_mem, h_in, size, cudaMemcpyHostToDevice, stream);
    
    // Dummy compute (memset on GPU via stream if we had a kernel, for now just copy back)
    // Async Copy D2H
    cudaMemcpyAsync(h_out, d_mem, size, cudaMemcpyDeviceToHost, stream);

    // Sync
    cudaStreamSynchronize(stream);

    // Verify
    for (size_t i = 0; i < size; ++i) {
        if (h_in[i] != h_out[i]) {
            std::cerr << "Mismatch at index " << i << ": " << (int)h_in[i] << " != " << (int)h_out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "SUCCESS: Async pinned memory copy test passed!" << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}
