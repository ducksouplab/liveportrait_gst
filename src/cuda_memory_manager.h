#ifndef __CUDA_MEMORY_MANAGER_H__
#define __CUDA_MEMORY_MANAGER_H__

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <string>

class CudaMemoryManager {
public:
    CudaMemoryManager();
    ~CudaMemoryManager();

    // Allocate pinned CPU memory
    void* allocatePinned(size_t size, const std::string& name);
    
    // Allocate GPU device memory
    void* allocateDevice(size_t size, const std::string& name);

    // Get buffer by name
    void* getBuffer(const std::string& name);

    // Cleanup all managed buffers
    void cleanup();

private:
    std::map<std::string, void*> cpu_buffers;
    std::map<std::string, void*> gpu_buffers;
};

#endif // __CUDA_MEMORY_MANAGER_H__
