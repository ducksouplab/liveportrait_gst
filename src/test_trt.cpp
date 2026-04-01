#include "trt_wrapper.h"
#include <iostream>
#include <vector>
#include <map>

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    try {
        // Load one of the engines we exported
        std::string engine_path = "/workspace/checkpoints/liveportrait_onnx/stitching_lip.trt";
        TRTWrapper trt(engine_path, stream);

        std::cout << "Successfully loaded engine: " << engine_path << std::endl;
        
        auto inputs = trt.getInputNames();
        auto outputs = trt.getOutputNames();

        std::cout << "Inputs: ";
        for (auto& n : inputs) std::cout << n << " ";
        std::cout << "\nOutputs: ";
        for (auto& n : outputs) std::cout << n << " ";
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaStreamDestroy(stream);
    return 0;
}
