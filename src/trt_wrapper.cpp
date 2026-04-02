#include "trt_wrapper.h"
#include <NvInferPlugin.h>
#include <fstream>
#include <iostream>
#include <dlfcn.h>
#include <stdexcept>

// Initialize static logger
TRTWrapper::Logger TRTWrapper::gLogger;

void TRTWrapper::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

TRTWrapper::TRTWrapper(const std::string& engine_path, cudaStream_t stream) : stream(stream) {
    // 1. Initialize plugins once
    static bool plugins_initialized = false;
    if (!plugins_initialized) {
        initLibNvInferPlugins(&gLogger, "");
        plugins_initialized = true;
    }

    // 2. Load custom GridSample3D plugin manually
    void* handle = dlopen("/opt/FasterLivePortrait/src/utils/libgrid_sample_3d_plugin.so", RTLD_GLOBAL | RTLD_NOW);
    if (!handle) {
        std::cerr << "Warning: Could not load libgrid_sample_3d_plugin.so: " << dlerror() << std::endl;
    }

    // 3. Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    // 4. Deserialize
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
    if (!engine) {
        throw std::runtime_error("Failed to deserialize CUDA engine: " + engine_path);
    }

    context.reset(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context for: " + engine_path);
    }

    // 5. Inspect bindings
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_names.push_back(name);
        } else {
            output_names.push_back(name);
        }
    }
}

TRTWrapper::~TRTWrapper() {}

bool TRTWrapper::execute(const std::map<std::string, void*>& inputs, 
                         const std::map<std::string, void*>& outputs) {
    
    // Set input shapes and addresses
    for (const auto& [name, ptr] : inputs) {
        if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            context->setInputShape(name.c_str(), engine->getTensorShape(name.c_str()));
            context->setTensorAddress(name.c_str(), ptr);
        } else {
            std::cerr << "Error: " << name << " is not an input tensor" << std::endl;
            return false;
        }
    }

    // Set output addresses
    for (const auto& [name, ptr] : outputs) {
        if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT) {
            context->setTensorAddress(name.c_str(), ptr);
        } else {
            std::cerr << "Error: " << name << " is not an output tensor" << std::endl;
            return false;
        }
    }

    return context->enqueueV3(stream);
}

std::vector<std::string> TRTWrapper::getInputNames() const { return input_names; }
std::vector<std::string> TRTWrapper::getOutputNames() const { return output_names; }

std::vector<int64_t> TRTWrapper::getTensorShape(const std::string& name) const {
    nvinfer1::Dims dims = engine->getTensorShape(name.c_str());
    std::vector<int64_t> shape;
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(dims.d[i]);
    }
    return shape;
}
