#ifndef __TRT_WRAPPER_H__
#define __TRT_WRAPPER_H__

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>
#include <memory>

class TRTWrapper {
public:
    TRTWrapper(const std::string& engine_path, cudaStream_t stream);
    ~TRTWrapper();

    bool execute(const std::map<std::string, void*>& inputs, 
                 const std::map<std::string, void*>& outputs);

    std::vector<std::string> getInputNames() const;
    std::vector<std::string> getOutputNames() const;
    std::vector<int64_t> getTensorShape(const std::string& name) const;

private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };
    
    static Logger gLogger;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

#endif // __TRT_WRAPPER_H__
