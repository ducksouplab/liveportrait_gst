import tensorrt as trt
import os

def build_engine(onnx_file, engine_file):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
            
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    
    print(f"Building engine: {engine_file}")
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_file, 'wb') as f:
        f.write(serialized_engine)
    print("Done.")
    return True

if __name__ == "__main__":
    build_engine("stitching_eye.onnx", "eyeblink.engine")
