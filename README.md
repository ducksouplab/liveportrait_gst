# gst-liveportrait

> [!WARNING]  
> **Status: Experimental / Non-Functional**  
> This project is currently under active development. While the C++ port of the LivePortrait logic is largely complete and achieving high frame rates, it is still being refined for full feature parity with the original Python implementation (specifically regarding gaze tracking accuracy).

A high-performance GStreamer video filter plugin for real-time head reenactment using **LivePortrait** and **TensorRT 10**.

## Overview

`gst-liveportrait` is a C++ GStreamer plugin designed to animate a static source image using a driving video stream. It leverages NVIDIA TensorRT for ultra-fast inference, achieving real-time performance on modern GPUs by orchestrating multiple specialized neural engines.

This implementation is based on the architecture and custom CUDA kernels defined in the [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) repository.

## Features

- **High Performance:** Achieves ~45 FPS (~22ms latency) on an NVIDIA RTX A5000.
- **Asynchronous Pipeline:** Utilizes Pinned Memory (`cudaMallocHost`) and `cudaMemcpyAsync` with a dedicated `cudaStream_t` to eliminate PCIe bottlenecks.
- **Full Engine Integration:** Orchestrates 7+ TensorRT engines (Appearance, Motion, Warping, Stitching, Landmark, etc.).
- **Specialized Retargeting:** Includes dedicated logic for Eye and Lip retargeting to ensure realistic facial expressions.
- **Stitching Engine:** Integrated alignment correction to maintain natural head proportions.
- **Motion Smoothing:** Implements the One-Euro filter to eliminate high-frequency facial jitter.
- **GStreamer Native:** Subclasses `GstVideoFilter` for easy integration into standard Linux video pipelines.

## Installation & Build

The plugin must be built within the provided Docker environment to ensure all dependencies (TensorRT 10, CUDA 12, GStreamer 1.28) are met.

### 1. Build the Docker Image
```bash
docker build -t gst-liveportrait-env .
```

### 2. Export TensorRT Engines
Follow the instructions in `INSTRUCTIONS.md` (Phase 1) inside the container to download ONNX models and export them to `.trt` files.

### 3. Compile the Plugin using Docker
Run the following command from the project root:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace gst-liveportrait-env bash -c "mkdir -p build && cd build && cmake .. && make -j$(nproc)"
```

## Usage (Experimental)

Once built, you can test the plugin using `gst-launch-1.0` within the Docker container.

### GStreamer Pipeline Example
```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace gst-liveportrait-env bash -c "\
    GST_PLUGIN_PATH=./build gst-launch-1.0 \
    filesrc location=assets/video_example.mp4 ! \
    decodebin ! videoconvert ! \
    videocrop left=280 right=280 ! \
    videoscale ! video/x-raw,width=512,height=512,format=RGB ! \
    liveportrait config-path=./checkpoints source-image=assets/test_image.jpg ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=outputs/output.mp4"
```

### Plugin Properties
- `config-path`: Path to the directory containing the TensorRT engines (e.g., `./checkpoints`).
- `source-image`: Path to the static source image (e.g., `assets/test_image.jpg`).

## Architecture

- **`CudaMemoryManager`**: Manages managed and pinned memory for zero-copy-like performance between host and device.
- **`TRTWrapper`**: Encapsulates TensorRT engine loading, execution, and I/O binding management with a centralized logger.
- **`LivePortraitPipeline`**: Orchestrates the complex inference flow and implements the relative motion/expression logic.
- **`image_proc.cu`**: Custom CUDA kernels for preprocessing (normalization, transpose), postprocessing, and keypoint transformations (rotation, stitching).

## Performance Profiling

The plugin includes built-in `cudaEvent` profiling. Typical results on an RTX A5000:
- **Total Latency:** ~22ms
- **Preprocessing:** ~0.2ms
- **Motion Extraction:** ~2.0ms
- **Warping (Core):** ~19.0ms
- **Postprocessing:** ~0.01ms

## Acknowledgments

- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) for the original algorithm and model architecture.
- [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin) for the custom TensorRT plugin.

## License

This project is licensed under the LGPL (consistent with GStreamer plugin standards).
