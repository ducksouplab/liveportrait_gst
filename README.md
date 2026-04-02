# gst-liveportrait

> [!IMPORTANT]  
> **Status: Highly Functional / Near Parity**  
> This C++ implementation has achieved near-perfect logic parity with the original Python implementation. It features accurate gaze tracking, realistic head pose alignment, and high-performance real-time inference (~45 FPS).

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

## Plugin Properties

| Property | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `config-path` | `string` | Path to the directory containing the TensorRT engines (e.g., `./checkpoints`). | `NULL` |
| `source-image` | `string` | Path to the static source image (e.g., `assets/test_image.jpg`). | `NULL` |

## Technical Insights (Learnings)

To achieve parity with the original repository, several critical nuances were implemented:

1.  **Landmark Resolution:** The Landmark engine strictly requires a **224x224** input resolution. Using 256x256 results in "wide open" eyes due to coordinate scaling mismatches.
2.  **Multi-Output Landmark handling:** The `landmark.trt` engine provides multiple output tensors. The actual 203 coordinates required for retargeting are located in the **third output tensor (index 2, named "856")**.
3.  **Rotation Matrix Order:** LivePortrait uses a specific Euler angle order: **`Ry * Rx * Rz`**. The final rotation matrix must also be **transposed** before being applied to the keypoints.
4.  **Relative Expression Formula:** To prevent "expression overdrive," the transformation strictly follows:  
    `x_d_i_new = x_s + (x_d_i_new - x_s) * multiplier`, where `x_s` is the source keypoint base.

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

## Usage

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

### Python Wrapper
A portable Python wrapper `liveportrait_process.py` is provided. It runs the entire GStreamer pipeline inside Docker, meaning you only need Python and Docker on your host machine.

```bash
# Basic Usage
python3 liveportrait_process.py \
    --input assets/video_example.mp4 \
    --output outputs/result.mp4 \
    --source assets/test_image.jpg \
    --config checkpoints/
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to driving video. | Required |
| `--output` | Path to save the result. | Required |
| `--source` | Path to source image. | Required |
| `--config` | Path to engines directory. | Required |
| `--crop-left` | Left crop for 1:1 aspect. | 280 |
| `--crop-right`| Right crop for 1:1 aspect. | 280 |
| `--plugin-path`| Path to the build dir. | `./build` |


## Architecture

- **`CudaMemoryManager`**: Manages managed and pinned memory for zero-copy-like performance between host and device.
- **`TRTWrapper`**: Encapsulates TensorRT engine loading and execution with support for multi-input/multi-output mapping.
- **`LivePortraitPipeline`**: Orchestrates the complex inference flow and implements the relative motion/expression logic.
- **`image_proc.cu`**: Custom CUDA kernels for preprocessing, postprocessing, and keypoint transformations.

## Performance Profiling

Typical results on an RTX A5000:
- **Total Latency:** ~22ms (~45 FPS)
- **Landmark/Motion Ext:** ~2.5ms
- **Warping (Core):** ~19.0ms
- **Stitching/Retarget:** ~0.5ms

## Acknowledgments

- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) for the original algorithm and model architecture.
- [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin) for the custom TensorRT plugin.

## License

This project is licensed under the LGPL.
