# gst-liveportrait: Project Instructions & C++ Architecture

## Core Mandate
This project implements a high-performance GStreamer video filter plugin (`gst-liveportrait`) using TensorRT. The implementation is strictly based on the C++ architecture, custom CUDA kernels (grid sample, warp affine), and tensor shapes defined in the `warmshao/FasterLivePortrait` repository.

## Critical Operating Rules
1. **Source of Truth:** All technical decisions must align with this file.
2. **State Tracking:** Current progress is tracked in `PROGRESS.md`. No jumping ahead to future phases.
3. **Memory Management:** To eliminate PCIe bottlenecks, strictly use **Pinned Memory** (`cudaMallocHost`) and **Asynchronous transfers** (`cudaMemcpyAsync`) with a dedicated `cudaStream_t`. Synchronous CPU-GPU transfers are prohibited in the processing loop.
4. **Environment:** Operating within Docker `ducksouplab/ducksoup:ducksoup_plugins_gst1.28.0`. No X11/Wayland display; all output must be routed to files (e.g., `filesink`).

## Project Phases

### Phase 1: Docker Setup & Repo Validation
- **Goal:** Extend the base image to include a functional `FasterLivePortrait` environment.
- **Tasks:**
    - Create a Dockerfile adding Python, PyTorch (CUDA), and TensorRT.
    - Clone `warmshao/FasterLivePortrait`.
    - Export models to TensorRT `.engine` files.
- **Validation:** Run baseline Python inference and output to a video file.

### Phase 2: CMake & GStreamer Boilerplate
- **Goal:** Establish the C++ build system and plugin skeleton.
- **Tasks:**
    - Create `CMakeLists.txt` finding GStreamer, CUDA, and TensorRT.
    - Implement `gstliveportrait.cpp/h` subclassing `GstVideoFilter`.
    - Instantiate a `cudaStream_t` in the element's state.
- **Validation:** Successfully run `gst-inspect-1.0 ./libgstliveportrait.so`.

### Phase 3: Pinned Memory Manager
- **Goal:** Implement the asynchronous memory backbone.
- **Tasks:**
    - Create `CudaMemoryManager` class.
    - Manage `cudaMallocHost` (CPU pinned) and `cudaMalloc` (GPU device) buffers.
- **Validation:** Execute a standalone C++ test verifying async memory copies.

### Phase 4: Async GStreamer Interception
- **Goal:** Integrate the GStreamer buffer flow with the CUDA stream.
- **Tasks:**
    - In `transform_frame`, map incoming buffer -> `cpu_pinned_in`.
    - Launch `cudaMemcpyAsync` to `gpu_device_in`.
    - Launch a dummy CUDA kernel.
    - Launch `cudaMemcpyAsync` to `cpu_pinned_out`.
    - `cudaStreamSynchronize` and write to the outgoing buffer.
- **Validation:** `gst-launch-1.0` pipeline with `liveportrait` producing a valid `.mp4`.

### Phase 5: TensorRT Wrapper
- **Goal:** Encapsulate TensorRT execution.
- **Tasks:**
    - Implement `TensorRTWrapper` class.
    - Load a dummy `.engine` and execute within the existing `cudaStream_t`.
- **Validation:** Run the Phase 4 pipeline with the dummy engine active.

### Phase 6: Full LivePortrait Logic
- **Goal:** Implement the complete inference pipeline.
- **Tasks:**
    - Integrate `FasterLivePortrait` custom `.cu` kernels.
    - Load Appearance, Motion, Warping, and Stitching engines.
    - Implement the logic: Appearance (init only), then Motion/Warping/Stitching per frame.
- **Validation:** Full pipeline converting a driving video and source image to a final animated `.mp4`.

### Phase 7: Profiling
- **Goal:** Performance optimization.
- **Tasks:**
    - Integrate `cudaEventRecord` for precise latency logging.
    - Identify bottlenecks to prepare for future NVMM zero-copy integration.
