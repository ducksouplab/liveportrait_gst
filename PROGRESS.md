# Project Progress Tracker: gst-liveportrait

| Phase | Description | Status |
| :--- | :--- | :--- |
| **1** | Docker Setup & Repo Validation | **Done** |
| **2** | CMake & GStreamer Boilerplate | **Done** |
| **3** | Pinned Memory Manager | **Done** |
| **4** | Async GStreamer Interception | **Done** |
| **5** | TensorRT Wrapper | **Done** |
| **6** | Full LivePortrait Logic | **Done** |
| **7** | Profiling | **Done** |
| **8** | Phase A: Eye Retargeting Export | **Done** |
| **9** | Phase B: C++ Eye Retargeting Integration | **Done** |

## Phase 1 Log
- [x] Create Dockerfile.
- [x] Build Docker image.
- [x] Export TensorRT engines.
- [x] Run Python validation test.

## Phase 8 Log: Phase A: Eye Retargeting Export
- [x] Write `export_eyeblink.py` to extract `opt_eyes` MLP.
- [x] Export to `eyeblink.onnx`.
- [x] Compile to `eyeblink.engine` using `trtexec` (via `compile_trt.py`).

## Phase 9 Log: Phase B: C++ Eye Retargeting Integration
- [x] Add GStreamer properties: `enable-eye-retargeting`, `eyes-open-ratio`, `gaze-x`, `gaze-y`.
- [x] Load `eyeblink.engine` in `LivePortraitPipeline`.
- [x] Implement latent space manipulation in `processFrame`.
- [x] Create and launch `add_latent_delta_kernel`.
- [x] Verify with test pipeline.

