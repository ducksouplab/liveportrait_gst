FROM ducksouplab/ducksoup:ducksoup_plugins_gst1.28.0

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies, CUDA Toolkit, and TensorRT Dev for building plugins
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ffmpeg \
    espeak-ng \
    build-essential \
    cmake \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    cuda-toolkit-12-3 \
    libnvinfer-dev \
    libnvonnxparsers-dev \
    libnvinfer-plugin-dev \
    python3-libnvinfer \
    && rm -rf /var/lib/apt/lists/*

# Use existing Python 3.11
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python symlink
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Install PyTorch and FasterLivePortrait dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy==1.26.4" \
    torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple \
    onnxruntime-gpu \
    nvidia-cudnn-cu12 \
    opencv-python \
    "huggingface_hub[cli]>=0.24.1" \
    tensorrt \
    colorama

# Clone FasterLivePortrait
WORKDIR /opt
RUN git clone https://github.com/warmshao/FasterLivePortrait.git

# Install FasterLivePortrait requirements
WORKDIR /opt/FasterLivePortrait
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip3 install --no-cache-dir --break-system-packages "numpy==1.26.4" torchgeometry

# Build Grid Sample Plugin
RUN git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin /opt/grid-sample3d-trt-plugin
WORKDIR /opt/grid-sample3d-trt-plugin
RUN mkdir build && cd build && \
    cmake .. -DCUDA_ARCHITECTURES=86 && \
    make -j$(nproc)

# Apply patches to FasterLivePortrait
WORKDIR /opt/FasterLivePortrait
RUN cp /opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so /opt/FasterLivePortrait/src/utils/ && \
    # Headless patch for run.py
    sed -i 's/cv2.imshow/# cv2.imshow/g' run.py && \
    sed -i 's/cv2.waitKey/# cv2.waitKey/g' run.py && \
    sed -i 's/cv2.destroyAllWindows/# cv2.destroyAllWindows/g' run.py && \
    # TensorRT 10 output order patch for motion_extractor_model.py
    sed -i "s/kp, pitch, yaw, roll, t, exp, scale = data/pitch, yaw, roll, t, exp, scale, kp = data/g" src/models/motion_extractor_model.py && \
    # TensorRT 10 compatibility for onnx2trt.py
    python3 -c 'import re; p="scripts/onnx2trt.py"; c=open(p).read(); c=re.sub(r"with self\.builder\.build_engine\(self\.network, self\.config\) as engine, open\(engine_path, \"wb\"\) as f:\s+f\.write\(engine\.serialize\(\)\)", "engine_bytes = self.builder.build_serialized_network(self.network, self.config)\n        with open(engine_path, \"wb\") as f:\n            f.write(engine_bytes)", c); c=c.replace("self.builder.max_batch_size = 1", "").replace("self.config.max_workspace_size = 12 * (2 ** 30)", "self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 * (2 ** 30))").replace("self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)", "").replace("./checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so", "/opt/FasterLivePortrait/src/utils/libgrid_sample_3d_plugin.so"); open(p, "w").write(c)'

WORKDIR /workspace
