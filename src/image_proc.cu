#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>
#include "image_proc.h"

__global__ void preprocess_kernel(const uint8_t* src, float* dst, int w, int h, bool bgr_to_rgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        int out_idx = y * w + x;
        int plane_size = w * h;
        float r, g, b;
        if (bgr_to_rgb) {
            b = (float)src[idx + 0] / 255.0f;
            g = (float)src[idx + 1] / 255.0f;
            r = (float)src[idx + 2] / 255.0f;
        } else {
            r = (float)src[idx + 0] / 255.0f;
            g = (float)src[idx + 1] / 255.0f;
            b = (float)src[idx + 2] / 255.0f;
        }
        dst[out_idx] = r;
        dst[out_idx + plane_size] = g;
        dst[out_idx + 2 * plane_size] = b;
    }
}

__global__ void postprocess_kernel(const float* src, uint8_t* dst, int w, int h, bool rgb_to_bgr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = y * w + x;
        int out_idx = (y * w + x) * 3;
        int plane_size = w * h;
        float r = src[idx];
        float g = src[idx + plane_size];
        float b = src[idx + 2 * plane_size];
        r = fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
        g = fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
        b = fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);
        if (rgb_to_bgr) {
            dst[out_idx + 0] = (uint8_t)b;
            dst[out_idx + 1] = (uint8_t)g;
            dst[out_idx + 2] = (uint8_t)r;
        } else {
            dst[out_idx + 0] = (uint8_t)r;
            dst[out_idx + 1] = (uint8_t)g;
            dst[out_idx + 2] = (uint8_t)b;
        }
    }
}

__global__ void transform_kp_kernel(const float* kp, const float* R, const float* exp, const float scale, const float* t, float* out, int num_kp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_kp) {
        float x = kp[i * 3 + 0];
        float y = kp[i * 3 + 1];
        float z = kp[i * 3 + 2];
        float rx = x * R[0] + y * R[3] + z * R[6];
        float ry = x * R[1] + y * R[4] + z * R[7];
        float rz = x * R[2] + y * R[5] + z * R[8];
        rx = scale * (rx + exp[i * 3 + 0]) + t[0];
        ry = scale * (ry + exp[i * 3 + 1]) + t[1];
        rz = scale * (rz + exp[i * 3 + 2]) + t[2];
        out[i * 3 + 0] = rx;
        out[i * 3 + 1] = ry;
        out[i * 3 + 2] = rz;
    }
}

__global__ void relative_expression_kernel(const float* exp_s, const float* exp_d_i, const float* exp_d_0, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = exp_s[i] + (exp_d_i[i] - exp_d_0[i]);
    }
}

__global__ void apply_stitching_kernel(float* kp, const float* delta, int num_kp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_kp) {
        kp[i * 3 + 0] += delta[i * 3 + 0];
        kp[i * 3 + 1] += delta[i * 3 + 1];
        kp[i * 3 + 2] += delta[i * 3 + 2];
        kp[i * 3 + 0] += delta[num_kp * 3 + 0];
        kp[i * 3 + 1] += delta[num_kp * 3 + 1];
    }
}

__global__ void add_deltas_kernel(float* kp, const float* d1, const float* d2, const float* d3, int num_kp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_kp) {
        kp[i * 3 + 0] += d1[i * 3 + 0] + d2[i * 3 + 0] + d3[i * 3 + 0];
        kp[i * 3 + 1] += d1[i * 3 + 1] + d2[i * 3 + 1] + d3[i * 3 + 1];
        kp[i * 3 + 2] += d1[i * 3 + 2] + d2[i * 3 + 2] + d3[i * 3 + 2];
        kp[i * 3 + 0] += d1[num_kp * 3 + 0] + d2[num_kp * 3 + 0] + d3[num_kp * 3 + 0];
        kp[i * 3 + 1] += d1[num_kp * 3 + 1] + d2[num_kp * 3 + 1] + d3[num_kp * 3 + 1];
    }
}

__global__ void concat_feat_kernel(const float* kp1, int size1, const float* kp2, int size2, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size1) {
        out[i] = kp1[i];
    } else if (i < size1 + size2) {
        out[i] = kp2[i - size1];
    }
}

extern "C" {

void launch_preprocess(const uint8_t* src, float* dst, int w, int h, bool bgr_to_rgb, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    preprocess_kernel<<<grid, block, 0, stream>>>(src, dst, w, h, bgr_to_rgb);
}

void launch_postprocess(const float* src, uint8_t* dst, int w, int h, bool rgb_to_bgr, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    postprocess_kernel<<<grid, block, 0, stream>>>(src, dst, w, h, rgb_to_bgr);
}

void launch_transform_kp(const float* kp, const float* R, const float* exp, float scale, const float* t, float* out, int num_kp, cudaStream_t stream) {
    int threads = 64;
    int blocks = (num_kp + threads - 1) / threads;
    transform_kp_kernel<<<blocks, threads, 0, stream>>>(kp, R, exp, scale, t, out, num_kp);
}

void launch_relative_expression(const float* exp_s, const float* exp_d_i, const float* exp_d_0, float* out, int size, cudaStream_t stream) {
    int threads = 64;
    int blocks = (size + threads - 1) / threads;
    relative_expression_kernel<<<blocks, threads, 0, stream>>>(exp_s, exp_d_i, exp_d_0, out, size);
}

void launch_apply_stitching(float* kp, const float* delta, int num_kp, cudaStream_t stream) {
    int threads = 64;
    int blocks = (num_kp + threads - 1) / threads;
    apply_stitching_kernel<<<blocks, threads, 0, stream>>>(kp, delta, num_kp);
}

void launch_add_deltas(float* kp, const float* d1, const float* d2, const float* d3, int num_kp, cudaStream_t stream) {
    int threads = 64;
    int blocks = (num_kp + threads - 1) / threads;
    add_deltas_kernel<<<blocks, threads, 0, stream>>>(kp, d1, d2, d3, num_kp);
}

void launch_concat_feat(const float* kp1, int size1, const float* kp2, int size2, float* out, cudaStream_t stream) {
    int total = size1 + size2;
    int threads = 64;
    int blocks = (total + threads - 1) / threads;
    concat_feat_kernel<<<blocks, threads, 0, stream>>>(kp1, size1, kp2, size2, out);
}

}
