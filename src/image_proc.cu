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

// Full transformation: x_final = scale * (kp @ R + exp) + t
__global__ void transform_kp_kernel(const float* kp, const float* R, const float* exp, const float scale, const float* t, float* out, int num_kp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_kp) {
        float x = kp[i * 3 + 0];
        float y = kp[i * 3 + 1];
        float z = kp[i * 3 + 2];

        // x @ R (Matrix multiplication 1x3 @ 3x3)
        float rx = x * R[0] + y * R[3] + z * R[6];
        float ry = x * R[1] + y * R[4] + z * R[7];
        float rz = x * R[2] + y * R[5] + z * R[8];

        // + exp
        rx += exp[i * 3 + 0];
        ry += exp[i * 3 + 1];
        rz += exp[i * 3 + 2];

        // * scale
        rx *= scale;
        ry *= scale;
        rz *= scale;

        // + t (only x and y)
        out[i * 3 + 0] = rx + t[0];
        out[i * 3 + 1] = ry + t[1];
        out[i * 3 + 2] = rz; // tz is typically ignored or kept as is
    }
}

// exp_final = exp_s + (exp_d_i - exp_d_0)
__global__ void relative_expression_kernel(const float* exp_s, const float* exp_d_i, const float* exp_d_0, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = exp_s[i] + (exp_d_i[i] - exp_d_0[i]);
    }
}

// apply stitching: kp = kp + delta_exp, kp.xy = kp.xy + delta_t
__global__ void apply_stitching_kernel(float* kp, const float* delta, int num_kp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_kp) {
        kp[i * 3 + 0] += delta[i * 3 + 0];
        kp[i * 3 + 1] += delta[i * 3 + 1];
        kp[i * 3 + 2] += delta[i * 3 + 2];
        
        // Apply global translation delta (last 2 elements of stitching output)
        kp[i * 3 + 0] += delta[num_kp * 3 + 0];
        kp[i * 3 + 1] += delta[num_kp * 3 + 1];
    }
}

// concat two keypoint sets: out = [kp1, kp2]
__global__ void concat_kp_kernel(const float* kp1, const float* kp2, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = kp1[i];
        out[i + size] = kp2[i];
    }
}

extern "C" {

void launch_concat_kp(const float* kp1, const float* kp2, float* out, int size, cudaStream_t stream) {
    int threads = 64;
    int blocks = (size + threads - 1) / threads;
    concat_kp_kernel<<<blocks, threads, 0, stream>>>(kp1, kp2, out, size);
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

}
