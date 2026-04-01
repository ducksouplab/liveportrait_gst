#ifndef __IMAGE_PROC_H__
#define __IMAGE_PROC_H__

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_preprocess(const uint8_t* src, float* dst, int w, int h, bool bgr_to_rgb, cudaStream_t stream);
void launch_postprocess(const float* src, uint8_t* dst, int w, int h, bool rgb_to_bgr, cudaStream_t stream);
void launch_transform_kp(const float* kp, const float* R, const float* exp, float scale, const float* t, float* out, int num_kp, cudaStream_t stream);
void launch_relative_expression(const float* exp_s, const float* exp_d_i, const float* exp_d_0, float* out, int size, cudaStream_t stream);
void launch_apply_stitching(float* kp, const float* delta, int num_kp, cudaStream_t stream);
void launch_concat_kp(const float* kp1, const float* kp2, float* out, int size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // __IMAGE_PROC_H__
