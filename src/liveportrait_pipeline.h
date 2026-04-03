#ifndef __LIVEPORTRAIT_PIPELINE_H__
#define __LIVEPORTRAIT_PIPELINE_H__

#include "trt_wrapper.h"
#include "cuda_memory_manager.h"
#include <string>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

// --- One Euro Filter Implementation ---
class OneEuroFilter {
public:
    OneEuroFilter(double freq = 25.0, float mincutoff = 1.0, float beta = 0.0, float dcutoff = 1.0)
        : freq(freq), mincutoff(mincutoff), beta(beta), dcutoff(dcutoff), 
          x_prev(0), dx_prev(0), first_time(true) {}

    float process(float x) {
        if (first_time) {
            x_prev = x;
            first_time = false;
            return x;
        }
        float dx = (x - x_prev) * freq;
        float edx = low_pass_filter(dx, dx_prev, alpha(freq, dcutoff));
        dx_prev = edx;
        float cutoff = mincutoff + beta * std::abs(edx);
        float ex = low_pass_filter(x, x_prev, alpha(freq, cutoff));
        x_prev = ex;
        return ex;
    }

private:
    double freq;
    float mincutoff, beta, dcutoff;
    float x_prev, dx_prev;
    bool first_time;

    float alpha(double freq, float cutoff) {
        float tau = 1.0 / (2 * 3.14159265358979323846 * cutoff);
        float te = 1.0 / freq;
        return 1.0 / (1.0 + tau / te);
    }

    float low_pass_filter(float x, float y_prev, float alpha) {
        return alpha * x + (1 - alpha) * y_prev;
    }
};

class LivePortraitPipeline {
public:
    LivePortraitPipeline(const std::string& checkpoints_dir, cudaStream_t stream);
    ~LivePortraitPipeline();

    bool initSource(const std::string& image_path);
    bool processFrame(const void* in_data, void* out_data, int width, int height,
                      bool enable_eye_retargeting = false, float eyes_open_ratio = 0.0f,
                      float eye_retargeting_strength = 1.0f,
                      float gaze_x = 0.0f, float gaze_y = 0.0f);

private:
    void preprocessImage(const cv::Mat& img, void* gpu_ptr, int target_w, int target_h, bool bgr_to_rgb);
    void computeStats(const std::string& name, void* device_ptr, size_t size);

    cudaStream_t stream;
    std::unique_ptr<CudaMemoryManager> mem;

    // Smoothing filters
    OneEuroFilter f_p, f_y, f_r;

    // Engines
    std::unique_ptr<TRTWrapper> appearance_engine;
    std::unique_ptr<TRTWrapper> motion_engine;
    std::unique_ptr<TRTWrapper> warping_engine;
    std::unique_ptr<TRTWrapper> stitching_engine;
    std::unique_ptr<TRTWrapper> stitching_eye_engine;
    std::unique_ptr<TRTWrapper> stitching_lip_engine;
    std::unique_ptr<TRTWrapper> landmark_engine;
    std::unique_ptr<TRTWrapper> face_det_engine;
    std::unique_ptr<TRTWrapper> face_pose_engine;
    std::unique_ptr<TRTWrapper> eyeblink_engine;

    // Intermediate and Source data
    cv::Mat src_img;
    void* f_s; 
    void* x_s; 
    void* gpu_kp_s_transformed; 
    void* exp_s;
    void* pitch_s;
    void* yaw_s;
    void* roll_s;
    void* t_s;
    void* scale_s;

    // Source ratios
    void* gpu_eye_ratio_s;
    void* gpu_lip_ratio_s;

    // Source values (CPU)
    float s_pitch_deg, s_yaw_deg, s_roll_deg;
    float s_t[3];
    float s_scale;
    float s_eye_ratio[2];
    float R_s[9];

    // Driving Reference (Frame 0)
    bool is_first_frame;
    float d_0_pitch_deg, d_0_yaw_deg, d_0_roll_deg;
    float d_0_t[3];
    float d_0_scale;
    void* gpu_exp_d_0;

    // Driving frame buffers (Device)
    void* gpu_input_motion_d;
    void* gpu_input_landmark_d;
    void* x_d; 
    void* exp_d;
    void* scale_d;
    void* pitch_d;
    void* yaw_d;
    void* roll_d;
    void* t_d;

    // Landmark buffers
    void* gpu_lmk_d_out1; // "output" (214)
    void* gpu_lmk_d_out2; // "853" (262)
    void* gpu_lmk_d_out3; // "856" (406) - The real landmarks
    
    void* gpu_eye_ratio_d;
    void* gpu_lip_ratio_d;
    void* gpu_eye_ratio_combined;
    void* gpu_lip_ratio_combined;

    // Final transformation buffers (Device)
    void* gpu_R_final;
    void* gpu_t_final;
    void* gpu_exp_rel;
    void* gpu_kp_rel;
    void* gpu_stitching_input;
    void* gpu_stitching_out;
    void* gpu_stitching_eye_out;
    void* gpu_stitching_lip_out;
    void* gpu_kp_final;
    void* gpu_out_frame;

    void* gpu_eye_params; // Now 66 floats: x_s (63) + params (3)
    void* gpu_eyeblink_delta;

    // CPU Pinned buffers for reading back small tensors
    float *h_pitch, *h_yaw, *h_roll, *h_t, *h_scale, *h_lmk;
    float *h_eye_params;

    // Profiling
    cudaEvent_t ev_start, ev_end;
    int frame_count;
};

#endif // __LIVEPORTRAIT_PIPELINE_H__
