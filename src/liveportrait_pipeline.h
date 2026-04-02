#ifndef __LIVEPORTRAIT_PIPELINE_H__
#define __LIVEPORTRAIT_PIPELINE_H__

#include "trt_wrapper.h"
#include "cuda_memory_manager.h"
#include <string>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

class LivePortraitPipeline {
public:
    LivePortraitPipeline(const std::string& checkpoints_dir, cudaStream_t stream);
    ~LivePortraitPipeline();

    bool initSource(const std::string& image_path);
    bool processFrame(const void* in_data, void* out_data, int width, int height);

private:
    void preprocessImage(const cv::Mat& img, void* gpu_ptr, int target_w, int target_h, bool bgr_to_rgb);
    void computeStats(const std::string& name, void* device_ptr, size_t size);

    cudaStream_t stream;
    std::unique_ptr<CudaMemoryManager> mem;

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

    // Source values (CPU)
    float s_pitch_deg, s_yaw_deg, s_roll_deg;
    float s_t[3];
    float s_scale;
    float R_s[9];

    // Driving Reference (Frame 0) - Now Instance Members
    bool is_first_frame;
    float d_0_pitch_deg, d_0_yaw_deg, d_0_roll_deg;
    float d_0_t[3];
    float d_0_scale;
    void* gpu_exp_d_0;

    // Driving frame buffers (Device)
    void* gpu_input_motion_d;
    void* x_d; 
    void* exp_d;
    void* scale_d;
    void* pitch_d;
    void* yaw_d;
    void* roll_d;
    void* t_d;

    // Landmark buffers
    void* gpu_lmk_d;
    void* gpu_eye_ratio;
    void* gpu_lip_ratio;

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

    // CPU Pinned buffers for reading back small tensors
    float *h_pitch, *h_yaw, *h_roll, *h_t, *h_scale, *h_lmk;

    // Profiling
    cudaEvent_t ev_start, ev_end;
};

#endif // __LIVEPORTRAIT_PIPELINE_H__
