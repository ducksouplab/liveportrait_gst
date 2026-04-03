#include "liveportrait_pipeline.h"
#include "image_proc.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <cmath>
#include <vector>
#include <gst/gst.h>

namespace fs = std::filesystem;

static void get_rotation_matrix(float pitch_deg, float yaw_deg, float roll_deg, float* R) {
    float PI = 3.14159265358979323846f;
    float p = pitch_deg * PI / 180.0f;
    float y = yaw_deg * PI / 180.0f;
    float r = roll_deg * PI / 180.0f;

    float cp = cosf(p), sp = sinf(p);
    float cy = cosf(y), sy = sinf(y);
    float cr = cosf(r), sr = sinf(r);

    // Python Logic: rot = rot_y @ rot_x @ rot_z; return rot.T
    float Rx[9] = {1, 0, 0, 0, cp, -sp, 0, sp, cp};
    float Ry[9] = {cy, 0, sy, 0, 1, 0, -sy, 0, cy};
    float Rz[9] = {cr, -sr, 0, sr, cr, 0, 0, 0, 1};

    float RyRx[9];
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j) {
            RyRx[i*3+j] = 0;
            for(int k=0; k<3; ++k) RyRx[i*3+j] += Ry[i*3+k] * Rx[k*3+j];
        }
    
    float R_full[9];
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j) {
            R_full[i*3+j] = 0;
            for(int k=0; k<3; ++k) R_full[i*3+j] += RyRx[i*3+k] * Rz[k*3+j];
        }

    // Return Transpose
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            R[j*3+i] = R_full[i*3+j];
}

static float headpose_pred_to_degree(const float* pred) {
    float max_val = pred[0];
    for(int i=1; i<66; ++i) if(pred[i] > max_val) max_val = pred[i];
    float sum_exp = 0;
    for(int i=0; i<66; ++i) sum_exp += expf(pred[i] - max_val);
    float degree = 0;
    for(int i=0; i<66; ++i) {
        float softmax = expf(pred[i] - max_val) / sum_exp;
        degree += softmax * i;
    }
    return degree * 3.0f - 97.5f;
}

LivePortraitPipeline::LivePortraitPipeline(const std::string& checkpoints_dir, cudaStream_t stream) 
    : stream(stream), is_first_frame(true), frame_count(0),
      f_p(25.0, 0.05, 0.005), f_y(25.0, 0.05, 0.005), f_r(25.0, 0.05, 0.005) {
    
    mem = std::make_unique<CudaMemoryManager>();
    std::string base = checkpoints_dir + "/liveportrait_onnx/";

    std::cout << "Loading LivePortrait engines from " << base << std::endl;

    appearance_engine = std::make_unique<TRTWrapper>(base + "appearance_feature_extractor.trt", stream);
    motion_engine = std::make_unique<TRTWrapper>(base + "motion_extractor.trt", stream);
    warping_engine = std::make_unique<TRTWrapper>(base + "warping_spade.trt", stream);
    stitching_engine = std::make_unique<TRTWrapper>(base + "stitching.trt", stream);
    stitching_eye_engine = std::make_unique<TRTWrapper>(base + "stitching_eye.trt", stream);
    stitching_lip_engine = std::make_unique<TRTWrapper>(base + "stitching_lip.trt", stream);
    landmark_engine = std::make_unique<TRTWrapper>(base + "landmark.trt", stream);
    face_det_engine = std::make_unique<TRTWrapper>(base + "retinaface_det_static.trt", stream);
    face_pose_engine = std::make_unique<TRTWrapper>(base + "face_2dpose_106_static.trt", stream);
    
    // Load official eyeblink engine (66 inputs)
    eyeblink_engine = std::make_unique<TRTWrapper>(base + "eyeblink.engine", stream);

    gpu_input_motion_d = mem->allocateDevice(3 * 256 * 256 * sizeof(float), "gpu_input_motion_d");
    gpu_input_landmark_d = mem->allocateDevice(3 * 224 * 224 * sizeof(float), "gpu_input_lmk_d");
    
    x_d = mem->allocateDevice(63 * sizeof(float), "x_d");
    exp_d = mem->allocateDevice(63 * sizeof(float), "exp_d");
    scale_d = mem->allocateDevice(1 * sizeof(float), "scale_d");
    pitch_d = mem->allocateDevice(66 * sizeof(float), "pitch_d");
    yaw_d = mem->allocateDevice(66 * sizeof(float), "yaw_d");
    roll_d = mem->allocateDevice(66 * sizeof(float), "roll_d");
    t_d = mem->allocateDevice(3 * sizeof(float), "t_d");

    gpu_exp_d_0 = mem->allocateDevice(63 * sizeof(float), "exp_d_0");
    gpu_exp_rel = mem->allocateDevice(63 * sizeof(float), "exp_rel");
    gpu_kp_rel = mem->allocateDevice(63 * sizeof(float), "kp_rel");
    gpu_kp_s_transformed = mem->allocateDevice(63 * sizeof(float), "kp_s_transformed");
    
    gpu_eye_ratio_s = mem->allocateDevice(2 * sizeof(float), "eye_ratio_s");
    gpu_lip_ratio_s = mem->allocateDevice(1 * sizeof(float), "lip_ratio_s");
    gpu_eye_ratio_d = mem->allocateDevice(2 * sizeof(float), "eye_ratio_d");
    gpu_lip_ratio_d = mem->allocateDevice(1 * sizeof(float), "lip_ratio_d");
    gpu_eye_ratio_combined = mem->allocateDevice(4 * sizeof(float), "eye_ratio_combined");
    gpu_lip_ratio_combined = mem->allocateDevice(2 * sizeof(float), "lip_ratio_combined");

    gpu_lmk_d_out1 = mem->allocateDevice(214 * sizeof(float), "lmk_out1");
    gpu_lmk_d_out2 = mem->allocateDevice(262 * sizeof(float), "lmk_out2");
    gpu_lmk_d_out3 = mem->allocateDevice(406 * sizeof(float), "lmk_out3");

    gpu_stitching_input = mem->allocateDevice(126 * sizeof(float), "stitching_input");
    gpu_stitching_out = mem->allocateDevice(65 * sizeof(float), "stitching_out");
    gpu_stitching_eye_out = mem->allocateDevice(65 * sizeof(float), "stitching_eye_out");
    gpu_stitching_lip_out = mem->allocateDevice(65 * sizeof(float), "stitching_lip_out");

    // Eyeblink Engine expects 66 inputs (63 source kp + 3 eye params)
    gpu_eye_params = mem->allocateDevice(66 * sizeof(float), "eye_params_concat");
    gpu_eyeblink_delta = mem->allocateDevice(63 * sizeof(float), "eyeblink_delta");

    gpu_kp_final = mem->allocateDevice(63 * sizeof(float), "kp_final");
    gpu_out_frame = mem->allocateDevice(3 * 512 * 512 * sizeof(float), "gpu_out_frame");

    h_pitch = (float*)mem->allocatePinned(66 * sizeof(float), "h_pitch");
    h_yaw = (float*)mem->allocatePinned(66 * sizeof(float), "h_yaw");
    h_roll = (float*)mem->allocatePinned(66 * sizeof(float), "h_roll");
    h_t = (float*)mem->allocatePinned(3 * sizeof(float), "h_t");
    h_scale = (float*)mem->allocatePinned(1 * sizeof(float), "h_scale");
    h_lmk = (float*)mem->allocatePinned(203 * 2 * sizeof(float), "h_lmk");
    h_eye_params = (float*)mem->allocatePinned(66 * sizeof(float), "h_eye_params_concat");
    
    gpu_R_final = mem->allocateDevice(9 * sizeof(float), "R_final");
    gpu_t_final = mem->allocateDevice(3 * sizeof(float), "t_final");

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    std::cout << "All engines and buffers initialized." << std::endl;
}

LivePortraitPipeline::~LivePortraitPipeline() {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

void LivePortraitPipeline::computeStats(const std::string& name, void* device_ptr, size_t size) {
    std::vector<float> host(size);
    cudaMemcpyAsync(host.data(), device_ptr, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double sum = 0, sq_sum = 0;
    for(float v : host) { sum += v; sq_sum += v*v; }
    double mean = sum / size;
    double std = std::sqrt(std::max(0.0, sq_sum / size - mean * mean));
    std::cout << "TELEMETRY: " << name << " mean=" << mean << ", std=" << std << std::endl;
}

void LivePortraitPipeline::preprocessImage(const cv::Mat& img, void* gpu_ptr, int target_w, int target_h, bool bgr_to_rgb) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);
    size_t img_size = target_w * target_h * 3;
    void* pinned_in = mem->allocatePinned(img_size, "tmp_pinned_in");
    memcpy(pinned_in, resized.data, img_size);
    void* gpu_raw = mem->allocateDevice(img_size, "tmp_gpu_raw");
    cudaMemcpyAsync(gpu_raw, pinned_in, img_size, cudaMemcpyHostToDevice, stream);
    launch_preprocess((uint8_t*)gpu_raw, (float*)gpu_ptr, target_w, target_h, bgr_to_rgb, stream);
}

bool LivePortraitPipeline::initSource(const std::string& image_path) {
    src_img = cv::imread(image_path);
    if (src_img.empty()) return false;
    is_first_frame = true;
    int w = src_img.cols, h = src_img.rows, size = std::min(w, h);
    cv::Mat src_square = src_img(cv::Rect((w - size) / 2, (h - size) / 2, size, size));

    void* gpu_input_feat = mem->allocateDevice(3 * 256 * 256 * sizeof(float), "src_feat_input");
    preprocessImage(src_square, gpu_input_feat, 256, 256, true);
    auto out_feat_shape = appearance_engine->getTensorShape("output");
    size_t out_feat_size = 1; for (auto d : out_feat_shape) out_feat_size *= d;
    f_s = mem->allocateDevice(out_feat_size * sizeof(float), "f_s");
    appearance_engine->execute({{"img", gpu_input_feat}}, {{"output", f_s}});

    void* gpu_input_lmk_s = mem->allocateDevice(3 * 224 * 224 * sizeof(float), "src_lmk_input");
    preprocessImage(src_square, gpu_input_lmk_s, 224, 224, true);
    landmark_engine->execute({{"input", gpu_input_lmk_s}}, {{"output", gpu_lmk_d_out1}, {"853", gpu_lmk_d_out2}, {"856", gpu_lmk_d_out3}});
    launch_calc_ratios((float*)gpu_lmk_d_out3, (float*)gpu_eye_ratio_s, (float*)gpu_lip_ratio_s, stream);
    // Read back source eye ratios to store for retargeting
    cudaMemcpyAsync(s_eye_ratio, gpu_eye_ratio_s, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);

    void* gpu_input_motion_s = mem->allocateDevice(3 * 256 * 256 * sizeof(float), "src_motion_input");
    preprocessImage(src_square, gpu_input_motion_s, 256, 256, true);
    pitch_s = mem->allocateDevice(66 * sizeof(float), "pitch_s"); yaw_s = mem->allocateDevice(66 * sizeof(float), "yaw_s");
    roll_s = mem->allocateDevice(66 * sizeof(float), "roll_s"); t_s = mem->allocateDevice(3 * sizeof(float), "t_s");
    exp_s = mem->allocateDevice(63 * sizeof(float), "exp_s"); scale_s = mem->allocateDevice(1 * sizeof(float), "scale_s");
    x_s = mem->allocateDevice(63 * sizeof(float), "x_s");
    motion_engine->execute({{"img", gpu_input_motion_s}}, {{"pitch", pitch_s}, {"yaw", yaw_s}, {"roll", roll_s}, {"t", t_s}, {"exp", exp_s}, {"scale", scale_s}, {"kp", x_s}});

    cudaMemcpyAsync(h_pitch, pitch_s, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_yaw, yaw_s, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_roll, roll_s, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_t, t_s, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_scale, scale_s, 1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    s_pitch_deg = headpose_pred_to_degree(h_pitch); s_yaw_deg = headpose_pred_to_degree(h_yaw); s_roll_deg = headpose_pred_to_degree(h_roll);
    s_scale = h_scale[0]; memcpy(s_t, h_t, 3 * sizeof(float)); get_rotation_matrix(s_pitch_deg, s_yaw_deg, s_roll_deg, R_s);
    void* gpu_R_s = mem->allocateDevice(9 * sizeof(float), "R_s_dev"); cudaMemcpyAsync(gpu_R_s, R_s, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    launch_transform_kp((float*)x_s, (float*)gpu_R_s, (float*)exp_s, s_scale, (float*)t_s, (float*)gpu_kp_s_transformed, 21, stream);
    return true;
}

bool LivePortraitPipeline::processFrame(const void* in_data, void* out_data, int width, int height,
                                        bool enable_eye_retargeting, float eyes_open_ratio,
                                        float eye_retargeting_strength,
                                        float gaze_x, float gaze_y) {
    if (src_img.empty()) return false;
    cv::Mat d_frame(height, width, CV_8UC3, (void*)in_data);
    preprocessImage(d_frame, gpu_input_motion_d, 256, 256, false);
    preprocessImage(d_frame, gpu_input_landmark_d, 224, 224, false);
    motion_engine->execute({{"img", gpu_input_motion_d}}, {{"pitch", pitch_d}, {"yaw", yaw_d}, {"roll", roll_d}, {"t", t_d}, {"exp", exp_d}, {"scale", scale_d}, {"kp", x_d}});
    landmark_engine->execute({{"input", gpu_input_landmark_d}}, {{"output", gpu_lmk_d_out1}, {"853", gpu_lmk_d_out2}, {"856", gpu_lmk_d_out3}});
    launch_calc_ratios((float*)gpu_lmk_d_out3, (float*)gpu_eye_ratio_d, (float*)gpu_lip_ratio_d, stream);

    if (is_first_frame) {
        cudaMemcpyAsync(h_pitch, pitch_d, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_yaw, yaw_d, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_roll, roll_d, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_t, t_d, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_scale, scale_d, 1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        d_0_pitch_deg = headpose_pred_to_degree(h_pitch); d_0_yaw_deg = headpose_pred_to_degree(h_yaw); d_0_roll_deg = headpose_pred_to_degree(h_roll);
        d_0_scale = h_scale[0]; memcpy(d_0_t, h_t, 3 * sizeof(float));
        cudaMemcpyAsync(gpu_exp_d_0, exp_d, 63 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        is_first_frame = false;
    }
    cudaMemcpyAsync(h_pitch, pitch_d, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_yaw, yaw_d, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_roll, roll_d, 66 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_t, t_d, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_scale, scale_d, 1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float R_d_i[9], R_d_0[9], R_rel[9], R_new[9];
    get_rotation_matrix(f_p.process(headpose_pred_to_degree(h_pitch)), f_y.process(headpose_pred_to_degree(h_yaw)), f_r.process(headpose_pred_to_degree(h_roll)), R_d_i);
    get_rotation_matrix(d_0_pitch_deg, d_0_yaw_deg, d_0_roll_deg, R_d_0);
    
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) { R_rel[i*3+j] = 0; for(int k=0; k<3; ++k) R_rel[i*3+j] += R_d_i[i*3+k] * R_d_0[k*3+j]; }
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) { R_new[i*3+j] = 0; for(int k=0; k<3; ++k) R_new[i*3+j] += R_rel[i*3+k] * R_s[k*3+j]; }
    
    float t_new[3] = {s_t[0] + (h_t[0] - d_0_t[0]), s_t[1] + (h_t[1] - d_0_t[1]), 0};
    float scale_new = s_scale * (h_scale[0] / d_0_scale);
    cudaMemcpyAsync(gpu_R_final, R_new, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_t_final, t_new, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    launch_relative_expression((float*)exp_s, (float*)exp_d, (float*)gpu_exp_d_0, (float*)gpu_exp_rel, 63, 1.0f, stream);
    launch_transform_kp((float*)x_s, (float*)gpu_R_final, (float*)gpu_exp_rel, scale_new, (float*)gpu_t_final, (float*)gpu_kp_rel, 21, stream);
    
    launch_concat_feat((float*)x_s, 63, (float*)gpu_kp_rel, 63, (float*)gpu_stitching_input, stream);
    stitching_engine->execute({{"input", gpu_stitching_input}}, {{"output", gpu_stitching_out}});
    launch_calc_ratios((float*)gpu_lmk_d_out3, (float*)gpu_eye_ratio_d, (float*)gpu_lip_ratio_d, stream);
    launch_concat_feat((float*)gpu_eye_ratio_s, 2, (float*)gpu_eye_ratio_d, 2, (float*)gpu_eye_ratio_combined, stream);
    launch_concat_feat((float*)gpu_lip_ratio_s, 1, (float*)gpu_lip_ratio_d, 1, (float*)gpu_lip_ratio_combined, stream);
    
    void* gpu_feat_eye = mem->getBuffer("feat_eye_diag"); if(!gpu_feat_eye) gpu_feat_eye = mem->allocateDevice(67*sizeof(float), "feat_eye_diag");
    void* gpu_feat_lip = mem->getBuffer("feat_lip_diag"); if(!gpu_feat_lip) gpu_feat_lip = mem->allocateDevice(65*sizeof(float), "feat_lip_diag");
    launch_concat_feat((float*)x_s, 63, (float*)gpu_eye_ratio_combined, 4, (float*)gpu_feat_eye, stream);
    launch_concat_feat((float*)x_s, 63, (float*)gpu_lip_ratio_combined, 2, (float*)gpu_feat_lip, stream);
    stitching_eye_engine->execute({{"input", gpu_feat_eye}}, {{"output", gpu_stitching_eye_out}});
    stitching_lip_engine->execute({{"input", gpu_feat_lip}}, {{"output", gpu_stitching_lip_out}});
    launch_add_deltas((float*)gpu_kp_rel, (float*)gpu_stitching_out, (float*)gpu_stitching_eye_out, (float*)gpu_stitching_lip_out, 21, stream);

    // Corrected Dynamic Eye Retargeting (66 inputs: source face + params [s_left, s_right, target])
    if (enable_eye_retargeting) {
        // Read back x_s to CPU to concatenate with params
        cudaMemcpyAsync(h_eye_params, x_s, 63 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        // Correct parameter mapping: [source_left, source_right, target]
        h_eye_params[63] = s_eye_ratio[0];
        h_eye_params[64] = s_eye_ratio[1];
        h_eye_params[65] = eyes_open_ratio;
        cudaMemcpyAsync(gpu_eye_params, h_eye_params, 66 * sizeof(float), cudaMemcpyHostToDevice, stream);
        // Corrected tensor names: "input" and "output"
        eyeblink_engine->execute({{"input", gpu_eye_params}}, {{"output", gpu_eyeblink_delta}});
        // Apply multiplier (strength) to achieves full blink
        launch_add_latent_delta((float*)gpu_kp_rel, (float*)gpu_eyeblink_delta, 21, eye_retargeting_strength, stream);
    }

    warping_engine->execute({{"feature_3d", f_s}, {"kp_source", gpu_kp_s_transformed}, {"kp_driving", gpu_kp_rel}}, {{"out", gpu_out_frame}});
    void* gpu_raw_out = mem->allocateDevice(512 * 512 * 3, "tmp_gpu_raw_out");
    launch_postprocess((float*)gpu_out_frame, (uint8_t*)gpu_raw_out, 512, 512, false, stream);
    void* pinned_out = mem->allocatePinned(512 * 512 * 3, "tmp_pinned_out");
    cudaMemcpyAsync(pinned_out, gpu_raw_out, 512 * 512 * 3, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    memcpy(out_data, pinned_out, 512 * 512 * 3);
    frame_count++;
    return true;
}
