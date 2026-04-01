# -*- coding: utf-8 -*-
import os
import argparse
import subprocess
import ffmpeg
import cv2
import time
import numpy as np
import os
import datetime
import platform
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm
from colorama import Fore, Back, Style
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


def run_with_video(args):
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)
    
    if not args.dri_video or not os.path.exists(args.dri_video):
        print("No driving video found! exit!")
        exit(1)
    
    vcap = cv2.VideoCapture(args.dri_video)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/latest_run"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vsave_crop_path = os.path.join(save_dir, f"output_crop.mp4")
    vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
    vsave_org_path = os.path.join(save_dir, f"output_org.mp4")
    vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0],
                                                                first_frame=first_frame)
        frame_ind += 1
        pbar.update(1)
        
        if out_crop is None:
            continue

        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])

        infer_times.append(time.time() - t0)
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        
        vout_crop.write(out_crop)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        vout_org.write(out_org)
        
    pbar.close()
    vcap.release()
    vout_crop.release()
    vout_org.release()
    
    if video_has_audio(args.dri_video):
        vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
        subprocess.call([FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video, "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
        print(f"Result with audio: {vsave_crop_path_new}")
    else:
        print(f"Result: {vsave_crop_path}")

    print("Inference median time: {} ms/frame".format(np.median(infer_times) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src_image', required=False, type=str, help='source image')
    parser.add_argument('--dri_video', required=False, type=str, help='driving video')
    parser.add_argument('--cfg', required=False, type=str, default="configs/trt_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference')
    parser.add_argument('--animal', action='store_true', help='use animal model')
    parser.add_argument('--paste_back', action='store_true', default=False, help='paste back to origin image')
    args, unknown = parser.parse_known_args()

    # Apply the TensorRT 10 output order patch globally for this run
    import src.models.motion_extractor_model as mem
    import inspect
    # We replace the class method directly if needed, but since we are running 
    # inside the container where we already patched the file, we just call it.
    
    run_with_video(args)
