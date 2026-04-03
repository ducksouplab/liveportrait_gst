#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def run_command(cmd, verbose=False):
    if verbose:
        print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        raise e

def process_liveportrait(
    input_path,
    output_path,
    source_path,
    config_path,
    plugin_path="./build",
    crop_left=280,
    crop_right=280,
    enable_eye_retargeting=False,
    eyes_open_ratio=0.0,
    eye_retargeting_strength=1.0,
    gaze_x=0.0,
    gaze_y=0.0,
    docker_image="ducksouplab/liveportrait_gst:latest",
    verbose=False
):
    """
    Programmatic interface to run the LivePortrait GStreamer plugin via Docker.
    """
    # Absolute paths setup
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    source_abs = os.path.abspath(source_path)
    config_abs = os.path.abspath(config_path)
    plugin_abs = os.path.abspath(plugin_path)
    
    # Directories and filenames
    input_dir = os.path.dirname(input_abs)
    input_file = os.path.basename(input_abs)
    
    output_dir = os.path.dirname(output_abs)
    output_file = os.path.basename(output_abs)
    
    source_dir = os.path.dirname(source_abs)
    source_file = os.path.basename(source_abs)
    
    # Docker mount points
    docker_work = "/work"
    docker_source = "/source"
    docker_config = "/checkpoints"
    docker_plugin = "/plugin"
    
    # Eye retargeting string
    eye_str = ""
    if enable_eye_retargeting:
        eye_str = (
            f"enable-eye-retargeting=true "
            f"eyes-open-ratio={eyes_open_ratio} "
            f"eye-retargeting-strength={eye_retargeting_strength} "
            f"gaze-x={gaze_x} gaze-y={gaze_y} "
        )

    # Prepare the pipeline string
    pipeline = (
        f"filesrc location={docker_work}/{input_file} ! "
        f"decodebin ! videoconvert ! "
        f"videocrop left={crop_left} right={crop_right} ! "
        f"videoscale ! video/x-raw,width=512,height=512,format=RGB ! "
        f"liveportrait config-path={docker_config} source-image={docker_source}/{source_file} {eye_str}! "
        f"videoconvert ! x264enc ! mp4mux ! "
        f"filesink location={docker_work}/{output_file}"
    )

    # Docker command construction
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{input_dir}:{docker_work}",
        "-v", f"{source_dir}:{docker_source}",
        "-v", f"{config_abs}:{docker_config}",
        "-v", f"{plugin_abs}:{docker_plugin}",
        "-e", f"GST_PLUGIN_PATH={docker_plugin}",
    ]
    
    # Pass through GST_DEBUG if set on host
    if os.getenv("GST_DEBUG"):
        docker_cmd.extend(["-e", f"GST_DEBUG={os.getenv('GST_DEBUG')}"])
    
    # Handle output directory if it differs from input directory
    if output_dir != input_dir:
        docker_cmd.extend(["-v", f"{output_dir}:/output_dir"])
        pipeline = pipeline.replace(f"location={docker_work}/{output_file}", f"location=/output_dir/{output_file}")

    docker_cmd.extend([
        docker_image,
        "gst-launch-1.0", "-q"
    ])
    
    docker_cmd.extend(pipeline.split())

    run_command(docker_cmd, verbose)
    return True

def main():
    parser = argparse.ArgumentParser(description="Wrapper for LivePortrait GStreamer plugin via Docker")
    
    # Core paths
    parser.add_argument("--input", required=True, help="Input driving video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--source", required=True, help="Source static image path")
    parser.add_argument("--config", required=True, help="Path to engines directory (e.g. checkpoints/)")
    
    # Plugin settings
    parser.add_argument("--plugin-path", default="./build", help="Path to libgstliveportrait.so on host")
    parser.add_argument("--crop-left", type=int, default=280, help="Left crop for aspect ratio correction")
    parser.add_argument("--crop-right", type=int, default=280, help="Right crop for aspect ratio correction")
    
    # Eye Retargeting
    parser.add_argument("--enable-eye-retargeting", action="store_true", help="Enable dynamic eye retargeting")
    parser.add_argument("--eyes-open-ratio", type=float, default=0.0, help="Target eye open ratio (0.0 to 1.0)")
    parser.add_argument("--eye-retargeting-strength", type=float, default=1.0, help="Multiplier for eye delta")
    parser.add_argument("--gaze-x", type=float, default=0.0, help="Gaze direction X")
    parser.add_argument("--gaze-y", type=float, default=0.0, help="Gaze direction Y")

    # Docker settings
    parser.add_argument("--docker-image", default="ducksouplab/liveportrait_gst:latest", help="Docker image name")
    parser.add_argument("--verbose", action="store_true", help="Print verbose execution info")

    args = parser.parse_args()

    try:
        process_liveportrait(
            input_path=args.input,
            output_path=args.output,
            source_path=args.source,
            config_path=args.config,
            plugin_path=args.plugin_path,
            crop_left=args.crop_left,
            crop_right=args.crop_right,
            enable_eye_retargeting=args.enable_eye_retargeting,
            eyes_open_ratio=args.eyes_open_ratio,
            eye_retargeting_strength=args.eye_retargeting_strength,
            gaze_x=args.gaze_x,
            gaze_y=args.gaze_y,
            docker_image=args.docker_image,
            verbose=args.verbose
        )
        print(f"Success! Output saved to {args.output}")
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
