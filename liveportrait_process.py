import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class LivePortraitProcess:
    def __init__(self, plugin_path="./build"):
        """
        Initializes GStreamer and sets the plugin path.
        """
        Gst.init(None)
        self.plugin_path = os.path.abspath(plugin_path)
        # Ensure the plugin path is in the environment
        os.environ["GST_PLUGIN_PATH"] = self.plugin_path
        
        # Check if the liveportrait element is available
        registry = Gst.Registry.get()
        plugin = registry.scan_path(self.plugin_path)
        if not Gst.ElementFactory.find("liveportrait"):
            raise RuntimeError(f"Could not find 'liveportrait' element in {self.plugin_path}")

    def process(self, input_video, output_video, source_image, config_path, crop=(280, 280)):
        """
        Runs the LivePortrait GStreamer pipeline.
        
        Args:
            input_video: Path to driving video.
            output_video: Path to save the result.
            source_image: Path to source static image.
            config_path: Path to TensorRT engines directory.
            crop: Tuple of (left, right) crop values for aspect ratio correction.
        """
        left, right = crop
        
        pipeline_str = (
            f"filesrc location={input_video} ! "
            f"decodebin ! videoconvert ! "
            f"videocrop left={left} right={right} ! "
            f"videoscale ! video/x-raw,width=512,height=512,format=RGB ! "
            f"liveportrait config-path={config_path} source-image={source_image} ! "
            f"videoconvert ! x264enc ! mp4mux ! "
            f"filesink location={output_video}"
        )
        
        print(f"Executing pipeline: {pipeline_str}")
        
        pipeline = Gst.parse_launch(pipeline_str)
        bus = pipeline.get_bus()
        
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            while True:
                msg = bus.timed_pop_filtered(
                    Gst.CLOCK_TIME_NONE,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS
                )
                
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        print(f"Error: {err.message}")
                        print(f"Debug: {debug}")
                        break
                    elif msg.type == Gst.MessageType.EOS:
                        print("End of stream reached.")
                        break
        finally:
            pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    # Example usage (can be used for basic testing)
    import argparse
    parser = argparse.ArgumentParser(description="LivePortrait Python Wrapper")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--config", required=True, help="Path to engines")
    parser.add_argument("--plugin-path", default="./build", help="Path to libgstliveportrait.so")
    
    args = parser.parse_args()
    
    processor = LivePortraitProcess(plugin_path=args.plugin_path)
    processor.process(args.input, args.output, args.source, args.config)
