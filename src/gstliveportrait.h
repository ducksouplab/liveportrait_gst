#ifndef __GST_LIVEPORTRAIT_H__
#define __GST_LIVEPORTRAIT_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <cuda_runtime.h>

G_BEGIN_DECLS

#define GST_TYPE_LIVEPORTRAIT (gst_liveportrait_get_type())
G_DECLARE_FINAL_TYPE (GstLivePortrait, gst_liveportrait, GST, LIVEPORTRAIT, GstVideoFilter)

struct _GstLivePortrait
{
  GstVideoFilter parent;

  /* Properties */
  gchar *source_image;
  gchar *config_path;

  /* Eye retargeting properties */
  gboolean enable_eye_retargeting;
  gfloat eyes_open_ratio;
  gfloat eye_retargeting_strength;
  gfloat gaze_x;
  gfloat gaze_y;

  /* CUDA state */
  cudaStream_t stream;
  gboolean cuda_initialized;

  /* Internal state */
  class LivePortraitPipeline *pipeline;
};

G_END_DECLS

#endif /* __GST_LIVEPORTRAIT_H__ */
