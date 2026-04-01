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

  /* CUDA state */
  cudaStream_t stream;
  gboolean cuda_initialized;

  /* Internal state (to be expanded in later phases) */
  void *memory_manager;
  void *trt_wrapper;
};

G_END_DECLS

#endif /* __GST_LIVEPORTRAIT_H__ */
