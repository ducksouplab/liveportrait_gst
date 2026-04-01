#include "gstliveportrait.h"
#include "cuda_memory_manager.h"
#include "liveportrait_pipeline.h"
#include <gst/video/video.h>
#include <cuda_runtime.h>
#include <iostream>

GST_DEBUG_CATEGORY_STATIC (gst_liveportrait_debug);
#define GST_CAT_DEFAULT gst_liveportrait_debug

enum
{
  PROP_0,
  PROP_SOURCE_IMAGE,
  PROP_CONFIG_PATH
};

#define VIDEO_CAPS \
    GST_VIDEO_CAPS_MAKE ("{ RGB, BGR }") ", " \
    "width = (int) 512, height = (int) 512"

G_DEFINE_TYPE_WITH_CODE (GstLivePortrait, gst_liveportrait, GST_TYPE_VIDEO_FILTER,
    GST_DEBUG_CATEGORY_INIT (gst_liveportrait_debug, "liveportrait", 0, "LivePortrait Filter"));

static void gst_liveportrait_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_liveportrait_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_liveportrait_finalize (GObject * object);

static gboolean gst_liveportrait_start (GstBaseTransform * trans);
static gboolean gst_liveportrait_stop (GstBaseTransform * trans);
static GstFlowReturn gst_liveportrait_transform_frame (GstVideoFilter * filter, GstVideoFrame * in_frame, GstVideoFrame * out_frame);

static void
gst_liveportrait_class_init (GstLivePortraitClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstBaseTransformClass *base_transform_class = (GstBaseTransformClass *) klass;
  GstVideoFilterClass *video_filter_class = (GstVideoFilterClass *) klass;

  gobject_class->set_property = gst_liveportrait_set_property;
  gobject_class->get_property = gst_liveportrait_get_property;
  gobject_class->finalize = gst_liveportrait_finalize;

  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_liveportrait_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_liveportrait_stop);

  video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_liveportrait_transform_frame);

  g_object_class_install_property (gobject_class, PROP_SOURCE_IMAGE,
      g_param_spec_string ("source-image", "Source Image", "Path to source image",
          NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_PATH,
      g_param_spec_string ("config-path", "Config Path", "Path to YAML config",
          NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_pad_template (GST_ELEMENT_CLASS (klass),
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, gst_caps_from_string (VIDEO_CAPS)));
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS (klass),
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS, gst_caps_from_string (VIDEO_CAPS)));

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "LivePortrait Filter", "Filter/Video", "Fast LivePortrait reenactment using TensorRT", "Gemini CLI / warmshao");
}

static void
gst_liveportrait_init (GstLivePortrait * self)
{
  self->source_image = NULL;
  self->config_path = NULL;
  self->cuda_initialized = FALSE;
  self->memory_manager = NULL;
  self->trt_wrapper = NULL;
}

static void
gst_liveportrait_finalize (GObject * object)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (object);

  g_free (self->source_image);
  g_free (self->config_path);

  if (self->memory_manager) {
    delete (CudaMemoryManager*)self->memory_manager;
    self->memory_manager = NULL;
  }

  if (self->trt_wrapper) {
    delete (LivePortraitPipeline*)self->trt_wrapper;
    self->trt_wrapper = NULL;
  }

  G_OBJECT_CLASS (gst_liveportrait_parent_class)->finalize (object);
}

static void
gst_liveportrait_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (object);

  switch (prop_id) {
    case PROP_SOURCE_IMAGE:
      g_free (self->source_image);
      self->source_image = g_value_dup_string (value);
      break;
    case PROP_CONFIG_PATH:
      g_free (self->config_path);
      self->config_path = g_value_dup_string (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_liveportrait_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (object);

  switch (prop_id) {
    case PROP_SOURCE_IMAGE:
      g_value_set_string (value, self->source_image);
      break;
    case PROP_CONFIG_PATH:
      g_value_set_string (value, self->config_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_liveportrait_start (GstBaseTransform * trans)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (trans);
  cudaError_t err;

  GST_DEBUG_OBJECT (self, "Starting LivePortrait plugin, initializing CUDA stream and pipeline");

  err = cudaStreamCreate (&self->stream);
  if (err != cudaSuccess) {
    GST_ERROR_OBJECT (self, "Failed to create CUDA stream: %s", cudaGetErrorString (err));
    return FALSE;
  }

  self->memory_manager = new CudaMemoryManager();
  
  if (self->config_path) {
    try {
        self->trt_wrapper = new LivePortraitPipeline(self->config_path, self->stream);
        if (self->source_image) {
            ((LivePortraitPipeline*)self->trt_wrapper)->initSource(self->source_image);
        }
    } catch (const std::exception& e) {
        GST_ERROR_OBJECT (self, "Failed to initialize LivePortrait pipeline: %s", e.what());
        return FALSE;
    }
  }

  self->cuda_initialized = TRUE;
  return TRUE;
}

static gboolean
gst_liveportrait_stop (GstBaseTransform * trans)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (trans);

  if (self->cuda_initialized) {
    cudaStreamDestroy (self->stream);
    self->cuda_initialized = FALSE;
  }

  if (self->memory_manager) {
    delete (CudaMemoryManager*)self->memory_manager;
    self->memory_manager = NULL;
  }

  if (self->trt_wrapper) {
    delete (LivePortraitPipeline*)self->trt_wrapper;
    self->trt_wrapper = NULL;
  }

  return TRUE;
}

static GstFlowReturn
gst_liveportrait_transform_frame (GstVideoFilter * filter, GstVideoFrame * in_frame, GstVideoFrame * out_frame)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (filter);
  LivePortraitPipeline *pipe = (LivePortraitPipeline*)self->trt_wrapper;

  if (pipe) {
      /* Full logic to be implemented in sub-tasks */
      pipe->processFrame(GST_VIDEO_FRAME_PLANE_DATA(in_frame, 0), 
                         GST_VIDEO_FRAME_PLANE_DATA(out_frame, 0),
                         GST_VIDEO_FRAME_WIDTH(in_frame),
                         GST_VIDEO_FRAME_HEIGHT(in_frame));
  } else {
      gst_video_frame_copy (out_frame, in_frame);
  }

  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "liveportrait", GST_RANK_NONE, GST_TYPE_LIVEPORTRAIT);
}

#ifndef PACKAGE
#define PACKAGE "gst-liveportrait"
#endif

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    liveportrait,
    "LivePortrait reenactment filter",
    plugin_init,
    "1.0",
    "LGPL",
    "gst-liveportrait",
    "https://github.com/warmshao/FasterLivePortrait"
)
