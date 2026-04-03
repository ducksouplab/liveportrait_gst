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
  PROP_CONFIG_PATH,
  PROP_ENABLE_EYE_RETARGETING,
  PROP_EYES_OPEN_RATIO,
  PROP_EYE_RETARGETING_STRENGTH,
  PROP_GAZE_X,
  PROP_GAZE_Y,
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
      g_param_spec_string ("config-path", "Config Path", "Path to checkpoints directory",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_EYE_RETARGETING,
      g_param_spec_boolean ("enable-eye-retargeting", "Enable Eye Retargeting",
          "Enable dynamic eyelid closure and gaze control", FALSE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EYES_OPEN_RATIO,
      g_param_spec_float ("eyes-open-ratio", "Eyes Open Ratio",
          "Eyes open ratio (0.0 to 1.0)", 0.0, 1.0, 0.0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EYE_RETARGETING_STRENGTH,
      g_param_spec_float ("eye-retargeting-strength", "Eye Retargeting Strength",
          "Multiplier for eye retargeting delta", 0.0, 10.0, 1.0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GAZE_X,
      g_param_spec_float ("gaze-x", "Gaze X",
          "Gaze direction X (-1.0 to 1.0)", -1.0, 1.0, 0.0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GAZE_Y,
      g_param_spec_float ("gaze-y", "Gaze Y",
          "Gaze direction Y (-1.0 to 1.0)", -1.0, 1.0, 0.0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

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
  self->enable_eye_retargeting = FALSE;
  self->eyes_open_ratio = 0.0f;
  self->eye_retargeting_strength = 1.0f;
  self->gaze_x = 0.0f;
  self->gaze_y = 0.0f;
  self->cuda_initialized = FALSE;
  self->pipeline = NULL;
}

static void
gst_liveportrait_finalize (GObject * object)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (object);

  g_free (self->source_image);
  g_free (self->config_path);

  if (self->pipeline) {
    delete self->pipeline;
    self->pipeline = NULL;
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
    case PROP_ENABLE_EYE_RETARGETING:
      self->enable_eye_retargeting = g_value_get_boolean (value);
      break;
    case PROP_EYES_OPEN_RATIO:
      self->eyes_open_ratio = g_value_get_float (value);
      break;
    case PROP_EYE_RETARGETING_STRENGTH:
      self->eye_retargeting_strength = g_value_get_float (value);
      break;
    case PROP_GAZE_X:
      self->gaze_x = g_value_get_float (value);
      break;
    case PROP_GAZE_Y:
      self->gaze_y = g_value_get_float (value);
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
    case PROP_ENABLE_EYE_RETARGETING:
      g_value_set_boolean (value, self->enable_eye_retargeting);
      break;
    case PROP_EYES_OPEN_RATIO:
      g_value_set_float (value, self->eyes_open_ratio);
      break;
    case PROP_EYE_RETARGETING_STRENGTH:
      g_value_set_float (value, self->eye_retargeting_strength);
      break;
    case PROP_GAZE_X:
      g_value_set_float (value, self->gaze_x);
      break;
    case PROP_GAZE_Y:
      g_value_set_float (value, self->gaze_y);
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

  if (self->config_path) {
    try {
        self->pipeline = new LivePortraitPipeline(self->config_path, self->stream);
        if (self->source_image) {
            self->pipeline->initSource(self->source_image);
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

  if (self->pipeline) {
    delete self->pipeline;
    self->pipeline = NULL;
  }

  if (self->cuda_initialized) {
    cudaStreamDestroy (self->stream);
    self->cuda_initialized = FALSE;
  }

  return TRUE;
}

static GstFlowReturn
gst_liveportrait_transform_frame (GstVideoFilter * filter, GstVideoFrame * in_frame, GstVideoFrame * out_frame)
{
  GstLivePortrait *self = GST_LIVEPORTRAIT (filter);

  if (self->pipeline) {
      self->pipeline->processFrame(GST_VIDEO_FRAME_PLANE_DATA(in_frame, 0), 
                         GST_VIDEO_FRAME_PLANE_DATA(out_frame, 0),
                         GST_VIDEO_FRAME_WIDTH(in_frame),
                         GST_VIDEO_FRAME_HEIGHT(in_frame),
                         self->enable_eye_retargeting,
                         self->eyes_open_ratio,
                         self->eye_retargeting_strength,
                         self->gaze_x,
                         self->gaze_y);
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
