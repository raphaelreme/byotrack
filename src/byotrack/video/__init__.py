"""Video package.

Provide a Video class that reads video from disk. It implements slicing and iterable protocols.
Also provide default transforms to scale video intensities and select a given channel.
"""

from byotrack.video.preprocessor.channel_projection import ChannelProjection
from byotrack.video.preprocessor.normalizer import IntensityNormalizer
from byotrack.video.preprocessor.preprocessor import VideoPreprocessor
from byotrack.video.preprocessor.slicer import FrameSlicer
from byotrack.video.preprocessor.spatial_projection import SpatialProjection
from byotrack.video.reader import ArrayVideoReader, OpenCVVideoReader, TiffVideoReader, VideoReader
from byotrack.video.video import Video, VideoTransformConfig, video_dtype, video_length, video_shape

__all__ = [
    "ArrayVideoReader",
    "ChannelProjection",
    "FrameSlicer",
    "IntensityNormalizer",
    "OpenCVVideoReader",
    "SpatialProjection",
    "TiffVideoReader",
    "Video",
    "VideoPreprocessor",
    "VideoReader",
    "VideoTransformConfig",
    "video_dtype",
    "video_length",
    "video_shape",
]
