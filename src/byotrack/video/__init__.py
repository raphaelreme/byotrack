"""Video package.

Provide a Video class that reads video from disk. It implements slicing and iterable protocols.
Multiple preprocessors can be added to the video to modify (crop, normalize, register, ...) the frames at reading time.
"""

from byotrack.video.preprocessor.channel_projection import ChannelProjection
from byotrack.video.preprocessor.normalizer import IntensityNormalizer
from byotrack.video.preprocessor.preprocessor import VideoPreprocessor
from byotrack.video.preprocessor.registrator import Registrator
from byotrack.video.preprocessor.slicer import FrameSlicer
from byotrack.video.preprocessor.spatial_projection import SpatialProjection
from byotrack.video.preprocessor.temporal_equalization import TemporalEqualizer
from byotrack.video.reader import ArrayVideoReader, OpenCVVideoReader, TiffVideoReader, VideoReader
from byotrack.video.video import Video, VideoTransformConfig, video_dtype, video_length, video_shape

__all__ = [
    "ArrayVideoReader",
    "ChannelProjection",
    "FrameSlicer",
    "IntensityNormalizer",
    "OpenCVVideoReader",
    "Registrator",
    "SpatialProjection",
    "TemporalEqualizer",
    "TiffVideoReader",
    "Video",
    "VideoPreprocessor",
    "VideoReader",
    "VideoTransformConfig",
    "video_dtype",
    "video_length",
    "video_shape",
]
