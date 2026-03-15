"""Video package.

Provide a Video class that reads video from disk. It implements slicing and iterable protocols.
Also provide default transforms to scale video intensities and select a given channel.
"""

from byotrack.video.video import Video, VideoTransformConfig

__all__ = ["Video", "VideoTransformConfig"]
