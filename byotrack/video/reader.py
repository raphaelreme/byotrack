from __future__ import annotations

import os
from typing import Dict, List, Tuple, Union

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore


class MetaVideoReader(type):
    """MetaClass for Video Readers

    Each VideoReader has to define a list of supported extensions.
    The last constructed VideoReader to claim an extension will be used to open the video.
    If no one has claimed an extension the default OpenCVVideoReader is used.
    """

    supported_extensions: List[str] = []
    extension_to_reader: Dict[str, MetaVideoReader] = {}

    def __init__(cls, cls_name: str, bases: tuple, attributes: dict) -> None:
        super().__init__(cls_name, bases, attributes)

        for extension in cls.supported_extensions:
            type(cls).extension_to_reader[extension] = cls


class VideoReader(metaclass=MetaVideoReader):
    """Unified video reader api

    Close to OpenCV API but few key differences:

    * There is always a frame loaded
    * Frame ids goes from 0 to length - 1
    * Read method is very different:

        * It retrieves the current frame then grabs the next (The other way around in opencv)
        * It returns therefore a ndarray and a bool rather than a bool and a ndarray
        * The boolean returned indicated if we can continue to read and not if the read operation has failed
    * Easy to check main attributes like:

        * frame_id
        * length
        * shape
        * fps if known (-1 otherwise)

    Return images in BGR by default like opencv as frames are mostly used with opencv afterwards for display.
    Can also return grayscale image (H, W, 1)

    Attributes:
        supported_extensions (List[str]): Static attribute used by `open` method to automatically choose
            which VideoReader to use.
        path (str | bytes | os.PathLike): Path of the current video
        released (bool): True when release has been called (close and release memory)
        fps (int): Frame rate (-1 if unknown)
        shape (Tuple[int, int]): Spatial dimensions of frames
        channels (int): Number of channels
        length (int): Number of frames
        frame_id (int): Current frame id

    """

    supported_extensions: List[str] = []

    def __init__(self, path: Union[str, os.PathLike], **kwargs):  # pylint: disable=unused-argument
        """Constructor: Open the video file

        Args:
            path (str | os.PathLike): Path to the video to read
            kwargs: Any additional kwargs are given to the underlying video reader

        """
        self.path = path
        self.released = False
        self.fps = -1
        self.shape = (0, 0)
        self.channels = 0
        self.length = 0
        self.frame_id = 0

    def release(self) -> None:
        """Close the file and free memory"""
        self.released = True

    def grab(self) -> bool:
        """Grab the next frame

        Can be faster than self.seek(self.frame_id + 1)

        Returns:
            bool: True if able to grab next frame

        """
        raise NotImplementedError()

    def retrieve(self) -> np.ndarray:
        """Retrieve the current frame

        Returns:
            np.ndarray: The current frame
                Shape: (H, W, 3) or (H, W, 1) (Grayscale)

        """
        raise NotImplementedError()

    def read(self) -> Tuple[np.ndarray, bool]:
        """Consume a frame. Is equivalent to retrieve + grab

        As in this implementation there is always a current frame. It reverses open cv implementation
        It first retrieves then grab next frame

        Returns:
            np.ndarray: The current frame
                Shape: (H, W, 3) or (H, W, 1) (Grayscale)
            bool: Whether there is a next frame to read

        """
        current_frame = self.retrieve()

        return current_frame, self.grab()

    def seek(self, frame_id: int) -> None:
        """Seek frame_id (will update the current frame)

        Valid frame ids from 0 to length - 1

        Raise:
            EOFError if seeking an invalid frame

        """
        raise NotImplementedError()

    def tell(self) -> int:
        """Returns self.frame_id

        Returns:
            int: current frame_id

        """
        return self.frame_id

    def _check_frame_id(self) -> None:
        """Check that the underlying data is at the right frame id"""

    @staticmethod
    def open(path: Union[str, os.PathLike], **kwargs) -> VideoReader:
        """Open a video file

        Use the extension to know which VideoReader to use

        Args:
            path (str | os.PathLike): File to open
            kwargs: Any additional args for the underlying video reader

        Returns:
            VideoReader

        """
        extension = str(path).rsplit(".", maxsplit=1)[-1]

        return VideoReader.extension_to_reader.get(extension, OpenCVVideoReader)(path, **kwargs)

    @staticmethod
    def ensure_3d(frame: np.ndarray) -> np.ndarray:
        """Ensure that frame is a 3 dimensional array

        Args:
            frame (np.ndarray): Frame to check

        Returns:
            np.ndarray: Valid frame with 3 dimensions (H, W, C)

        """
        if len(frame.shape) == 2:
            frame = frame[..., None]

        if len(frame.shape) != 3:
            raise ValueError(f"Unhandled dimensions for the given frame: {frame.shape}")

        return frame


class OpenCVVideoReader(VideoReader):
    """Wrapper around opencv VideoCapture

    Default VideoReader when opening a file.

    """

    def __init__(self, path: Union[str, os.PathLike], **kwargs):
        super().__init__(path, **kwargs)
        self.video = cv2.VideoCapture(str(path), **kwargs)
        assert self.video.grab(), "No frames found in the video"

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.shape = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self) -> None:
        super().release()
        self.video.release()

    def grab(self) -> bool:
        grabbed = self.video.grab()
        if grabbed:
            self.frame_id += 1
        else:
            self.seek(self.frame_id)
        return grabbed

    def retrieve(self) -> np.ndarray:
        return self.ensure_3d(self.video.retrieve()[1])

    def seek(self, frame_id: int) -> None:
        if not 0 <= frame_id < self.length:
            raise EOFError(f"Seeking outside of video limits: {frame_id} not in [{0}, {self.length}[")
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id + 1)
        self.frame_id = frame_id

    def _check_frame_id(self) -> None:
        assert self.frame_id == self.video.get(cv2.CAP_PROP_POS_FRAMES) - 1


class TiffVideoReader(VideoReader):
    """Special VideoReader for multi images tif files, not supported by opencv

    Uses PIL as backend

    """

    supported_extensions = ["tif", "tiff"]

    def __init__(self, path: Union[str, os.PathLike], **kwargs):
        super().__init__(path, **kwargs)
        self.video = Image.open(path, **kwargs)
        self.shape = self.video.size
        self.length = self.video.n_frames

    def release(self) -> None:
        super().release()
        self.video.close()

    def grab(self) -> bool:
        try:
            self.seek(self.frame_id + 1)
            return True
        except EOFError:
            return False

    def retrieve(self) -> np.ndarray:
        frame = self.ensure_3d(np.array(self.video))
        return np.flip(frame, 2)  # In PIL, RGB images are read as RGB. Let's flip it to BGR

    def seek(self, frame_id: int) -> None:
        self.video.seek(frame_id)
        self.frame_id = frame_id

    def _check_backend_frame_id(self) -> None:
        assert self.frame_id == self.video.tell()
