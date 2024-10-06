from __future__ import annotations

import os
import pathlib
from typing import Callable, Iterable, Dict, List, Optional, Tuple, Union

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore
import tifffile  # type: ignore

from .. import utils

# TODO: TiffReader read slice from inside a page ? (directly in going through the bytes ?)
# Hard and probably not so useful as from what we saw, pages are usually only 2D and therefore not so large


def slice_length(slice_: slice, shape: int) -> int:
    """Compute the number of element in a slice"""
    start, stop, step = slice_.indices(shape)
    return max((stop - start - (step > 0) + (step < 0)) // step + 1, 0)


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
    * Frames are loaded in RGB.
    * It support any number of channels and 2D/3D
    * Read method is very different:

        * It retrieves the current frame then grabs the next (The other way around in opencv)
        * It returns therefore a ndarray and a bool rather than a bool and a ndarray
        * The boolean returned indicated if we can continue to read and not if the read operation has failed
    * Easy to check main attributes like:

        * frame_id
        * length
        * channels
        * shape
        * fps if known (-1 otherwise)

    Attributes:
        supported_extensions (List[str]): Static attribute used by `open` method to automatically choose
            which VideoReader to use.
        path (pathlib.Path): Path of the current video
        released (bool): True when release has been called (close and release memory)
        fps (int): Frame rate (-1 if unknown)
        shape (Tuple[int, ...]): Spatial dimensions of frames (Height, Width[, Depth])
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
        self.path = pathlib.Path(path)
        self.released = False
        self.fps = -1
        self.shape: Tuple[int, ...] = (0, 0)
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
                Shape: ([D, ]H, W, C)

        """
        raise NotImplementedError()

    def read(self) -> Tuple[np.ndarray, bool]:
        """Consume a frame. Is equivalent to retrieve + grab

        As in this implementation there is always a current frame. It reverses OpenCV implementation
        It first retrieves then grab next frame

        Returns:
            np.ndarray: The current frame - Shape: ([D, ]H, W, C)
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
        path = pathlib.Path(path)
        if path.is_dir():
            return MultiFrameReader(path, **kwargs)

        return VideoReader.extension_to_reader.get(path.suffix, OpenCVVideoReader)(path, **kwargs)


class OpenCVVideoReader(VideoReader):
    """Wrapper around opencv VideoCapture

    Default VideoReader when opening a file.

    It only supports 2D images (grayscale or RGB).

    Attributes:
        video (cv2.VideoCapture): VideoCapture from opencv

    """

    def __init__(self, path: Union[str, os.PathLike], **kwargs):
        super().__init__(path, **kwargs)
        self.video = cv2.VideoCapture(str(path), **kwargs)
        assert self.video.grab(), "No frame found in the video"
        self.channels = self.retrieve().shape[-1]

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.shape = (int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)))

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
        frame = self.video.retrieve()[1]

        if len(frame.shape) == 2:
            return frame[..., None]

        return np.flip(frame, -1)

    def seek(self, frame_id: int) -> None:
        if not 0 <= frame_id < self.length:
            raise EOFError(f"Seeking outside of video limits: {frame_id} not in [{0}, {self.length}[")
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id + 1)
        self.frame_id = frame_id

    def _check_frame_id(self) -> None:
        assert self.frame_id == self.video.get(cv2.CAP_PROP_POS_FRAMES) - 1


class PILVideoReader(VideoReader):
    """Old PIL video reader. Works well for 2D multi frames Tiff files that are not supported by OpenCV.

    IT only supports 2D videos. TiffVideoReader should have a larger support for TiffFiles

    See `VideoReader` for inherited attributes.

    Attributes:
        video (PIL.Image.Image): PIL image (animated)

    """

    supported_extensions = [".tif", ".tiff"]

    def __init__(self, path: Union[str, os.PathLike], **kwargs):
        super().__init__(path, **kwargs)
        self.video = Image.open(path, **kwargs)
        self.channels = self.retrieve().shape[-1]
        self.shape = self.video.size[::-1]
        self.length = self.video.n_frames  # type: ignore

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
        frame = np.array(self.video)

        if len(frame.shape) == 2:
            return frame[..., None]

        return frame

    def seek(self, frame_id: int) -> None:
        self.video.seek(frame_id)
        self.frame_id = frame_id

    def _check_frame_id(self) -> None:
        assert self.frame_id == self.video.tell()


class TiffVideoReader(VideoReader):
    """Tiff video reader with tifffile. Handle 2D and 3D videos with any channels

    Axes are inferred from the tifffile metadata and convert into (T, [D, ]H, W, C) (<=> T[Z]YXC).
    We may not support all formats, or your specific metadata can be wrong/missing. In this case,
    you can also provide the expected axes of the tifffile using an ordered string.

    For example: "TYX" for 2D videos without channel, "TCZYX" for 3D videos with channels
    (ordered by time, channel then stack), "ZTYX" for 3D videos without channels (ordered by stack then time).

    Note:
        With tifffile syntax, we use X for width, Y for height, Z for depth and
        C/S for channels (C and S are not supported together) and T for time.
        Any other letter (I, O, Q, ...) is first interpret as T if it is missing,
        then Z if it is missing, and finally it will yield an error.

    It also supports to read the tiff at a specific resolution level

    See `VideoReader` for inherited attributes.

    Attributes:
        out_axes (str): Axes order of the outputs
        level (int): Resolution level (if any)
            Default: 0 (finest level)
        in_axes (Dict[str, int]): Parsed in axes
        ax_slice (Dict[str, slice]): Optional slices to use on axes.

    """

    supported_extensions = [".tif", ".tiff"]

    out_axes = "TZYXC"  # TZYXC = TDHWC

    def __init__(
        self,
        path: Union[str, os.PathLike],
        level=0,
        axes: Optional[str] = None,
        ax_slice: Optional[Dict[str, slice]] = None,
        **kwargs,
    ):
        """Constructor

        Args:
            axes (Optional[str]): Override the axes found in the tiff metadata.
                Default: None (Parse the metadata)
        """
        super().__init__(path, **kwargs)

        assert self.out_axes[-1] == "C", "Channels have to be the last out axes"

        self.video = tifffile.TiffFile(path, **kwargs).series[0].levels[level]  # pylint: disable=unsubscriptable-object
        self.level = level
        self.ax_slice = ax_slice if ax_slice is not None else {}
        self._page_dim = len(self.video.keyframe.axes)
        self.in_axes = self._extract_axes(axes)

        self._check_slice(self.ax_slice, self.out_axes)

        shape = tuple(
            shape if axis not in self.ax_slice else slice_length(self.ax_slice[axis], shape)
            for shape, axis in zip(self.video._shape_squeezed, self.axes)  # pylint: disable=protected-access
        )

        self.channels = shape[self.in_axes["C"]] if "C" in self.in_axes else 1
        self.shape = tuple(
            shape[self.in_axes[axis]] for axis in self.out_axes if axis in ("X", "Y", "Z") and axis in self.in_axes
        )
        self.length = shape[self.in_axes["T"]]

        self._current = np.zeros((*shape[: self.in_axes["T"]], *shape[self.in_axes["T"] + 1 :]), dtype=self.video.dtype)
        self._load()

    @staticmethod
    def _check_slice(ax_slice: Dict[str, slice], axes: str):
        if "T" in ax_slice:
            raise ValueError("Temporal slicing is not supported for VideoReader. It can be done though Video slicing.")

        for axis, slice_ in ax_slice.items():
            assert axis in axes, f"Unknown axis: {axis} not in {axes}"
            if slice_.step is not None and slice_.step < 1:
                raise ValueError(
                    "Negative step slicing is not supported for VideoReader. It can be done though Video slicing"
                )

    def _extract_axes(self, axes: Optional[str] = None) -> Dict[str, int]:
        if axes:
            # Check that the provided axes are valid
            assert len(set(axes)) == len(axes), "Duplicated axes"
            assert "T" in axes and "X" in axes and "Y" in axes, "Axes should include T, X and Y"
            assert len(axes) <= 5, "The reader can only handle up to 3D + T + C"
            for axis in axes:
                assert axis in self.out_axes, f"Unknown axis: {axis} not in {self.out_axes}"
            assert "T" not in axes[-self._page_dim :], "The temporal axis cannot be in the pages dimensions"
            return {axis: i for i, axis in enumerate(axes)}

        axes_squeezed = self.video._axes_squeezed  # pylint: disable=protected-access
        in_axes = {axis: i for i, axis in enumerate(axes_squeezed)}
        if "S" in in_axes:  # Interpret S as C if possible
            if "C" in in_axes:
                raise ValueError("Unable to parse data with two channel dimensions (C and S)")
            in_axes["C"] = in_axes.pop("S")

        # Check that everything is fine
        assert len(in_axes) == len(axes_squeezed), "Duplicated axes"
        assert 3 <= len(in_axes) <= 5, "The reader expects at least 2D + T and at most 3D + T + C"
        assert "X" in in_axes and "Y" in in_axes, "Missing a spatial dimension X or Y"
        # T is checked afterwards

        for axis in list(in_axes.keys()):
            if axis not in self.out_axes:  # An unexpected key
                if "T" not in in_axes:
                    in_axes["T"] = in_axes.pop(axis)  # Let's assume that it means T first
                elif "Z" not in in_axes:
                    in_axes["Z"] = in_axes.pop(axis)  # Let's assume that it means Z then
                else:
                    raise ValueError(f"Unable to parse the tiff axes: {axis} ({axes_squeezed}) not in {self.out_axes}.")

        if "T" not in in_axes:
            if "Z" not in in_axes:
                raise ValueError(f"Unable to find a temporal axis in {axes_squeezed}")

            in_axes["T"] = in_axes.pop("Z")  # Let's assume that Z is T

        assert in_axes["T"] < len(in_axes) - self._page_dim, "The temporal axis cannot be in the pages dimensions"

        return in_axes

    @property
    def axes(self) -> Iterable[str]:
        return (axis for axis, _ in sorted(self.in_axes.items(), key=lambda k_v: k_v[1]))

    # def _load(self):  # Old load without slices
    #     temporal_axis = self.in_axes["T"]
    #     shape = self.video._shape_squeezed  # pylint: disable=protected-access
    #     before = int(np.prod(shape[: -self._page_dim][:temporal_axis]))
    #     after = int(np.prod(shape[: -self._page_dim][temporal_axis + 1 :]))

    #     # Create a view on current
    #     current = self._current.reshape((before, after, *self.video.keyframe.shape))

    #     for i in range(before):
    #         for j in range(after):
    #             current[i, j] = self.video[(i * self.length + self.frame_id) * after + j].asarray()

    def _load(self):
        temporal_axis = self.in_axes["T"]

        true_shape = self.video._shape_squeezed  # pylint: disable=protected-access
        slices = tuple(slice(None) if axis not in self.ax_slice else self.ax_slice[axis] for axis in self.axes)

        # Compute stride and offset for the out of page dimensions
        strides = np.cumprod((1, *true_shape[-self._page_dim - 1 : 0 : -1]))[::-1]
        offset = np.array([0 if slice_.start is None else slice_.start for slice_ in slices])[: -self._page_dim]
        offset = (offset * strides).sum()
        strides *= np.array([1 if slice_.step is None else slice_.step for slice_ in slices])[: -self._page_dim]

        for index in np.ndindex(*self._current.shape[: -self._page_dim]):
            index_list = list(index)
            index_list.insert(temporal_axis, self.frame_id)

            i = int(offset + np.sum(np.array(index_list) * strides))
            self._current[index] = self.video[i].asarray()[slices[-self._page_dim :]]  # type: ignore

    def release(self) -> None:
        super().release()
        self.video.parent.close()

    def grab(self) -> bool:
        if self.frame_id + 1 >= self.length:
            return False

        self.frame_id += 1
        self._load()
        return True

    def seek(self, frame_id: int) -> None:
        if not 0 <= frame_id < self.length:
            raise EOFError(f"Seeking outside of video limits: {frame_id} not in [{0}, {self.length}[")

        if frame_id != self.frame_id:
            self.frame_id = frame_id
            self._load()

    def retrieve(self) -> np.ndarray:
        # Permute axes into [Z]YXC
        frame = self._current.transpose(
            tuple(
                self.in_axes[axis] - (self.in_axes[axis] > self.in_axes["T"])  # After T => -1 in axis as T is gone
                for axis in self.out_axes
                if axis in self.in_axes and axis != "T"
            )
        )

        if "C" not in self.in_axes:
            frame = frame[..., None]

        # Return a contiguous copy of self._current
        return frame.copy("C")


class FrameTiffLoader:  # pylint: disable=too-few-public-methods
    """Load a single frame stored in a TiffFile with tifffile.

    It handle 2D and 3D videos with any channels. Axes are inferred from the tifffile
    metadata and convert into ([D, ]H, W, C) (<=> [Z]YXC).

    We may not support all formats, or your specific metadata can be wrong/missing. In this case,
    you can also provide the expected axes of the tifffile using an ordered string.

    For example: "YX" for 2D videos without channel, "CZYX" for 3D videos with channels
    (ordered by channel then stack).

    Note:
        With tifffile syntax, we use X for width, Y for height, Z for depth and
        C/S for channels (C and S are not supported together). Any other letter
        (T, I, O, Q, ...) is either intepreted as Z if it is missing, or it will
        yield an error.

    It also supports to read the tiff at a specific resolution level.

    Attributes:
        out_axes (str): Axes order of the outputs
        level (int): Resolution level (if any)
            Default: 0 (finest level)
        axes (Optional[str]): Override the axes found in the tiff metadata.
            Default: None (Parse the metadata)

    """

    out_axes = "ZYXC"  # ZYXC = DHWC

    def __init__(self, level=0, axes: Optional[str] = None, ax_slice: Optional[Dict[str, slice]] = None):
        self.level = level
        self.axes = axes
        self.ax_slice = {} if ax_slice is None else ax_slice

        TiffVideoReader._check_slice(self.ax_slice, self.out_axes)

        assert self.out_axes[-1] == "C", "Channels have to be the last out axis"

        if self.axes:
            # Check that the provided axes are valid

            assert len(set(self.axes)) == len(self.axes), "Duplicated axes"
            assert "X" in self.axes and "Y" in self.axes, "Missing a spatial dimension X or Y"
            assert len(self.axes) <= 4, "The frame loader can only load up to 3D + C images"
            for axis in self.axes:
                assert axis in self.out_axes, f"Unknown axis: {axis} not in {self.out_axes}"

    def _extract_axes(self, serie: tifffile.TiffPageSeries) -> Dict[str, int]:
        if self.axes:
            return {axis: i for i, axis in enumerate(self.axes)}

        axes_squeezed = serie._axes_squeezed  # pylint: disable=protected-access
        axes = {axis: i for i, axis in enumerate(axes_squeezed)}
        if "S" in axes:
            if "C" in axes:
                raise ValueError("Unable to parse data with two channel dimensions (C and S)")
            axes["C"] = axes.pop("S")

        # Check that everything is fine
        assert len(axes) == len(axes_squeezed), "Duplicated axes"
        assert "X" in axes and "Y" in axes, "Missing a spatial dimension X or Y"
        assert len(serie.axes) <= 4, "The frame loader can only load up to 3D + C images"

        for axis in list(axes.keys()):
            if axis not in self.out_axes:  # An unexpected key
                if "Z" not in axes:
                    axes["Z"] = axes.pop(axis)  # Let's assume that it means Z
                else:
                    raise ValueError(f"Unable to parse the tiff axes: {axis} ({axes_squeezed}) not in {self.out_axes}.")

        return axes

    def __call__(self, path: Union[str, os.PathLike]) -> np.ndarray:
        tiff = tifffile.TiffFile(path)
        serie = tiff.series[0].levels[self.level]  # pylint: disable=unsubscriptable-object
        axes = self._extract_axes(serie)

        if not self.ax_slice:
            output = tiff.asarray(series=0, level=self.level, squeeze=True)
            assert output.shape == serie._shape_squeezed, "Unable to read at the expected shape"
        else:
            slices = tuple(
                slice(None) if axis not in self.ax_slice else self.ax_slice[axis]
                for axis in (axis for axis, _ in sorted(axes.items(), key=lambda k_v: k_v[1]))
            )
            output = self._read_slice(serie, slices)

        # Permute axes into [Z]YXC
        output = output.transpose(tuple(axes[axis] for axis in self.out_axes if axis in axes))

        if "C" not in axes:
            output = output[..., None]

        return output

    @staticmethod
    def _read_slice(serie: tifffile.TiffPageSeries, slices: Tuple[slice, ...]) -> np.ndarray:
        """Read the serie using the given slice"""

        page_dim = len(serie.keyframe.axes)

        true_shape = serie._shape_squeezed  # pylint: disable=protected-access
        final_shape = tuple(slice_length(slice_, shape) for slice_, shape in zip(slices, true_shape))

        # Compute stride and offset for the out of page dimensions
        strides = np.cumprod((1, *true_shape[-page_dim - 1 : 0 : -1]))[::-1]
        offset = np.array([0 if slice_.start is None else slice_.start for slice_ in slices])[:-page_dim]
        offset = (offset * strides).sum()
        strides *= np.array([1 if slice_.step is None else slice_.step for slice_ in slices])[:-page_dim]

        output = np.zeros(final_shape, dtype=serie.keyframe.dtype)

        for index in np.ndindex(*final_shape[:-page_dim]):
            i = int(offset + np.sum(np.array(index) * strides))
            output[index] = serie[i].asarray()[slices[-page_dim:]]  # type: ignore

        return output


def pil_loader(path: Union[str, os.PathLike]) -> np.ndarray:
    """Load an image with PIL. Shape: (H, W, C)

    It only supports 2D images
    """
    frame: np.ndarray = np.array(Image.open(path))

    if len(frame.shape) == 2:
        return frame[..., None]

    return frame


class MultiFrameReader(VideoReader):
    """Read video from a list of files inside a folder

    By default, it will find the alphanumerically sorted list of paths
    that shares the most common extension in the folder. The extension may be provided by the user.

    You can provide your own list of paths (absolute paths). The folder path is then ignored.

    Finally, you may also provide your own loading function to load each frame as a numpy array.

    See `VideoReader` for inherited attributes.

    Attributes:
        paths (List[pathlib.Path]): Sorted list of Paths to each frame of the video.
        frame_loader (Callable[[Union[str, os.PathLike]], np.ndarray]): Loads frame from their associated files.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        paths: Optional[List[Union[str, os.PathLike]]] = None,
        extension: Optional[str] = None,
        frame_loader: Optional[Callable[[Union[str, os.PathLike]], np.ndarray]] = None,
        **kwargs,
    ):
        super().__init__(path, **kwargs)

        if paths:
            self.paths = [pathlib.Path(path) for path in paths]
            assert len({path.suffix for path in self.paths}) == 1, "Found several extensions in paths"
        else:
            files = list(file for file in self.path.iterdir() if file.is_file())
            if extension is None:
                extensions: Dict[str, int] = {}
                for file in files:
                    extensions[file.suffix] = extensions.get(file.suffix, 0) + 1

                extension = max(extensions.items(), key=lambda k_v: k_v[1])[0]

            self.paths = utils.sorted_alphanumeric(file for file in files if file.suffix == extension)

            assert self.paths, "No frame found in the video"

        if frame_loader is None:
            if self.paths[0].suffix in (".tif", ".tiff"):
                frame_loader = FrameTiffLoader(**kwargs)
            else:
                frame_loader = pil_loader
        self.frame_loader = frame_loader

        self.length = len(self.paths)

        first_frame = self.retrieve()
        self.shape = first_frame.shape[:-1]
        self.channels = first_frame.shape[-1]

    def grab(self) -> bool:
        if self.frame_id + 1 >= self.length:
            return False

        self.frame_id += 1
        return True

    def seek(self, frame_id: int) -> None:
        if not 0 <= frame_id < self.length:
            raise EOFError(f"Seeking outside of video limits: {frame_id} not in [{0}, {self.length}[")

        self.frame_id = frame_id

    def retrieve(self) -> np.ndarray:
        # TODO: Do the read in grab/seek and store in a _current attribute to have a fast retrieve ?
        return self.frame_loader(self.paths[self.frame_id])
