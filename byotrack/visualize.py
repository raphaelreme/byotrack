import functools
from typing import Collection, Dict, Optional, Sequence, Tuple, Union
import warnings

import cv2
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import tqdm

import byotrack


_hsv = mpl.colormaps["hsv"]
_colors = list(map(lambda x: tuple(int(c * 255) for c in x[:3]), map(_hsv, [i / 200 for i in range(200)])))


def _convert_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Conversion from a floating or integer frame to uint8 displayable image

    Args:
        frame (np.ndarray): Video frame 2D (If 3D video, the caller should reduce the depth dimension)
            Integer (resp. Floating) frames are expected to be normalized in [0, 255] (resp. [0.0, 1.0])
            If this is not the case, a simple normalization is performed before conversion.
            Shape: (H, W, C), dtype: float | int

    Returns:
        np.ndarray: Displayable frame in uint8
            Shape: (H, W, C), dtype: uint8

    """
    if np.issubdtype(frame.dtype, np.floating):
        if frame.max() > 1.0:
            warnings.warn(
                "Floating videos are expected to be normalized in [0, 1].\n"
                "The simple normalization implemented here is suboptimal."
            )
            frame = frame / frame.max()
        frame = (frame * 255).round()
    else:
        if frame.max() > 255:
            warnings.warn(
                "Integer videos are expected to be normalized in [0, 255].\n"
                "The simple normalization implemented here is suboptimal."
            )

            frame = (frame * (255 / frame.max())).round()

    return frame.astype(np.uint8)


def display_lifetime(tracks: Collection[byotrack.Track]):
    """Display the lifetime of tracks

    Active tracks are in white. (Tracks on x-axis, Frames on y-axis)

    Args:
        tracks (Collection[byotrack.Track]): Tracks

    """
    start = min(track.start for track in tracks)
    tracks_tensor = byotrack.Track.tensorize(tracks)

    mask = (~torch.isnan(tracks_tensor).any(dim=2, keepdim=True)).numpy() * 255
    mask = np.concatenate([mask] * 3, axis=2)

    fig = plt.figure(figsize=(24, 16), dpi=100)
    plt.xlabel("Track id")
    plt.ylabel("Frame")
    plt.imshow(mask)
    # Hacky axe relabelling
    fig.axes[0].set_yticks(fig.axes[0].get_yticks()[1:-1])
    fig.axes[0].set_yticklabels(fig.axes[0].get_yticks().astype(np.int32) + start)
    plt.show()


def temporal_projection(
    tracks_tensor: torch.Tensor,
    colors: Sequence[Tuple[int, int, int]] = ((255, 255, 255),),
    background: Optional[np.ndarray] = None,
    color_by_time=False,
) -> np.ndarray:
    """Project all given tracks into a single image

    A track is displayed as the line of its consecutive positions. Track's undefined positions
    are not displayed.

    3d tracks and images are not supported

    Args:
        tracks_tensor (torch.Tensor): Tracks tensor (See `byotrack.Track.tensorize`)
            Positions of each track at each frame. (NaN if not defined).
            Shape: (T, N, 2)
        colors (Sequence[Tuple[int, int, int]]): Color sequence. When we color each track independently,
            each track is associated with a color (ith track with ith color, it loops if more tracks than colors).
            For time coloring, each frame has its own color (same across all tracks).
            Default: ((255, 255, 255),) (Everything is white)
        background (Optional[np.ndarray]): Optional frame to display behind the tracks.
            If not given, we set the background as 0, with the smallest size to contain all tracks.
            Default: None
            Shape: (H, W[, C])
        color_by_time (bool): If set to True, each frame has its own color. Otherwise, each track has its own color.
            Color by time is much slower (cv2 do not allow complex multilines draw)
            Default: False

    Returns:
        np.ndarray: Projection image
            Shape: (H, W, 3), dtype: np.uint8
    """

    is_defined = ~torch.isnan(tracks_tensor).any(dim=-1)
    frames = torch.arange(len(tracks_tensor))

    if background is not None:
        if background.ndim == 2:
            background = background[..., None]
        image = _convert_to_uint8(background)
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        offset = torch.zeros(2)
    else:
        offset = torch.min(tracks_tensor[is_defined], dim=0).values
        shape = (torch.max(tracks_tensor[is_defined], dim=0).values - offset).round().to(torch.int32) + 1
        image = np.zeros((*shape, 3), dtype=np.uint8)

    for track_i, points in enumerate(tracks_tensor.permute(1, 0, 2)):
        if not is_defined[:, track_i].any():  # No points for this track
            continue

        points = points[is_defined[:, track_i]] - offset

        if color_by_time:
            valid_frames = frames[is_defined[:, track_i]]

            previous = points[0]
            for frame_id, point in zip(valid_frames[1:], points[1:]):
                cv2.line(
                    image,
                    previous.numpy().round().astype(np.int32)[::-1],
                    point.numpy().round().astype(np.int32)[::-1],
                    colors[frame_id % len(colors)],
                )
                previous = point
        else:
            cv2.polylines(
                image,
                points.numpy().round().astype(np.int32)[None, :, None, ::-1],
                False,
                colors[track_i % len(colors)],
            )

    return image


# TODO: Add a display where we can see the tracks segmentation ?
# And the detections with colors picked like for tracks ?
class InteractiveVisualizer:  # pylint: disable=too-many-instance-attributes
    """Interactive visualization with opencv

    Supports 3D stacks (but you will only see one plane at a time in 2D).

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * b/n: Move backward/forward in depth (Z axis)
        * d: Switch detections display mode (Not displayed, Mask, Segmentation) if available
        * t: Switch on/off the display of tracks if available
        * v: Switch on/off the display of the video

    Keys can be modified in the dict `keys` (PAUSE, UP, DOWN, RIGHT, LEFT, DET, TRA, VID).

    Attributes:
        video (Sequence[np.ndarray] | np.ndarray): Optional video to display.
            Shape: (T, [D, ], H, W, C)
            Default: () (no frames)
        detections_sequence (Sequence[byotrack.Detections]): Optional detections to display
            Default: () (no detections)
        tracks (Collection[byotrack.Tracks]): Optional tracks to display
            Default: () (no tracks)
        frame_shape (Tuple[int, int]): Shape of frames
        n_frames (int): Number of frames
        tracks_colors (List[Tuple[int, int, int]]): Colors to use for tracks
            Colors are picked circularly given their position in the `tracks` list.
            By default we use the hsv colormaps from matplotlib
        scale (int): Up or Down scale the display.
            Default: 1 (no scaling)
    """

    # KEY MAPPING
    # TODO: Use arrow or adapt to keyboard ?
    keys = {
        "QUIT": "q",
        "PAUSE": " ",
        "UP": "n",
        "DOWN": "b",
        "RIGHT": "x",
        "LEFT": "w",
        "DET": "d",
        "TRA": "t",
        "VID": "v",
    }

    window_name = "ByoTrack ViZ"

    def __init__(
        self,
        video: Union[Sequence[np.ndarray], np.ndarray] = (),
        detections_sequence: Sequence[byotrack.Detections] = (),
        tracks: Collection[byotrack.Track] = (),
    ) -> None:
        assert len(video) != 0 or detections_sequence or tracks, "No data to display"

        self.video = video
        self.detections_sequence = detections_sequence
        self.tracks = tracks

        self.frame_shape = self._get_frame_shape()
        self.n_frames = self._get_n_frames()

        self.tracks_colors = _colors[:: max(1, len(_colors) // (len(tracks) + 1))]
        self.scale = 1
        self.interpolation = cv2.INTER_NEAREST

        self._frame_id = 0
        self._video_frame = self.video[self._frame_id] if len(self.video) != 0 else np.zeros((*self.frame_shape, 0))
        self._stack_id = 0 if len(self.frame_shape) != 3 else self.frame_shape[0] // 2
        self._display_video = int(len(video) != 0)
        self._display_detections = int(bool(detections_sequence))
        self._display_tracks = int(bool(tracks))
        self._running = False

    def run(self, frame_id=0, fps=20) -> None:
        """Run the visualization

        Args:
            frame_id (int): Starting frame_id
            fps (int): Frame rate
        """
        try:
            self._frame_id = frame_id
            self._run(fps)
        finally:
            cv2.destroyWindow(self.window_name)

    def _run(self, fps=20) -> None:  # pylint: disable=too-many-branches,too-many-statements
        self._video_frame = self.video[self._frame_id] if len(self.video) != 0 else np.zeros((*self.frame_shape, 0))
        while True:
            frame = np.zeros((*self.frame_shape[-2:], 3), dtype=np.uint8)

            if self._display_video and len(self.video) != 0:
                _frame = self._video_frame
                _frame = _frame[self._stack_id] if len(self.frame_shape) == 3 else _frame
                frame[:] = _convert_to_uint8(_frame)  # We only support grayscale (C=1) and RGB (C=3)

            if self._display_detections and 0 <= self._frame_id < len(self.detections_sequence):
                frame = self.draw_segmentation(frame)

            if self.scale != 1:
                frame = cv2.resize(  # type: ignore
                    frame, None, fx=self.scale, fy=self.scale, interpolation=self.interpolation
                )

            if self._display_tracks:
                frame = self.draw_tracks(frame)

            # Display the resulting frame
            cv2.imshow(self.window_name, np.flip(frame, 2))

            title = f"Frame {self._frame_id} / {self.n_frames}"
            if len(self.frame_shape) == 3:
                title += f", Stack {self._stack_id} / {self.frame_shape[0]}"
            cv2.setWindowTitle(self.window_name, title)

            # Handle user actions
            key = cv2.waitKey(1000 // fps) & 0xFF
            if self.handle_actions(key):
                break

    def draw_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Draw the segmentation on the frame

        It will draw the segmentation for `frame_id` and `stack_id`.

        Args:
            frame (np.ndarray): The 2D frame to draw on (will not be modified)
                Shape: (H, W), dtype: np.uint8

        Returns:
            np.ndarray: The resulting frame with the segmentation drawned on
        """
        segmentation = np.zeros_like(frame)
        if self.detections_sequence[self._frame_id].dim == 3:
            if self._stack_id < self.detections_sequence[self._frame_id].shape[0]:
                segmentation_ = self.detections_sequence[self._frame_id].segmentation[self._stack_id].clone().numpy()
            else:
                # Out of the 3D segmentation. Then, seg is simply 0
                segmentation_ = np.zeros_like(frame)[..., 0]
        else:
            segmentation_ = self.detections_sequence[self._frame_id].segmentation.clone().numpy()

        if self._display_detections == 1:  # Display segmentation as mask
            segmentation_ = segmentation_ != 0
            segmentation_ = segmentation_.astype(np.uint8) * 255
        else:
            mask = segmentation_ == 0
            segmentation_ = (segmentation_ % 206) + 50
            segmentation_[mask] = 0
            segmentation_ = segmentation_.astype(np.uint8)

        # TODO: Add another mode to see detections by color or with circle ?
        # Draw in blue and mix with the frame if _display_video
        segmentation[: segmentation_.shape[0], : segmentation_.shape[1], 2] = segmentation_[
            : frame.shape[0], : frame.shape[1]
        ]

        if self._display_video:
            frame = frame / 2 + segmentation / 2
            frame = frame.astype(np.uint8)
        else:
            frame = segmentation

        return frame

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw the tracks on the frame

        It will draw the tracks for `frame_id` and `stack_id`.

        Args:
            frame (np.ndarray): The 2D frame to draw on (will not be modified)
                Shape: (H, W), dtype: np.uint8

        Returns:
            np.ndarray: The resulting frame with the drawn tracks
        """
        frame = frame.copy()  # Do not draw inplace
        for track in self.tracks:
            point = track[self._frame_id] * self.scale
            if torch.isnan(point).any():
                continue

            if len(point) == 3:
                if (point[0] / self.scale - self._stack_id).abs() > 5:
                    continue  # Do not display tracks that are more than 5 stacks away

                point = point[1:]  # Remove depth axis

            i, j = point.round().to(torch.int).tolist()

            color = self.tracks_colors[track.identifier % len(self.tracks_colors)]

            cv2.circle(frame, (j, i), 5, color)
            cv2.putText(frame, str(track.identifier % 100), (j + 4, i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

        return frame

    def handle_actions(self, key: int) -> bool:  # pylint: disable=too-many-branches
        """Handle inputs from user

        Return True to quit

        Args:
            key (int): Key input from user

        Returns:
            bool: True to quit visualization
        """
        if key == ord(self.keys["QUIT"]):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(self.keys["PAUSE"]):
            self._running = not self._running

        old_frame_id = self._frame_id

        if not self._running and key == ord(self.keys["LEFT"]):  # Prev
            self._frame_id = (self._frame_id - 1) % self.n_frames

        if not self._running and key == ord(self.keys["RIGHT"]):  # Next
            self._frame_id = (self._frame_id + 1) % self.n_frames

        if self._running:  # Stop running if we reach the last frame
            self._frame_id += 1
            if self._frame_id >= self.n_frames:
                self._running = False
                self._frame_id = self.n_frames - 1

        if self._frame_id != old_frame_id and len(self.video) != 0:
            self._video_frame = self.video[self._frame_id]  # Read video only once when we change change frame_id

        if len(self.frame_shape) == 3 and key == ord(self.keys["DOWN"]):
            self._stack_id = (self._stack_id - 1) % self.frame_shape[0]

        if len(self.frame_shape) == 3 and key == ord(self.keys["UP"]):
            self._stack_id = (self._stack_id + 1) % self.frame_shape[0]

        if key == ord(self.keys["DET"]):
            self._display_detections = (self._display_detections + 1) % 3

        if key == ord(self.keys["TRA"]):
            self._display_tracks = 1 - self._display_tracks

        if key == ord(self.keys["VID"]):
            self._display_video = 1 - self._display_video

        return False

    def _get_frame_shape(self) -> Tuple[int, ...]:
        """Find the frame shape to use for the visualization

        It uses the shape of the video is given. Otherwise falls back to
        the max detection shape.

        If no video, neither detection are given, it is extrapolated as
        the maximum values (+ some margin) in tracks positions.
        """
        dim = 0
        if len(self.video) != 0:
            frame_shape = self.video[0].shape[:-1]
            dim = len(frame_shape)

        if self.detections_sequence:
            if dim:  # Shape is already defined from the video
                # If 3D video, with 2D detections, the vizualizer will still work by extending the 2D dets
                assert self.detections_sequence[0].dim <= dim, "Unable to handle 3D detections with 2D videos"
            else:
                frame_shape = self.detections_sequence[0].shape

                # Find the maximal frame shape in the detections
                def _reduce(reduced: Tuple[int, ...], shape: Tuple[int, ...]) -> Tuple[int, ...]:
                    assert len(reduced) == len(shape), "Detections should share the same dimension"
                    return tuple(max(s_r, s) for s_r, s in zip(reduced, shape))

                frame_shape = functools.reduce(
                    _reduce, (detections.shape for detections in self.detections_sequence), frame_shape
                )
                dim = len(frame_shape)

        if self.tracks:
            track_tensor = byotrack.Track.tensorize(self.tracks)
            positions = track_tensor[~torch.isnan(track_tensor).any(dim=2)]
            minimum: torch.Tensor = positions.min(dim=0).values.round()
            maximum: torch.Tensor = positions.max(dim=0).values.round()

            if not dim:
                frame_shape = tuple(map(int, maximum + 20))  # Add some margin
                dim = len(frame_shape)

            if (minimum < 0).any() or (maximum[:dim] >= torch.tensor(frame_shape)[:dim]).any():
                print("Some tracks are out of frame")

        return frame_shape

    def _get_n_frames(self) -> int:
        if len(self.video) != 0:
            return len(self.video)

        n_frames = 1
        if self.tracks:
            n_frames = max(n_frames, max(track.start + len(track) for track in self.tracks))

        if self.detections_sequence:
            n_frames = max(n_frames, len(self.detections_sequence))

        return n_frames


class InteractiveFlowVisualizer:  # pylint: disable=too-many-instance-attributes
    """Interactive optical flow visualization using a moving grid above the video

    The video is displayed with a uniform grid (control points) above that moves according to the computed flow map.
    If the optical flow algorithm is good, the points should moves along the video.

    It only supports 2D videos.

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * g: Reset the control points as a grid

    Keys can be modified in the dict `keys` (PAUSE, RIGHT, LEFT, GRID).

    Note:
        To prevent lagging, even when precompute is False, optical flow map are saved and never computed twice.
        You should not use this function with too large videos. (Use slicing to reduce the size of a video)

    Attributes:
        video (Sequence[np.ndarray] | np.ndarray): The video of interest (2D)
            Shape: (T, H, W, C)
        optflow (byotrack.OpticalFlow): Optical flow algorithm to visualize
        grid_step (int): size (in pixels) between control points
        points (np.ndarray): Control points (grid)
            Shape: (N, 2), dtype: float64
        precompute (bool): Compute all the flows first (bi directionnal) to prevent lagging
            Default: False
        tracks_colors (List[Tuple[int, int, int]]): Colors to use for tracks
            Colors are picked circularly given the track identifier.
            By default we use the hsv colormaps from matplotlib
        scale (int): Up or Down scale the display.
            Default: 1 (no scaling)
    """

    # KEY MAPPING
    keys = {
        "QUIT": "q",
        "PAUSE": " ",
        "RIGHT": "x",
        "LEFT": "w",
        "GRID": "g",
    }

    window_name = "ByoTrack FlowViz"

    def __init__(
        self,
        video: Union[Sequence[np.ndarray], np.ndarray],
        optflow: byotrack.OpticalFlow,
        grid_step=20,
        precompute=False,
    ):
        self.video = video
        self.optflow = optflow
        self.grid_step = grid_step
        self.precompute = precompute
        # Cache
        self._flow_maps: Dict[Tuple[int, int], np.ndarray] = {}

        self.frame_shape = video[0].shape[:-1]
        if len(self.frame_shape) != 2:
            raise ValueError("This visualization only supports 2D videos")

        self.n_frames = len(video)
        self.colors = _colors
        self.scale = 1
        self._grid = (
            np.indices(self.frame_shape, dtype=np.float64)[:, ::grid_step, ::grid_step].reshape(2, -1).transpose()
        )
        self.points = self._grid.copy()

        self._frame_id = 0
        self._running = False

        if precompute:
            self._precompute()

    def _precompute(self):
        """Precompute all flow maps (forward and backward)"""
        src = self.optflow.preprocess(self.video[0])

        for frame_id, frame in enumerate(tqdm.tqdm(self.video[1:])):
            dst = self.optflow.preprocess(frame)

            self._flow_maps[(frame_id, frame_id + 1)] = self.optflow.compute(src, dst)
            self._flow_maps[(frame_id + 1, frame_id)] = self.optflow.compute(dst, src)

            src = dst

    def run(self, frame_id=0, fps=20) -> None:
        """Run the visualization

        Args:
            frame_id (int): Starting frame_id
            fps (int): Frame rate
        """
        try:
            self._frame_id = frame_id
            self._run(fps)
        finally:
            cv2.destroyWindow(self.window_name)

    def _run(self, fps=20) -> None:
        self.points = self._grid

        src_frame_id = self._frame_id
        src = self.optflow.preprocess(self.video[self._frame_id])
        dst = src

        while True:
            self._frame_id += self._running

            if self._frame_id >= self.n_frames:
                self._running = False
                self._frame_id = self.n_frames - 1

            frame = self.video[self._frame_id]

            if src_frame_id != self._frame_id:  # Change of frame, let's update
                dst = self.optflow.preprocess(frame)
                if (src_frame_id, self._frame_id) not in self._flow_maps:
                    self._flow_maps[(src_frame_id, self._frame_id)] = self.optflow.compute(src, dst)

                self.points = self.optflow.transform(self._flow_maps[(src_frame_id, self._frame_id)], self.points)

                src = dst
                src_frame_id = self._frame_id

            # Drawing
            draw = _convert_to_uint8(frame)
            assert draw.shape[2] in (1, 3), "We only support Grayscale images (C=1) or RGB images (C=3)"

            if draw.shape[2] == 1:
                draw = np.concatenate([draw] * 3, axis=2)

            if self.scale != 1:
                draw = cv2.resize(draw, None, fx=self.scale, fy=self.scale)

            for k, (i, j) in enumerate(self.points * self.scale):
                cv2.circle(draw, (round(j), round(i)), 3, self.colors[k % len(self.colors)])

            # Display the resulting frame
            cv2.imshow(self.window_name, np.flip(draw, 2))
            cv2.setWindowTitle(self.window_name, f"Frame {self._frame_id} / {self.n_frames}")

            # Handle user actions
            key = cv2.waitKey(1000 // fps) & 0xFF
            if self.handle_actions(key):
                break

    def handle_actions(self, key: int) -> bool:
        """Handle inputs from user

        Return True to quit

        Args:
            key (int): Key input from user

        Returns:
            bool: True to quit visualization
        """
        if key == ord(self.keys["QUIT"]):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(self.keys["PAUSE"]):
            self._running = not self._running

        if not self._running and key == ord(self.keys["LEFT"]):  # Prev
            self._frame_id = (self._frame_id - 1) % self.n_frames

        if not self._running and key == ord(self.keys["RIGHT"]):  # Next
            self._frame_id = (self._frame_id + 1) % self.n_frames

        if key == ord(self.keys["GRID"]):
            self.points = self._grid

        return False


def flow_to_rgb(flow: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
    """Converts a 2D optical flow into a RBG map

    The angle is converted into a color, the magnitude into luminosity.
    (Follow what is done by OpenCV for visualization)

    See `get_flow_wheel` to get the color wheel for each direction.

    Args:
        flow (np.ndarray): Flow to convert to rgb
            Shape: (2, H, W)
        clip (Optional[float]): Clip the magnitude to this maximum value
            Default: None

    Returns:
        np.ndarray: RGB flow visualization
            Shape: (H, W, 3)

    """
    hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)

    magnitude, angle = cv2.cartToPolar(flow[1], flow[0])
    if clip is not None:
        magnitude = np.clip(magnitude, 0, clip)
    else:
        clip = magnitude.max()

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = magnitude / clip * 255

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_flow_wheel(size=128, mask=True) -> np.ndarray:
    """Build the color wheel that is used for `flow_to_rgb`

    Args:
        size (int): The generated wheel will have 2 * size + 1 width and height.
            Default: 128
        mask (bool): Mask values outside of the circle centered on the middle.
            Default: True

    Returns:
        np.ndarray: Color wheel
            Shape: (2 * size + 1, 2 * size + 1, 3)

    """
    wheel = np.indices((2 * size, 2 * size), dtype=np.float32) - size

    if mask:
        mask = np.sqrt(np.sum(wheel**2, axis=0)) > size
        wheel[:, mask] = 0

    return flow_to_rgb(wheel, size)
