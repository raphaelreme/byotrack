from typing import Collection, Dict, Optional, Sequence, Tuple, Union

import cv2
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import tqdm

import byotrack

_hsv = mpl.colormaps["hsv"]
_colors = list(map(lambda x: tuple(int(c * 255) for c in x[:3]), map(_hsv, [i / 200 for i in range(200)])))


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
        if np.issubdtype(background.dtype, np.floating):
            background = (background * 255).round().astype(np.uint8)
        if background.ndim == 2:
            background = background[..., None]
        if background.shape[2] == 1:  # type: ignore  # (Mypy-1.4.1 bugs in python3.7)
            background = np.concatenate([background] * 3, axis=-1)
        image = np.asarray(background)  # Mypy is lost also here...
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


class InteractiveVisualizer:  # pylint: disable=too-many-instance-attributes
    """Interactive visualization with opencv

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * d: Switch detections display mode (Not displayed, Mask, Segmentation) if available
        * t: Switch on/off the display of tracks if available
        * v: Switch on/off the display of the video

    Attributes:
        video (Sequence[np.ndarray] | np.ndarray): Optional video to display.
            Default: () (no frames)
        detections_sequence (Sequence[byotrack.Detections]): Optional detections to display
            Default: () (no detections)
        tracks (Collection[byotrack.Tracks]): Optional tracks to display
            Default: () (no tracks)
        frame_shape (Tuple[int, int]): Shape of frames
        n_frames (int): Number of frames
        tracks_colors (List[Tuple[int, int, int]]): Colors to use for tracks
            Colors are picked circularly given the track identifier.
            By default we use the hsv colormaps from matplotlib
        scale (int): Up or Down scale the display.
            Default: 1 (no scaling)
    """

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
        self.tracks_colors = _colors
        self.scale = 1

        self._frame_id = 0
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

    def _run(self, fps=20) -> None:  # pylint: disable=too-many-branches
        while True:
            frame = np.zeros((*self.frame_shape, 3), dtype=np.uint8)

            if self._display_video and len(self.video) != 0:
                _frame = self.video[self._frame_id]
                _frame = _frame * 255 if np.issubdtype(_frame.dtype, np.floating) else _frame
                frame = _frame.astype(np.uint8)
                assert frame.shape[2] in (1, 3)

                if frame.shape[2] == 1:
                    frame = np.concatenate([frame] * 3, axis=2)

            if self._display_detections and 0 <= self._frame_id < len(self.detections_sequence):
                segmentation: np.ndarray = self.detections_sequence[self._frame_id].segmentation.clone().numpy()
                if self._display_detections == 1:  # Display segmentation as mask
                    segmentation = segmentation != 0
                    segmentation = segmentation.astype(np.uint8)[..., None] * 255
                else:
                    segmentation = (segmentation % 206) + 50
                    segmentation[self.detections_sequence[self._frame_id].segmentation == 0] = 0
                    segmentation = segmentation.astype(np.uint8)[..., None]

                segmentation = np.concatenate(
                    [segmentation, np.zeros_like(segmentation), np.zeros_like(segmentation)], axis=2
                )

                if self._display_video:
                    frame = frame / 2 + segmentation / 2
                    frame = frame.astype(np.uint8)
                else:
                    frame = segmentation

            if self.scale != 1:
                frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)  # type: ignore

            if self._display_tracks:
                for track in self.tracks:
                    point = track[self._frame_id] * self.scale
                    if torch.isnan(point).any():
                        continue

                    i, j = point.round().to(torch.int).tolist()

                    color = self.tracks_colors[track.identifier % len(self.tracks_colors)]

                    cv2.circle(frame, (j, i), 5, color)
                    cv2.putText(
                        frame, str(track.identifier % 100), (j + 4, i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color
                    )

            # Display the resulting frame
            cv2.imshow(self.window_name, frame)
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
        if key == ord("q"):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(" "):
            self._running = not self._running

        if not self._running and key == ord("w"):  # Prev
            self._frame_id = (self._frame_id - 1) % self.n_frames

        if not self._running and key == ord("x"):  # Next
            self._frame_id = (self._frame_id + 1) % self.n_frames

        if self._running:  # Stop running if we reach the last frame
            self._frame_id += 1
            if self._frame_id >= self.n_frames:
                self._running = False
                self._frame_id = self.n_frames - 1

        if key == ord("d"):
            self._display_detections = (self._display_detections + 1) % 3

        if key == ord("t"):
            self._display_tracks = 1 - self._display_tracks

        if key == ord("v"):
            self._display_video = 1 - self._display_video

        return False

    def _get_frame_shape(self) -> Tuple[int, int]:
        frame_shape: Tuple[int, int] = (1, 1)
        if len(self.video) != 0:
            frame_shape = np.broadcast_shapes(frame_shape, self.video[0].shape[:2])  # type: ignore
        if self.detections_sequence:
            frame_shape = np.broadcast_shapes(frame_shape, self.detections_sequence[0].shape)  # type: ignore
        if self.tracks:
            track_tensor = byotrack.Track.tensorize(self.tracks)
            positions = track_tensor[~torch.isnan(track_tensor).any(dim=2)]
            minimum: torch.Tensor = positions.min(dim=0).values.round()
            maximum: torch.Tensor = positions.max(dim=0).values.round()

            if frame_shape == (1, 1):
                size = (maximum + 20).to(torch.int).tolist()  # Add some margin
                frame_shape = (size[0], size[1])

            if (minimum < 0).any() or maximum[0] >= frame_shape[0] or maximum[1] >= frame_shape[1]:
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

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * g: Reset the control points as a grid

    Note:
        To prevent lagging, even when precompute is False, optical flow map are saved and never computed twice.
        You should not use this function with too large videos. (Use slicing to reduce the size of a video)

    Attributes:
        video (Sequence[np.ndarray] | np.ndarray): The video of interest
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

        self.frame_shape = video[0].shape[:2]
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

            frame = self.video[self._frame_id]

            if src_frame_id != self._frame_id:  # Change of frame, let's update
                dst = self.optflow.preprocess(frame)
                if (src_frame_id, self._frame_id) not in self._flow_maps:
                    self._flow_maps[(src_frame_id, self._frame_id)] = self.optflow.compute(src, dst)

                self.points = self.optflow.transform(self._flow_maps[(src_frame_id, self._frame_id)], self.points)

                src = dst
                src_frame_id = self._frame_id

            # Drawing
            draw = frame * 255 if np.issubdtype(frame.dtype, np.floating) else frame
            draw = draw.astype(np.uint8)
            assert draw.shape[2] in (1, 3)

            if draw.shape[2] == 1:
                draw = np.concatenate([draw] * 3, axis=2)

            if self.scale != 1:
                draw = cv2.resize(draw, None, fx=self.scale, fy=self.scale)

            for k, (i, j) in enumerate(self.points * self.scale):
                cv2.circle(draw, (round(j), round(i)), 3, self.colors[k % len(self.colors)])

            # Display the resulting frame
            cv2.imshow(self.window_name, draw)
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
        if key == ord("q"):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(" "):
            self._running = not self._running

        if not self._running and key == ord("w"):  # Prev
            self._frame_id = (self._frame_id - 1) % self.n_frames

        if not self._running and key == ord("x"):  # Next
            self._frame_id = (self._frame_id + 1) % self.n_frames

        if key == ord("g"):
            self.points = self._grid

        return False
