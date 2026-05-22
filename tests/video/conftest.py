from __future__ import annotations

import cv2
import numpy as np
import PIL.Image
import pytest
import tifffile


@pytest.fixture
def video_2d() -> np.ndarray:
    return np.random.randint(500, 3000, (10, 20, 30, 3), dtype=np.uint16)


@pytest.fixture
def video_3d() -> np.ndarray:
    return np.random.randint(0, 256, (8, 5, 20, 30, 2), dtype=np.uint8)


@pytest.fixture
def tiff_2d(tmp_path, video_2d):
    """2D video as TIFF with axes TYXC."""
    path = tmp_path / "video_2d.tiff"
    tifffile.imwrite(path, video_2d, photometric="rgb", metadata={"axes": "TYXC"})
    return path, video_2d


@pytest.fixture
def tiff_3d(tmp_path, video_3d):
    """3D video as TIFF with axes TZYXC."""
    path = tmp_path / "video_3d.tiff"
    tifffile.imwrite(path, video_3d, metadata={"axes": "TZYXC"})
    return path, video_3d


@pytest.fixture
def pil_tiff_gray(tmp_path, video_2d):
    """Multi-frame grayscale TIFF written with PIL (uses first channel of video_2d)."""
    data = video_2d[..., 0].astype(np.uint8)  # (T, H, W)
    path = tmp_path / "pil_gray.tiff"
    frames = [PIL.Image.fromarray(data[i]) for i in range(len(data))]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    return path, data[..., None]  # expected shape: (T, H, W, 1)


@pytest.fixture
def pil_tiff_rgb(tmp_path):
    """Multi-frame RGB TIFF written with PIL."""
    data = np.random.randint(0, 256, (5, 10, 12, 3), dtype=np.uint8)
    path = tmp_path / "pil_rgb.tiff"
    frames = [PIL.Image.fromarray(data[i]) for i in range(len(data))]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    return path, data


@pytest.fixture
def png_folder(tmp_path):
    """Folder of PNG frames (RGB uint8)."""
    data = np.random.randint(0, 256, (5, 10, 12, 3), dtype=np.uint8)
    folder = tmp_path / "frames"
    folder.mkdir()
    for i, frame in enumerate(data):
        PIL.Image.fromarray(frame).save(folder / f"frame_{i:03d}.png")
    return folder, data


@pytest.fixture
def avi_2d(tmp_path):
    """Small RGB AVI written with cv2 MJPG codec."""
    data = np.random.randint(50, 200, (5, 10, 12, 3), dtype=np.uint8)
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (12, 10))  # type: ignore[attr-defined]
    for frame in data:
        writer.write(frame[..., ::-1])  # RGB→BGR for cv2
    writer.release()
    return path, data


@pytest.fixture
def avi_gray(tmp_path):
    """Small grayscale AVI written with cv2 MJPG codec."""
    data = np.random.randint(50, 200, (5, 10, 12), dtype=np.uint8)
    path = tmp_path / "gray.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (12, 10), isColor=False)  # type: ignore[attr-defined]
    for frame in data:
        writer.write(frame)
    writer.release()
    return path, data
