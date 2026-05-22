from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import PIL.Image
import pytest
import tifffile

from byotrack.video.reader import (
    ArrayVideoReader,
    FrameTiffLoader,
    MultiFrameReader,
    OpenCVVideoReader,
    PILVideoReader,
    TiffVideoReader,
    VideoReader,
    pil_loader,
    slice_length,
)

if TYPE_CHECKING:
    import pathlib


## slice_length


def test_slice_length_empty():
    assert slice_length(slice(5, 5), 10) == 0
    assert slice_length(slice(10, 0), 10) == 0
    assert slice_length(slice(0, 0, 1), 10) == 0


def test_slice_length_forward():
    assert slice_length(slice(None), 10) == 10
    assert slice_length(slice(2, 8), 10) == 6
    assert slice_length(slice(0, 10), 10) == 10


def test_slice_length_reverse():
    assert slice_length(slice(9, None, -1), 10) == 10
    assert slice_length(slice(9, 4, -2), 10) == 3


def test_slice_length_step():
    assert slice_length(slice(0, 10, 2), 10) == 5
    assert slice_length(slice(1, 10, 3), 10) == 3


## VideoReader.open() factory / MetaVideoReader registry


def test_video_reader_open_tiff_uses_tiff_reader(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    reader = VideoReader.open(path)
    assert isinstance(reader, TiffVideoReader)


def test_video_reader_open_unknown_extension_uses_opencv(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = VideoReader.open(path)
    assert isinstance(reader, OpenCVVideoReader)


def test_video_reader_open_directory_uses_multi_frame_reader(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, _ = png_folder
    reader = VideoReader.open(folder)
    assert isinstance(reader, MultiFrameReader)
    assert reader.frame_loader is pil_loader


## ArrayVideoReader


def test_array_reader_2d_attributes():
    data = np.random.randint(0, 256, (10, 20, 30, 3), dtype=np.uint8)
    reader = ArrayVideoReader("", data)
    assert reader.length == 10
    assert reader.shape == (20, 30)
    assert reader.channels == 3
    assert reader.dtype == np.uint8
    assert reader.frame_id == 0


def test_array_reader_3d_attributes():
    data = np.random.randint(0, 256, (8, 5, 20, 30, 2), dtype=np.uint16)
    reader = ArrayVideoReader("", data)
    assert reader.length == 8
    assert reader.shape == (5, 20, 30)
    assert reader.channels == 2
    assert reader.dtype == np.uint16
    assert reader.frame_id == 0


def test_array_reader_2d_warns_no_channel():
    data = np.random.randint(0, 256, (5, 10, 12), dtype=np.uint8)  # 2D without channels
    with pytest.warns(UserWarning, match="Channel dimension not found"):
        reader = ArrayVideoReader("", data)

    assert reader.channels == 1
    assert reader.shape == (10, 12)

    assert reader.retrieve().shape == (10, 12, 1)


def test_array_reader_2d_with_many_channels_warns_and_convert_to_3d_single_channel():
    data = np.random.randint(0, 256, (3, 10, 12, 60), dtype=np.uint8)  # 3D without channels is converted
    with pytest.warns(UserWarning, match="channel dimension is missing"):
        reader = ArrayVideoReader("", data)

    # The large channel dim has been converted into a spatial dim.
    assert reader.channels == 1
    assert reader.shape == (10, 12, 60)

    assert reader.retrieve().shape == (10, 12, 60, 1)


def test_array_reader_3d_with_many_channels_is_not_converted():
    data = np.random.randint(0, 256, (3, 5, 10, 12, 60), dtype=np.uint8)  # 3D with many channels
    reader = ArrayVideoReader("", data)

    assert reader.channels == 60
    assert reader.shape == (5, 10, 12)

    assert reader.retrieve().shape == (5, 10, 12, 60)


def test_array_reader_nonempty_path_warns():
    data = np.zeros((3, 4, 4, 1), dtype=np.uint8)
    with pytest.warns(UserWarning, match="will not use the given path"):
        ArrayVideoReader("some/path", data)


def test_array_reader_too_few_dims_raises():
    with pytest.raises(ValueError, match="at least"):
        ArrayVideoReader("", np.zeros((10, 10), dtype=np.uint8))


def test_array_reader_too_many_dims_raises():
    with pytest.raises(ValueError, match="at most"):
        ArrayVideoReader("", np.zeros((2, 2, 2, 2, 2, 2), dtype=np.uint8))


def test_array_reader_grab_advances():
    data = np.arange(24, dtype=np.uint8).reshape(3, 2, 4, 1)
    reader = ArrayVideoReader("", data)
    assert reader.frame_id == 0
    assert reader.grab() is True
    assert reader.frame_id == 1
    assert reader.grab() is True
    assert reader.frame_id == 2
    assert reader.grab() is False  # at last frame
    assert reader.frame_id == 2  # stays put


def test_array_reader_seek_valid():
    data = np.arange(24, dtype=np.uint8).reshape(4, 2, 3, 1)
    reader = ArrayVideoReader("", data)
    reader.seek(3)
    assert reader.frame_id == 3
    assert (reader.retrieve() == data[3]).all()


def test_array_reader_seek_same_frame_noop():
    data = np.zeros((4, 2, 3, 1), dtype=np.uint8)
    reader = ArrayVideoReader("", data)
    reader.seek(2)
    current_before = reader._current
    reader.seek(2)  # no-op
    assert reader._current is current_before


def test_array_reader_seek_out_of_range_raises():
    reader = ArrayVideoReader("", np.zeros((3, 4, 4, 1), dtype=np.uint8))
    with pytest.raises(EOFError):
        reader.seek(3)
    with pytest.raises(EOFError):
        reader.seek(-1)


def test_array_reader_retrieve_2d_copy():
    data = np.ones((3, 4, 4, 2), dtype=np.uint16) * 42
    reader = ArrayVideoReader("", data)
    frame = reader.retrieve()

    assert frame.shape == (4, 4, 2)
    assert frame.dtype == np.uint16
    assert (frame == data[0]).all()

    # Verify it's a copy
    frame[:] = 0
    assert (reader.retrieve() == data[0]).all()


def test_array_reader_read():
    data = np.zeros((3, 4, 4, 1), dtype=np.uint8)
    reader = ArrayVideoReader("", data)
    frame, has_next = reader.read()

    assert frame.shape == (4, 4, 1)
    assert has_next is True
    assert reader.frame_id == 1

    # Read last frame
    reader.seek(2)
    _, has_next = reader.read()

    assert has_next is False


def test_array_reader_tell():
    reader = ArrayVideoReader("", np.zeros((3, 4, 4, 1), dtype=np.uint8))
    assert reader.tell() == 0

    reader.seek(2)
    assert reader.tell() == 2


def test_array_reader_release():
    data = np.zeros((3, 4, 4, 1), dtype=np.uint8)
    reader = ArrayVideoReader("", data)
    reader.release()

    assert reader.released is True
    assert not hasattr(reader, "video")


## OpenCVVideoReader


def test_opencv_reader_attributes(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, data = avi_2d
    reader = OpenCVVideoReader(path)

    assert reader.length == len(data)
    assert reader.shape == (10, 12)  # H, W
    assert reader.channels == 3
    assert reader.dtype == np.dtype(np.uint8)


def test_opencv_reader_retrieve_rgb(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = OpenCVVideoReader(path)
    frame = reader.retrieve()

    assert frame.shape == (10, 12, 3)
    assert frame.dtype == np.uint8


def test_opencv_reader_grab_advances(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = OpenCVVideoReader(path)

    for i in range(reader.length - 1):
        assert reader.grab() is True
        assert reader.frame_id == i + 1
    assert reader.grab() is False


def test_opencv_reader_seek_valid(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = OpenCVVideoReader(path)
    reader.seek(3)

    assert reader.frame_id == 3


def test_opencv_reader_seek_out_of_range_raises(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = OpenCVVideoReader(path)

    with pytest.raises(EOFError):
        reader.seek(reader.length)
    with pytest.raises(EOFError):
        reader.seek(-1)


def test_opencv_reader_empty_raises(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="No frame"):
        OpenCVVideoReader(tmp_path / "nonexistent.avi")


def test_opencv_reader_grayscale_opens(avi_gray: tuple[pathlib.Path, np.ndarray]):
    # MJPG codec converts grayscale to 3-channel BGR on read; verify the reader at least opens
    path, _ = avi_gray
    reader = OpenCVVideoReader(path)

    assert reader.length > 0
    assert reader.shape == (10, 12)


def test_opencv_reader_check_frame_id(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = OpenCVVideoReader(path)
    reader.seek(2)
    reader._check_frame_id()  # should not raise


def test_opencv_reader_release(avi_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = avi_2d
    reader = OpenCVVideoReader(path)
    reader.release()
    assert reader.released is True


## PILVideoReader


def test_pil_reader_gray_attributes(pil_tiff_gray: tuple[pathlib.Path, np.ndarray]):
    path, expected = pil_tiff_gray
    reader = PILVideoReader(path)
    assert reader.length == len(expected)
    assert reader.shape == expected.shape[1:3]  # (H, W)
    assert reader.channels == 1


def test_pil_reader_gray_retrieve(pil_tiff_gray: tuple[pathlib.Path, np.ndarray]):
    path, expected = pil_tiff_gray
    reader = PILVideoReader(path)
    frame = reader.retrieve()
    assert frame.shape == (expected.shape[1], expected.shape[2], 1)
    assert (frame == expected[0]).all()


def test_pil_reader_rgb_channels(pil_tiff_rgb: tuple[pathlib.Path, np.ndarray]):
    path, _ = pil_tiff_rgb
    reader = PILVideoReader(path)
    assert reader.channels == 3


def test_pil_reader_grab_sequential(pil_tiff_gray: tuple[pathlib.Path, np.ndarray]):
    path, _ = pil_tiff_gray
    reader = PILVideoReader(path)

    for i in range(reader.length - 1):
        assert reader.grab() is True
        assert reader.frame_id == i + 1
    assert reader.grab() is False


def test_pil_reader_seek(pil_tiff_gray: tuple[pathlib.Path, np.ndarray]):
    path, expected = pil_tiff_gray
    reader = PILVideoReader(path)
    reader.seek(4)

    assert reader.frame_id == 4

    frame = reader.retrieve()
    assert (frame == expected[4]).all()


def test_pil_reader_check_frame_id(pil_tiff_gray: tuple[pathlib.Path, np.ndarray]):
    path, _ = pil_tiff_gray
    reader = PILVideoReader(path)
    reader.seek(3)
    reader._check_frame_id()  # should not raise


def test_pil_reader_release(pil_tiff_gray: tuple[pathlib.Path, np.ndarray]):
    path, _ = pil_tiff_gray
    reader = PILVideoReader(path)
    reader.release()
    assert reader.released is True


## TiffVideoReader


def test_tiff_reader_2d_attributes(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, data = tiff_2d
    reader = TiffVideoReader(path)

    assert reader.length == data.shape[0]
    assert reader.shape == (data.shape[1], data.shape[2])
    assert reader.channels == data.shape[3]
    assert reader.dtype == data.dtype


def test_tiff_reader_2d_retrieve(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, data = tiff_2d
    reader = TiffVideoReader(path)
    frame = reader.retrieve()

    assert frame.shape == data.shape[1:]  # (H, W, C)
    assert (frame == data[0]).all()


def test_tiff_reader_2d_all_frames(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, data = tiff_2d
    reader = TiffVideoReader(path)

    for i in range(reader.length):
        reader.seek(i)
        assert (reader.retrieve() == data[i]).all()


def test_tiff_reader_3d_attributes(tiff_3d: tuple[pathlib.Path, np.ndarray]):
    path, data = tiff_3d
    reader = TiffVideoReader(path)

    assert reader.length == data.shape[0]
    assert reader.shape == (data.shape[1], data.shape[2], data.shape[3])
    assert reader.channels == data.shape[4]


def test_tiff_reader_3d_retrieve(tiff_3d: tuple[pathlib.Path, np.ndarray]):
    path, data = tiff_3d
    reader = TiffVideoReader(path)
    frame = reader.retrieve()

    assert frame.shape == data.shape[1:]  # (D, H, W, C)
    assert (frame == data[0]).all()


def test_tiff_reader_no_channel(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (4, 3, 10, 12), dtype=np.uint8)
    path = tmp_path / "tzyx.tiff"
    tifffile.imwrite(path, data, photometric=True, metadata={"axes": "TZYX"})
    reader = TiffVideoReader(path)

    assert reader.channels == 1
    frame = reader.retrieve()
    assert frame.shape == (3, 10, 12, 1)


def test_tiff_reader_grab(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    reader = TiffVideoReader(path)

    for i in range(reader.length - 1):
        assert reader.grab() is True
        assert reader.frame_id == i + 1
    assert reader.grab() is False


def test_tiff_reader_seek_valid(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, data = tiff_2d
    reader = TiffVideoReader(path)
    reader.seek(5)

    assert reader.frame_id == 5
    assert (reader.retrieve() == data[5]).all()


def test_tiff_reader_seek_same_frame_noop(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    reader = TiffVideoReader(path)
    reader.seek(3)

    current_before = reader._current
    reader.seek(3)

    assert reader._current is current_before


def test_tiff_reader_seek_out_of_range_raises(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    reader = TiffVideoReader(path)

    with pytest.raises(EOFError):
        reader.seek(reader.length)
    with pytest.raises(EOFError):
        reader.seek(-1)


def test_tiff_reader_release(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    reader = TiffVideoReader(path)
    reader.release()
    assert reader.released is True


def test_tiff_reader_permuted_explicit_axes(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (5, 10, 12, 2), dtype=np.uint8)
    path = tmp_path / "v.tiff"
    tifffile.imwrite(path, data)
    print(tifffile.TiffFile(path).series[0].levels[0].axes)

    reader = TiffVideoReader(path, axes="YTXC")
    assert reader.length == 10
    assert reader.channels == 2
    assert reader.shape == (5, 12)


def test_tiff_reader_ax_slice_spatial(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (5, 10, 12, 2), dtype=np.uint8)
    path = tmp_path / "v.tiff"
    tifffile.imwrite(path, data, metadata={"axes": "TYXC"})
    reader = TiffVideoReader(path, ax_slice={"Y": slice(2, 8), "X": slice(3, 9)})

    assert reader.shape == (6, 6)

    frame = reader.retrieve()
    assert frame.shape == (6, 6, 2)


def test_tiff_reader_ax_slice_temporal_raises(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    with pytest.raises(ValueError, match="Temporal slicing"):
        TiffVideoReader(path, ax_slice={"T": slice(0, 3)})


def test_tiff_reader_ax_slice_negative_step_raises(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d
    with pytest.raises(ValueError, match="Negative step"):
        TiffVideoReader(path, ax_slice={"Y": slice(None, None, -1)})


def test_tiff_reader_s_axis_as_c(tmp_path: pathlib.Path):
    # Writing with photometric="rgb" produces axes "QYXS" (tifffile detected);
    # TiffVideoReader interprets S->C and Q->T, giving the correct length/channels.
    data = np.random.randint(0, 256, (5, 10, 12, 3), dtype=np.uint8)
    path = tmp_path / "rgb.tiff"
    tifffile.imwrite(path, data, photometric="rgb")
    reader = TiffVideoReader(path)

    assert reader.length == 5
    assert reader.channels == 3
    assert reader.shape == (10, 12)


def test_tiff_reader_invalid_out_axes(tiff_2d: tuple[pathlib.Path, np.ndarray]):
    path, _ = tiff_2d

    TiffVideoReader.out_axes = "TZYX"

    with pytest.raises(ValueError, match="Channels have to be the last"):
        TiffVideoReader(path)

    TiffVideoReader.out_axes = "TZYXC"


def test_tiff_reader_c_and_s_raises(tmp_path: pathlib.Path):
    # imagej=True with photometric='rgb' produces axes with both C and S -> ValueError
    data = np.random.randint(0, 256, (5, 10, 12, 3), dtype=np.uint8)
    path = tmp_path / "cs.tiff"
    tifffile.imwrite(path, data, imagej=True, photometric="rgb")

    with pytest.raises(ValueError, match="two channel dimensions"):
        TiffVideoReader(path)


def test_tiff_reader_no_temporal_axis_raises(tmp_path: pathlib.Path):
    # YXC format: no T, no Z -> ValueError "Unable to find a temporal axis"
    data = np.random.randint(0, 256, (10, 12, 3), dtype=np.uint8)
    path = tmp_path / "yxc.tiff"
    tifffile.imwrite(path, data, photometric="rgb", metadata={"axes": "YXC"})

    with pytest.raises(ValueError, match="Unable to find a temporal axis"):
        TiffVideoReader(path)


def test_tiff_reader_z_axis_treated_as_t(tmp_path: pathlib.Path):
    # ZYXC format: no T axis present -> Z is treated as T
    data = np.random.randint(0, 256, (4, 10, 12, 2), dtype=np.uint8)
    path = tmp_path / "zyxc.tiff"
    tifffile.imwrite(path, data, metadata={"axes": "ZYXC"})
    reader = TiffVideoReader(path)

    assert reader.length == 4
    assert reader.shape == (10, 12)
    assert reader.channels == 2


def test_frame_tiff_reader_unknown_axes_as_t_and_z(tmp_path: pathlib.Path):
    # QIYX format: neither T nor Z -> Q will be T and I -> Z
    data = np.random.randint(0, 256, (2, 3, 10, 12), dtype=np.uint8)
    path = tmp_path / "qiyx.tiff"
    tifffile.imwrite(path, data, photometric=True, metadata={"axes": "QIYX"})
    reader = TiffVideoReader(path)

    assert reader.length == 2
    assert reader.shape == (3, 10, 12)
    assert reader.channels == 1


def test_frame_tiff_reader_too_many_unknown_axes_raises(tmp_path: pathlib.Path):
    # QIZYX format -> T is Q, but who is I?
    data = np.random.randint(0, 256, (2, 3, 3, 10, 12), dtype=np.uint8)
    path = tmp_path / "qizyx.tiff"
    tifffile.imwrite(path, data, photometric=True, metadata={"axes": "QIZYX"})

    with pytest.raises(ValueError, match="Unable to auto-assign axe"):
        TiffVideoReader(path)


## FrameTiffLoader


def test_frame_tiff_loader_yx(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (10, 12), dtype=np.uint8)
    path = tmp_path / "frame.tiff"
    tifffile.imwrite(path, data, metadata={"axes": "YX"})
    loader = FrameTiffLoader()
    frame = loader(path)

    assert frame.shape == (10, 12, 1)
    assert (frame[..., 0] == data).all()


def test_frame_tiff_loader_yxc(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (10, 12, 3), dtype=np.uint8)
    path = tmp_path / "frame.tiff"
    tifffile.imwrite(path, data, photometric="rgb", metadata={"axes": "YXC"})
    loader = FrameTiffLoader()
    frame = loader(path)

    assert frame.shape == (10, 12, 3)
    assert (frame == data).all()


def test_frame_tiff_loader_zyxc(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (4, 10, 12, 2), dtype=np.uint8)
    path = tmp_path / "frame3d.tiff"
    tifffile.imwrite(path, data, metadata={"axes": "ZYXC"})
    loader = FrameTiffLoader()
    frame = loader(path)

    assert frame.shape == (4, 10, 12, 2)
    assert (frame == data).all()


def test_frame_tiff_loader_ax_slice(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (10, 12, 2), dtype=np.uint8)
    path = tmp_path / "frame.tiff"
    tifffile.imwrite(path, data, metadata={"axes": "YXC"})
    loader = FrameTiffLoader(ax_slice={"Y": slice(2, 8), "X": slice(3, 9)})
    frame = loader(path)
    assert frame.shape == (6, 6, 2)
    assert (frame == data[2:8, 3:9]).all()


def test_frame_tiff_loader_invalid_ax_slice_raises():
    with pytest.raises(ValueError, match="Negative step"):
        FrameTiffLoader(ax_slice={"Y": slice(None, None, -1)})


def test_frame_tiff_loader_permuted_explicit_axes(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (10, 12), dtype=np.uint8)
    path = tmp_path / "frame.tiff"
    tifffile.imwrite(path, data)
    loader = FrameTiffLoader(axes="XY")
    frame = loader(path)

    assert frame.shape == (12, 10, 1)


def test_tiff_loader_s_axis_as_c(tmp_path: pathlib.Path):
    # Writing with photometric="rgb" produces axes "QYXS" (tifffile detected);
    # FrameTiffLoader interprets S->C and Q->Z, giving the correct depth/channels.
    data = np.random.randint(0, 256, (5, 10, 12, 3), dtype=np.uint8)
    path = tmp_path / "rgb.tiff"
    tifffile.imwrite(path, data, photometric="rgb")
    frame = FrameTiffLoader()(path)

    assert (frame == data).all()


def test_frame_tiff_loader_c_and_s_raises(tmp_path: pathlib.Path):
    # A TIFF with both C and S axes (via explicit metadata) -> ValueError
    data = np.random.randint(0, 256, (2, 10, 12, 3), dtype=np.uint8)
    path = tmp_path / "cyxs.tiff"
    tifffile.imwrite(path, data, metadata={"axes": "CYXS"})
    loader = FrameTiffLoader()

    with pytest.raises(ValueError, match="two channel dimensions"):
        loader(path)


def test_frame_tiff_loader_unknown_axis_treated_as_z(tmp_path: pathlib.Path):
    # An axis not in out_axes="ZYXC" and Z absent → that axis becomes Z
    data = np.random.randint(0, 256, (3, 10, 12), dtype=np.uint8)
    path = tmp_path / "qyx.tiff"
    tifffile.imwrite(path, data, photometric=True, metadata={"axes": "QYX"})
    loader = FrameTiffLoader()

    frame = loader(path)
    assert frame.shape == (3, 10, 12, 1)


def test_frame_tiff_loader_unknown_axis_with_z_raises(tmp_path):
    # Unknown axis AND Z already present => ValueError
    data = np.random.randint(0, 256, (2, 3, 10, 12), dtype=np.uint8)
    path = tmp_path / "qzyx.tiff"
    tifffile.imwrite(path, data, photometric=True, metadata={"axes": "QZYX"})
    loader = FrameTiffLoader()

    with pytest.raises(ValueError, match="Unable to parse"):
        loader(path)


## pil_loader


def test_pil_loader_gray(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (10, 12), dtype=np.uint8)
    path = tmp_path / "gray.png"
    PIL.Image.fromarray(data).save(path)
    frame = pil_loader(path)
    assert frame.shape == (10, 12, 1)
    assert (frame[..., 0] == data).all()


def test_pil_loader_rgb(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (10, 12, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    PIL.Image.fromarray(data).save(path)
    frame = pil_loader(path)

    assert frame.shape == (10, 12, 3)
    assert (frame == data).all()


# ---------------------------------------------------------------------------
# MultiFrameReader
# ---------------------------------------------------------------------------


def test_multi_frame_reader_png_folder(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, data = png_folder
    reader = MultiFrameReader(folder)

    assert reader.length == len(data)
    assert reader.shape == (10, 12)
    assert reader.channels == 3
    assert reader.dtype == data.dtype


def test_multi_frame_reader_retrieve(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, data = png_folder
    reader = MultiFrameReader(folder)
    frame = reader.retrieve()

    assert frame.shape == (10, 12, 3)
    assert (frame == data[0]).all()


def test_multi_frame_reader_all_frames(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, data = png_folder
    reader = MultiFrameReader(folder)

    for i in range(reader.length):
        reader.seek(i)
        assert (reader.retrieve() == data[i]).all()


def test_multi_frame_reader_grab(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, _ = png_folder
    reader = MultiFrameReader(folder)

    for i in range(reader.length - 1):
        assert reader.grab() is True
        assert reader.frame_id == i + 1
    assert reader.grab() is False


def test_multi_frame_reader_seek_out_of_range_raises(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, _ = png_folder
    reader = MultiFrameReader(folder)

    with pytest.raises(EOFError):
        reader.seek(reader.length)
    with pytest.raises(EOFError):
        reader.seek(-1)


def test_multi_frame_reader_explicit_and_implicit_extension(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, data = png_folder
    # Add a stray .txt file that should be ignored
    (folder / "notes.txt").write_text("ignore me")
    reader = MultiFrameReader(folder, extension=".png")
    assert reader.length == len(data)

    reader_implicit = MultiFrameReader(folder)
    assert reader_implicit.paths == reader.paths


def test_multi_frame_reader_explicit_paths(tmp_path: pathlib.Path, png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, data = png_folder
    paths = sorted(folder.glob("*.png"))

    # Use only frames 1, 2, 3
    reader = MultiFrameReader(tmp_path, paths=paths[1:4])

    assert reader.length == 3
    assert (reader.retrieve() == data[1]).all()


def test_multi_frame_reader_custom_frame_loader(png_folder: tuple[pathlib.Path, np.ndarray]):
    folder, _ = png_folder
    calls = []

    def my_loader(path):
        calls.append(path)
        return pil_loader(path)

    reader = MultiFrameReader(folder, frame_loader=my_loader)
    _ = reader.retrieve()

    assert len(calls) >= 1


def test_multi_frame_reader_empty_folder_raises(tmp_path: pathlib.Path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="No frame"):
        MultiFrameReader(empty)


def test_multi_frame_reader_tiff_folder_uses_frame_tiff_loader(tmp_path: pathlib.Path):
    data = np.random.randint(0, 256, (4, 10, 12, 2), dtype=np.uint8)
    folder = tmp_path / "tiffs"
    folder.mkdir()
    for i, frame in enumerate(data):
        tifffile.imwrite(folder / f"frame_{i:03d}.tif", frame, metadata={"axes": "YXC"})

    reader = MultiFrameReader(folder)

    assert reader.length == 4
    assert isinstance(reader.frame_loader, FrameTiffLoader)

    frame = reader.retrieve()
    assert frame.shape == (10, 12, 2)
    assert (frame == data[0]).all()
