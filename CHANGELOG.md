# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2026-07-17

### Breaking Changes

- **`ctc.save_tracks`**: `anisotropy` is now a `tuple[float, float, float]`
  (`ani_z, ani_y, ani_x`) instead of a single float (the relative
  depth-to-xy pixel size). It now relies on the new
  `update_detections_from_tracks` to draw disks/relabel detections.
- **`byotrack.geff.save_tracks_to_geff`**: `drop_nan` and `split_channels`
  now default to `True` instead of `False`, producing napari-geff
  compatible files by default. Pass `drop_nan=False, split_channels=False`
  to keep the previous ByoTrack round-trip behavior.

### Added

- `byotrack.update_detections_from_tracks`: relabels detections with their
  matched track identifiers, optionally drops unmatched detections (false
  positives) and draws disks for tracks with a known position but no
  matching detection (false negatives).
- `Registrator` video preprocessor: online translation registration of a
  video against a reference frame using phase cross-correlation
  (`skimage.registration.phase_cross_correlation` + `scipy.ndimage.shift`).
- `color_from_labels` option in `byotrack.napari` (`add_detections`,
  `detections_to_napari_segmentation`) to color segmentation layers using
  the `Detections.labels` attribute (e.g. track identifiers) instead of
  per-frame detection identifiers.

### Fixed

- `DetectionsFilter` was combining `min_area`/`max_area` constraints with
  `&` instead of `|`, so detections were never actually filtered out.

## [2.0.1] - 2026-07-13

### Breaking Changes

- **CTC dataset format**: `save_detections`, `save_tracks` and the private
  `_save_metadata` now take the object to save as the first positional
  argument and the path as the second (`save(obj, path)`), matching the
  convention used elsewhere in the library. `detections_sequence` in
  `save_tracks` is now keyword-only.
- **`EMHTParameters`**: renamed `expected_initial_particles` to
  `expected_initial_targets` and `expected_new_particles` to
  `expected_new_targets`.

### Added

- New `byotrack.geff` package: IO support for the
  [Graph Exchange File Format (GEFF)](https://liveimagetrackingtools.org/geff/latest/),
  with `save_tracks_to_geff` / `load_tracks_from_geff`,
  `save_video_to_zarr` / `load_video_from_zarr` /
  `load_video_from_geff`, and `save_detections_to_zarr` /
  `load_detections_from_zarr` / `load_detections_from_geff`.
- `TrackingGraph.from_tracks` now accepts a `drop_nan` keyword argument to
  drop undefined (NaN) positions instead of keeping them as NaN-valued
  nodes (useful for software that doesn't support NaN node positions).
- `ctc.load_detections`: wraps `GroundTruthDetector` to load CTC-format
  detections in one call.
- `byotrack.video.video_length`, `video_shape` and `video_dtype`
  utilities to introspect a video (duck-typed) without necessarily loading
  its first frame, and without requiring `__len__` support — enabling
  future support for zarr-based videos.

### Changed

- Docstrings and the top-level `byotrack` package example now consistently
  use "targets" instead of "particles", and the getting-started example
  was updated to reflect the current API (`VideoProcessor`, `napari`
  visualization, `KalmanLinker`).

### Fixed

- `Detector.run`, `OnlineLinker.run` and `MultiStepTracker`/
  `BatchMultiStepTracker` now use the new video introspection utilities
  instead of `len(video)`, so they work with videos that don't support
  `__len__` (e.g. zarr-backed videos).

## [2.0.0] - 2026-07-06

### Breaking Changes

- **Detections API**: Replaced the single `Detections` class with multiple
  typed implementations (`PointDetections`, `BBoxDetections`,
  `SegmentationDetections`). Added the `DetectionsLike` protocol and an
  `as_detections()` helper for automatic conversion. Linkers now accept `None`
  as the video argument.
- **Video API**:
  - `shape` now includes the channel axis: `(T, H, W, C)` in 2D, matching
    `np.asarray(video).shape`. The old `shape` was `(T, H, W)`.
  - `channels` property removed (use `shape[-1]` instead).
  - `VideoTransformConfig` / `set_transform` are **deprecated** (they still
    work but will be removed in a future release). Use the new
    `VideoProcessor` API instead: `video.normalize()`,
    `video.add_preprocessor()`, and channel/spatial slicing.

### Added

- New `VideoProcessor` API: a modular, chainable preprocessing pipeline
  (`IntensityNormalizer`, `ChannelProjection`, `SpatialProjection`,
  `FrameSlicer`).
- New `byotrack.napari` package for visualizing videos, detections, optical
  flow, and tracks in [napari](https://napari.org/). Supports 2D and 3D data,
  split/merge events, anisotropy, and lazy visualization.
- `ArrayVideoReader`: wrap any numpy array (or array-like) as a `Video`.
- `dtype` property on `VideoReader` and `Video`.
- Extended video slicing: channel selection, spatial projection, and ellipsis
  (`...`) are now supported.
- `GroundTruthDetector` moved to the public `byotrack.api` package; it now
  supports `BatchMultiStepTracker` with a separate segmentation video.
- `Track.dim` property.
- Parameter estimators for ByoTrack linkers (SKT/KOFT).
- `DetectionsFilter` (renamed from `FilterDetections`).
- Comprehensive test suite across the api, dataset, fiji, icy, and video
  packages.
- Weekly CI run against the latest dependency versions.

### Fixed

- Mean projection overflow for large pixel values.
- Numba segmentation function returning wrong values for unsigned integer
  dtypes (`-1` was interpreted as `255` for `uint8`).
- CTC example script updated to work with the new Detections and Video APIs.
- Examples notebooks updated to work with the APIs.
