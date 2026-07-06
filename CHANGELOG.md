# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
