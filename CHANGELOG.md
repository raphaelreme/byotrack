# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.5] - 2026-08-19

### Breaking Changes

- **`byotrack.napari.tracks_to_napari_tracks`**: no longer returns a separate
  `lineage_ids` array. It now returns a 3-tuple `(points, graph,
  features_points)`, with lineage ids folded into
  `features_points["lineage_ids"]` (computed automatically unless already
  provided through the `features` argument).
- **`icy.load_tracks`**: track identifiers are no longer parsed from Icy's
  XML `id` attribute, whose format turned out to be a badly interpreted
  value producing large, nonsensical identifiers that broke napari track
  visualization. Tracks are now assigned sequential identifiers (0 to N-1)
  in file order until the original format is understood.

### Added

- New `TrackAstraLinker` (`byotrack.implementation.linker.trackastra`):
  wraps the official TrackAstra graph-optimization solver (ILP or greedy,
  with or without division support) directly, as an alternative to the
  existing online `TrackOnStraLinker`. Requires `pip install trackastra`
  (and the `trackastra[ilp]` extra for the ILP solver).

## [2.0.4] - 2026-08-17

### Added

- `byotrack.Video.__array__`: efficient conversion to a numpy array. `np.asarray(video)`/
  `np.array(video)` now preallocate the output array and fill it frame by frame, instead
  of falling back to numpy's generic sequence conversion (which first collects every
  frame into a Python list), roughly halving peak memory usage. Raises `ValueError` if
  called with `copy=False`, since a copy cannot be avoided.

### Changed

- `TrackingGraph.from_tracks`: reworked to build nodes and edges in batches
  (`add_nodes_from`/`add_edges_from`) with vectorized NaN filtering instead of a
  per-point Python loop, significantly reducing computation time on large track sets.
- `byotrack.fiji.save_detections`: the segmentations array is now preallocated and
  filled frame by frame instead of built through `np.concatenate`, reducing peak memory
  usage.

## [2.0.3] - 2026-08-17

### Breaking Changes

- **`byotrack.napari`**: `add_detections`/`visualize`'s `color_from_labels`
  argument is renamed to `color_detections_from_labels`. `add_optical_flow`/
  `visualize_flow_deformation`'s `size` argument is renamed to `node_size`
  (and a new `edge_width` argument controls the grid wireframe width, now
  defaulting to a thinner `0.1`).
- **`update_detections_from_tracks`**: `radius` no longer accepts a
  broadcastable `torch.Tensor`. It is now either a scalar, or a mapping from
  `track.identifier` to a scalar or a per-track-local-frame sequence, so
  that the radius no longer depends on tracks iteration order.

### Added

- `byotrack.napari.add_tracks`/`visualize`: new `track_features` argument to
  register additional per-track (or per-track-and-frame) features on the
  Napari Tracks layer, enabling coloring tracks by arbitrary features (not
  only by identifier/lineage).
- `ctc.load_detections`: new `correct_for_missing_frames` argument to
  support CTC gold ground-truth segmentations where some frames (or, in 3D,
  some Z-planes) are missing annotation files. Missing frames/planes are
  filled with empty detections so the returned sequence spans every index
  up to the maximum frame found.
- `KalmanLinkerParameters.estimate_process_std_from_tracks` and
  `KOFTLinkerParameters.estimate_flow_std_from_tracks`: new `mini` argument
  to clip the estimated std to a minimum value (default: 0.5), avoiding
  degenerate (too small) estimates.

### Changed

- KOFT now estimates its association threshold using the grounded
  likelihood `p(z_t^pos | past)` (restricted to the position, since velocity
  is not used in the association cost), instead of relying on the previous
  KOFT implementation's projected precision. The association threshold is
  adjusted accordingly.

### Fixed

- `update_detections_from_tracks` no longer relies on the iteration order
  of `tracks` to resolve per-track disk radii.

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
