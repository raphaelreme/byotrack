import napari
from typing import  Sequence
import byotrack
from byotrack.api.detections.segmentation_detections import SegmentationDetections

def NapariViewer(video: byotrack.Video=None, detections_sequence: Sequence[byotrack.Detections] = None, tracks: Sequence[byotrack.Track] = None, detections_display_mode: str = 'mask'):
    """ Visualize a byotrack.Video, a sequence of byotrack.Detections and a sequence of byotrack.Track in napari

    """

    if detections_display_mode not in ['mask', 'points', 'auto']:
        raise ValueError(f"Invalid detections_display_mode: {detections_display_mode}. Must be one of 'mask', 'points' or 'auto'")
    

    viewer = napari.Viewer()
    if video is not None:
        napari_video = _video_to_napari(video)
        viewer.add_image(napari_video, name='video')
    if detections_sequence is not None:
        if detections_display_mode=="auto":  # not implemented yet, for now we default to mask
            detections_display_mode = 'mask'
        if detections_display_mode == 'mask':
            masks = _detections_to_masks(detections_sequence)
            viewer.add_labels(masks, name='detections')
        elif detections_display_mode == 'points':
            points = _detections_to_spots(detections_sequence)
            viewer.add_points(points, name='points')
    if tracks is not None:
        napari_tracks = _tracks_to_napari(tracks)
        viewer.add_tracks(napari_tracks, name='tracks')

    return viewer




