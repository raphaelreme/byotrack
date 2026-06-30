import napari
import numpy as np
from torch import isnan

from typing import List, Sequence
import torch
from tqdm import tqdm

import byotrack
from byotrack.api.detections.segmentation_detections import SegmentationDetections

def _video_to_napari(video: byotrack.Video) -> np.ndarray:
    """ Convert a byotrack.Video to a format compatible with napari (T, H, W) or (T, D, H, W)"""
    napari_video = []
    for frame in range(len(video)):
        napari_video.append(video[frame].squeeze())
    return np.array(napari_video)


def _detections_to_masks(detections: Sequence[byotrack.Detections]) -> np.ndarray:
    """Convert byotrack.Detections to masks

    """
    masks = []
    for detection in detections:
        masks.append(detection.segmentation.numpy())
    return np.array(masks)

def _masks_to_detections(masks: np.ndarray) -> Sequence[byotrack.Detections]:
    """Convert masks to byotrack.Detections

"""
    mask_final=[]
    for i in range(0,len(masks)):
        mask_tensor= (masks[i])
    #make tensor as dytype torch.int32      
        mask_tensor=mask_tensor.astype('int32')
        mask_tensor=torch.from_numpy(np.array(mask_tensor))


        mask_final.append(SegmentationDetections(mask_tensor))
    return mask_final


def _detections_to_spots(detections: Sequence[byotrack.Detections]) -> np.ndarray:
    """Convert byotrack.Detections to a format compatible with napari Spots layer (T, N, D)"""
    spots = []
    for frame,detection in enumerate(detections):
    
        for i in range(len(detection.position)):
            spots.append([frame, *detection.position.numpy()[i]])

    return np.array(spots)


def _tracks_to_napari(tracks: Sequence[byotrack.Track]) -> np.ndarray:
    """ Convert byotrack tracks to a format compatible with napari (T, D, H, W)"""
    napari_tracks = []
    for i,track in enumerate(tracks):
        for n_alive in range(len(track)):

            if track.points[n_alive][0].isnan():
                # The track is not alive at this time point, we skip it
                continue

            napari_tracks.append([i, track.start + n_alive, *track.points[n_alive].cpu().numpy()])
    return np.array(napari_tracks)

#Only for the forward flow for now
def optical_flow_for_napari(video: byotrack.Video, optflow: byotrack.OpticalFlow, grid_step: int = 32):
    """Process the video with the optical flow algorithm and return the transformed points and flow maps

    Args:
        video (byotrack.Video): The input video
        optflow (byotrack.OpticalFlow): The optical flow algorithm to use
        grid_step (int, optional): The step size for the grid of points to transform. Defaults to 32.

    Returns:
        np.ndarray: The transformed points with shape (N, 3) where N is the total number of points across all frames and the 3 columns correspond to (frame_id, x, y)
        List[np.ndarray]: The list of flow maps for each frame with shape (H, W, 2) for the x and y flow components
    """
    all_points= []
    
    src_shape = video[0].squeeze().shape
    
    _grid = np.indices(src_shape, dtype=np.float64)
    _grid = _grid[:,::grid_step, ::grid_step].reshape(2, -1).T

    _frame_id=np.zeros(_grid.shape[0], dtype=np.int64)

    points=_grid.copy()
    points = np.hstack([_frame_id[:,None], points])  # (N, 3) with (frame_id, x, y)
    all_points=points.copy()
    _points=_grid.copy()
    for frame_id in tqdm(range(0, len(video)-1), desc="Preprocessing optical flow"):
        flow_maps = []
        dst = optflow.preprocess(video[frame_id+1])

        src = optflow.preprocess(video[frame_id])
        flow_maps.append(optflow.compute(src, dst))

        _points = optflow.transform(flow_maps[-1], _points)
        _frame_id= np.ones(_points.shape[0], dtype=np.int64) * (frame_id+1)
        points = np.hstack([_frame_id[:,None], _points]) 

        all_points= np.vstack([all_points, points])        
    return all_points,flow_maps

