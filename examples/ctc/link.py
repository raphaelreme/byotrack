"""Script based on ByoTrack for solving the Cell Linking Benchmark (CLB) of the Cell Tracking Challenge (CTC)

Submitted to the CLB in Nov. 2024. Cleaned in Jan. 2025.
"""

import argparse
import pathlib
from typing import List, Optional, Sequence

import cv2
import numpy as np
import skimage  # type: ignore
import torch

import byotrack
import byotrack.api.features_extractor
import byotrack.dataset.ctc as ctc_data
from byotrack.implementation.optical_flow import skimage as sk, opencv
from byotrack.implementation.linker.frame_by_frame import nearest_neighbor, kalman_linker, koft
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater
import byotrack.metrics.ctc as ctc_metrics


def get_average_size(detections_sequence: List[byotrack.Detections]) -> float:
    """Get the average size of cells in the dataset"""

    total_size = 0
    count = 0
    for detections in detections_sequence:
        if len(detections) <= 0:
            continue

        count += len(detections)
        total_size += int(detections.mass.sum().item())

    return total_size / (count + (count == 0))


def get_average_min_dist(detections_sequence: List[byotrack.Detections]) -> float:
    """Get the average minimal distance between cells in the dataset"""
    sum_min_dist = 0.0
    count = 0
    for detections in detections_sequence:
        if len(detections) <= 1:
            continue

        count += 1
        sum_min_dist += torch.cdist(detections.position, detections.position).sort(dim=1).values[:, 1].median().item()

    return sum_min_dist / (count + (count == 0))


# def mean_max_min_displacementw(detections_sequence: List[byotrack.Detections]) -> float:
#     """Roughly estimate cell motion"""
#     sum_min_dist = 0.0
#     count = 0
#     max_min_dist = 0.0
#     for detections in detections_sequence:
#         if len(detections) <= 1:
#             continue

#         count += len(detections)
#         sum_min_dist += torch.cdist(detections.position, detections.position).sort(dim=1).values[:, 1].sum().item()

#     return sum_min_dist / (count + (count == 0))


def link(video: byotrack.Video, detections_sequence: Sequence[byotrack.Detections], **kwargs) -> List[byotrack.Track]:
    specs: kalman_linker.FrameByFrameLinkerParameters
    linker: kalman_linker.FrameByFrameLinker
    optflow: byotrack.OpticalFlow
    if kwargs["linker"] == "NN":  # NN
        specs = nearest_neighbor.NearestNeighborParameters(
            association_threshold=kwargs["association_threshold"],  # Greedy is good
            association_method="sparse_opt_smooth",  # Sparse linking (faster)
            n_valid=1,  # No spurious detections
            n_gap=kwargs["n_gap"],  # Few missing detections
            anisotropy=(kwargs["anisotropy"], 1.0, 1.0),  # For 3D anisotrope datasets
            split_factor=kwargs["split_factor"],  # Allows splits
        )

        linker = nearest_neighbor.NearestNeighborLinker(specs)
        # Runs with a fake video (as it is unused) to reduce run time (reading video can be expensive)
        tracks = linker.run([np.zeros((1, 1, 1, 1)) for _ in range(len(detections_sequence))], detections_sequence)
        return ForwardBackwardInterpolater(method="constant").run(
            [np.zeros((1, 1, 1, 1)) for _ in range(len(detections_sequence))], tracks
        )

    if kwargs["linker"] == "SKT":  # Standard Kalman Tracking from KOFT paper
        specs = kalman_linker.KalmanLinkerParameters(
            association_threshold=kwargs["association_threshold"],  # Greedy is good
            detection_std=kwargs["detection_std"],  # Detections precisions. In CTC they are quite precise.
            process_std=kwargs["process_std"],  # ~ size of unmodeled displacement
            kalman_order=kwargs[  # 0 (Brownian) is better in CTC
                "kalman_order"
            ],  # Order of the kalman filter (0: Brownian, 1: Directed, 2: Accelerated, ...)
            cost="euclidean",  # Use euclidean distance (likelihood should be better, but harder to set threshold)
            association_method="sparse_opt_smooth",  # Sparse linking (faster)
            n_valid=1,  # No spurious detections
            n_gap=kwargs["n_gap"],  # Few missing detections
            anisotropy=(kwargs["anisotropy"], 1.0, 1.0),  # For 3D anisotrope datasets
            split_factor=kwargs["split_factor"],  # Allows splits
            track_building="smoothed",  # RTS smoothing
        )

        linker = kalman_linker.KalmanLinker(specs)

        # Runs with a fake video (as it is unused) to reduce run time (reading video can be expensive)
        return list(
            linker.run(
                [np.zeros((1, 1, 1, 1)) for _ in range(len(detections_sequence))],
                detections_sequence,
            )
        )

    # KOFT
    if video.ndim == 5:  # 3D, but tvl1 is quite slow and unprecise in CTC
        optflow = sk.SkimageOpticalFlow(
            skimage.registration.optical_flow_tvl1,
            downscale=4,
            parameters={"attachment": kwargs["attachment"]},
        )
    else:
        optflow = opencv.OpenCVOpticalFlow(cv2.FarnebackOpticalFlow.create(winSize=kwargs["win_size"]), downscale=4)

    specs = koft.KOFTLinkerParameters(
        association_threshold=kwargs["association_threshold"],  # Greedy is good
        detection_std=kwargs["detection_std"],  # Detections precisions. In CTC they are quite precise.
        process_std=kwargs["process_std"],  #  ~size of unmodeled displacement
        flow_std=kwargs["flow_std"],  # ~ Optical flow errors (quite low performances in CTC)
        kalman_order=kwargs[  # 0 (Brownian) is better in CTC
            "kalman_order"
        ],  # Order of the kalman filter (0: Brownian, 1: Directed, 2: Accelerated, ...)
        cost="euclidean",  # Use euclidean distance (likelihood should be better, but harder to set threshold)
        association_method="sparse_opt_smooth",  # Sparse linking (faster)
        n_valid=1,  # No spurious detections
        n_gap=kwargs["n_gap"],  # Few missing detections
        anisotropy=(kwargs["anisotropy"], 1.0, 1.0),  # For 3D anisotrope datasets
        split_factor=kwargs["split_factor"],  # Allows splits
        track_building="smoothed",  # RTS smoothing
    )

    linker = koft.KOFTLinker(specs, optflow)

    return list(linker.run(video, detections_sequence))


def main(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    data_path: str,
    dataset: str,
    seq: int,
    *,
    linker: Optional[str] = None,
    association_threshold=0.0,
    detection_std=0.0,
    process_std=0.0,
    flow_std=0.0,
    kalman_order=0,
    anisotropy=0.0,
    split_factor=-1.0,
    n_gap=1,
    win_size=0,
    attachment=10.0,
    detections_sequence=None,
    eval_software: Optional[str] = None,
):
    path = pathlib.Path(data_path) / dataset
    n_digit = len(next((path / f"{seq:02}").glob("t*.tif")).stem[1:])

    # Load the video and normalize it
    video = byotrack.Video(path / f"{seq:02}")  # Load videos
    video.set_transform(
        byotrack.VideoTransformConfig(
            aggregate=True,
            normalize=True,
            compute_stats_on=50 if video.ndim == 4 else 10,
        )
    )

    # Load segmentations
    if not detections_sequence:
        detections_sequence = ctc_data.GroundTruthDetector().run(byotrack.Video(path / f"{seq:02}_ERR_SEG"))

    # Set default parameters
    if video.ndim == 4:  # 2D
        anisotropy = 1.0

    # Anisotropy is computed if not given based on detections (Depends on direciton)
    if video.ndim == 5 and anisotropy <= 0.0:  # 3D
        sizes = sum(
            detections.bbox[:, detections.dim :].to(torch.float32).mean(axis=0) for detections in detections_sequence
        ) / len(detections_sequence)
        anisotropy = float(sizes[1:].mean() / sizes[0])

    # Useful features for setting the parameters
    spot_size = get_average_size(detections_sequence)
    closest_spot_dist = get_average_min_dist(detections_sequence)
    cell_increase = (len(detections_sequence[-1]) - len(detections_sequence[0])) / len(detections_sequence[0])

    if video.ndim == 5:  # 3D + T + C
        spot_radius = float((spot_size * anisotropy * 3 / 4 / np.pi) ** (1 / 3))  # area = 4/3 pi R^3 / ani
    else:
        spot_radius = float(np.sqrt(spot_size / np.pi))  # pi R^2

    print("===========Dataset features============")
    print("Spot radius: ", spot_radius)
    print("Closest spot dist:", closest_spot_dist)
    print("Anisotropy of the Z axis:", anisotropy)
    print("Cell increase between first and last frame:", cell_increase * 100, "%")

    # For farneback, we set by default the winsize ~= cell diameter (after a downscale of 4)
    if win_size == 0:
        win_size = max(10, int(spot_radius / 2))

    # Detections std ~= 1/2 cell radius
    if detection_std <= 0.0:
        detection_std = spot_radius / 2

    # Process_std ~= 3 spot radius
    if process_std <= 0.0:
        process_std = 3 * spot_radius

    # flow_std = process_std (noisy flow that we trust as much as our process)
    if flow_std <= 0.0:
        flow_std = process_std

    if video.ndim == 5 and anisotropy != 1.0:  # Anisotrope 3D, we scale errors on the Z axis
        detection_std = torch.tensor(
            (detection_std / anisotropy, detection_std, detection_std),
            dtype=torch.float32,
        )
        process_std = torch.tensor((process_std / anisotropy, process_std, process_std), dtype=torch.float32)
        flow_std = torch.tensor(
            (flow_std, flow_std, flow_std), dtype=torch.float32
        )  # The flow has no reason to be scaled

    if split_factor < 0:
        if (
            cell_increase > 0.3 and cell_increase * len(detections_sequence[0]) > 1
        ):  # At least 30% of augmentation of cells to activate mitose
            split_factor = 1.0
        else:
            split_factor = 0.0

    # For 3D videos, we could do a conversion 3D to 2D for simple 3D (but it is complex
    # and useful only for computational time)
    # Or use a 3D optical flow, but we did not manage to find one that was accurate and fast enough on CTC.
    # Therefore in 3D, by default, we use SKT which is much faster and still has very good results.
    linker = linker if linker is not None else ("KOFT" if video.ndim == 4 else "SKT")

    # For the association threshold we just set at 3 times the spot radius
    association_threshold = (
        association_threshold if association_threshold > 0.0 else max(spot_radius * 3, closest_spot_dist)
    )

    parameters = {
        "linker": linker,
        "association_threshold": association_threshold,
        "detection_std": detection_std,
        "process_std": process_std,
        "flow_std": flow_std,
        "kalman_order": kalman_order,
        "split_factor": split_factor,
        "anisotropy": anisotropy,
        "n_gap": n_gap,
        "win_size": win_size,
        "attachment": attachment,
    }

    print("==============parameters================")
    print(parameters)

    print("===============Running==================")
    # Let's track!
    tracks = link(video, detections_sequence, **parameters)
    print(f"Produced {len(tracks)} tracks.")

    # Save tracks
    ctc_data.save_tracks(
        path / f"{seq:02}_RES",
        tracks,
        detections_sequence=detections_sequence,
        default_radius=spot_radius * 3,
        shape=video.shape[1:],
        n_digit=n_digit,
        anisotropy=anisotropy,
    )

    if eval_software is not None:  # Evaluate if the path to the software is given
        metric = ctc_metrics.CTCMetrics(eval_software)

        tra = metric.compute_tracking_metric(
            "TRA",
            tracks,
            path / f"{seq:02}_GT" / "TRA",
            detections_sequence=detections_sequence,
            anisotropy=anisotropy,
            default_radius=spot_radius * 3,
        )

        print("TRA:", tra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KOFT on CTC")

    parser.add_argument(
        "--data_path",
        default="../",
        help="Path to the CTC datasets where each dataset is stored",
    )
    parser.add_argument("--dataset", default="BF-C2DL-HSC", help="Dataset name")
    parser.add_argument("--seq", type=int, default=1, help="Sequence to analyze")
    parser.add_argument("--linker", type=str, default=None, help="Linker to run (KOFT | SKT)")
    parser.add_argument("--association_threshold", type=float, default=0.0, help="Association threshold")
    parser.add_argument(
        "--detection_std",
        type=float,
        default=0.0,
        help="Localization errors of the detections",
    )
    parser.add_argument("--process_std", type=float, default=0.0, help="Modelisation errors")
    parser.add_argument("--flow_std", type=float, default=0.0, help="Flows errors")
    parser.add_argument("--kalman_order", type=int, default=0, help="Order of the Kalman filter")
    parser.add_argument(
        "--anisotropy",
        type=float,
        default=0.0,
        help="Anisotropy of Z axis (1 in z = ani in X or Y)",
    )
    parser.add_argument(
        "--split_factor",
        type=float,
        default=-1,
        help="Allows splits with split_factor * asso_thresh",
    )
    parser.add_argument(
        "--n_gap",
        type=float,
        default=1,
        help="Number of miss detected allowed in tracks",
    )
    parser.add_argument("--win_size", type=int, default=0, help="Farneback window")
    parser.add_argument("--attachment", type=float, default=10.0, help="Regularity of TVL1")
    parser.add_argument("--eval_software", type=str, default=None, help="Path to the evalutation software")

    args = parser.parse_args()
    print(args)

    main(
        args.data_path,
        args.dataset,
        args.seq,
        linker=args.linker,
        association_threshold=args.association_threshold,
        detection_std=args.detection_std,
        process_std=args.process_std,
        flow_std=args.flow_std,
        kalman_order=args.kalman_order,
        anisotropy=args.anisotropy,
        split_factor=args.split_factor,
        n_gap=args.n_gap,
        win_size=args.win_size,
        attachment=args.attachment,
        eval_software=args.eval_software,
    )
