"""Wrappers around CTC metrics [10]

We currently only wrap the CTC software providing SEG, DET and TRA metrics.

In the future, we may wrap the Fiji plugin to access BIO metrics.
"""

import os
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Collection, Union, Sequence
import warnings


import byotrack
from byotrack import fiji
from byotrack.dataset import ctc
from byotrack.utils import sorted_alphanumeric


class CTCSoftwareRunner:  # pylint: disable=too-few-public-methods
    """Runs CTC software (TRA/DET/SEG) [10]

    It wraps the TRA/DET/SEG softwares that are distributed at https://celltrackingchallenge.net/evaluation-methodology
    It is up to the user to download the correct executable and ensure that it works.

    These softwares take as inputs the path to the folder of the dataset (with ground-truth and results tracks),
    the sequence id to evaluate and the number of digits used to encode time in file names. (3 or 4)

    They output a string with the measure score, that we parse.

    Note:
        This code requires CTC softwares as extra dependencies.
        They can be downloaded at https://celltrackingchallenge.net/evaluation-methodology.

    Attributes:
        ctc_software (pathlib.Path): Path to the ctc software folder.
            It should be the root folder containing Win/Linux/Mac subfolders with their executables.
        system (str): The user system in the CTC nomenclature. One of Linux, Mac or Win.
        last_log (str): Logs of the last computed metrics

    """

    cmd = './{executable} "{dataset}" {seq:02} {n_digit}'

    def __init__(self, ctc_software: Union[str, os.PathLike]):
        self.ctc_software = pathlib.Path(ctc_software)
        self.last_log = ""

        self.system = platform.system()
        if self.system == "Darwin":
            self.system = "Mac"
        elif self.system == "Windows":
            self.system = "Win"
        # else: Linux is already good, Java is not supported so let's just try Linux ?

        assert self.ctc_software.is_dir(), "The path to the unzipped CTC `evaluation_software` is required"
        assert (self.ctc_software / self.system).is_dir(), f"Softwares for your system ({self.system}) are not found"

    def run(self, metric: str, dataset: Union[str, os.PathLike], seq=1, n_digit=4) -> float:
        """Run the CTC software for the given metric and dataset

        Args:
            metric (str): The metric to evaluate. One of (TRA, DET, SEG).
            dataset (Union[str, os.PathLike]): Path to the dataset to evaluate.
            seq (int): Sequence to evaluate inside the dataset.
                CTC softwares will compare {dataset}/{seq:02}_RES with {dataset}/{seq:02}_GT
                Default: 1
            n_digit (int): Number of digits used to encode time in file names.
                It is dataset dependant, but in ByoTrack, by default we use 4 digits.
                Default: 4

        Returns:
            float: The evaluated metric

        """
        assert metric in ("SEG", "DET", "TRA"), "Only SEG, DET and TRA metrics are available with this software"

        executable = metric + "Measure"
        if self.system == "WIN":
            executable = executable + ".exe"

        cmd = self.cmd.format(executable=executable, dataset=dataset, seq=seq, n_digit=n_digit)

        if self.system == "WIN":
            cmd = cmd[2:]  # Strip ./ on windows

        print("Calling CTC softwares with:", cmd)
        outputs = subprocess.run(
            cmd, check=False, cwd=f"{self.ctc_software}/{self.system}", shell=True, stdout=subprocess.PIPE
        )
        output_string = outputs.stdout.decode("utf-8")

        # Let's parse the output
        try:
            value = float(output_string.split()[-1])
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Cannot parse outputs, the CTC software probably found an error: " + output_string
            ) from exc

        self.last_log = (pathlib.Path(dataset) / f"{seq:02}_RES" / f"{metric}_log.txt").read_text("utf-8")

        return value


class CTCMetrics(CTCSoftwareRunner):
    """Wrapper around CTC metrics. [10]

    It wraps the TRA/DET/SEG softwares that are distributed at https://celltrackingchallenge.net/evaluation-methodology
    It is up to the user to download the correct executable and ensure that it works.

    To compute a metric, the code creates a temporary folder, where it saves the ground-truth tracks
    alongside the predicted tracks. Then it runs the required CTC software, and finally remove the
    temporary folder. See `CTCSoftwareRunner`.

    Note:
        This code requires CTC softwares as extra dependencies.
        They can be downloaded at https://celltrackingchallenge.net/evaluation-methodology.

    We provide two distincts methods to evaluate the metrics. One based on tracks, the other on detections:

    1. `compute_detection_metrics` compute the DET and SEG using the results of a segmentation algorithm (no tracks
    information is required). Note that TRA cannot be computed in this case.
    2. `compute_tracking_metrics` compute the metrics for tracks. One can compute the DET and SEG for the tracks,
    but these will not match with the ones computed for detections. Indeed, tracks filter out some detections
    and may add points to tracks to fill missing detections. Usually, this leads to better DET/SEG metrics
    as it reduces false positive and false negative.

    It will store the logs of the last called metric in `self.last_log`

    """

    def compute_tracking_metric(
        self,
        metric: str,
        tracks: Collection[byotrack.Track],
        ground_truth_tracks: Union[str, os.PathLike, Collection[byotrack.Track]],
        *,
        detections_sequence: Sequence[byotrack.Detections] = (),
        ground_truth_detections_sequence: Sequence[byotrack.Detections] = (),
        **kwargs,
    ) -> float:
        """Compute the given metric for the given tracks and ground truthes.

        It will create a temporary folder, to store the tracks and ground-truthes in the right format and then
        execute the CTC software. The temporary folder is removed at the end.

        See `run` to simply compute the metric of an already existing folder.

        Note:
            In CTC, matching of predicted tracks with GT ones is done based on a kind of IOU.
            Therefore, it may be useful to provide the detections_sequence associated with the tracks.
            You may also tweak the `default_radius` arguments (in kwargs). (See `ctc.save_tracks`)

        Note:
            This function computes the tracking metrics, which are different than
            those computed with `compute_detection_metric` for SEG and DET.
            For tracking metrics, we export the segmentation of tracks, rather than the detections_sequence.
            Tracks segmentation differs from the predicted segmentation, as the tracks processus
            filters out detections and may predict tracks position for missing ones. (It usually improves
            the segmentation metrics, as it reduces false positive and false negative)

        Args:
            metric (str): The metric to evaluate. One of (TRA, DET, SEG).
            tracks (Collection[byotrack.Track]): Predicted tracks to evaluate.
            ground_truth_tracks (Union[str, os.PathLike, Collection[byotrack.Track]]): Ground truth data.
                It is either a path to the GT tracks folder, which will be copied in our temporary folder.
                Or it is a list of ByoTrack.Track, that will be saved in the temporary folder.
            detections_sequence (Sequence[byotrack.Detections]): Optional detections, used when saving the tracks.
                Default: ()  # No detections and tracks segmentations will be disk of radius `default_radius`.
            ground_truth_detections_sequence (Sequence[byotrack.Detections]): Optional detections for ground-truth.
                When saving the GT tracks, these detections are used. See `ctc.save_tracks`.
                Default: ()  # No detections and tracks segmentations will be disk of radius `default_radius`.
            **kwargs: Additional arguments to provide to the `ctc.save_tracks` function.
                'shape': Provide the shape of saved image. It is mandatory if no detections is provided.
                'default_radius': Radius of the disk drawn for tracks that have no detections.
                'last': Last frame to consider. (Typically to shorten the sequences,
                or if no object is tracked on the last frames, this will enforce the creation of empty tiff files)

        """
        if "shape" in kwargs:
            shape = kwargs.pop("shape")
        else:
            if detections_sequence:
                shape = detections_sequence[0].shape
            elif ground_truth_detections_sequence:
                shape = ground_truth_detections_sequence[0].shape
            else:
                raise ValueError("Without any detections, `shape` must be provided")

        with tempfile.TemporaryDirectory(prefix="ByoTrack-CTC-Metrics") as output_dir:
            output_path = pathlib.Path(output_dir)
            ground_truth_path = output_path / "01_GT" / ("SEG" if metric == "SEG" else "TRA")
            ctc.save_tracks(
                output_path / "01_RES", tracks, detections_sequence, as_res=True, shape=shape, n_digit=4, **kwargs
            )
            if isinstance(ground_truth_tracks, (str, os.PathLike)):
                if ground_truth_detections_sequence:
                    warnings.warn(
                        "When using a saved GT folder, it will be copied and ground-truth detections are ignored"
                    )
                self.copy_ground_truth(ground_truth_tracks, ground_truth_path, as_seg=metric == "SEG")
            else:
                ctc.save_tracks(
                    ground_truth_path,
                    ground_truth_tracks,
                    ground_truth_detections_sequence,
                    as_res=False,
                    as_seg=metric == "SEG",
                    shape=shape,
                    **kwargs,
                )

            return self.run(metric, output_path, 1)

    def compute_detection_metric(
        self,
        metric: str,
        detections_sequence: Sequence[byotrack.Detections],
        ground_truth_detections_sequence: Union[str, os.PathLike, Sequence[byotrack.Detections]],
    ) -> float:
        """Compute the given metric for the given detections and ground truthes.

        It will create a temporary folder, to store the detections and ground-truthes in the right format and then
        execute the CTC software. The temporary folder is removed at the end.

        See `run` to simply compute the metric of an already existing folder.

        Note:
            This function truly computes the detections metrics, which are different than
            those computed with `compute_tracking_metric` for SEG and DET.
            For tracking metrics, we export the segmentation of tracks, rather than the detections_sequence.
            Tracks segmentation differs from the predicted segmentation, as the tracks processus
            filters out detections and may predict tracks position for missing ones. (It usually improves
            the segmentation metrics, as it reduces false positive and false negative)

        Args:
            metric (str): The metric to evaluate. One of (TRA, DET, SEG).
            tracks (Collection[byotrack.Track]): Predicted tracks to evaluate.
            ground_truth_tracks (Union[str, os.PathLike, Collection[byotrack.Track]]): Ground truth data.
                It is either a path to the GT tracks folder (), which will be copied in our temporary folder.
                Or it is a list of ByoTrack.Track, that will be saved in the temporary folder.
            detections_sequence (Sequence[byotrack.Detections]): Optional detections, used when saving the tracks.
                Default: ()  # No detections and tracks segmentations will be disk of radius `default_radius`.
            ground_truth_detections_sequence (Sequence[byotrack.Detections]): Optional detections for ground-truth.
                When saving the GT tracks, these detections are used. See `ctc.save_tracks`.
                Default: ()  # No detections and tracks segmentations will be disk of radius `default_radius`.

        """
        assert metric != "TRA", "TRA metric cannot be computed for detections"

        with tempfile.TemporaryDirectory(prefix="ByoTrack-CTC-Metrics") as output_dir:
            output_path = pathlib.Path(output_dir)
            ground_truth_path = output_path / "01_GT" / ("SEG" if metric == "SEG" else "TRA")
            ctc.save_detections(output_path / "01_RES", detections_sequence)
            if isinstance(ground_truth_detections_sequence, (str, os.PathLike)):
                self.copy_ground_truth(ground_truth_detections_sequence, ground_truth_path, as_seg=metric == "SEG")
            else:
                ctc.save_detections(
                    ground_truth_path, ground_truth_detections_sequence, as_res=False, as_seg=metric == "SEG"
                )

            return self.run(metric, output_path, 1)

    @staticmethod
    def copy_ground_truth(
        ground_truth_path: Union[str, os.PathLike], target_path: Union[str, os.PathLike], as_seg=False
    ) -> None:
        """Copy the ground-truth to the target

        It will copy any files matching maskT.tif, res_track.txt, man_trackT.tif, man_segT.tif, man_track.txt.
        Nomenclature of files is adapted to match ground truth one. Also, files with 3 digits
        to encode the time information, will be saved with 4 digits.

        Note:
            It is probably more permissive than the software, allowing to convert from/to SEG to/from TRA annotations.
            It also supports converting results into annotations (SEG and TRA).
            In some cases, it may produce invalid data for CTC softwares, for instance converting SEG GT into TRA ones
            only works if SEG GT are fully annotated and it only allows to run DET measures, as the
            metadata txt file would be missing.

        Args:
            ground_truth_path (Union[str, os.PathLike]): Path to ground truthes.
                We support any directory containing potential files to copy. (In particular
                it can be 01_GT/SEG or 01_GT/TRA)
            target_path (Union[str, os.PathLike]): Path where to copy files.
            as_seg (bool): Save target_files as man_segT.tif instead of man_trackT.tif
                Default: False
        """

        ground_truth_path = pathlib.Path(ground_truth_path)
        target_path = pathlib.Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

        # Search for tiff files
        gt_seg_tiff_paths = sorted_alphanumeric(ground_truth_path.glob("man_seg*.tif"))
        gt_track_tiff_paths = sorted_alphanumeric(ground_truth_path.glob("man_track*.tif"))
        res_tiff_paths = sorted_alphanumeric(ground_truth_path.glob("mask*.tif"))

        # Find the inputs type: either res or annotations, and either SEG or TRA for annotations
        if len(gt_seg_tiff_paths) and len(gt_track_tiff_paths):
            warnings.warn(
                "Found both man_trackT.tif and man_segT.tif files in the GT folder. The most common ones will be copied"
            )

        is_seg = len(gt_seg_tiff_paths) > len(gt_track_tiff_paths)

        if max(len(gt_seg_tiff_paths), len(gt_track_tiff_paths)) and len(res_tiff_paths):
            warnings.warn(
                "Found both ground truth and results tiff files in the GT folder. The most common ones will be used."
            )

        is_res = len(res_tiff_paths) > max(len(gt_seg_tiff_paths), len(gt_track_tiff_paths))

        # Handle optional metadata txt file required for TRA
        if is_res:
            if (ground_truth_path / "res_track.txt").exists():
                shutil.copy(ground_truth_path / "res_track.txt", target_path / "man_track.txt")
        else:
            if (ground_truth_path / "man_track.txt").exists():
                shutil.copy(ground_truth_path / "man_track.txt", target_path / "man_track.txt")

        # Handle SEG case on its own
        if is_seg and not is_res:
            for path in gt_seg_tiff_paths:
                # Parse file name, as it can be man_segT.tif or sometimes man_seg_T_Z.tif in 3D
                stem = path.stem[len("man_seg") :]

                if "_" in stem:
                    assert as_seg, "Cannot convert a partial SEG ground truth into a valid DET/TRA ones"
                    frame_id, slice_id = (int(word) for word in stem[1:].split("_"))
                    shutil.copy(path, target_path / f"man_seg_{frame_id:04}_{slice_id:04}.tif")
                else:
                    frame_id = int(stem)
                    shutil.copy(path, target_path / f"man_{'seg' if as_seg else 'track'}{frame_id:04}.tif")

            return

        # Finally handle the rest
        for path in res_tiff_paths if is_res else gt_track_tiff_paths:
            frame_id = int(path.stem[4 if is_res else 9 :])
            shutil.copy(path, target_path / f"man_{'seg' if as_seg else 'track'}{frame_id:04}.tif")


class BioMetrics:
    """Wrapper around the CTC "Biological measures" Fiji plugin [10]

    It allows the computations of BIO metrics for the Cell Tracking Challenge [10].

    Note:
        This implementation requires Fiji to be installed (https://imagej.net/downloads)
        with the CTC plugins (https://github.com/CellTrackingChallenge/fiji-plugins)

    Attributes:
        runner (byotrack.fiji.FijiRunner): Fiji runner
        last_metrics (Tuple[float, float, float, float]): Sub metrics information for the last computed BIO
            It consists of ("CT", "TF", "BC(i)", "CCA"). Their average gives the BIO metric.

    """

    plugin_name = "Biological measures"
    # Reg exp for parsing outputs: find the last 4 numbers in the output line such in the format 'a, b, c, d]]'
    output_regexp = re.compile(r"([+-]?\d*\.\d+), ([+-]?\d*\.\d+), ([+-]?\d*\.\d+), ([+-]?\d*\.\d+)]]$")

    def __init__(self, fiji_path: Union[str, os.PathLike]) -> None:
        """Constructor

        Args:
            fiji_path (str | os.PathLike): Path to the fiji executable
                The executable can be found inside the installation folder of Fiji.
                Linux: Fiji.app/ImageJ-<os>
                Windows: Fiji.app/ImageJ-<os>.exe
                MacOs: Fiji.app/Contents/MacOs/ImageJ-<os>

        """
        self.runner = fiji.FijiRunner(fiji_path, capture_outputs=True)
        self.last_metrics = (0.0, 0.0, 0.0, 0.0)

    def run(self, dataset: Union[str, os.PathLike], seq=1, n_digit=4) -> float:
        """Run the CTC "Biological measures" plugin on the given dataset.

        The dataset should already have results stored in it. It expects the CTC format.

        Args:
            metric (str): The metric to evaluate. One of (TRA, DET, SEG).
            dataset (Union[str, os.PathLike]): Path to the dataset to evaluate.
            seq (int): Sequence to evaluate inside the dataset.
                The plugin will compare {dataset}/{seq:02}_RES with {dataset}/{seq:02}_GT
                Default: 1
            n_digit (int): Number of digits used to encode time in file names.
                It is dataset dependant, but in ByoTrack, by default we use 4 digits.
                Default: 4

        Returns:
            float: The evaluated metric

        """
        path = pathlib.Path(dataset)

        # Let's do output redirection
        self.runner.run(
            "Biological measures", resPath=path / f"{seq:02}_RES", gtPath=path / f"{seq:02}_GT", noOfDigits=n_digit
        )

        line = self.runner.last_outputs.stdout.decode().strip().split("\n")[-1]
        match = self.output_regexp.search(line)
        if not match:
            raise RuntimeError("Cannot parse outputs, the CTC software probably found an error: " + line)

        self.last_metrics = tuple(  # type: ignore
            (float(group) if float(group) >= 0.0 else 0.0) for group in match.groups()
        )

        return sum(self.last_metrics) / 4

    def compute(
        self,
        tracks: Collection[byotrack.Track],
        ground_truth_tracks: Union[str, os.PathLike, Collection[byotrack.Track]],
        *,
        detections_sequence: Sequence[byotrack.Detections] = (),
        ground_truth_detections_sequence: Sequence[byotrack.Detections] = (),
        **kwargs,
    ) -> float:
        """Compute BIO metric for the given tracks and ground truthes.

        It will create a temporary folder, to store the tracks and ground-truthes in the right format and then
        execute the CTC plugins. The temporary folder is removed at the end.

        See `run` to simply compute the metric of an already existing folder.

        Note:
            In CTC, matching of predicted tracks with GT ones is done based on a kind of IOU.
            Therefore, it may be useful to provide the detections_sequence associated with the tracks.
            You may also tweak the `default_radius` arguments (in kwargs). (See `ctc.save_tracks`)

        Args:
            tracks (Collection[byotrack.Track]): Predicted tracks to evaluate.
            ground_truth_tracks (Union[str, os.PathLike, Collection[byotrack.Track]]): Ground truth data.
                It is either a path to the GT tracks folder, which will be copied in our temporary folder.
                Or it is a list of ByoTrack.Track, that will be saved in the temporary folder.
            detections_sequence (Sequence[byotrack.Detections]): Optional detections, used when saving the tracks.
                Default: ()  # No detections and tracks segmentations will be disk of radius `default_radius`.
            ground_truth_detections_sequence (Sequence[byotrack.Detections]): Optional detections for ground-truth.
                When saving the GT tracks, these detections are used. See `ctc.save_tracks`.
                Default: ()  # No detections and tracks segmentations will be disk of radius `default_radius`.
            **kwargs: Additional arguments to provide to the `ctc.save_tracks` function.
                'shape': Provide the shape of saved image. It is mandatory if no detections is provided.
                'default_radius': Radius of the disk drawn for tracks that have no detections.
                'last': Last frame to consider. (Typically to shorten the sequences,
                or if no object is tracked on the last frames, this will enforce the creation of empty tiff files)

        """
        if "shape" in kwargs:
            shape = kwargs.pop("shape")
        else:
            if detections_sequence:
                shape = detections_sequence[0].shape
            elif ground_truth_detections_sequence:
                shape = ground_truth_detections_sequence[0].shape
            else:
                raise ValueError("Without any detections, `shape` must be provided")

        with tempfile.TemporaryDirectory(prefix="ByoTrack-CTC-Metrics") as output_dir:
            output_path = pathlib.Path(output_dir)
            ground_truth_path = output_path / "01_GT" / "TRA"
            ctc.save_tracks(
                output_path / "01_RES", tracks, detections_sequence, as_res=True, shape=shape, n_digit=4, **kwargs
            )
            if isinstance(ground_truth_tracks, (str, os.PathLike)):
                if ground_truth_detections_sequence:
                    warnings.warn(
                        "When using a saved GT folder, it will be copied and ground-truth detections are ignored"
                    )
                CTCMetrics.copy_ground_truth(ground_truth_tracks, ground_truth_path)
            else:
                ctc.save_tracks(
                    ground_truth_path,
                    ground_truth_tracks,
                    ground_truth_detections_sequence,
                    as_res=False,
                    shape=shape,
                    **kwargs,
                )

            return self.run(output_path, 1)
