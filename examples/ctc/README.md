# Cell Linking Benchmark

`link.py` is the python script that we submitted to the Cell Linking Benchmark ([CLB](https://celltrackingchallenge.net/latest-clb-results/)).

It is based on [KOFT](https://ieeexplore.ieee.org/abstract/document/10635656/) algorithm. The goal is to link given cell detections into tracks.
The script reads the given detections and link them to tracks using KOFT algorithm and then save the resulting tracks alongside the input data (following CTC format).

This guide will help you reproduce our results.

## Environment creation and installation

We recommend using conda to set up the environment. Here are the few command lines to reproduce:

```bash
$ conda create -n byotrack python=3.10
$ conda activate byotrack
$ pip install -r requirements.txt  # Without conda, you can simply run this line to install the requirements on your python environment
```

## Running the linker

To run a single tracking on a specific dataset:

```bash
$ # Activate your environement if not already done
$ conda activate byotrack
$
$ # Run the script for the specific dataset.
$ # Replace $CTC_PATH by the path to the parent folder of each CTC-like datasets.
$ # The code expects to find a dataset with CTC format at {data_path}/{dataset}
$ # and stores the tracking results in {data_path}/{dataset}/{seq_id}_RES/
$ # Some extra parameters can be fixed for some specific dataset (see `hyper_parameters.json` or run `link.py` with --help)
$ python link.py --data_path $CTC_PATH --dataset BF-C2DL-HSC --seq 1
```

To run on all datasets of the CLB, we provided the `run_clb.py` script:

```bash
$ # Activate your environement if not already done
$ conda activate byotrack
$
$ # Run on all datasets/seq.
$ # Use --default_parameters to build a true general submission (non-dataset specific, even for hyper parameters)
$ python run_clb.py --data_path $CTC_PATH  # --default-parameters
```

## Data format

We use the data format of the Cell Tracking Challenge that can be found [here](https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf).

You can download the CTC datasets or build your own dataset by following the CTC format:

```
<DATA_PATH>/<dataset_0>/{seq_id}/t{frame_id}.tif             # Each frame for sequence seq_id
                      /{seq_id}_ERR_SEG/mask{frame_id}.tif  # Each segmentation for sequence seq_id (to be linked)
                      /{seq_id}_GT/                         # Optional ground truth (see CTC)
                      /{seq_id}_RES/mask{frame_id}.tif      # Tracking output from the `link.py` script
                                    res_track.txt           # Tracks metadata

          /<dataset_1>/ ....
```

The script `link.py` expects as input the data path to the data folder, the name of the dataset inside the data folder and the sequence to link in the dataset. It stores the tracking results inside the input dataset in the {seq_id}_RES folder.

### Outputs

Tracking results are stored alongside the dataset following CTC format (see above). Given a data_path, dataset and seq_id,
it loads the segmentations (from {data_path}/{dataset}/{seq_id}_ERR_SEG) of the video, links them through time and saves the tracks in the {data_path}/{dataset}/{seq_id}_RES folder.

> [!WARNING]
> Running several time the script for the same video will overwrite previous outputs.


## Method

![KOFT](koft.png)

We propose to use our Bayesian linking algorithms [KOFT
and SKT](https://ieeexplore.ieee.org/abstract/document/10635656/) that are implemented in ByoTrack. They both rely on
Kalman filters to model particle motion, and solve the tracks-to-detection
association frame by frame with Jonker-Volgenant algorithm to find a solution
to the Linear Association Problem (LAP). Whereas classical
Bayesian approaches (like SKT) measure only the position
(sometimes the intensity) of the tracked objects, KOFT
uses optical flow to also measure the velocity of these
objects. More precisely, its Kalman filter is designed with a 2-
steps update at time t: a first update is done with the position of
the associated detection, a second update measures the future
velocity of the track using optical flow between frame t and
t + 1 at the estimated localization of the track.

Cell mitosis events are detected through a second LAP at each
frame between linked tracks and non-linked detections

We smooth the track positions after tracking. We assume a
Gaussian positional noise and use a optimal Rauch-Tung-Striebel(RTS)
smoother. This slightly improves localization of tracks,
but does not change any association.

The method is further detailed in `method.pdf`.


## Cite Us

```bibtex
@inproceedings{reme2024particle,
  title={Particle tracking in biological images with optical-flow enhanced kalman filtering},
  author={Reme, Raphael and Newson, Alasdair and Angelini, Elsa and Olivo-Marin, Jean-Christophe and Lagache, Thibault},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
