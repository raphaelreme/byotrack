"""Provide some example data to test ByoTrack"""

import os
import urllib.request

import platformdirs

import byotrack


data_dir = platformdirs.user_data_dir("ByoTrack", roaming=True)
os.makedirs(data_dir, exist_ok=True)


def hydra_neurons() -> byotrack.Video:
    """2D time-lapse sequence of Hydra Vulgaris neurons (TdTomato).

    This data comes from: Hanson, A., Reme, R., Telerman, N., Yamamoto, W., Olivo-Marin, J. C.,
    Lagache, T., & Yuste, R. (2024). "Automatic monitoring of neural activity with single-cell resolution
    in behaving Hydra". Scientific Reports, 14(1), 5083.

    Please cite the paper if you use this data.

    It downloads the tdt_contrxn-1.avi file (https://datadryad.org/stash/dataset/doi:10.5061/dryad.h9w0vt4q3)
    into your user data folder if not already downloaded. The file is then read as a Video.

    It this fails, you can download the data manually and then open it using byotrack.Video(path).

    Returns:
        byotrack.Video: Video of Hydra Vulgaris neurons.
            Shape: (T=1000, H=848, W=1024, C=3)

    """
    file = os.path.join(data_dir, "tdt_contrxn-1.avi")
    if not os.path.exists(file):
        print("Downloading data from https://github.com/raphaelreme/byotrack/ ...")
        urllib.request.urlretrieve(
            "https://github.com/raphaelreme/byotrack/raw/main/example_data/tdt_contrxn-1.avi", file
        )

    try:
        return byotrack.Video(file)
    except:
        print("The downloaded data may be corrupted. You can manually download it from the ByoTrack github.")
        raise
