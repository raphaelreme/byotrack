"""Provide some example data to test ByoTrack."""

import pathlib
import urllib.request

import platformdirs

import byotrack

data_dir = pathlib.Path(platformdirs.user_data_dir("ByoTrack", roaming=True))
data_dir.mkdir(parents=True, exist_ok=True)


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
    file = data_dir / "tdt_contrxn-1.avi"
    if not file.exists():
        print("Downloading data from https://github.com/raphaelreme/byotrack/ ...")  # noqa: T201
        urllib.request.urlretrieve(
            "https://github.com/raphaelreme/byotrack/raw/main/example_data/tdt_contrxn-1.avi", file
        )

    try:
        return byotrack.Video(file)
    except Exception as exc:
        raise RuntimeError(
            "The downloaded data is corrupted. Consider a manual download from the ByoTrack github."
        ) from exc
