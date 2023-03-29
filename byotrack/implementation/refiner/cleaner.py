from typing import Collection, Iterable, List

import numpy as np
import torch

import byotrack


class Cleaner(byotrack.Refiner):
    """Cleaner refiner

    Split tracklet when the distance between two consecutive points is greater than `max_dist`.
    And delete tracks that are shorter than `min_length`.

    Warning: Does not really support tracks with nan points (will just not split when a nan is met)

    Attributes:
        min_length (int): Minimum length of tracks kept
        max_dist (float): Maximum distance between two consecutive points in a track

    """

    def __init__(self, min_length: int, max_dist: float) -> None:
        super().__init__()
        self.min_length = min_length
        self.max_dist = max_dist

    def run(self, video: Iterable[np.ndarray], tracks: Collection[byotrack.Track]) -> List[byotrack.Track]:
        return self.clean_tracks(tracks, self.min_length, self.max_dist)

    @staticmethod
    def clean_tracks(tracks: Collection[byotrack.Track], min_length: int, max_dist: float) -> List[byotrack.Track]:
        """Clean tracks

        Split tracklet when the distance between two consecutive points is greater than `max_dist`.
        And delete tracks that are shorter than `min_length`.

        Warning: Does not really support tracks with nan points (will just not split when a nan is met)

        Args:
            tracks (Collection[Track]): Tracks to clean
            min_length (int): Minimum length of tracks kept
            max_dist (float): Maximum distance between two consecutive points in a track
                If max_dist <= 0, splitting is not done

        Returns:
            List[Tracklet]: Cleaned tracks

        """
        n_split = 0
        n_filtered = 0
        cleaned_tracks: List[byotrack.Track] = []

        for track in tracks:
            cleaned = []
            if max_dist <= 0:
                cleaned.append(track)
            else:
                speed = torch.norm(track.points[1:] - track.points[:-1], dim=1)
                first = 0
                for i in range(len(track) - 1):
                    if speed[i] > max_dist:  # Break (i -> i + 1)
                        cleaned.append(
                            byotrack.Track(
                                track.start + first,
                                track.points[first : i + 1],
                                identifier=track.identifier,
                            )
                        )
                        first = i + 1
                        n_split += 1

                # There is at least one element left
                cleaned.append(byotrack.Track(track.start + first, track.points[first:], identifier=track.identifier))

            n_filtered += len(cleaned)
            cleaned = list(filter(lambda tracklet: len(tracklet) >= min_length, cleaned))
            n_filtered -= len(cleaned)

            cleaned_tracks.extend(cleaned)

        print(f"Cleaning: Split {n_split} tracks and dropped {n_filtered} resulting ones")
        print(f"Cleaning: From {len(tracks)} to {len(cleaned_tracks)} tracks")
        return cleaned_tracks
