import glob
import cv2

from pathlib import Path

import pandas as pd
import yt_dlp

from tracklab.datastruct import (
    TrackingDataset,
    TrackingSet,
)
import logging

log = logging.getLogger(__name__)

def write_video_images_to_disk(video_path):
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    # Check if camera opened successfully
    assert cap.isOpened(), "Error opening video stream or file"

    tmp_video_folder = Path("tmp", video_name)
    log.info("Dumping video frames to {}".format(tmp_video_folder.resolve()))
    tmp_video_folder.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image_path = tmp_video_folder / "{}_{:06d}.jpg".format(
                video_name, frame_idx
            )
            cv2.imwrite(str(image_path), frame)
            frame_idx += 1
        else:
            break
    cap.release()
    return tmp_video_folder


class ExternalVideo(TrackingDataset):
    """
    A class to use PbTrack at inference on any .mp4 video.
    The .mp4 image frames are first saved individually to disk in 'working_dir/tmp/' dir.
    A tracking test set is then created based on these images files, with video_metadata and image_metadata, but without
    detections.
    TODO dirty/easy implementation, we should refactor the tracking engine to make it load directly the images from the
        mp4. Each video in video_metadata should specify its type: folder of images or mp4 video or youtube link, etc,
        and tracking engine should adapt its batch loop accordingly.
    """

    annotations_dir = "posetrack_data"

    def __init__(self, dataset_path: str, video_path: str, *args, **kwargs):
        if video_path.startswith("http"):
            yt_params = {"noplaylist": True, "restrictfilenames": True}
            with yt_dlp.YoutubeDL(yt_params) as ydl:
                info_dict = ydl.extract_info(video_path)
                video_path = ydl.prepare_filename(info_dict)
        self.video_path = Path(video_path)
        video_name = self.video_path.stem
        assert self.video_path.exists(), "Video does not exist ('{}')".format(
            self.video_path
        )

        nframes = self.get_frame_count(self.video_path)
        video_id = 0
        image_metadata = pd.DataFrame(
            [
                {
                    "id": i,
                    "name": f"{video_name}_{i}",
                    "frame": i,
                    "nframes": nframes,
                    "video_id": video_id,
                    "file_path": f"vid://{self.video_path}:{i}",
                }
                for i in range(nframes)
            ]
        )

        video_metadata = pd.DataFrame([{"id": video_name, "name": video_name}])

        val_set = TrackingSet(
            video_metadata,
            image_metadata,
            None,
        )

        super().__init__(dataset_path, None, val_set, None, *args, **kwargs)

    @staticmethod
    def get_frame_count(video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
        cap.release()
        return frames

