from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

import cv2
import av

from tracklab.callbacks import Progressbar, Callback
from tracklab.core.visualizer import Visualizer
from tracklab.datastruct import TrackerState
from tracklab.utils.cv2 import final_patch, cv2_load_image


class VisualizationEngine(Callback):
    """ Visualization engine from list of visualizers.

    Args:
        visualizers: a list of visualizer instances, which must implement `draw_frame`,
                     or subclass :class:`DetectionVisualizer` and implement
                     `draw_detection`.
        save_images: whether to save the visualization as image files (.jpeg)
        save_videos: whether to save the visualization as video files (.mp4)
        process_n_videos: number of videos to visualize. Will visualize the first N videos.
        process_n_frames_by_video: number of frames per video to visualize. Will visualize
                                   frames every N/n frames (not first n frames)
    """

    def __init__(self,
                 visualizers: Dict[str, Visualizer],
                 save_images: bool = False,
                 save_videos: bool = False,
                 video_fps: int = 25,
                 process_n_videos: Optional[int] = None,
                 process_n_frames_by_video: Optional[int] = None,
                 **kwargs
                 ):
        self.visualizers = visualizers
        self.save_dir = Path("visualization")
        self.save_images = save_images
        self.save_videos = save_videos
        self.video_fps = video_fps
        self.max_videos = process_n_videos
        self.max_frames = process_n_frames_by_video
        for visualizer in visualizers.values():
            visualizer.post_init(**kwargs)

    def on_video_loop_end(self, engine, video_metadata, video_idx, detections,
                          image_pred):
        if self.save_videos or self.save_images:
            progress = engine.callbacks.get("progress", Progressbar(dummy=True))
            self.visualize(engine.tracker_state, video_idx, detections, image_pred, progress)

    def visualize(self, tracker_state: TrackerState, video_id, detections, image_preds, progress=None):
        image_metadatas = tracker_state.image_metadatas[
            tracker_state.image_metadatas.video_id == video_id
            ]
        image_gts = tracker_state.image_gt[tracker_state.image_gt.video_id == video_id]
        nframes = len(image_metadatas)
        video_name = tracker_state.video_metadatas.loc[video_id].name
        total = self.max_frames or len(image_metadatas.index)
        progress.init_progress_bar("vis", "Visualization", total)
        detection_preds_by_image = detections.groupby("image_id")
        detection_gts_by_image = tracker_state.detections_gt.groupby("image_id")
        args = [create_draw_args(
            image_id,
            self,
            image_metadatas,
            detection_preds_by_image.get_group(image_id),
            detection_gts_by_image.get_group(image_id),
            image_gts,
            image_preds,
            nframes,
        ) for image_id in islice(image_metadatas.index, 0, None, nframes//total)
        ]
        
        # Initialize PyAV video container if saving videos
        video_writer = None
        if self.save_videos:
            image = cv2_load_image(image_metadatas.iloc[0].file_path)
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create PyAV container for H.264 encoding
            container = av.open(str(filepath), mode='w')
            stream = container.add_stream('h264', rate=self.video_fps)
            
            # Set resolution to 720p (1280x720)
            stream.width = 1280
            stream.height = 720
            stream.pix_fmt = 'yuv420p'
            # Set reasonable default bitrate for good quality
            stream.bit_rate = 10000  # Increased bitrate for 720p
            video_writer = (container, stream)

        with Pool() as p:
            for output_image, file_name in p.imap(process_frame, args):
                if self.save_images:
                    filepath = self.save_dir / "images" / str(video_name) / file_name
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(filepath), output_image)
                
                if self.save_videos:
                    container, stream = video_writer
                    
                    # Resize frame to 720p
                    resized_image = cv2.resize(output_image, (1280, 720), interpolation=cv2.INTER_AREA)
                    
                    # Convert BGR (OpenCV) to RGB (PyAV)
                    frame_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    
                    # Create PyAV frame and encode
                    frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                    packet = stream.encode(frame)
                    if packet:
                        container.mux(packet)
                
                progress.on_module_step_end(None, "vis", None, None)
                
        # Flush the encoder and close the container
        if self.save_videos:
            container, stream = video_writer
            # Flush remaining packets
            for packet in stream.encode():
                container.mux(packet)
            container.close()

    def draw_frame(self, image_metadata, detections_pred, detections_gt,
                   image_pred, image_gt, nframes):
        image = cv2_load_image(image_metadata.file_path)
        for visualizer in self.visualizers.values():
            visualizer.draw_frame(image, detections_pred, detections_gt,
                                  image_pred, image_gt)

        return final_patch(image)


def create_draw_args(image_id, instance, image_metadatas, detections_pred, detections_gt,
                     image_gts, image_preds, nframes):
    image_metadata = image_metadatas.loc[image_id]
    image_gt = image_gts.loc[image_id]
    image_pred = image_preds.loc[image_id]
    return (instance, image_metadata, detections_pred, detections_gt, image_pred,
            image_gt, nframes)


def process_frame(args):
    (instance, image_metadata, detections_pred, detections_gt,
     image_pred, image_gt, nframes) = args
    frame = instance.draw_frame(image_metadata, detections_pred, detections_gt,
                                image_pred, image_gt, nframes)

    return frame, Path(image_metadata.file_path).name
