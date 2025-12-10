from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional
import logging

import cv2
import numpy as np
import pandas as pd

from tracklab.callbacks import Progressbar, Callback
from tracklab.visualization import Visualizer
from tracklab.datastruct import TrackerState
from tracklab.utils.cv2 import final_patch, cv2_load_image

log = logging.getLogger(__name__)

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

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        if self.save_videos or self.save_images:
            log.info(f"Visualization output at : {self.save_dir.absolute()}")

    def on_video_loop_end(self, engine, video_metadata, video_idx, detections,
                          image_pred):
        if self.save_videos or self.save_images:
            progress = engine.callbacks.get("progress", Progressbar(dummy=True))
            self.visualize(engine.tracker_state, video_idx, detections, image_pred, progress)
            progress.on_module_end(None, "vis", None)

    """ 
    #TODO implement the online visualization
    previous code:
        if self.cfg.show_online:
        tracker_state = engine.tracker_state
        if tracker_state.detections_gt is not None:
            ground_truths = tracker_state.detections_gt[
                tracker_state.detections_gt.image_id == image_metadata.name
            ]
        else:
            ground_truths = None
        if len(detections) == 0:
            image = image
        else:
            detections = detections[detections.image_id == image_metadata.name]
            image = self.draw_frame(image_metadata,
                                    detections, ground_truths, "inf", image=image)
        if platform.system() == "Linux" and self.video_name not in self.windows:
            self.windows.append(self.video_name)
            cv2.namedWindow(str(self.video_name),
                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(self.video_name), image.shape[1], image.shape[0])
        cv2.imshow(str(self.video_name), image)
        cv2.waitKey(1)
    """

    def visualize(self, tracker_state: TrackerState, video_id, detections, image_preds, progress=None):
        log.info(f"[DEBUG] detections columns at vis: {list(detections.columns)}")
        log.info(f"[DEBUG] has bbox_pitch? {'bbox_pitch' in detections.columns}")
        counts = detections.groupby("image_id").size()
        log.info(f"[DEBUG] non-empty frames: {(counts > 0).sum()} / {len(counts)}")
        has_pitch_per_det = detections['bbox_pitch'].apply(lambda v: isinstance(v, dict))
        pitch_frames = has_pitch_per_det.groupby(detections['image_id']).any()
        log.info(f"[DEBUG] frames with any bbox_pitch dict: {int(pitch_frames.sum())} / {len(pitch_frames)}")
        first_bad = pitch_frames.index[~pitch_frames].tolist()[:5]
        log.info(f"[DEBUG] first frames missing bbox_pitch: {first_bad}")
        
        def _extract_points(df_slice):
            pts = []
            for _, row in df_slice.iterrows():
                bp = row.get('bbox_pitch', None)
                if isinstance(bp, dict):
                    x = bp.get('x_bottom_middle')
                    y = bp.get('y_bottom_middle')
                    if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                        pts.append((float(x), float(y)))
            return pts
        
        image_metadatas = tracker_state.image_metadatas[tracker_state.image_metadatas.video_id == video_id]
        image_gts = tracker_state.image_gt[tracker_state.image_gt.video_id == video_id]
        nframes = len(image_metadatas)
        video_name = tracker_state.video_metadatas.loc[video_id]["name"]
        for visualizer in self.visualizers.values():
            try:
                visualizer.preproces(detections, tracker_state.detections_gt, image_preds, tracker_state.image_gt)
            except Exception as e:
                log.warning(f"Visualizer {Visualizer} raised error : {e} during preprocess.")
        total = self.max_frames or len(image_metadatas.index)
        progress.init_progress_bar("vis", "Visualization", total)
        detection_preds_by_image = detections.groupby("image_id")
        detection_gts_by_image = tracker_state.detections_gt.groupby("image_id")
        step = max(1, nframes // total) 
        args = [create_draw_args(
            image_id,
            self,
            image_metadatas,
            get_group(detection_preds_by_image, image_id),
            get_group(detection_gts_by_image, image_id),
            image_gts,
            image_preds,
            nframes,
        ) for image_id in islice(image_metadatas.index, 0, None, step)]
        if self.save_videos:
            image = cv2_load_image(image_metadatas.iloc[0].file_path)
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.video_fps),
                (image.shape[1], image.shape[0]),
            )
        with Pool() as p:
            for output_image, file_name in p.imap(process_frame, args):
                if self.save_images:
                    filepath = self.save_dir / "images" / str(video_name) / file_name
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(filepath), output_image)
                if self.save_videos:
                    video_writer.write(output_image)
                progress.on_module_step_end(None, "vis", None, None)

    def draw_frame(self, image_metadata, detections_pred, detections_gt,
                   image_pred, image_gt, nframes):
        image = cv2_load_image(image_metadata.file_path)
        
        # --- 追加: bbox_pitch からピッチ上の描画点を取り出す ---
        pitch_pts = []
        try:
            if len(detections_pred) > 0 and 'bbox_pitch' in detections_pred.columns:
                for _, row in detections_pred.iterrows():
                    bp = row['bbox_pitch']
                    if isinstance(bp, dict):
                        x = bp.get('x_bottom_middle')
                        y = bp.get('y_bottom_middle')
                        if x is not None and y is not None:
                            # NaN/inf は弾く
                            if np.isfinite(x) and np.isfinite(y):
                                pitch_pts.append((float(x), float(y)))
        except Exception:
            # ロバストにスルー（可視化を止めない）
            pitch_pts = []

        # --- 追加: image_pred に差し込んで可視化側へ渡す ---
        # Seriesでもdictでも安全にコピーして新しいキーを追加
        if isinstance(image_pred, pd.Series):
            image_pred = image_pred.copy()
            image_pred.loc['bbox_pitch_points'] = pitch_pts
        elif isinstance(image_pred, dict):
            image_pred = dict(image_pred)
            image_pred['bbox_pitch_points'] = pitch_pts
        else:
            # 想定外型でも可視化が落ちないようにする
            pass
        
        for visualizer in self.visualizers.values():
            try:
                visualizer.draw_frame(image, detections_pred, detections_gt, image_pred, image_gt)
            except Exception as e:
                log.warning(f"Visualizer {Visualizer} raised error : {e} during drawing.")
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


def get_group(g, key):
    if key in g.groups:
        return g.get_group(key)
    # 空でも後段が期待する列を持たせる（列が無いせいで描画がスキップされるのを防ぐ）
    return pd.DataFrame({
        "bbox_ltwh": pd.Series(dtype=object),
        "bbox_pitch": pd.Series(dtype=object),
        "role": pd.Series(dtype=object),
        "team": pd.Series(dtype=object),
        "jersey_number": pd.Series(dtype=float),
    })