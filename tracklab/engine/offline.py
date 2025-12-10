import logging

from tracklab.engine import TrackingEngine
from tracklab.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, tracker_state, video, video_id):
        # ★ 最初に、どの video を処理しているかログ
        log.info(
            f"[OfflineEngine] === Start video_loop: "
            f"video_id={video_id}, video={getattr(video, 'name', video)} ==="
        )

        # 各モジュールのリセット
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                log.debug(f"[OfflineEngine] Resetting model: {name}")
                model.reset()

        detections, image_pred = tracker_state.load()
        if len(self.module_names) == 0:
            log.info(
                f"[OfflineEngine] No modules in pipeline. "
                f"Returning raw detections for video_id={video_id}"
            )
            return detections, image_pred

        image_filepaths = {idx: fn for idx, fn in image_pred["file_path"].items()}
        model_names = self.module_names
        num_models = len(model_names)

        log.info(
            f"[OfflineEngine] video_id={video_id}: "
            f"{len(image_filepaths)} frames, {num_models} modules in pipeline: "
            + " -> ".join(model_names)
        )

        for model_idx, model_name in enumerate(model_names, start=1):
            model = self.models[model_name]
            level = getattr(model, "level", "unknown")

            log.info(
                f"[OfflineEngine] [video_id={video_id}] "
                f"({model_idx}/{num_models}) Starting module: "
                f"{model_name} (level={level})"
            )

            try:
                # -----------------------------
                # video-level module
                # -----------------------------
                if level == "video":
                    detections = model.process(detections, image_pred)
                    log.info(
                        f"[OfflineEngine] [video_id={video_id}] "
                        f"Finished video-level module {model_name} "
                        f"(num_dets={len(detections)})"
                    )
                    continue

                # -----------------------------
                # detection/image-level module
                # -----------------------------
                self.datapipes[model_name].update(
                    image_filepaths, image_pred, detections
                )

                self.callback(
                    "on_module_start",
                    task=model_name,
                    dataloader=self.dataloaders[model_name],
                )

                # バッチごとに処理。データ数が多いので N バッチごとにログする
                for batch_idx, batch in enumerate(
                    self.dataloaders[model_name], start=1
                ):
                    if batch_idx % 100 == 1:
                        log.info(
                            f"[OfflineEngine] [video_id={video_id}] "
                            f"module={model_name}: processing batch {batch_idx}"
                        )

                    detections, image_pred = self.default_step(
                        batch, model_name, detections, image_pred
                    )

                self.callback(
                    "on_module_end", task=model_name, detections=detections
                )

                log.info(
                    f"[OfflineEngine] [video_id={video_id}] "
                    f"Finished module {model_name} "
                    f"(num_dets={len(detections)})"
                )

                if detections.empty:
                    log.warning(
                        f"[OfflineEngine] [video_id={video_id}] "
                        f"Detections are empty after module {model_name}. "
                        f"Stopping pipeline for this video."
                    )
                    return detections, image_pred

            except Exception as e:
                # Python 例外で落ちた場合はここに来る
                log.exception(
                    f"[OfflineEngine] ERROR in module {model_name} "
                    f"on video_id={video_id}: {e}"
                )
                # そのまま上に投げる（挙動は今まで通り）
                raise

        log.info(
            f"[OfflineEngine] === Finished video_loop: "
            f"video_id={video_id}, num_dets={len(detections)} ==="
        )
        return detections, image_pred