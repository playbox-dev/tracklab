import re
from abc import ABC, ABCMeta
from typing import List

import logging

log = logging.getLogger(__name__)


class MetaModule(ABCMeta):
    @property
    def name(cls):
        name = cls.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(cls):
        name = cls.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]


class Module(metaclass=ABCMeta):
    input_columns = None
    output_columns = None
    training_enabled = False
    forget_columns = []

    @property
    def name(self):
        name = self.__class__.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(self):
        name = self.__class__.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]

    def validate_input(self, dataframe):
        assert self.input_columns is not None, "Every model should define its inputs"
        for col in self.input_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The input detection should contain {col}.")

    def validate_output(self, dataframe):
        assert self.output_columns is not None, "Every model should define its outputs"
        for col in self.output_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The output detection should contain {col}.")

    def get_input_columns(self, level):
        if isinstance(self.input_columns, list):
            return self.input_columns if level == "detection" else []
        elif isinstance(self.input_columns, dict):
            return self.input_columns.get(level, [])

    def get_output_columns(self, level):
        if isinstance(self.output_columns, list):
            return self.output_columns if level == "detection" else []
        elif isinstance(self.output_columns, dict):
            return self.output_columns.get(level, [])


class Pipeline:
    def __init__(self, models: List[Module]):
        self.models = [model for model in models if model.name != "skip"]
        log.info("Pipeline: " + " -> ".join(model.name for model in self.models))

    def validate(self, load_columns: dict[str, set]):
        columns = {k: set(v) for k, v in load_columns.items()}
        for level in ["image", "detection"]:
            for model in self.models:
                if model.input_columns is None or model.output_columns is None:
                    raise AttributeError(
                        f"{type(model)} should contain input_ and output_columns"
                    )
                if not set(model.get_input_columns(level)).issubset(columns[level]):
                    raise AttributeError(
                        f"The {model.name} model doesn't have "
                        "all the input needed, "
                        f"needed {model.get_input_columns(level)}, provided {columns[level]}"
                    )
                columns[level].update(model.get_output_columns(level))
        log.info(f"Pipeline has been validated")
        
    def __iter__(self):
        """
        `for model in pipeline:` で回されたときに、
        各モジュールの開始／終了ログを出すためのイテレータ
        """
        total = len(self.models)
        for idx, model in enumerate(self.models, start=1):
            log.info(
                f"[Pipeline] Starting module {idx}/{total}: {model.name}"
            )
            yield model
            log.info(
                f"[Pipeline] Finished module {idx}/{total}: {model.name}"
            )

    def __str__(self):
        return " -> ".join(model.name for model in self.models)

    def __getitem__(self, item: int):
        return self.models[item]

    def is_empty(self):
        return len(self.models) == 0
    
    def run(self, tracker_state):
        """
        パイプライン内の各 Module を順番に実行しつつ、
        どのモジュールを実行しているかをログに出す。
        """
        if self.is_empty():
            log.info("[PIPELINE] No modules to run (pipeline is empty).")
            return tracker_state

        log.info(
            "[PIPELINE] Start pipeline with %d modules: %s",
            len(self.models),
            " -> ".join(m.name for m in self.models),
        )

        for idx, model in enumerate(self.models, start=1):
            log.info(
                "[PIPELINE] (%d/%d) Running module: %s",
                idx, len(self.models), model.name
            )
            tracker_state = model(tracker_state)
            log.info(
                "[PIPELINE] (%d/%d) Finished module: %s",
                idx, len(self.models), model.name
            )

        log.info("[PIPELINE] All modules finished.")
        return tracker_state


class Skip(Module):
    def __init__(self, **kwargs):
        pass

    @property
    def name(self):
        return "skip"
