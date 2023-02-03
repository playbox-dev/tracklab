from typing import Mapping, Sequence
import wandb
from omegaconf import OmegaConf
import pandas as pd

keep_dict = {
    "dataset": ["dataset_path", "nframes", "nvid", "vids_dict"],
    "detect": ["bbox", "predict", "train"],
    "eval": ["mot"],
    "reid": ["data", "loss", "model", "sampler", "test", "train", "dataset"],
    "track": True,
}


def normalize_subdict(subdict):
    subdict["target"] = subdict.pop("_target_")
    if "cfg" in subdict:
        for k, v in subdict["cfg"].items():
            subdict[k] = v
        del subdict["cfg"]
    return subdict


def init(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = {}
    for key, values in keep_dict.items():
        subdict = cfg[key]
        subdict = normalize_subdict(subdict)
        if values is True:
            new_cfg[key] = subdict
        else:
            new_subdict = {}
            for value in values + ["target"]:
                if value in subdict:
                    new_subdict[value] = subdict[value]
                    if value == "target":
                        new_subdict[value] = new_subdict[value].split(".")[-1]

            new_cfg[key] = new_subdict

    wandb.init(project=cfg["experiment_name"], entity="pbtrack", config=new_cfg)


def log(res_dict, name, video_dict=None):
    wandb.log(
        {f"{name}/{k}": v for k, v in res_dict.items()},
        step=0,
    )
    if video_dict is not None:
        video_df = pd.DataFrame.from_dict(video_dict, orient="index")
        video_df.insert(0, "video", video_df.index)
        wandb.log({f"{name}/videos": video_df}, step=0)


def apply_recursively(d, f=lambda v: v, filter=lambda k, v: True, always_filter=False):
    """
    Apply a function to leaf values of a dict recursively and/or filter dict

    Args:
        f: function taking the value of a leaf as argument and returning
           a transformation of that value
        filter: condition to apply f to only (sub-)branches of the tree
        always_filter: if true filter sub-branches, if false only filter leaves.
    Returns:
        transformed and filtered dict
    """
    for k, v in d.items():
        if isinstance(v, Mapping):
            if always_filter:
                d[k] = apply_recursively(v, f, filter) if filter(k, v) else v
            else:
                d[k] = apply_recursively(v, f, filter, always_filter)
        elif isinstance(v, str):
            d[k] = f(v) if filter(k, v) else v
        elif isinstance(v, Sequence):
            if len(v) == 0:
                d[k] = v
            elif isinstance(v[0], Mapping):
                d[k] = [apply_recursively(val, f, filter) for val in v]
            else:
                d[k] = [f(val) if filter(k, val) else val for val in v]
        else:
            d[k] = f(v) if filter(k, v) else v
    return d
