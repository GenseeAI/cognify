import json
import logging
import importlib

from .common import CogBase

logger = logging.getLogger(__name__)


def dump_params(params: list[CogBase], log_path: str):
    logger.debug(f"---- Dumping parameters to {log_path} ----")
    ps = [param.to_dict() for param in params]
    with open(log_path, "w+") as f:
        json.dump(ps, f, indent=4)


def load_params(log_path: str) -> list[CogBase]:
    logger.debug(f"---- Loading parameters from {log_path} ----")
    with open(log_path, "r") as f:
        data = json.load(f)
    params = []
    for dat in data:
        params.append(build_param(dat))
    return params


def build_param(data: dict):
    module_name = data.pop("__module__")
    class_name = data.pop("__class__")
    module = importlib.import_module(module_name)

    cls = getattr(module, class_name)
    return cls.from_dict(data)
