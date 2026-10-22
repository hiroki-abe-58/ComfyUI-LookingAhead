"""Stand-ins for the few diffusers and transformers names the vendored Looking-Ahead files import, so the pack needs
neither package. Only what those files use is provided: `config` with the constructor arguments, the scheduler's output
type and an FA4 availability probe."""
import functools
import inspect
import types
from dataclasses import dataclass
from enum import Enum

import torch


def is_flash_attn_4_available():
    try:
        from transformers.utils.import_utils import is_flash_attn_4_available as probe
    except ImportError:
        return False
    return probe()


def register_to_config(init):
    """Like diffusers' decorator: store the constructor arguments (defaults included) in `self.config`."""
    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        object.__setattr__(self, "config", types.SimpleNamespace(**values))
        init(self, *args, **kwargs)

    return wrapper


class ConfigMixin:
    pass


class ModelMixin(torch.nn.Module):
    pass


class SchedulerMixin:
    pass


@dataclass
class SchedulerOutput:
    prev_sample: torch.Tensor


class KarrasDiffusionSchedulers(Enum):
    """Empty: the vendored scheduler only lists it as `_compatibles`."""
