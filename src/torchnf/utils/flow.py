import functools
from typing import Union

import torch

from torchnf.abc import DensityTransform, Transformer


def invert(
    cls: Union[type[DensityTransform], type[Transformer]]
) -> Union[type[DensityTransform], type[Transformer]]:
    """
    Swaps the ``forward`` and ``inverse`` methods of the given class.
    """

    class Inverted(cls):
        forward = cls.inverse
        inverse = cls.forward

    return Inverted


def eval_mode(meth):
    """
    Decorator which sets a model to eval mode for the duration of the method.
    """

    @functools.wraps(meth)
    def wrapper(model: torch.nn.Module, *args, **kwargs):
        original_state = model.training
        model.eval()
        out = meth(model, *args, **kwargs)
        model.train(original_state)
        return out

    return wrapper
