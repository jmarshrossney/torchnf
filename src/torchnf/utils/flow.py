from typing import Union

from torchnf.abc import DensityTransform, Transformer


def invert(cls: Union[type[DensityTransform], type[Transformer]]) -> type:
    """
    Swaps the ``forward`` and ``inverse`` methods of the given class.
    """

    class Inverted(cls):
        forward = cls.inverse
        inverse = cls.forward

    return Inverted
