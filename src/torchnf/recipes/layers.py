import torch

import torchnf.conditioners
import torchnf.flow
import torchnf.transformers
import torchnf.recipes.networks


class CouplingLayerDenseNet:
    def __init__(
        self,
        transformer: torchnf.transformers.Transformer,
        net: torchnf.recipes.networks.DenseNet,
        mask: torch.BoolTensor,
    ) -> None:
        self.transformer = transformer  # deepcopy this?
        self.net = net
        self.mask = mask

        self.size_in = int(mask.sum())
        self.size_out = transformer.n_params * int(mask.logical_not().sum())

    @property
    def conditioner(self) -> torchnf.conditioners.MaskedConditioner:
        return torchnf.conditioners.MaskedConditioner(
            self.net(self.size_in, self.size_out), self.mask, mask_mode="index"
        )

    def __call__(self) -> torchnf.flow.FlowLayer:
        return torchnf.flow.FlowLayer(self.transformer, self.conditioner)


class CouplingLayerConvNet:
    def __init__(
        self,
        transformer: torchnf.transformers.Transformer,
        net: torchnf.recipes.networks.ConvNet,
        mask: torch.BoolTensor,
        *,
        create_channel_dim: bool = True,
    ) -> None:
        self.transformer = transformer  # deepcopy this?
        self.net = net
        self.mask = mask

        self.size_in = None  # lazily infer
        self.size_out = transformer.n_params

        self.create_channel_dim = create_channel_dim

    @property
    def conditioner(self) -> torchnf.conditioners.MaskedConditioner:
        return torchnf.conditioners.MaskedConditioner(
            self.net(self.size_in, self.size_out),
            self.mask,
            mask_mode="fill",
            create_channel_dim=self.create_channel_dim,
        )

    def __call__(self) -> torchnf.flow.FlowLayer:
        return torchnf.flow.FlowLayer(self.transformer, self.conditioner)
