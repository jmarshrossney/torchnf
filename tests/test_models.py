import torch

from torchnf.models import BoltzmannGenerator
from torchnf.prior import SimplePrior
from torchnf.core import Flow, FlowLayer
from torchnf.transformers import AffineTransform
from torchnf.conditioners import SimpleConditioner, MaskedConditioner
from torchnf.networks import DenseNet

data_shape = (6, 6)
prior = SimplePrior(
    torch.distributions.Normal(0, 1), batch_size=100, expand_shape=data_shape
)
target = torch.distributions.Normal(
    loc=torch.full(data_shape, 1), scale=torch.full(data_shape, 0.5)
)


class Target:
    dist = torch.distributions.Normal(
        loc=torch.full(data_shape, 1), scale=torch.full(data_shape, 0.5)
    )

    def __call__(self, sample):
        log_prob = self.dist.log_prob(sample)
        return log_prob.flatten(start_dim=1).sum(dim=1)


target = Target()

n_layers = 1
layers = [
    FlowLayer(AffineTransform(), SimpleConditioner([0, 0]))
    for _ in range(n_layers)
]
flow = Flow(*layers)


class Model(BoltzmannGenerator):
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )


model = Model(prior, target, flow)

initial_metrics = model.validate()

model.fit(1000)

final_metrics = model.validate()

print(initial_metrics)

print(final_metrics)
