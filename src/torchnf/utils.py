import torch


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1).sum(dim=1)


def expand_elements(
    x: torch.Tensor, shape: torch.Size, stack_dim: int
) -> torch.Tensor:
    return torch.stack(
        [el.expand(shape) for el in x.split(1)],
        dim=stack_dim,
    )


def simple_net_builder(net_spec: list[dict]) -> torch.nn.Sequential:

    # TODO: augment namespace with custom modules to allow user
    # to request custom layers?

    layers = []
    for layer_spec in net_spec:

        cls = getattr(torch.nn, layer_spec["class"])
        args = layer_spec["args"]
        layer = cls(**args)
        layers.append(layer)

    return torch.nn.Sequential(*layers)
