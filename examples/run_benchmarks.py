import moons

# TODO seed all and test results
# TODO benchmark timings

TRAINER_CONFIG = {
    "logger": False,
    "limit_val_batches": 0,  # disable validation
}


def _moons_translation():
    config = {
        "epochs": 5,
        "flow": {"transformer": "torchnf.transformers.Translation"},
    }
    metrics = moons.main(config, TRAINER_CONFIG)
    assert metrics["test_loss"] < 2


def _moons_affine():
    config = {
        "epochs": 5,
        "flow": {"transformer": "torchnf.transformers.AffineTransform"},
    }
    metrics = moons.main(config, TRAINER_CONFIG)
    assert metrics["test_loss"] < 1.5


def _moons_spline():
    config = {
        "epochs": 5,
        "flow": {
            "transformer": {
                "class_path": "torchnf.transformers.RQSplineTransform",
                "init_args": {
                    "n_segments": 4,
                    "interval": [-2, 2],
                },
            },
        },
    }
    metrics = moons.main(config, TRAINER_CONFIG)
    assert metrics["test_loss"] < 1.5


def main():

    _moons_translation()
    _moons_affine()
    _moons_spline()


if __name__ == "__main__":
    main()
