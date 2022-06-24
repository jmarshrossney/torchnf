import moons

# TODO seed all and test results
# TODO benchmark timings


def _moons_translation():
    config = {
        "flow": {"transformer": "torchnf.transformers.Translation"},
    }
    metrics = moons.main(config)
    assert metrics["Test/loss"] < 2.5


def _moons_affine():
    config = {
        "flow": {"transformer": "torchnf.transformers.AffineTransform"},
    }
    metrics = moons.main(config)
    assert metrics["Test/loss"] < 2


def main():

    _moons_translation()
    _moons_affine()


if __name__ == "__main__":
    main()
