import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tracking as tracking_module


@pytest.mark.parametrize("error", [None, RuntimeError("virtual training failed")])
def test_fit_always_finishes_tracking(monkeypatch, error):
    events = []

    class FakeTracking:
        def __init__(self, **kwargs):
            events.append(("started", kwargs["experiment_name"]))

        def finish(self):
            events.append(("finished", None))

    def virtual_fit(self, logger):
        events.append(("training_returned", None))
        if error is not None:
            raise error
        return "done"

    monkeypatch.setattr(tracking_module, "Tracking", FakeTracking)
    monkeypatch.setattr(RayPPOTrainer, "_fit", virtual_fit)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "virtual-project",
                "experiment_name": "virtual-run",
                "logger": ["wandb"],
            }
        }
    )

    if error is None:
        assert trainer.fit() == "done"
    else:
        with pytest.raises(RuntimeError, match="virtual training failed"):
            trainer.fit()

    assert events == [
        ("started", "virtual-run"),
        ("training_returned", None),
        ("finished", None),
    ]


def test_tracking_finish_is_idempotent(monkeypatch):
    finish_codes = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: None,
        finish=lambda exit_code: finish_codes.append(exit_code),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracker = tracking_module.Tracking(
        project_name="virtual-project",
        experiment_name="virtual-run",
        default_backend=["wandb"],
    )
    tracker.finish()
    tracker.finish()

    assert finish_codes == [0]
