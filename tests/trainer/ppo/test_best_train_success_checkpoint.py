import json
import os
from types import MethodType

from omegaconf import OmegaConf
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_best_train_success_overwrites_fixed_checkpoint(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "save_best_train_success": True,
                "best_train_success_metric": "episode/success_rate",
                "default_local_dir": str(tmp_path),
            }
        }
    )
    trainer.best_train_success = float("-inf")
    trainer.global_steps = 3
    saved_steps = []

    def fake_save_checkpoint(self, checkpoint_name=None, update_latest=True):
        checkpoint_path = os.path.join(
            self.config.trainer.default_local_dir, checkpoint_name
        )
        os.makedirs(checkpoint_path)
        with open(os.path.join(checkpoint_path, "payload.txt"), "w") as f:
            f.write(str(self.global_steps))
        saved_steps.append(self.global_steps)

    trainer._save_checkpoint = MethodType(fake_save_checkpoint, trainer)

    assert trainer._maybe_save_best_train_success(
        {"episode/success_rate": 0.25}
    )
    assert not trainer._maybe_save_best_train_success(
        {"episode/success_rate": 0.20}
    )

    trainer.global_steps = 7
    assert trainer._maybe_save_best_train_success(
        {"episode/success_rate": 0.50}
    )

    checkpoint_path = tmp_path / "best_train_success"
    with open(checkpoint_path / "best_train_success.json") as f:
        metadata = json.load(f)

    assert saved_steps == [3, 7]
    assert metadata == {
        "metric": "episode/success_rate",
        "value": 0.5,
        "global_step": 7,
    }
    assert (checkpoint_path / "payload.txt").read_text() == "7"


def test_load_best_train_success_metadata(tmp_path):
    metadata_path = tmp_path / "best_train_success" / "best_train_success.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        json.dumps(
            {
                "metric": "episode/success_rate",
                "value": 0.75,
                "global_step": 9,
            }
        )
    )

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"trainer": {"default_local_dir": str(tmp_path)}}
    )
    trainer.best_train_success = float("-inf")

    trainer._load_best_train_success()

    assert trainer.best_train_success == 0.75


def test_resume_path_accepts_best_train_success_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "best_train_success"
    actor_path = checkpoint_path / "actor"
    actor_path.mkdir(parents=True)
    (checkpoint_path / "best_train_success.json").write_text(
        json.dumps(
            {
                "metric": "episode/success_rate",
                "value": 0.75,
                "global_step": 9,
            }
        )
    )
    torch.save({"num_yielded": 9}, checkpoint_path / "data.pt")

    class ActorWorker:
        def __init__(self):
            self.loaded_path = None

        def load_checkpoint(self, path, del_local_after_load=False):
            self.loaded_path = path

    class DataLoader:
        def __init__(self):
            self.state = None

        def load_state_dict(self, state):
            self.state = state

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "resume_path",
                "resume_from_path": str(checkpoint_path),
                "default_hdfs_dir": None,
                "default_local_dir": str(tmp_path),
                "val_only": False,
                "del_local_ckpt_after_load": False,
            }
        }
    )
    trainer.global_steps = 0
    trainer.total_training_steps = 20
    trainer.use_critic = False
    trainer.actor_rollout_wg = ActorWorker()
    trainer.train_dataloader = DataLoader()

    trainer._load_checkpoint()

    assert trainer.global_steps == 9
    assert trainer.actor_rollout_wg.loaded_path == str(actor_path)
    assert trainer.train_dataloader.state == {"num_yielded": 9}
