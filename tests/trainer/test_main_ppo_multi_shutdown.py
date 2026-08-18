"""CPU-only smoke test for independently owned main_ppo Ray clusters."""

import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory


def _driver(name, ray_dir, ready, release):
    import ray
    from omegaconf import OmegaConf

    from verl.trainer import main_ppo

    class SmokeRunner:
        def run(self, config):
            ping = ray.remote(lambda value: value)
            assert ray.get(ping.remote(f"{name}-before")) == f"{name}-before"
            Path(ready).touch()
            deadline = time.monotonic() + 60
            while not Path(release).exists():
                assert time.monotonic() < deadline
                time.sleep(0.05)
            assert ray.get(ping.remote(f"{name}-after")) == f"{name}-after"
            print(f"{name}: second ping passed", flush=True)

    main_ppo.TaskRunner = SmokeRunner
    main_ppo.run_ppo(
        OmegaConf.create(
            {
                "ray_init": {
                    "address": "local",
                    "num_cpus": 1,
                    "include_dashboard": False,
                    "_temp_dir": ray_dir,
                }
            }
        )
    )


def _wait_ready(processes, markers):
    deadline = time.monotonic() + 90
    while not all(marker.exists() for marker in markers):
        for process in processes:
            assert process.poll() is None, process.communicate()[0]
        assert time.monotonic() < deadline
        time.sleep(0.1)


def test_two_main_ppo_drivers_shutdown_independently(tmp_path):
    processes = []
    markers = []
    ray_temp_dirs = []
    try:
        for name in ("A", "B"):
            ray_temp = TemporaryDirectory(prefix=f"r2{name}-", dir="/tmp")
            ray_temp_dirs.append(ray_temp)
            ready = tmp_path / f"{name}.ready"
            release = tmp_path / f"{name}.release"
            markers.append(ready)
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, "--driver", name, ray_temp.name, str(ready), str(release)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        _wait_ready(processes, markers)
        (tmp_path / "A.release").touch()
        output_a = processes[0].communicate(timeout=60)[0]
        assert processes[0].returncode == 0, output_a

        # B remains connected and performs its second remote call after A's
        # main_ppo finally block has already called ray.shutdown().
        (tmp_path / "B.release").touch()
        output_b = processes[1].communicate(timeout=60)[0]
        assert processes[1].returncode == 0, output_b
        assert "A: second ping passed" in output_a
        assert "B: second ping passed" in output_b
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        for ray_temp in ray_temp_dirs:
            ray_temp.cleanup()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--driver":
        _driver(*sys.argv[2:])
    else:
        with TemporaryDirectory(prefix="main-ppo-multi-") as directory:
            test_two_main_ppo_drivers_shutdown_independently(Path(directory))
        print("multi-main_ppo shutdown test: PASS", flush=True)
