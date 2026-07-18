"""Capture one outcome-blind, settled PyBullet start state for paired MEM runs."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from shelf_gym.utils.cnabu_mem_experiment_control import (
    apply_initial_state_snapshot,
    build_initial_state_snapshot,
    capture_runtime_physics_state,
    configure_controlled_mem,
    configure_deterministic_process_environment,
    physics_state_sha256,
    write_initial_state_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    return {
        "path": str(REPO_ROOT),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predefined-scene", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    scene_path = args.predefined_scene.resolve()
    output_path = args.output_json.resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))

    process_environment = configure_deterministic_process_environment()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    from shelf_gym.scripts.data_generation.pushing_collection import (
        PushingCollection,
    )

    environment = PushingCollection(
        render=bool(args.render),
        show_vis=False,
        use_ycb=True,
        max_obj_num=25,
        max_occupancy_threshold=0.4,
        use_occupancy_for_placing=True,
    )
    try:
        control = configure_controlled_mem(environment, seed=int(args.seed))
        with scene_path.open("rb") as handle:
            arrangement = pickle.load(handle)
        environment.restore_shelf_state(arrangement)
        snapshot = build_initial_state_snapshot(
            environment,
            scene_path=scene_path,
            seed=int(args.seed),
        )
        precanonical_state_sha256 = snapshot["state_sha256"]
        first_application = apply_initial_state_snapshot(
            environment,
            snapshot,
            scene_path=scene_path,
            require_exact=False,
        )
        if not first_application["matches_snapshot"]:
            snapshot["state"] = capture_runtime_physics_state(environment)
            snapshot["state_sha256"] = physics_state_sha256(snapshot["state"])
        verification = apply_initial_state_snapshot(
            environment,
            snapshot,
            scene_path=scene_path,
            require_exact=True,
        )
        snapshot["canonicalization"] = {
            "precanonical_state_sha256": precanonical_state_sha256,
            "first_application": first_application,
            "fixed_point_verification": verification,
        }
        snapshot.update(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "python": sys.executable,
                "command": " ".join(sys.argv),
                "repo": _git_state(),
                "experiment_control": control,
                "process_environment": process_environment,
                "planner_executed": False,
                "mapping_model_loaded": False,
                "graph_model_loaded": False,
                "gt_used_for_planner_input": False,
            }
        )
        write_initial_state_snapshot(snapshot, output_path)
    finally:
        environment.close()

    print(
        json.dumps(
            {
                "output": str(output_path),
                "state_sha256": snapshot["state_sha256"],
                "object_count": len(snapshot["state"]["objects"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
