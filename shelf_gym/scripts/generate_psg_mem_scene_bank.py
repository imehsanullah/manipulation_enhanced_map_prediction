#!/usr/bin/env python3
"""Generate a hash-pinned PSG-MEM scene-only tuning bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


SCHEMA = "psg_mem_scene_bank_generation_manifest_v1"
COMPLETION_SCHEMA = "psg_mem_scene_bank_generation_completion_v1"
FAILURE_SCHEMA = "psg_mem_scene_bank_generation_failure_v1"
BANK_ID = "psg_mem_x1_tuning_20260719_v1"
GROUP_COUNT = 21
GENERATOR_SEED_BASE = 2026071900
OCCUPANCY_STRATA = (
    ("low_official_range", 0.35),
    ("medium_official_range", 0.375),
    ("high_official_range", 0.40),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()

    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=repo)
    raw_untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo
    )
    untracked = []
    for raw_name in raw_untracked.split(b"\0"):
        if not raw_name:
            continue
        relative = raw_name.decode("utf-8")
        path = repo / relative
        if path.is_symlink():
            payload = ("symlink:" + str(path.readlink())).encode("utf-8")
            kind = "symlink"
            digest = hashlib.sha256(payload).hexdigest()
            size = len(payload)
        elif path.is_file():
            kind = "file"
            digest = _sha256(path)
            size = path.stat().st_size
        else:
            raise ValueError(f"unsupported untracked repository entry: {path}")
        untracked.append(
            {
                "path": relative,
                "kind": kind,
                "size_bytes": int(size),
                "sha256": digest,
            }
        )
    state = {
        "path": str(repo.resolve()),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "head_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked,
    }
    state["worktree_state_sha256"] = hashlib.sha256(
        _canonical_bytes(
            {
                "commit": state["commit"],
                "head_diff_sha256": state["head_diff_sha256"],
                "untracked_files": untracked,
            }
        )
    ).hexdigest()
    return state


def build_scene_groups() -> list[Dict[str, Any]]:
    groups = []
    for replicate in range(7):
        for stratum, occupancy in OCCUPANCY_STRATA:
            index = len(groups)
            groups.append(
                {
                    "scene_group_id": f"PSG_X1_TUNE_{index:03d}",
                    "scene_role": "fresh_tuning_x1",
                    "generator_seed": GENERATOR_SEED_BASE + index,
                    "occupancy_stratum": stratum,
                    "occupancy_target": float(occupancy),
                    "alignment": 0,
                    "replicate": replicate,
                }
            )
    return groups


def _verified_manifest(path: Path, expected_sha256: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if _sha256(resolved) != expected_sha256:
        raise ValueError("scene-bank manifest hash mismatch")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return dict(manifest)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected scene-bank generation schema")
    if manifest.get("bank_id") != BANK_ID:
        raise ValueError("unexpected scene-bank ID")
    if manifest.get("run_type") != "x1_fresh_tuning_scene_generation":
        raise ValueError("unexpected scene-bank run type")
    if manifest.get("host") != socket.gethostname():
        raise ValueError("scene-bank host mismatch")
    groups = manifest.get("scene_groups")
    if groups != build_scene_groups() or manifest.get("group_count") != GROUP_COUNT:
        raise ValueError("scene-bank groups differ from the frozen X1 scope")
    expected_group_hash = hashlib.sha256(_canonical_bytes(groups)).hexdigest()
    if manifest.get("scene_groups_sha256") != expected_group_hash:
        raise ValueError("scene-bank group hash mismatch")
    seeds = {int(group["generator_seed"]) for group in groups}
    if seeds & set(range(2026071800, 2026071818)):
        raise ValueError("scene-bank seeds overlap the opened necessity bank")
    config = manifest.get("generator_config")
    expected_config = {
        "use_ycb": True,
        "use_occupancy_for_placing": True,
        "max_obj_num": 25,
        "max_occupancy_threshold": 0.4,
        "hard_only": False,
        "planner_executed": False,
        "mapping_model_loaded": False,
        "graph_model_loaded": False,
    }
    if config != expected_config:
        raise ValueError("scene-bank generator configuration changed")
    root = manifest.get("output_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("scene-bank output root must be absolute")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"generator"}:
        raise ValueError("scene-bank manifest requires the exact generator pin")
    identity = artifacts["generator"]
    this_file = Path(__file__).resolve()
    if (
        not isinstance(identity, Mapping)
        or Path(str(identity.get("path", ""))).resolve() != this_file
        or identity.get("sha256") != _sha256(this_file)
    ):
        raise ValueError("scene-bank generator identity mismatch")
    source = manifest.get("source_repository")
    repo = this_file.parents[2]
    if (
        not isinstance(source, Mapping)
        or Path(str(source.get("path", ""))).resolve() != repo
    ):
        raise ValueError("scene-bank source repository mismatch")
    state = _git_state(repo)
    if source.get("worktree_state_sha256") != state["worktree_state_sha256"]:
        raise ValueError("scene-bank source repository state changed")
    policy = manifest.get("artifact_policy")
    if policy != {
        "scene_arrangements": True,
        "physics_snapshots": True,
        "compact_records": True,
        "raw_maps": False,
        "datasets_or_hdf5": False,
        "checkpoints_or_models": False,
        "planner_or_oracle": False,
    }:
        raise ValueError("scene-bank artifact policy changed")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def generate_bank(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    output_root: Path,
) -> Dict[str, Any]:
    validate_manifest(manifest)
    root = output_root.expanduser().resolve()
    if root != Path(str(manifest["output_root"])).resolve():
        raise ValueError("requested scene-bank output root does not match its pin")
    if root.exists():
        raise FileExistsError(f"refusing to reuse scene-bank output root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _write_new_json(root / "approved_manifest.json", dict(manifest))

    from shelf_gym.utils.cnabu_mem_experiment_control import (
        apply_initial_state_snapshot,
        build_initial_state_snapshot,
        capture_runtime_physics_state,
        configure_controlled_mem,
        configure_deterministic_process_environment,
        physics_state_sha256,
    )

    process_environment = configure_deterministic_process_environment()
    from shelf_gym.scripts.data_generation.pushing_collection import PushingCollection

    environment: Any = None
    records = []
    active_group = None
    try:
        config = manifest["generator_config"]
        environment = PushingCollection(
            render=False,
            show_vis=False,
            use_ycb=bool(config["use_ycb"]),
            max_obj_num=int(config["max_obj_num"]),
            max_occupancy_threshold=float(config["max_occupancy_threshold"]),
            use_occupancy_for_placing=bool(config["use_occupancy_for_placing"]),
        )
        for group in manifest["scene_groups"]:
            active_group = str(group["scene_group_id"])
            group_root = root / active_group
            group_root.mkdir(parents=False, exist_ok=False)
            seed = int(group["generator_seed"])
            random.seed(seed)
            np.random.seed(seed)
            runtime_control = configure_controlled_mem(environment, seed=seed)
            environment.reset_env(
                occupancy_threshold=float(group["occupancy_target"]),
                alignment=int(group["alignment"]),
                hard_only=bool(config["hard_only"]),
            )
            scene_path = group_root / "placed_objects.pkl"
            _write_new_bytes(
                scene_path,
                pickle.dumps(
                    environment.sampled_objects, protocol=pickle.HIGHEST_PROTOCOL
                ),
            )
            arrangement = pickle.loads(scene_path.read_bytes())
            environment.restore_shelf_state(arrangement)
            snapshot = build_initial_state_snapshot(
                environment, scene_path=scene_path, seed=seed
            )
            precanonical_hash = snapshot["state_sha256"]
            first_application = apply_initial_state_snapshot(
                environment,
                snapshot,
                scene_path=scene_path,
                require_exact=False,
            )
            if not first_application["matches_snapshot"]:
                snapshot["state"] = capture_runtime_physics_state(environment)
                snapshot["state_sha256"] = physics_state_sha256(snapshot["state"])
            fixed_point = apply_initial_state_snapshot(
                environment,
                snapshot,
                scene_path=scene_path,
                require_exact=True,
            )
            snapshot.update(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "host": socket.gethostname(),
                    "python": sys.executable,
                    "scene_group_id": active_group,
                    "scene_role": group["scene_role"],
                    "generator_seed": seed,
                    "occupancy_stratum": group["occupancy_stratum"],
                    "occupancy_target": float(group["occupancy_target"]),
                    "canonicalization": {
                        "precanonical_state_sha256": precanonical_hash,
                        "first_application": first_application,
                        "fixed_point_verification": fixed_point,
                    },
                    "process_environment": process_environment,
                    "runtime_control": runtime_control,
                    "planner_executed": False,
                    "mapping_model_loaded": False,
                    "graph_model_loaded": False,
                    "raw_maps_written": False,
                    "gt_used_for_planner_input": False,
                }
            )
            snapshot_path = group_root / "initial_state.json"
            _write_new_json(snapshot_path, snapshot)
            record = {
                **dict(group),
                "scene_path": str(scene_path),
                "scene_sha256": _sha256(scene_path),
                "snapshot_path": str(snapshot_path),
                "snapshot_sha256": _sha256(snapshot_path),
                "state_sha256": snapshot["state_sha256"],
                "object_count": len(snapshot["state"]["objects"]),
                "planner_executed": False,
                "mapping_model_loaded": False,
                "graph_model_loaded": False,
                "raw_maps_written": False,
            }
            record_path = group_root / "generation_record.json"
            _write_new_json(record_path, record)
            record["generation_record_path"] = str(record_path)
            record["generation_record_sha256"] = _sha256(record_path)
            records.append(record)
    except BaseException as error:
        failure_path = root / "generation_failure.json"
        if not failure_path.exists():
            _write_new_json(
                failure_path,
                {
                    "schema": FAILURE_SCHEMA,
                    "bank_id": BANK_ID,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "active_scene_group_id": active_group,
                    "completed_group_count": len(records),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "artifact_policy": dict(manifest["artifact_policy"]),
                },
            )
        raise
    finally:
        if environment is not None:
            environment.close()

    completion = {
        "schema": COMPLETION_SCHEMA,
        "bank_id": BANK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "group_count": len(records),
        "groups": records,
        "process_environment": process_environment,
        "artifact_policy": dict(manifest["artifact_policy"]),
    }
    _write_new_json(root / "generation_completion.json", completion)
    return completion


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _verified_manifest(manifest_path, args.manifest_sha256)
    completion = generate_bank(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(completion, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
