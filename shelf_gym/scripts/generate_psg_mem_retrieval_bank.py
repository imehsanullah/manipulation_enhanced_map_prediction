#!/usr/bin/env python3
"""Generate a hash-pinned PSG-MEM retrieval scene bank and target catalogs."""

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

from shelf_gym.scripts.prepare_psg_mem_x3_episode import (
    TARGET_CATALOG_SCHEMA,
    evaluator_target_catalog_from_arrays,
)


MANIFEST_SCHEMA = "psg_mem_retrieval_bank_generation_manifest_v1"
COMPLETION_SCHEMA = "psg_mem_retrieval_bank_generation_completion_v1"
FAILURE_SCHEMA = "psg_mem_retrieval_bank_generation_failure_v1"
GENERATOR_CONFIG = {
    "use_ycb": True,
    "use_occupancy_for_placing": True,
    "max_obj_num": 25,
    "max_occupancy_threshold": 0.4,
    "hard_only": False,
    "planner_executed": False,
    "mapping_model_loaded": False,
    "graph_model_loaded": False,
}
ARTIFACT_POLICY = {
    "scene_arrangements": True,
    "physics_snapshots": True,
    "compact_records": True,
    "evaluator_target_catalogs": True,
    "simulator_instance_ids_evaluator_only": True,
    "raw_maps": False,
    "datasets_or_hdf5": False,
    "checkpoints_or_models": False,
    "planner_or_action_oracle": False,
}
_ALLOWED_RUN_TYPES = {
    "m7_retrieval_candidate_bank_generation": "m7_candidate",
    "m11_confirmatory_candidate_bank_generation": "m11_candidate",
}
_OPENED_GENERATOR_SEEDS = set(range(2026071800, 2026071818)) | set(
    range(2026071900, 2026071921)
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


def _identity(path: str | Path) -> Dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


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
            size = len(payload)
            digest = hashlib.sha256(payload).hexdigest()
        elif path.is_file():
            kind = "file"
            size = path.stat().st_size
            digest = _sha256(path)
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


def _verify_identity(identity: Mapping[str, Any], expected: Path, *, name: str) -> None:
    if (
        not isinstance(identity, Mapping)
        or Path(str(identity.get("path", ""))).resolve() != expected.resolve()
        or identity.get("sha256") != _sha256(expected.resolve())
    ):
        raise ValueError(f"retrieval-bank {name} identity changed")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected retrieval-bank generation schema")
    bank_id = manifest.get("bank_id")
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError("retrieval-bank ID must be non-empty")
    run_type = manifest.get("run_type")
    if run_type not in _ALLOWED_RUN_TYPES:
        raise ValueError("unexpected retrieval-bank run type")
    if (
        manifest.get("host") != socket.gethostname()
        or manifest.get("host") != "moncheri"
    ):
        raise ValueError("retrieval-bank host must be moncheri")
    groups = manifest.get("scene_groups")
    if not isinstance(groups, list) or not 30 <= len(groups) <= 300:
        raise ValueError("retrieval bank requires 30..300 candidate groups")
    if manifest.get("group_count") != len(groups):
        raise ValueError("retrieval-bank group count changed")
    if (
        manifest.get("scene_groups_sha256")
        != hashlib.sha256(_canonical_bytes(groups)).hexdigest()
    ):
        raise ValueError("retrieval-bank group hash mismatch")
    expected_role = _ALLOWED_RUN_TYPES[str(run_type)]
    ids = []
    seeds = []
    allowed_strata = {
        "low_official_range": 0.35,
        "medium_official_range": 0.375,
        "high_official_range": 0.4,
    }
    for index, raw in enumerate(groups):
        if not isinstance(raw, Mapping):
            raise ValueError("retrieval-bank group rows must be mappings")
        required = {
            "scene_group_id",
            "scene_role",
            "generator_seed",
            "occupancy_stratum",
            "occupancy_target",
            "alignment",
            "replicate",
        }
        if set(raw) != required:
            raise ValueError("retrieval-bank group row shape changed")
        group_id = raw.get("scene_group_id")
        seed = raw.get("generator_seed")
        replicate = raw.get("replicate")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("retrieval-bank scene_group_id must be non-empty")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("retrieval-bank generator seed must be non-negative")
        if (
            isinstance(replicate, bool)
            or not isinstance(replicate, int)
            or replicate < 0
        ):
            raise ValueError("retrieval-bank replicate must be non-negative")
        if raw.get("scene_role") != expected_role:
            raise ValueError("retrieval-bank scene role differs from run type")
        stratum = raw.get("occupancy_stratum")
        if (
            stratum not in allowed_strata
            or float(raw.get("occupancy_target", -1.0)) != allowed_strata[stratum]
        ):
            raise ValueError("retrieval-bank occupancy stratum changed")
        if raw.get("alignment") != 0:
            raise ValueError("retrieval-bank alignment must remain zero")
        ids.append(group_id)
        seeds.append(seed)
        if index != replicate * 3 + list(allowed_strata).index(str(stratum)):
            raise ValueError("retrieval-bank replicate/stratum order changed")
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError("retrieval-bank groups and seeds must be unique")
    if set(seeds) & _OPENED_GENERATOR_SEEDS:
        raise ValueError("retrieval-bank seeds overlap an opened bank")
    if manifest.get("generator_config") != GENERATOR_CONFIG:
        raise ValueError("retrieval-bank generator configuration changed")
    output = manifest.get("output_root")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ValueError("retrieval-bank output root must be absolute")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "generator",
        "target_catalog_builder",
    }:
        raise ValueError("retrieval-bank artifact pins changed")
    this_file = Path(__file__).resolve()
    catalog_builder = this_file.with_name("prepare_psg_mem_x3_episode.py")
    _verify_identity(artifacts["generator"], this_file, name="generator")
    _verify_identity(
        artifacts["target_catalog_builder"],
        catalog_builder,
        name="target catalog builder",
    )
    source = manifest.get("source_repository")
    repository = this_file.parents[2]
    if (
        not isinstance(source, Mapping)
        or Path(str(source.get("path", ""))).resolve() != repository
        or source.get("worktree_state_sha256")
        != _git_state(repository)["worktree_state_sha256"]
    ):
        raise ValueError("retrieval-bank source repository state changed")
    if manifest.get("artifact_policy") != ARTIFACT_POLICY:
        raise ValueError("retrieval-bank artifact policy changed")
    authorization = manifest.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("retrieval-bank authorization is absent")
    expected_authorization = {
        "authorized_by": "Ehsan",
        "authorization": "all work needed through completion of the given goal",
        "machine": "moncheri",
        "devices": ["cuda:0", "egl:gpu0"],
        "script": str(this_file),
        "dataset_path": None,
        "output_path": str(Path(output).resolve()),
        "run_type": "comparison",
        "model_outputs_allowed": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("retrieval-bank exact authorization tuple changed")


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    return np.asarray(value)


def build_target_catalog_payload(
    *, scene_group_id: str, instance_maps: Any, semantic_2d: Any
) -> Dict[str, Any]:
    instances = _host_array(instance_maps)
    semantics = _host_array(semantic_2d)
    targets = evaluator_target_catalog_from_arrays(instances, semantics)
    raw_shape = list(np.asarray(semantics).shape)
    if len(raw_shape) != 2:
        raise ValueError("live target catalog requires a 2D semantic map")
    return {
        "schema": TARGET_CATALOG_SCHEMA,
        "scene_group_id": str(scene_group_id),
        "source": "live_gt_height_map_evaluator_only",
        "raw_shape_hw": [int(value) for value in raw_shape],
        "depth_partition": "raw_y_thirds_v1",
        "targets": targets,
        "runtime_visible": False,
        "contains_simulator_instance_ids": True,
        "planner_may_read": False,
    }


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


def _verified_manifest(path: Path, expected_sha256: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if _sha256(resolved) != str(expected_sha256).lower():
        raise ValueError("retrieval-bank manifest hash mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return dict(payload)


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
        raise ValueError("requested retrieval-bank output differs from its pin")
    if root.exists():
        raise FileExistsError(f"refusing to reuse retrieval-bank output: {root}")
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
            before_catalog_hash = physics_state_sha256(
                capture_runtime_physics_state(environment)
            )
            gt_data = environment.get_gt_height_map(no_tqdm=True)
            catalog = build_target_catalog_payload(
                scene_group_id=active_group,
                instance_maps=gt_data["instance_maps"],
                semantic_2d=gt_data["semantic_gt"],
            )
            after_catalog_hash = physics_state_sha256(
                capture_runtime_physics_state(environment)
            )
            if before_catalog_hash != after_catalog_hash:
                raise RuntimeError(
                    "target-catalog GT scan changed object physics state"
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
                    "target_catalog": {
                        "evaluator_only": True,
                        "physics_state_sha256_before": before_catalog_hash,
                        "physics_state_sha256_after": after_catalog_hash,
                    },
                }
            )
            snapshot_path = group_root / "initial_state.json"
            catalog_path = group_root / "target_catalog.json"
            _write_new_json(snapshot_path, snapshot)
            _write_new_json(catalog_path, catalog)
            region_counts = {
                region: sum(
                    row["coarse_region"] == region for row in catalog["targets"]
                )
                for region in ("front", "mid", "back")
            }
            record = {
                **dict(group),
                "scene_path": str(scene_path),
                "scene_sha256": _sha256(scene_path),
                "snapshot_path": str(snapshot_path),
                "snapshot_sha256": _sha256(snapshot_path),
                "target_catalog_path": str(catalog_path),
                "target_catalog_sha256": _sha256(catalog_path),
                "state_sha256": snapshot["state_sha256"],
                "object_count": len(snapshot["state"]["objects"]),
                "eligible_target_count": len(catalog["targets"]),
                "target_region_counts": region_counts,
                "target_catalog_physics_state_restored": True,
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
            print(
                f"generated {active_group}: objects={record['object_count']} "
                f"targets={record['eligible_target_count']} regions={region_counts}",
                flush=True,
            )
    except BaseException as error:
        failure_path = root / "generation_failure.json"
        if not failure_path.exists():
            _write_new_json(
                failure_path,
                {
                    "schema": FAILURE_SCHEMA,
                    "bank_id": manifest["bank_id"],
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

    totals = {
        region: sum(row["target_region_counts"][region] for row in records)
        for region in ("front", "mid", "back")
    }
    availability = {
        region: sum(row["target_region_counts"][region] > 0 for row in records)
        for region in ("front", "mid", "back")
    }
    completion = {
        "schema": COMPLETION_SCHEMA,
        "bank_id": manifest["bank_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "group_count": len(records),
        "groups": records,
        "target_region_totals": totals,
        "scene_availability_by_region": availability,
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
