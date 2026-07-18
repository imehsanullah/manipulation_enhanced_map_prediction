"""Controlled-state utilities for paired CNABU/MEM experiments.

These helpers are deliberately outside the planner.  They make independent
processes start from the same settled PyBullet state and seed every local RNG
used by the current MEM evaluation path; they do not change candidate
allocation, action scoring, or feasibility.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np


SNAPSHOT_SCHEMA = "cnabu_mem_initial_physics_state_v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PYBULLET_OPTIONS = "--numThreads=1"
STATE_HASH_QUANTIZATION_DECIMALS = 12
DETERMINISTIC_HEIGHTMAP_FLAG = "1"


def configure_deterministic_process_environment() -> Dict[str, str]:
    """Set process variables that must precede CUDA-backed Torch work."""

    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG is already set to incompatible value {!r}".format(
                existing
            )
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    existing_pybullet = os.environ.get("SHELF_GYM_PYBULLET_OPTIONS")
    if existing_pybullet not in (None, PYBULLET_OPTIONS):
        raise RuntimeError(
            "SHELF_GYM_PYBULLET_OPTIONS is already set to incompatible value {!r}".format(
                existing_pybullet
            )
        )
    os.environ["SHELF_GYM_PYBULLET_OPTIONS"] = PYBULLET_OPTIONS
    existing_heightmap = os.environ.get("SHELF_GYM_DETERMINISTIC_HEIGHTMAP")
    if existing_heightmap not in (None, DETERMINISTIC_HEIGHTMAP_FLAG):
        raise RuntimeError(
            "SHELF_GYM_DETERMINISTIC_HEIGHTMAP is already set to incompatible value {!r}".format(
                existing_heightmap
            )
        )
    os.environ["SHELF_GYM_DETERMINISTIC_HEIGHTMAP"] = (
        DETERMINISTIC_HEIGHTMAP_FLAG
    )
    return {
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        "SHELF_GYM_PYBULLET_OPTIONS": PYBULLET_OPTIONS,
        "SHELF_GYM_DETERMINISTIC_HEIGHTMAP": DETERMINISTIC_HEIGHTMAP_FLAG,
    }


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonicalize_state_for_hash(value: Any) -> Any:
    """Remove sub-1e-12 quaternion normalization jitter from state hashes."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_state_for_hash(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_state_for_hash(item) for item in value]
    if isinstance(value, float):
        return float(round(value, STATE_HASH_QUANTIZATION_DECIMALS))
    return value


def _float_list(values: Sequence[Any]) -> list[float]:
    return [float(value) for value in values]


def _body_state(client: Any, body_id: int) -> Dict[str, Any]:
    position, orientation = client.getBasePositionAndOrientation(int(body_id))
    linear_velocity, angular_velocity = client.getBaseVelocity(int(body_id))
    return {
        "position": _float_list(position),
        "orientation_xyzw": _float_list(orientation),
        "linear_velocity": _float_list(linear_velocity),
        "angular_velocity": _float_list(angular_velocity),
    }


def _robot_state(environment: Any) -> Dict[str, Any]:
    client = environment._p
    robot_id = int(environment.robot_id)
    movable_joints = []
    for joint_index in range(int(client.getNumJoints(robot_id))):
        joint_info = client.getJointInfo(robot_id, joint_index)
        if int(joint_info[2]) == int(client.JOINT_FIXED):
            continue
        joint_state = client.getJointState(robot_id, joint_index)
        movable_joints.append(
            {
                "joint_index": int(joint_index),
                "position": float(joint_state[0]),
                "velocity": float(joint_state[1]),
            }
        )
    return {
        "body": _body_state(client, robot_id),
        "movable_joints": movable_joints,
    }


def capture_runtime_physics_state(environment: Any) -> Dict[str, Any]:
    """Capture the state fields needed to reproduce a MEM episode start."""

    instance_to_class = environment.obj.get_id_to_class_dict()
    objects = []
    for object_index, body_id in enumerate(environment.current_obj_ids):
        objects.append(
            {
                "object_index": int(object_index),
                "class_id": int(instance_to_class[int(body_id)]),
                "body": _body_state(environment._p, int(body_id)),
            }
        )
    return {
        "objects": objects,
        "robot": _robot_state(environment),
    }


def physics_state_sha256(state: Mapping[str, Any]) -> str:
    canonical_state = _canonicalize_state_for_hash(state)
    return hashlib.sha256(_canonical_json_bytes(canonical_state)).hexdigest()


def build_initial_state_snapshot(
    environment: Any,
    *,
    scene_path: Path,
    seed: int,
) -> Dict[str, Any]:
    scene_path = scene_path.resolve()
    state = capture_runtime_physics_state(environment)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "scene": {
            "path": str(scene_path),
            "sha256": _sha256_file(scene_path),
        },
        "capture_seed": int(seed),
        "state": state,
        "state_sha256": physics_state_sha256(state),
        "state_hash_quantization_decimals": STATE_HASH_QUANTIZATION_DECIMALS,
        "selection_policy": "first_state_restored_under_frozen_control_without_planner_outcomes",
    }


def write_initial_state_snapshot(snapshot: Mapping[str, Any], output_path: Path) -> None:
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_initial_state_snapshot(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported initial-state snapshot schema")
    if payload.get("state_hash_quantization_decimals") != STATE_HASH_QUANTIZATION_DECIMALS:
        raise ValueError("unsupported initial-state hash quantization convention")
    expected_hash = physics_state_sha256(payload["state"])
    if payload.get("state_sha256") != expected_hash:
        raise ValueError("initial-state snapshot state hash does not verify")
    return payload


def _apply_body_state(client: Any, body_id: int, state: Mapping[str, Any]) -> None:
    client.resetBasePositionAndOrientation(
        int(body_id),
        state["position"],
        state["orientation_xyzw"],
    )
    client.resetBaseVelocity(
        int(body_id),
        linearVelocity=state["linear_velocity"],
        angularVelocity=state["angular_velocity"],
    )


def _state_difference_summary(expected: Any, actual: Any) -> Dict[str, Any]:
    changed = []
    structural_mismatches = []

    def visit(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                structural_mismatches.append(path or "<root>")
                return
            for key in sorted(left):
                visit(left[key], right[key], "{}.{}".format(path, key).lstrip("."))
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                structural_mismatches.append(path)
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, "{}[{}]".format(path, index))
            return
        if left == right:
            return
        absolute_difference = None
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            absolute_difference = abs(float(left) - float(right))
        changed.append(
            {
                "path": path,
                "expected": left,
                "actual": right,
                "absolute_difference": absolute_difference,
            }
        )

    visit(expected, actual, "")
    numeric_differences = [
        item["absolute_difference"]
        for item in changed
        if item["absolute_difference"] is not None
    ]
    return {
        "changed_scalar_count": int(len(changed)),
        "structural_mismatches": structural_mismatches,
        "max_absolute_difference": (
            float(max(numeric_differences)) if numeric_differences else None
        ),
        "examples": changed[:12],
    }


def apply_initial_state_snapshot(
    environment: Any,
    snapshot: Mapping[str, Any],
    *,
    scene_path: Path,
    require_exact: bool = True,
) -> Dict[str, Any]:
    """Apply and byte-hash verify a captured state after normal restoration."""

    scene_path = scene_path.resolve()
    scene_sha256 = _sha256_file(scene_path)
    if snapshot["scene"]["sha256"] != scene_sha256:
        raise ValueError("initial-state snapshot belongs to a different scene")
    objects = snapshot["state"]["objects"]
    if len(objects) != len(environment.current_obj_ids):
        raise ValueError("initial-state object count differs from restored scene")

    instance_to_class = environment.obj.get_id_to_class_dict()
    for expected_index, (record, body_id) in enumerate(
        zip(objects, environment.current_obj_ids)
    ):
        if int(record["object_index"]) != int(expected_index):
            raise ValueError("initial-state object order is not contiguous")
        runtime_class = int(instance_to_class[int(body_id)])
        if int(record["class_id"]) != runtime_class:
            raise ValueError("initial-state object class order differs from scene")
        _apply_body_state(environment._p, int(body_id), record["body"])

    robot = snapshot["state"]["robot"]
    _apply_body_state(environment._p, int(environment.robot_id), robot["body"])
    for joint in robot["movable_joints"]:
        environment._p.resetJointState(
            int(environment.robot_id),
            int(joint["joint_index"]),
            targetValue=float(joint["position"]),
            targetVelocity=float(joint["velocity"]),
        )
    environment._p.performCollisionDetection()

    applied_state = capture_runtime_physics_state(environment)
    applied_hash = physics_state_sha256(applied_state)
    matches_snapshot = applied_hash == snapshot["state_sha256"]
    difference = (
        None
        if matches_snapshot
        else _state_difference_summary(snapshot["state"], applied_state)
    )
    if require_exact and not matches_snapshot:
        raise RuntimeError(
            "applied initial state does not exactly match snapshot: {} != {}; {}".format(
                applied_hash,
                snapshot["state_sha256"],
                json.dumps(difference, sort_keys=True, allow_nan=False),
            )
        )
    return {
        "scene_sha256": scene_sha256,
        "state_sha256": applied_hash,
        "matches_snapshot": bool(matches_snapshot),
        "difference": difference,
        "object_count": int(len(objects)),
    }


def configure_controlled_mem(environment: Any, *, seed: int) -> Dict[str, Any]:
    """Apply the common RNG, Torch, and PyBullet comparison policy."""

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    environment._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    seeded_generators = []
    for owner, attribute in (
        (environment, "rng"),
        (environment.ps, "rng"),
        (getattr(environment, "dataset", None), "rng"),
    ):
        if owner is None or not hasattr(owner, attribute):
            continue
        setattr(owner, attribute, np.random.default_rng(int(seed)))
        seeded_generators.append(
            "{}.{}".format(type(owner).__name__, attribute)
        )
    return {
        "seed": int(seed),
        "seeded_local_generators": seeded_generators,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "pybullet_connection_options": os.environ.get(
            "SHELF_GYM_PYBULLET_OPTIONS"
        ),
        "pybullet_deterministic_overlapping_pairs": 1,
        "process_environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "SHELF_GYM_PYBULLET_OPTIONS": os.environ.get(
                "SHELF_GYM_PYBULLET_OPTIONS"
            ),
            "SHELF_GYM_DETERMINISTIC_HEIGHTMAP": os.environ.get(
                "SHELF_GYM_DETERMINISTIC_HEIGHTMAP"
            ),
        },
    }


def array_sha256(value: Any) -> Dict[str, Any]:
    if not isinstance(value, np.ndarray):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "numpy"):
            value = value.numpy()
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object-dtype arrays do not have a stable byte contract")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return {
        "sha256": digest.hexdigest(),
        "shape": [int(size) for size in array.shape],
        "dtype": array.dtype.str,
    }


def initial_observation_hashes(
    camera_data: Mapping[str, Any],
    gt_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Hash all numeric initial camera-array and GT fields."""

    def hash_mapping(mapping: Mapping[str, Any]) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {}
        for key in sorted(mapping):
            try:
                result[str(key)] = array_sha256(mapping[key])
            except (TypeError, ValueError):
                continue
        return result

    camera_hashes = hash_mapping(camera_data)
    gt_hashes = hash_mapping(gt_data)
    combined = {"camera_data": camera_hashes, "gt_data": gt_hashes}
    return {
        **combined,
        "combined_sha256": hashlib.sha256(
            _canonical_json_bytes(combined)
        ).hexdigest(),
    }


def candidate_set_fingerprint(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Compactly fingerprint feasible candidate paths without retaining maps."""

    paths = result.get("paths")
    path_records = []
    if paths is not None:
        for path in paths:
            path_records.append(array_sha256(path))
    motion = result.get("motion_parametrization")
    motion_record = None if motion is None else array_sha256(motion)
    annotations = result.get("path_annotations")
    annotation_hash = (
        None
        if annotations is None
        else hashlib.sha256(_canonical_json_bytes(annotations)).hexdigest()
    )
    fingerprint_payload = {
        "path_count": int(len(path_records)),
        "path_hashes": [record["sha256"] for record in path_records],
        "motion_parametrization": motion_record,
        "path_annotations_sha256": annotation_hash,
    }
    return {
        **fingerprint_payload,
        "combined_sha256": hashlib.sha256(
            _canonical_json_bytes(fingerprint_payload)
        ).hexdigest(),
    }
