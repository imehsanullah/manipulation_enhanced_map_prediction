#!/usr/bin/env python3
"""Prepare one hash-pinned X3 episode specification without running it.

The target catalog is derived from an existing evaluator-only GT height-map
artifact.  The resulting planner query contains only semantic class and a
coarse depth region.  The hidden simulator instance ID stays under the task's
evaluation token and is removed by the X3 driver before the runtime adapter is
constructed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


EPISODE_SPEC_SCHEMA = "psg_mem_x3_episode_spec_v1"
RUNTIME_SPEC_SCHEMA = "shelf_gym_psg_mem_runtime_v1"
TARGET_CATALOG_SCHEMA = "psg_mem_evaluator_target_catalog_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_file(path: str | Path, *, name: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _identity(path: Path) -> Dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _merge_instance_stack(value: Any, *, background_value: int = -1) -> np.ndarray:
    stack = np.asarray(value)
    if stack.ndim == 2:
        return stack.copy()
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError("GT instance_maps must have shape [H,W] or [V,H,W]")
    merged = np.zeros_like(stack[0])
    for layer in stack:
        np.copyto(merged, layer, where=layer != int(background_value))
    return merged


def evaluator_target_catalog_from_arrays(
    instance_maps: Any, semantic_2d: Any
) -> list[Dict[str, Any]]:
    """Return evaluator-only target rows with frozen raw-map depth thirds."""

    instances = _merge_instance_stack(instance_maps)
    semantics = np.asarray(semantic_2d)
    if instances.shape != semantics.shape or instances.ndim != 2:
        raise ValueError("GT instance and semantic maps must be aligned 2D arrays")
    height = int(instances.shape[0])
    rows = []
    for raw_id in np.unique(instances):
        instance_id = int(raw_id)
        if instance_id in {-1, 0}:
            continue
        mask = instances == raw_id
        semantic_ids, counts = np.unique(semantics[mask], return_counts=True)
        class_id = int(semantic_ids[int(np.argmax(counts))])
        if not 0 <= class_id < 14:
            continue
        centroid_y = float(np.nonzero(mask)[0].mean())
        depth_fraction = centroid_y / max(float(height), 1.0)
        region = (
            "front"
            if depth_fraction < 1.0 / 3.0
            else "mid" if depth_fraction < 2.0 / 3.0 else "back"
        )
        rows.append(
            {
                "evaluation_object_id": instance_id,
                "class_id": class_id,
                "coarse_region": region,
            }
        )
    rows.sort(key=lambda row: int(row["evaluation_object_id"]))
    if not rows:
        raise ValueError("GT height-map contains no eligible object target")
    return rows


def evaluator_target_catalog(gt_hms_path: str | Path) -> list[Dict[str, Any]]:
    """Load target rows from an evaluator-only GT height-map artifact."""

    path = _absolute_file(gt_hms_path, name="gt_hms")
    with np.load(path, allow_pickle=False) as data:
        if "instance_maps" not in data.files or "semantic_2d" not in data.files:
            raise KeyError("GT height-map requires instance_maps and semantic_2d")
        return evaluator_target_catalog_from_arrays(
            data["instance_maps"], data["semantic_2d"]
        )


def load_evaluator_target_catalog(path: str | Path) -> list[Dict[str, Any]]:
    """Load a compact target catalog that is forbidden to the runtime path."""

    resolved = _absolute_file(path, name="target_catalog")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TARGET_CATALOG_SCHEMA
    ):
        raise ValueError("unexpected evaluator target-catalog schema")
    if payload.get("source") != "live_gt_height_map_evaluator_only":
        raise ValueError("target catalog has an unexpected source")
    if payload.get("depth_partition") != "raw_y_thirds_v1":
        raise ValueError("target catalog changed the raw-map depth partition")
    raw_shape = payload.get("raw_shape_hw")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in raw_shape
        )
    ):
        raise ValueError("target catalog requires a positive raw_shape_hw")
    if (
        payload.get("runtime_visible") is not False
        or payload.get("contains_simulator_instance_ids") is not True
        or payload.get("planner_may_read") is not False
    ):
        raise ValueError("target catalog violates its evaluator-only boundary")
    raw_rows = payload.get("targets")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("target catalog requires non-empty targets")
    rows = []
    seen = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "evaluation_object_id",
            "class_id",
            "coarse_region",
        }:
            raise ValueError("target catalog rows have an unexpected shape")
        object_id = raw.get("evaluation_object_id")
        class_id = raw.get("class_id")
        region = raw.get("coarse_region")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id < 1
            or object_id in seen
        ):
            raise ValueError(
                "target catalog object IDs must be unique positive integers"
            )
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 14
        ):
            raise ValueError("target catalog class IDs must be integers in [0,13]")
        if region not in {"front", "mid", "back"}:
            raise ValueError("target catalog coarse region must be front/mid/back")
        seen.add(object_id)
        rows.append(
            {
                "evaluation_object_id": object_id,
                "class_id": class_id,
                "coarse_region": region,
            }
        )
    rows.sort(key=lambda row: int(row["evaluation_object_id"]))
    return rows


def build_episode_spec(
    *,
    episode_id: str,
    arm_id: str,
    scene_path: str | Path,
    initial_state_snapshot: str | Path,
    gt_hms_path: str | Path | None,
    experiment_config_path: str | Path,
    scene_graph_mem_src: str | Path,
    target_seed: int,
    planner_seed: int,
    requested_region: Optional[str],
    action_budget: int,
    max_sampled_pushes: int,
    relation_device: str,
    render: bool = False,
    target_catalog_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Build and strict-JSON validate one non-executing X3 specification."""

    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be non-empty")
    arm = str(arm_id).lower()
    if arm not in set("abcdefgh"):
        raise ValueError("arm_id must be a..h")
    for name, value in (
        ("target_seed", target_seed),
        ("planner_seed", planner_seed),
        ("action_budget", action_budget),
        ("max_sampled_pushes", max_sampled_pushes),
    ):
        minimum = 0 if name.endswith("seed") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if relation_device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("relation_device must be cpu/cuda/cuda:0")
    if not isinstance(render, bool):
        raise ValueError("render must be boolean")

    source_root = Path(scene_graph_mem_src).expanduser()
    if not source_root.is_absolute():
        raise ValueError("scene_graph_mem_src must be absolute")
    source_root = source_root.resolve()
    if not (source_root / "scene_graph_mem").is_dir():
        raise ValueError("scene_graph_mem_src is not a package source root")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from scene_graph_mem.experiments.protocol import (
        load_experiment_config,
        validate_experiment_config,
    )
    from scene_graph_mem.runtime.retrieval_experiment import (
        sample_stratified_target,
    )

    config_path = _absolute_file(experiment_config_path, name="experiment_config")
    config = load_experiment_config(config_path)
    validation = validate_experiment_config(config, verify_files=True)
    if validation["experiment_id"] != "x3":
        raise ValueError("episode preparation requires the X3 experiment config")
    failure_diagnostics = config.get("failure_diagnostics")
    if failure_diagnostics is not None and not isinstance(
        failure_diagnostics, Mapping
    ):
        raise ValueError("failure_diagnostics must be a mapping")
    arms = {str(row["id"]): dict(row) for row in config["arms"]}
    arm_config = arms[arm]
    artifacts = dict(config["artifacts"])
    scene = _absolute_file(scene_path, name="scene")
    snapshot = _absolute_file(initial_state_snapshot, name="initial_state_snapshot")
    if (gt_hms_path is None) == (target_catalog_path is None):
        raise ValueError(
            "exactly one of gt_hms_path or target_catalog_path is required"
        )
    gt_hms = None if gt_hms_path is None else _absolute_file(gt_hms_path, name="gt_hms")
    compact_catalog = (
        None
        if target_catalog_path is None
        else _absolute_file(target_catalog_path, name="target_catalog")
    )
    target_catalog = (
        evaluator_target_catalog(gt_hms)
        if gt_hms is not None
        else load_evaluator_target_catalog(compact_catalog)
    )
    task = sample_stratified_target(
        target_catalog,
        seed=int(target_seed),
        requested_region=requested_region,
    )

    tracker_config = dict(config["tracker"])
    tracker_config.pop("enabled", None)
    planner_overrides = dict(config["planner"])
    for name in ("representation", "reasoning_mode"):
        if name in arm_config:
            planner_overrides[name] = arm_config[name]
    spec = {
        "schema": EPISODE_SPEC_SCHEMA,
        "episode_id": episode_id,
        "arm_id": arm,
        "task": task,
        "episode_config": {
            "budget": int(action_budget),
            "seed": int(planner_seed),
            "stop_when_accessible": True,
            "stop_on_collision": True,
            "capture_infrastructure_failures": True,
            "enforce_arm_graph_semantics": True,
        },
        "planner_overrides": planner_overrides,
        "adapter_artifacts": {
            "runtime": dict(artifacts["shelf_gym_runtime_adapter"]),
            "evaluator": dict(artifacts["shelf_gym_evaluator_adapter"]),
        },
        "integration": {
            "schema": RUNTIME_SPEC_SCHEMA,
            "scene": _identity(scene),
            "initial_state_snapshot": _identity(snapshot),
            "splitter_checkpoint": dict(artifacts["splitter_checkpoint"]),
            "relation_checkpoint": dict(artifacts["relation_checkpoint"]),
            "relation_config": dict(artifacts["relation_config"]),
            "scene_graph_python": dict(artifacts["scene_graph_python"]),
            "scene_graph_sidecar": dict(artifacts["scene_graph_sidecar"]),
            "scene_graph_mem_src": str(source_root),
            "seed": int(planner_seed),
            "action_budget": int(action_budget),
            "max_sampled_pushes": int(max_sampled_pushes),
            "render": bool(render),
            "relation_device": relation_device,
            "raw_shape_hw": [140, 200],
            "crop_rows": [10, 130],
            "graph_config": dict(config["graph"]),
            "tracker_config": tracker_config,
            "extractor_config": dict(config["extractor"]),
            "sidecar_startup_timeout_seconds": float(
                config["sidecar"]["startup_timeout_seconds"]
            ),
            "sidecar_request_timeout_seconds": float(
                config["sidecar"]["request_timeout_seconds"]
            ),
        },
        "evaluation": {
            **(
                {"gt_hms": _identity(gt_hms)}
                if gt_hms is not None
                else {"target_catalog": _identity(compact_catalog)}
            ),
            "target_sampling": {
                "policy": "uniform_within_requested_raw_depth_third_v1",
                "target_seed": int(target_seed),
                "requested_region": requested_region,
                "catalog_size": len(target_catalog),
                "runtime_receives_catalog": False,
                "runtime_receives_hidden_target_id": False,
            },
            **(
                {
                    "failure_diagnostics": copy.deepcopy(
                        dict(failure_diagnostics)
                    )
                }
                if failure_diagnostics is not None
                else {}
            ),
        },
        "preparation_provenance": {
            "schema": "psg_mem_x3_episode_preparation_v1",
            "experiment_config": _identity(config_path),
            "resolved_experiment_config_sha256": validation["config_sha256"],
            "executes_simulation": False,
            "writes_dataset": False,
            "writes_checkpoint_or_model": False,
        },
    }
    if "reasoning_checkpoint" in artifacts:
        spec["integration"]["reasoning_checkpoint"] = dict(
            artifacts["reasoning_checkpoint"]
        )
    if "runtime_graph_variant" in config:
        variant = config["runtime_graph_variant"]
        if not isinstance(variant, Mapping):
            raise ValueError("runtime_graph_variant must be a mapping")
        spec["integration"]["runtime_graph_variant"] = copy.deepcopy(
            dict(variant)
        )
    if "component_config" in config:
        component_config = config["component_config"]
        if not isinstance(component_config, Mapping):
            raise ValueError("component_config must be a mapping")
        spec["integration"]["component_config"] = copy.deepcopy(
            dict(component_config)
        )
    if "evaluation_object_id" in json.dumps(spec["integration"], sort_keys=True):
        raise AssertionError("hidden target identity leaked into runtime integration")
    json.dumps(spec, sort_keys=True, allow_nan=False)
    return spec


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--arm-id", choices=list("abcdefgh"), required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--initial-state-snapshot", type=Path, required=True)
    target_source = parser.add_mutually_exclusive_group(required=True)
    target_source.add_argument("--gt-hms", type=Path)
    target_source.add_argument("--target-catalog", type=Path)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--scene-graph-mem-src", type=Path, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--requested-region", choices=["front", "mid", "back"])
    parser.add_argument("--action-budget", type=int, required=True)
    parser.add_argument("--max-sampled-pushes", type=int, required=True)
    parser.add_argument(
        "--relation-device", choices=["cpu", "cuda", "cuda:0"], required=True
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output = args.output_json.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")
    spec = build_episode_spec(
        episode_id=args.episode_id,
        arm_id=args.arm_id,
        scene_path=args.scene,
        initial_state_snapshot=args.initial_state_snapshot,
        gt_hms_path=args.gt_hms,
        experiment_config_path=args.experiment_config,
        scene_graph_mem_src=args.scene_graph_mem_src,
        target_seed=args.target_seed,
        planner_seed=args.planner_seed,
        requested_region=args.requested_region,
        action_budget=args.action_budget,
        max_sampled_pushes=args.max_sampled_pushes,
        relation_device=args.relation_device,
        render=bool(args.render),
        target_catalog_path=args.target_catalog,
    )
    output.write_text(
        json.dumps(spec, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
