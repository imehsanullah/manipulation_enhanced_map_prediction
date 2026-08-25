"""Evaluator-only temporal identity measurements for PSG-MEM X2.

Runtime graphs are complete before this adapter reads live GT maps or physics
state. Stable physical object indices, simulator body IDs, and raw masks stay
inside this evaluator boundary; public rows contain only per-episode
pseudonyms and runtime track IDs.
"""

from __future__ import annotations

import copy
import json
import math
import time
from typing import Any, Dict, Mapping, MutableSequence, Sequence

import numpy as np

from shelf_gym.utils.action_conditioned_relation_oracle import merge_instance_stack
from shelf_gym.utils.cnabu_mem_experiment_control import (
    capture_runtime_physics_state,
    physics_state_sha256,
)
from shelf_gym.utils.psg_mem_live_registry import get_live_episode
from scene_graph_mem.nodes.node_proposal import Node, evaluate_nodes
from scene_graph_mem.runtime.cnabu_scene_graph import decode_binary_mask_rle


EVALUATION_SCHEMA = "psg_mem_x2_evaluation_config_v1"
STEP_SCHEMA = "psg_mem_x2_step_evaluation_v1"
DEFAULT_CONFIG = {
    "schema": EVALUATION_SCHEMA,
    "node_iou_threshold": 0.50,
    "memory_duplicate_overlap_fraction_threshold": 0.20,
    "world_motion_threshold_m": 0.01,
    "predicted_displacement_threshold_pixels": 1.0,
}
_PRIVATE_FIELDS = {
    "body_id",
    "evaluation_object_id",
    "gt_instance_id",
    "gt_mask",
    "instance_id",
    "simulator_instance_id",
    "target_mask",
}


def _typed_key(value: Any) -> tuple[type, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("runtime track IDs must be strings or integers")
    return type(value), value


def _public_track_id(value: Any, *, step: int, persistent: bool) -> Any:
    _typed_key(value)
    if persistent:
        return value.item() if isinstance(value, np.generic) else value
    return "rebuild:{}:{}:{}".format(step, type(value).__name__, value)


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    return np.asarray(value)


def _mask_geometry(mask: np.ndarray) -> tuple[list[int], list[float], int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("X2 object masks must be non-empty")
    return (
        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        [float(ys.mean()), float(xs.mean())],
        int(mask.sum()),
    )


def _node(*, ordinal: int, class_id: int, mask: np.ndarray, score: float) -> Node:
    bbox, centroid, area = _mask_geometry(mask)
    return Node(
        id=int(ordinal),
        class_id=int(class_id),
        class_name=str(int(class_id)),
        mask=np.asarray(mask, dtype=bool),
        bbox_xyxy_abs=bbox,
        centroid_yx=centroid,
        area=area,
        score=float(score),
        was_split=False,
    )


def _validate_config(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG if value is None else value)
    if config != DEFAULT_CONFIG:
        raise ValueError("X2 evaluator configuration differs from the frozen contract")
    return config


def _physics_positions(state: Mapping[str, Any]) -> Dict[int, np.ndarray]:
    result = {}
    for row in state.get("objects") or []:
        object_index = int(row["object_index"])
        position = np.asarray((row.get("body") or {})["position"], dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("X2 physics state requires finite 3D object positions")
        result[object_index] = position
    return result


def _physical_objects(
    mem: Any,
    *,
    positions: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, list[Dict[str, Any]], list[Node]]:
    payload = mem.get_gt_height_map(no_tqdm=True)
    if not isinstance(payload, Mapping):
        raise ValueError("X2 evaluator requires the live GT height-map mapping")
    if "instance_maps" not in payload or "semantic_gt" not in payload:
        raise ValueError("X2 evaluator requires GT instances and semantics")
    instance_map = np.asarray(merge_instance_stack(payload["instance_maps"]))
    semantics = _host_array(payload["semantic_gt"])
    if instance_map.ndim != 2 or semantics.shape != instance_map.shape:
        raise ValueError("X2 GT instance and semantic maps must be aligned 2D arrays")
    class_by_body = {
        int(key): int(value) for key, value in mem.obj.get_id_to_class_dict().items()
    }
    records = []
    nodes = []
    for object_index, raw_body_id in enumerate(mem.current_obj_ids):
        body_id = int(raw_body_id)
        mask = instance_map == body_id
        if not mask.any():
            continue
        if body_id not in class_by_body or object_index not in positions:
            raise ValueError("X2 live object is absent from class/physics state")
        class_id = int(class_by_body[body_id])
        semantic_ids, counts = np.unique(semantics[mask], return_counts=True)
        if int(semantic_ids[int(np.argmax(counts))]) != class_id:
            raise ValueError("X2 GT semantic and simulator class labels disagree")
        records.append(
            {
                "object_index": int(object_index),
                "class_id": class_id,
                "mask": np.asarray(mask, dtype=bool),
                "position": np.asarray(positions[object_index], dtype=np.float64),
            }
        )
        nodes.append(
            _node(
                ordinal=len(nodes),
                class_id=class_id,
                mask=mask,
                score=1.0,
            )
        )
    if not records:
        raise ValueError("X2 evaluator found no visible physical objects")
    return instance_map, records, nodes


def _runtime_objects(
    graph: Mapping[str, Any],
    *,
    raw_shape: tuple[int, int],
    step: int,
    persistent: bool,
    current_only: bool,
) -> list[Dict[str, Any]]:
    rows = []
    for raw in graph.get("nodes", []):
        if raw.get("node_type") != "object":
            continue
        if (
            current_only
            and persistent
            and raw.get("tracking_state")
            in {
                "unobserved",
                "possibly_moved",
            }
        ):
            continue
        encoded = raw.get("footprint_mask")
        if not isinstance(encoded, Mapping):
            raise ValueError("X2 runtime object requires footprint_mask")
        mask = np.asarray(decode_binary_mask_rle(encoded), dtype=bool)
        if mask.shape != raw_shape:
            raise ValueError("X2 runtime and evaluator masks use different raw frames")
        source_id = raw.get("node_id")
        rows.append(
            {
                "track_id": _public_track_id(
                    source_id, step=step, persistent=persistent
                ),
                "source_track_id": source_id,
                "class_id": int(raw["class_id"]),
                "mask": mask,
                "score": float((raw.get("source_payload") or {}).get("score", 1.0)),
            }
        )
    rows.sort(key=lambda row: repr(_typed_key(row["track_id"])))
    return rows


def _as_nodes(records: Sequence[Mapping[str, Any]]) -> list[Node]:
    return [
        _node(
            ordinal=index,
            class_id=int(row["class_id"]),
            mask=np.asarray(row["mask"], dtype=bool),
            score=float(row.get("score", 1.0)),
        )
        for index, row in enumerate(records)
    ]


def _overlap_fraction(first: np.ndarray, second: np.ndarray) -> float:
    denominator = min(int(first.sum()), int(second.sum()))
    if denominator <= 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / denominator)


def _displacement_detected(
    graph: Mapping[str, Any],
    *,
    source_track_id: Any,
    step: int,
    threshold_pixels: float,
) -> bool:
    source_key = _typed_key(source_track_id)
    for edge in graph.get("edges", []):
        if (
            edge.get("edge_type") != "same_object_as"
            or int(edge.get("target_step", -1)) != int(step)
            or _typed_key(edge.get("target")) != source_key
        ):
            continue
        displacement = np.asarray(edge.get("displacement_xy"), dtype=np.float64)
        if displacement.shape == (2,) and np.isfinite(displacement).all():
            return bool(np.linalg.norm(displacement) >= float(threshold_pixels))
    return False


def _evaluate_step(
    mem: Any,
    graph: Mapping[str, Any],
    *,
    episode_id: str,
    step: int,
    persistent: bool,
    previous_action_kind: str,
    config: Mapping[str, Any],
    rebuild_memory: MutableSequence[Dict[str, Any]],
    previous_positions: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    started = time.perf_counter()
    before_state = capture_runtime_physics_state(mem)
    before_hash = physics_state_sha256(before_state)
    positions = _physics_positions(before_state)
    instance_map, physical, references = _physical_objects(mem, positions=positions)
    current = _runtime_objects(
        graph,
        raw_shape=instance_map.shape,
        step=step,
        persistent=persistent,
        current_only=True,
    )
    if persistent:
        memory = _runtime_objects(
            graph,
            raw_shape=instance_map.shape,
            step=step,
            persistent=True,
            current_only=False,
        )
    else:
        rebuild_memory.extend(copy.deepcopy(current))
        memory = list(rebuild_memory)
    predictions = _as_nodes(current)
    metrics = evaluate_nodes(
        predictions,
        references,
        thresholds=[float(config["node_iou_threshold"])],
        overlap_fraction_threshold=float(
            config["memory_duplicate_overlap_fraction_threshold"]
        ),
    )
    threshold_key = f"{float(config['node_iou_threshold']):.2f}"
    raw_matches = list(metrics["thresholds"][threshold_key].pop("matches"))
    primary_by_physical = {
        int(row["gt_index"]): int(row["pred_index"]) for row in raw_matches
    }
    after_push = str(previous_action_kind) == "push"
    identity_rows = []
    for physical_index, reference in enumerate(physical):
        current_overlaps = [
            row["track_id"]
            for row in current
            if int(row["class_id"]) == int(reference["class_id"])
            and (
                np.logical_and(row["mask"], reference["mask"]).sum()
                / max(
                    np.logical_or(row["mask"], reference["mask"]).sum(),
                    1,
                )
            )
            >= float(config["node_iou_threshold"])
        ]
        memory_overlaps = [
            row["track_id"]
            for row in memory
            if int(row["class_id"]) == int(reference["class_id"])
            and _overlap_fraction(row["mask"], reference["mask"])
            >= float(config["memory_duplicate_overlap_fraction_threshold"])
        ]
        pred_index = primary_by_physical.get(physical_index)
        primary = None if pred_index is None else current[pred_index]
        old_position = previous_positions.get(int(reference["object_index"]))
        moved = bool(
            old_position is not None
            and np.linalg.norm(reference["position"] - old_position)
            >= float(config["world_motion_threshold_m"])
        )
        detected = bool(
            moved
            and primary is not None
            and persistent
            and _displacement_detected(
                graph,
                source_track_id=primary["source_track_id"],
                step=step,
                threshold_pixels=float(
                    config["predicted_displacement_threshold_pixels"]
                ),
            )
        )
        identity_rows.append(
            {
                "episode_id": episode_id,
                "step": int(step),
                "gt_object_id": f"physical_object_{int(reference['object_index']):03d}",
                "primary_track_id": (None if primary is None else primary["track_id"]),
                "track_ids": current_overlaps,
                "memory_track_ids": memory_overlaps,
                "gt_moved": moved,
                "displacement_detected": detected,
                "after_push": after_push,
            }
        )
    previous_positions.clear()
    previous_positions.update(
        {
            index: np.asarray(value, dtype=np.float64)
            for index, value in positions.items()
        }
    )
    after_state = capture_runtime_physics_state(mem)
    after_hash = physics_state_sha256(after_state)
    if before_hash != after_hash:
        raise RuntimeError("X2 evaluator changed the live physics state")
    result = {
        "schema": STEP_SCHEMA,
        "step": int(step),
        "persistent": bool(persistent),
        "previous_action_kind": str(previous_action_kind),
        "identity_rows": identity_rows,
        "current_track_count": len(current),
        "memory_track_count": len(memory),
        "visible_physical_object_count": len(physical),
        "matched_object_count": len(raw_matches),
        "node_metrics": metrics,
        "physics_state_sha256_before": before_hash,
        "physics_state_sha256_after": after_hash,
        "physics_state_exactly_restored": True,
        "evaluator_runtime_seconds": float(time.perf_counter() - started),
        "evaluation_boundary": {
            "runtime_graph_uses_gt": False,
            "simulator_ids_returned": False,
            "raw_gt_maps_returned": False,
            "read_only": True,
        },
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    for field in _PRIVATE_FIELDS:
        if f'"{field}"' in encoded:
            raise AssertionError(f"X2 evaluator leaked private field {field}")
    return result


def build_evaluator_adapter(spec: Mapping[str, Any]) -> Dict[str, Any]:
    episode_id = spec.get("episode_id")
    arm_id = str(spec.get("arm_id", ""))
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("X2 evaluator requires episode_id")
    if arm_id not in {"tracker", "rebuild"}:
        raise ValueError("X2 evaluator arm_id must be tracker or rebuild")
    persistent = arm_id == "tracker"
    config = _validate_config(spec.get("evaluation"))
    handle = get_live_episode(episode_id)
    rebuild_memory: list[Dict[str, Any]] = []
    previous_positions: Dict[int, np.ndarray] = {}

    def evaluate_graph(
        *, graph: Mapping[str, Any], step: int, previous_action_kind: str
    ) -> Dict[str, Any]:
        if graph.get("episode_id") != episode_id or int(graph.get("step", -1)) != int(
            step
        ):
            raise ValueError("X2 graph episode/step does not match evaluator")
        if int(step) != int(handle.bridge.state.action_count):
            raise ValueError("X2 evaluator step disagrees with live MEM state")
        graph_persistent = bool(
            (graph.get("metadata") or {}).get("persistent_memory", False)
        )
        if graph_persistent != persistent:
            raise ValueError("X2 graph persistence differs from evaluator arm")
        return _evaluate_step(
            handle.mem,
            graph,
            episode_id=episode_id,
            step=int(step),
            persistent=persistent,
            previous_action_kind=str(previous_action_kind),
            config=config,
            rebuild_memory=rebuild_memory,
            previous_positions=previous_positions,
        )

    return {
        "evaluate_graph": evaluate_graph,
        "close": lambda: None,
        "provenance": {
            "schema": "psg_mem_x2_evaluator_provenance_v1",
            "arm_id": arm_id,
            "persistent": persistent,
            "evaluation_config": copy.deepcopy(config),
            "read_only": True,
            "returns_simulator_ids": False,
            "returns_raw_gt_maps": False,
        },
    }


__all__ = [
    "DEFAULT_CONFIG",
    "EVALUATION_SCHEMA",
    "STEP_SCHEMA",
    "build_evaluator_adapter",
]
