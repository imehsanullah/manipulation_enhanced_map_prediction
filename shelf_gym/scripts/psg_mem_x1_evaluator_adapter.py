"""Evaluator-only node and relation measurements for PSG-MEM X1.

The runtime graph is constructed before this adapter is called and contains no
GT inputs.  This module is loaded separately by the X1 driver, reads live GT
maps and simulator body IDs only inside the evaluator boundary, and returns a
de-identified metric payload.  The physical access oracle restores the scene
after every target query; an additional whole-evaluation physics hash makes
that inertness explicit.
"""

from __future__ import annotations

import copy
import json
import math
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from shelf_gym.utils.action_conditioned_relation_oracle import (
    evaluate_live_target_access_feasibility,
    merge_instance_stack,
)
from shelf_gym.utils.cnabu_mem_experiment_control import (
    capture_runtime_physics_state,
    physics_state_sha256,
)
from shelf_gym.utils.psg_mem_live_registry import get_live_episode
from scene_graph_mem.nodes.node_proposal import Node, evaluate_nodes
from scene_graph_mem.relations.blocker_ranking import (
    compute_target_query_ranking_metrics,
)
from scene_graph_mem.runtime.cnabu_scene_graph import decode_binary_mask_rle


EVALUATION_SCHEMA = "psg_mem_x1_evaluation_config_v1"
STEP_SCHEMA = "psg_mem_x1_step_evaluation_v1"
DEFAULT_CONFIG = {
    "schema": EVALUATION_SCHEMA,
    "node_iou_threshold": 0.50,
    "merge_overlap_fraction_threshold": 0.20,
    "physical_relation_binary_threshold": 0.40,
    "prediction_binary_threshold": 0.90,
    "ranking_k": 3,
}


def _typed_key(value: Any) -> tuple[type, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("runtime node IDs must be strings or integers")
    return type(value), value


def _validate_config(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG if value is None else value)
    if config != DEFAULT_CONFIG:
        raise ValueError("X1 evaluator configuration differs from the frozen contract")
    return config


def _mask_geometry(mask: np.ndarray) -> tuple[list[int], list[float], int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("X1 object masks must be non-empty")
    return (
        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        [float(ys.mean()), float(xs.mean())],
        int(mask.sum()),
    )


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    return np.asarray(value)


def _node(
    *,
    ordinal: int,
    class_id: int,
    mask: np.ndarray,
    score: float,
) -> Node:
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


def _predicted_objects(
    graph: Mapping[str, Any], raw_shape: tuple[int, int]
) -> tuple[list[Mapping[str, Any]], list[Node]]:
    objects = [
        copy.deepcopy(dict(node))
        for node in graph.get("nodes", [])
        if node.get("node_type") == "object"
    ]
    objects.sort(key=lambda row: repr(_typed_key(row.get("node_id"))))
    if len({_typed_key(row.get("node_id")) for row in objects}) != len(objects):
        raise ValueError("X1 runtime object node IDs must be unique")
    nodes = []
    for ordinal, row in enumerate(objects):
        encoded = row.get("footprint_mask")
        if not isinstance(encoded, Mapping):
            raise ValueError("X1 runtime object requires footprint_mask")
        mask = np.asarray(decode_binary_mask_rle(encoded), dtype=bool)
        if mask.shape != raw_shape:
            raise ValueError("X1 runtime and evaluator masks use different raw frames")
        nodes.append(
            _node(
                ordinal=ordinal,
                class_id=int(row["class_id"]),
                mask=mask,
                score=float((row.get("source_payload") or {}).get("score", 1.0)),
            )
        )
    return objects, nodes


def _physical_objects(mem: Any) -> tuple[np.ndarray, list[Dict[str, Any]], list[Node]]:
    payload = mem.get_gt_height_map(no_tqdm=True)
    if not isinstance(payload, Mapping):
        raise ValueError("X1 evaluator requires the live GT height-map mapping")
    if "instance_maps" not in payload or "semantic_gt" not in payload:
        raise ValueError("X1 evaluator requires live GT instance_maps and semantic_gt")
    instance_map = np.asarray(merge_instance_stack(payload["instance_maps"]))
    semantics = _host_array(payload["semantic_gt"])
    if instance_map.ndim != 2 or semantics.shape != instance_map.shape:
        raise ValueError("X1 GT instance and semantic maps must be aligned 2D arrays")
    class_by_body = {
        int(key): int(value) for key, value in mem.obj.get_id_to_class_dict().items()
    }
    records = []
    nodes = []
    for body_id in sorted(int(value) for value in mem.current_obj_ids):
        mask = instance_map == body_id
        if not mask.any():
            continue
        if body_id not in class_by_body:
            raise ValueError("X1 live object is absent from the simulator class map")
        class_id = int(class_by_body[body_id])
        visible_semantics, counts = np.unique(semantics[mask], return_counts=True)
        majority = int(visible_semantics[int(np.argmax(counts))])
        if majority != class_id:
            raise ValueError("X1 GT semantic and simulator class labels disagree")
        ordinal = len(records)
        records.append({"body_id": body_id, "class_id": class_id, "mask": mask})
        nodes.append(_node(ordinal=ordinal, class_id=class_id, mask=mask, score=1.0))
    if not records:
        raise ValueError("X1 evaluator found no visible physical objects")
    return instance_map, records, nodes


def _safe_node_metrics(
    metrics: Mapping[str, Any],
    *,
    predicted: Sequence[Mapping[str, Any]],
    threshold_key: str,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    result = copy.deepcopy(dict(metrics))
    raw_matches = list(result["thresholds"][threshold_key].pop("matches"))
    safe_matches = []
    for row in raw_matches:
        pred_index = int(row["pred_index"])
        safe_matches.append(
            {
                "source_node_id": predicted[pred_index]["node_id"],
                "predicted_index": pred_index,
                "physical_ordinal": int(row["gt_index"]),
                "iou": float(row["iou"]),
            }
        )
    # Physical ordinals are process-local array positions, not simulator IDs;
    # they are needed only until relation rows are built and are stripped from
    # the public payload below.
    public_matches = [
        {"node_id": row["source_node_id"], "iou": row["iou"]} for row in safe_matches
    ]
    result["matched_objects"] = public_matches
    return result, safe_matches


def _relation_lookup(graph: Mapping[str, Any]) -> Dict[tuple[tuple, tuple], float]:
    result: Dict[tuple[tuple, tuple], float] = {}
    for edge in graph.get("edges", []):
        if edge.get("edge_type") != "blocks_access_to":
            continue
        source = _typed_key(edge.get("source"))
        target = _typed_key(edge.get("target"))
        pair = (source, target)
        if pair in result:
            raise ValueError("X1 graph contains a duplicate directed relation pair")
        score = float(edge.get("score"))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("X1 relation scores must be finite in [0,1]")
        result[pair] = score
    return result


def _evaluate_step(
    mem: Any,
    graph: Mapping[str, Any],
    *,
    step: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    before_hash = physics_state_sha256(capture_runtime_physics_state(mem))
    instance_map, physical, references = _physical_objects(mem)
    predicted_records, predictions = _predicted_objects(graph, instance_map.shape)
    node_metrics = evaluate_nodes(
        predictions,
        references,
        thresholds=[float(config["node_iou_threshold"])],
        overlap_fraction_threshold=float(config["merge_overlap_fraction_threshold"]),
    )
    threshold_key = f"{float(config['node_iou_threshold']):.2f}"
    safe_node_metrics, matches = _safe_node_metrics(
        node_metrics,
        predicted=predicted_records,
        threshold_key=threshold_key,
    )
    relation_scores = _relation_lookup(graph)
    edge_rows = []
    ranking_rows = []
    oracle_runtime_seconds = 0.0
    for target in matches:
        target_physical = physical[int(target["physical_ordinal"])]
        oracle = evaluate_live_target_access_feasibility(
            mem,
            target_instance_id=int(target_physical["body_id"]),
            include_evaluation_private_blockers=True,
        )
        if oracle.get("environment_state_restored") is not True:
            raise RuntimeError("X1 physical evaluator did not restore the scene")
        oracle_runtime_seconds += float(oracle["runtime_seconds"])
        private = dict(oracle.get("_evaluation_private") or {})
        denominator = int(private.get("eligible_candidate_count", 0))
        counts = dict(private.get("blocker_candidate_counts") or {})
        target_id = target["source_node_id"]
        target_key = _typed_key(target_id)
        rank_scores = []
        relevance = []
        labels = []
        source_ids = []
        for source in matches:
            if source is target:
                continue
            source_id = source["source_node_id"]
            pair = (_typed_key(source_id), target_key)
            if pair not in relation_scores:
                raise ValueError(
                    "X1 requires every matched directed pair; set edge_min_score=0"
                )
            source_physical = physical[int(source["physical_ordinal"])]
            count = int(counts.get(str(int(source_physical["body_id"])), 0))
            physical_score = 0.0 if denominator <= 0 else float(count / denominator)
            label = bool(
                physical_score >= float(config["physical_relation_binary_threshold"])
            )
            probability = float(relation_scores[pair])
            edge_rows.append(
                {
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "probability": probability,
                    "physical_score": physical_score,
                    "label": label,
                    "prediction": bool(
                        probability >= float(config["prediction_binary_threshold"])
                    ),
                    "source_match_iou": float(source["iou"]),
                    "target_match_iou": float(target["iou"]),
                }
            )
            rank_scores.append(probability)
            relevance.append(physical_score)
            labels.append(label)
            source_ids.append(source_id)
        if source_ids:
            ranking = compute_target_query_ranking_metrics(
                rank_scores,
                relevance,
                [True] * len(source_ids),
                stable_node_ids=source_ids,
                blocking_probabilities=rank_scores,
                binary_labels=labels,
                ks=[int(config["ranking_k"])],
                min_candidates=1,
            )
            ranking_rows.append(
                {
                    "target_node_id": target_id,
                    "target_match_iou": float(target["iou"]),
                    **ranking,
                }
            )
    after_hash = physics_state_sha256(capture_runtime_physics_state(mem))
    if after_hash != before_hash:
        raise RuntimeError("X1 evaluator changed the live physics state")
    result = {
        "schema": STEP_SCHEMA,
        "step": int(step),
        "node_metrics": safe_node_metrics,
        "matched_object_count": len(matches),
        "matched_pair_count": len(edge_rows),
        "edge_rows": edge_rows,
        "ranking_rows": ranking_rows,
        "physical_oracle_call_count": len(matches),
        "physical_oracle_runtime_seconds": float(oracle_runtime_seconds),
        "evaluator_runtime_seconds": float(time.perf_counter() - started),
        "physics_state_sha256_before": before_hash,
        "physics_state_sha256_after": after_hash,
        "physics_state_exactly_restored": True,
        "evaluation_boundary": {
            "runtime_graph_uses_gt": False,
            "simulator_ids_returned": False,
            "raw_gt_maps_returned": False,
            "physical_oracle_read_only": True,
        },
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    for forbidden in ("body_id", "instance_id", "evaluation_object_id"):
        if forbidden in encoded:
            raise AssertionError(f"X1 evaluator leaked private field {forbidden}")
    return result


def build_evaluator_adapter(spec: Mapping[str, Any]) -> Dict[str, Any]:
    episode_id = spec.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("X1 evaluator adapter requires episode_id")
    config = _validate_config(spec.get("evaluation"))
    handle = get_live_episode(episode_id)

    def evaluate_graph(*, graph: Mapping[str, Any], step: int) -> Dict[str, Any]:
        if not isinstance(graph, Mapping):
            raise TypeError("X1 evaluator requires a graph mapping")
        if graph.get("episode_id") != episode_id or int(graph.get("step", -1)) != int(
            step
        ):
            raise ValueError("X1 graph episode/step does not match the evaluator")
        if int(step) != int(handle.bridge.state.action_count):
            raise ValueError("X1 evaluator step disagrees with live MEM state")
        return _evaluate_step(
            handle.mem,
            graph,
            step=int(step),
            config=config,
        )

    return {
        "evaluate_graph": evaluate_graph,
        "close": lambda: None,
        "provenance": {
            "schema": "psg_mem_x1_evaluator_provenance_v1",
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
