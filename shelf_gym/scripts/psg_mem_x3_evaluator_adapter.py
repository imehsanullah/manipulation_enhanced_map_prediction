"""Evaluator-only adapter for the live PSG-MEM X3 driver.

This module receives the hidden simulator target token.  The separately loaded
runtime adapter never imports it or sees that token.  Physical access is scored
with the frozen 3x3 collision oracle and all evaluator perturbations are restored
before control returns.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from shelf_gym.utils.action_conditioned_relation_oracle import (
    evaluate_live_target_access_feasibility,
    merge_instance_stack,
)
from shelf_gym.utils.psg_mem_live_registry import get_live_episode
from scene_graph_mem.nodes.node_proposal import Node, evaluate_nodes, pairwise_iou
from scene_graph_mem.runtime.cnabu_scene_graph import decode_binary_mask_rle


FAILURE_DIAGNOSTICS_SCHEMA = "psg_mem_x10_failure_diagnostics_config_v1"
FAILURE_DIAGNOSTIC_STEP_SCHEMA = "psg_mem_x10_failure_diagnostic_step_v1"
DEFAULT_FAILURE_DIAGNOSTICS = {
    "schema": FAILURE_DIAGNOSTICS_SCHEMA,
    "enabled": False,
    "node_iou_threshold": 0.50,
    "merge_overlap_fraction_threshold": 0.20,
    "physical_relation_binary_threshold": 0.40,
    "prediction_binary_threshold": 0.80,
}


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    return np.asarray(value)


def _current_evaluator_maps(mem: Any) -> tuple[np.ndarray, np.ndarray]:
    payload = mem.get_gt_height_map(no_tqdm=True)
    if (
        not isinstance(payload, Mapping)
        or "instance_maps" not in payload
        or "semantic_gt" not in payload
    ):
        raise ValueError("live evaluator requires aligned GT instance/semantic maps")
    instances = np.asarray(merge_instance_stack(payload["instance_maps"]))
    semantics = _host_array(payload["semantic_gt"])
    if instances.ndim != 2 or semantics.shape != instances.shape:
        raise ValueError("live evaluator GT maps must be aligned 2D arrays")
    return instances, semantics


def _current_instance_map(mem: Any) -> np.ndarray:
    payload = mem.get_gt_height_map(no_tqdm=True)
    if not isinstance(payload, Mapping) or "instance_maps" not in payload:
        raise ValueError("live evaluator requires GT instance maps")
    return np.asarray(merge_instance_stack(payload["instance_maps"]))


def _failure_diagnostics_config(value: Any) -> Dict[str, Any]:
    if value is None:
        return copy.deepcopy(DEFAULT_FAILURE_DIAGNOSTICS)
    if not isinstance(value, Mapping):
        raise ValueError("failure_diagnostics must be a mapping")
    config = {**DEFAULT_FAILURE_DIAGNOSTICS, **dict(value)}
    if set(config) != set(DEFAULT_FAILURE_DIAGNOSTICS):
        raise ValueError("failure_diagnostics contains an undeclared field")
    if config["schema"] != FAILURE_DIAGNOSTICS_SCHEMA or not isinstance(
        config["enabled"], bool
    ):
        raise ValueError("failure_diagnostics schema/enabled contract changed")
    for name, expected in (
        ("node_iou_threshold", 0.50),
        ("merge_overlap_fraction_threshold", 0.20),
        ("physical_relation_binary_threshold", 0.40),
        ("prediction_binary_threshold", 0.80),
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"failure_diagnostics {name} must be numeric")
        if float(value) != expected:
            raise ValueError(f"failure_diagnostics {name} differs from the freeze")
        config[name] = float(value)
    return config


def _typed_key(value: Any) -> tuple[type, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("runtime node IDs must be strings or integers")
    return type(value), value


def _mask_geometry(mask: np.ndarray) -> tuple[list[int], list[float], int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("diagnostic object masks must be non-empty")
    return (
        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        [float(ys.mean()), float(xs.mean())],
        int(mask.sum()),
    )


def _metric_node(ordinal: int, class_id: int, mask: np.ndarray) -> Node:
    bbox, centroid, area = _mask_geometry(mask)
    return Node(
        id=int(ordinal),
        class_id=int(class_id),
        class_name=str(int(class_id)),
        mask=np.asarray(mask, dtype=bool),
        bbox_xyxy_abs=bbox,
        centroid_yx=centroid,
        area=area,
        score=1.0,
        was_split=False,
    )


def _runtime_objects(
    graph: Mapping[str, Any], shape: tuple[int, int]
) -> tuple[list[Dict[str, Any]], list[Node]]:
    rows = [
        copy.deepcopy(dict(node))
        for node in graph.get("nodes") or []
        if node.get("node_type") == "object"
    ]
    rows.sort(key=lambda row: repr(_typed_key(row.get("node_id"))))
    if len({_typed_key(row.get("node_id")) for row in rows}) != len(rows):
        raise ValueError("diagnostic runtime node IDs must be unique")
    nodes = []
    for index, row in enumerate(rows):
        mask = np.asarray(decode_binary_mask_rle(row["footprint_mask"]), dtype=bool)
        if mask.shape != shape:
            raise ValueError("runtime and evaluator masks use different raw frames")
        nodes.append(_metric_node(index, int(row["class_id"]), mask))
    return rows, nodes


def _physical_objects(
    instance_map: np.ndarray, semantic_map: np.ndarray
) -> tuple[list[Dict[str, Any]], list[Node]]:
    rows = []
    nodes = []
    for raw_id in sorted(int(value) for value in np.unique(instance_map)):
        if raw_id in {-1, 0}:
            continue
        mask = instance_map == raw_id
        semantic_ids, counts = np.unique(semantic_map[mask], return_counts=True)
        class_id = int(semantic_ids[int(np.argmax(counts))])
        if not 0 <= class_id < 14:
            continue
        rows.append({"private_object_id": raw_id, "class_id": class_id})
        nodes.append(_metric_node(len(nodes), class_id, mask))
    return rows, nodes


def _relation_lookup(graph: Mapping[str, Any]) -> Dict[tuple[tuple, tuple], float]:
    result: Dict[tuple[tuple, tuple], float] = {}
    for edge in graph.get("edges") or []:
        if edge.get("edge_type") != "blocks_access_to":
            continue
        pair = (_typed_key(edge.get("source")), _typed_key(edge.get("target")))
        if pair in result:
            raise ValueError("diagnostic graph has duplicate directed relations")
        score = float(edge.get("score"))
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("diagnostic relation scores must be finite probabilities")
        result[pair] = score
    return result


def _assert_public_diagnostic(value: Any) -> None:
    forbidden = {
        "evaluation_object_id",
        "instance_id",
        "simulator_instance_id",
        "body_id",
        "gt_object_id",
        "instance_map",
        "semantic_map",
        "target_mask",
        "gt_mask",
    }

    def walk(raw: Any) -> None:
        if isinstance(raw, Mapping):
            for key, child in raw.items():
                if str(key) in forbidden:
                    raise AssertionError(
                        f"failure diagnostic leaked private field {key}"
                    )
                walk(child)
        elif isinstance(raw, (list, tuple)):
            for child in raw:
                walk(child)

    walk(value)
    json.dumps(value, sort_keys=True, allow_nan=False)


def _failure_diagnostic(
    *,
    graph: Mapping[str, Any],
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    private: Mapping[str, Any],
    target_private_id: int,
    selected_action: Mapping[str, Any],
    config: Mapping[str, Any],
    state: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Build a de-identified evaluator diagnostic for one pre-action graph."""

    runtime_rows, predictions = _runtime_objects(graph, instance_map.shape)
    physical_rows, references = _physical_objects(instance_map, semantic_map)
    metrics = evaluate_nodes(
        predictions,
        references,
        thresholds=[float(config["node_iou_threshold"])],
        overlap_fraction_threshold=float(
            config["merge_overlap_fraction_threshold"]
        ),
    )
    threshold_key = f"{float(config['node_iou_threshold']):.2f}"
    raw_matches = list(metrics["thresholds"][threshold_key]["matches"])
    iou, _predicted_fraction, reference_fraction = pairwise_iou(
        predictions, references
    )
    overlap = float(config["merge_overlap_fraction_threshold"])
    merge_by_prediction = (
        (reference_fraction >= overlap).sum(axis=1)
        if references
        else np.zeros(len(predictions), dtype=np.int64)
    )
    split_by_reference = (
        (reference_fraction >= overlap).sum(axis=0)
        if predictions
        else np.zeros(len(references), dtype=np.int64)
    )
    match_by_reference = {
        int(row["gt_index"]): {
            "predicted_index": int(row["pred_index"]),
            "iou": float(row["iou"]),
        }
        for row in raw_matches
    }
    private_to_runtime: Dict[int, Dict[str, Any]] = {}
    current_associations = {}
    for reference_index, match in match_by_reference.items():
        runtime = runtime_rows[match["predicted_index"]]
        private_id = int(physical_rows[reference_index]["private_object_id"])
        private_to_runtime[private_id] = {
            "node_id": copy.deepcopy(runtime["node_id"]),
            "iou": float(match["iou"]),
            "predicted_index": int(match["predicted_index"]),
            "reference_index": int(reference_index),
        }
        current_associations[private_id] = _typed_key(runtime["node_id"])

    prior_associations = state.setdefault("association_by_private_object", {})
    switch_count_current = 0
    target_switch_current = False
    for private_id, runtime_key in current_associations.items():
        prior = prior_associations.get(private_id)
        switched = prior is not None and prior != runtime_key
        switch_count_current += int(switched)
        target_switch_current = bool(
            target_switch_current
            or (private_id == int(target_private_id) and switched)
        )
        prior_associations[private_id] = runtime_key
    duplicate_count_current = int(np.sum(split_by_reference >= 2))
    state["association_switch_count"] = int(
        state.get("association_switch_count", 0) + switch_count_current
    )
    state["association_duplicate_count"] = int(
        state.get("association_duplicate_count", 0) + duplicate_count_current
    )
    state["target_association_switch_count"] = int(
        state.get("target_association_switch_count", 0)
        + int(target_switch_current)
    )

    target_reference_index = next(
        (
            index
            for index, row in enumerate(physical_rows)
            if int(row["private_object_id"]) == int(target_private_id)
        ),
        None,
    )
    target_match = (
        None
        if target_reference_index is None
        else match_by_reference.get(target_reference_index)
    )
    target_runtime = (
        None
        if target_reference_index is None or target_match is None
        else runtime_rows[int(target_match["predicted_index"])]
    )
    target_node_id = (
        None if target_runtime is None else copy.deepcopy(target_runtime["node_id"])
    )
    target_candidate_edge_present = False
    if target_node_id is not None:
        target_candidate_edge_present = any(
            edge.get("edge_type") == "candidate_of"
            and _typed_key(edge.get("source")) == _typed_key(target_node_id)
            for edge in graph.get("edges") or []
        )

    denominator = int(private.get("eligible_candidate_count", 0))
    counts = dict(private.get("blocker_candidate_counts") or {})
    blocker_private_ids = {
        int(value) for value in private.get("physical_blocker_instance_ids", [])
    }
    physical_blockers = []
    physical_score_by_source: Dict[tuple, float] = {}
    for private_id in sorted(blocker_private_ids):
        match = private_to_runtime.get(private_id)
        if match is None:
            continue
        score = (
            0.0
            if denominator <= 0
            else float(int(counts.get(str(private_id), 0)) / denominator)
        )
        source_id = copy.deepcopy(match["node_id"])
        physical_score_by_source[_typed_key(source_id)] = score
        physical_blockers.append(
            {
                "source_node_id": source_id,
                "score": score,
                "evaluation_match_iou": float(match["iou"]),
            }
        )
    physical_blockers.sort(
        key=lambda row: (-float(row["score"]), repr(row["source_node_id"]))
    )

    relation_rows = []
    false_positive_count = 0
    false_negative_count = 0
    relation_scores = _relation_lookup(graph)
    relation_defined = target_node_id is not None
    if relation_defined:
        target_key = _typed_key(target_node_id)
        source_keys = {
            _typed_key(row["node_id"]): copy.deepcopy(row["node_id"])
            for row in runtime_rows
            if _typed_key(row["node_id"]) != target_key
        }
        for source_key, source_id in sorted(
            source_keys.items(), key=lambda item: repr(item[0])
        ):
            probability = float(relation_scores.get((source_key, target_key), 0.0))
            physical_score = float(physical_score_by_source.get(source_key, 0.0))
            predicted_positive = bool(
                probability >= float(config["prediction_binary_threshold"])
            )
            physical_positive = bool(
                physical_score
                >= float(config["physical_relation_binary_threshold"])
            )
            false_positive = bool(predicted_positive and not physical_positive)
            false_negative = bool(physical_positive and not predicted_positive)
            false_positive_count += int(false_positive)
            false_negative_count += int(false_negative)
            if predicted_positive or physical_positive:
                relation_rows.append(
                    {
                        "source_node_id": source_id,
                        "target_node_id": copy.deepcopy(target_node_id),
                        "probability": probability,
                        "physical_score": physical_score,
                        "predicted_positive": predicted_positive,
                        "physical_positive": physical_positive,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                    }
                )

    selected_source = selected_action.get("source_node_id")
    selected_source_is_physical = None
    selected_relation_false_positive = None
    if selected_action.get("kind") == "push" and selected_source is not None:
        selected_key = _typed_key(selected_source)
        selected_physical_score = float(
            physical_score_by_source.get(selected_key, 0.0)
        )
        selected_source_is_physical = bool(
            selected_physical_score
            >= float(config["physical_relation_binary_threshold"])
        )
        if relation_defined:
            probability = float(
                relation_scores.get((selected_key, _typed_key(target_node_id)), 0.0)
            )
            selected_relation_false_positive = bool(
                probability >= float(config["prediction_binary_threshold"])
                and not selected_source_is_physical
            )

    target_merge = bool(
        target_match is not None
        and merge_by_prediction[int(target_match["predicted_index"])] >= 2
    )
    target_split = bool(
        target_reference_index is not None
        and split_by_reference[int(target_reference_index)] >= 2
    )
    result = {
        "schema": FAILURE_DIAGNOSTIC_STEP_SCHEMA,
        "graph_step": int(graph.get("step", -1)),
        "runtime_object_count": len(runtime_rows),
        "physical_visible_object_count": len(physical_rows),
        "matched_object_count": len(raw_matches),
        "node_miss_count": int(
            metrics["thresholds"][threshold_key]["missed_gt"]
        ),
        "node_false_positive_count": int(
            metrics["thresholds"][threshold_key]["false_positives"]
        ),
        "node_merge_count": int(metrics["merge_indicators"]),
        "node_split_count": int(metrics["over_split_indicators"]),
        "target_visible_in_evaluator_map": target_reference_index is not None,
        "target_matched_runtime_node": target_node_id is not None,
        "target_node_id": target_node_id,
        "target_match_iou": (
            None if target_match is None else float(target_match["iou"])
        ),
        "target_merge": target_merge,
        "target_split": target_split,
        "target_candidate_edge_present": bool(target_candidate_edge_present),
        "association_switch_count_current": switch_count_current,
        "association_switch_count_cumulative": int(
            state["association_switch_count"]
        ),
        "association_duplicate_count_current": duplicate_count_current,
        "association_duplicate_count_cumulative": int(
            state["association_duplicate_count"]
        ),
        "target_association_switch_count_cumulative": int(
            state["target_association_switch_count"]
        ),
        "physical_blocker_rows": physical_blockers,
        "edge_diagnostic_defined": bool(relation_defined),
        "edge_false_positive_count": false_positive_count,
        "edge_false_negative_count": false_negative_count,
        "edge_rows": relation_rows,
        "selected_source_is_physical_blocker": selected_source_is_physical,
        "selected_relation_false_positive": selected_relation_false_positive,
        "read_only": True,
        "environment_state_restored": True,
        "runtime_received_diagnostic": False,
        "evaluator_only_ground_truth": True,
        "simulator_ids_returned": False,
        "raw_maps_returned": False,
    }
    _assert_public_diagnostic(result)
    return result


def _oracle_rows(
    graph: Mapping[str, Any],
    private: Mapping[str, Any],
    instance_map: np.ndarray,
) -> list[Dict[str, Any]]:
    objects = [
        node for node in graph.get("nodes", []) if node.get("node_type") == "object"
    ]
    objects.sort(key=lambda node: repr(_typed_key(node["node_id"])))
    denominator = int(private.get("eligible_candidate_count", 0))
    counts = dict(private.get("blocker_candidate_counts") or {})
    physical_rows = []
    for raw_instance_id in sorted(
        {int(value) for value in private.get("physical_blocker_instance_ids", [])}
    ):
        physical = instance_map == raw_instance_id
        if physical.any():
            physical_rows.append((raw_instance_id, physical))
    predicted_masks = [
        np.asarray(decode_binary_mask_rle(node["footprint_mask"]), dtype=bool)
        for node in objects
    ]
    if any(mask.shape != instance_map.shape for mask in predicted_masks):
        raise ValueError("runtime and evaluator masks use different raw frames")
    ious = np.zeros((len(physical_rows), len(objects)), dtype=np.float64)
    for physical_index, (_instance_id, physical) in enumerate(physical_rows):
        for object_index, predicted in enumerate(predicted_masks):
            union = int(np.logical_or(predicted, physical).sum())
            if union:
                ious[physical_index, object_index] = float(
                    np.logical_and(predicted, physical).sum() / union
                )
    assignments = []
    if ious.size:
        physical_indices, object_indices = linear_sum_assignment(1.0 - ious)
        assignments = list(zip(physical_indices.tolist(), object_indices.tolist()))
    rows = []
    for physical_index, object_index in assignments:
        instance_id = physical_rows[physical_index][0]
        node = objects[object_index]
        iou = float(ious[physical_index, object_index])
        if iou <= 0.0:
            continue
        count = int(counts.get(str(instance_id), 0))
        rows.append(
            {
                "source_node_id": node["node_id"],
                "score": 0.0 if denominator <= 0 else float(count / denominator),
                "evaluation_match_iou": iou,
                "source": "physical_v1_evaluator_only",
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), repr(row["source_node_id"])))
    return rows


def build_evaluator_adapter(spec: Mapping[str, Any]) -> Dict[str, Any]:
    episode_id = spec.get("episode_id")
    evaluation_token = spec.get("evaluation_token")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("evaluator adapter requires episode_id")
    if not isinstance(evaluation_token, Mapping):
        raise ValueError("evaluator adapter requires the hidden evaluation token")
    target_id = evaluation_token.get("evaluation_object_id")
    if isinstance(target_id, bool) or not isinstance(target_id, (int, np.integer)):
        raise ValueError("evaluation_object_id must be an integer")
    target_id = int(target_id)
    handle = get_live_episode(episode_id)
    evaluation_config = spec.get("evaluation")
    if evaluation_config is not None and not isinstance(evaluation_config, Mapping):
        raise ValueError("evaluator configuration must be a mapping")
    diagnostic_config = _failure_diagnostics_config(
        (evaluation_config or {}).get("failure_diagnostics")
    )
    semantic_map_cache: Dict[int, np.ndarray] = {}
    diagnostic_state: Dict[str, Any] = {}

    def evaluate(actions_taken: int) -> Dict[str, Any]:
        actions = int(actions_taken)
        if actions != int(handle.bridge.state.action_count):
            raise ValueError("evaluator action count disagrees with live MEM state")
        if actions not in handle.evaluation_cache:
            details = evaluate_live_target_access_feasibility(
                handle.mem,
                target_instance_id=target_id,
                include_evaluation_private_blockers=True,
            )
            if diagnostic_config["enabled"]:
                instance_map, semantic_map = _current_evaluator_maps(handle.mem)
                semantic_map_cache[actions] = semantic_map
            else:
                instance_map = _current_instance_map(handle.mem)
            details["target_ever_visible"] = bool(np.any(instance_map == target_id))
            handle.evaluation_cache[actions] = details
            handle.instance_map_cache[actions] = instance_map
        return handle.evaluation_cache[actions]

    def access_evaluator(**kwargs: Any) -> Dict[str, Any]:
        token = kwargs.get("evaluation_token")
        if (
            not isinstance(token, Mapping)
            or int(token.get("evaluation_object_id", -1)) != target_id
        ):
            raise ValueError("access evaluator received a different hidden target")
        actions = int(kwargs["actions_taken"])
        details = evaluate(actions)
        if details.get("environment_state_restored") is not True:
            raise RuntimeError(
                "live access evaluator did not attest simulator-state restoration"
            )
        mechanism_consistent = None
        failure_diagnostic = None
        if actions > 0 and handle.latest_graph is not None and handle.bridge.history:
            prior = handle.evaluation_cache.get(actions - 1)
            selected = handle.bridge.history[-1].get("selected_action") or {}
            selected_source = selected.get("source_node_id")
            if (
                prior is not None
                and selected.get("kind") == "push"
                and selected_source is not None
            ):
                rows = _oracle_rows(
                    handle.latest_graph,
                    prior.get("_evaluation_private") or {},
                    handle.instance_map_cache[actions - 1],
                )
                mechanism_consistent = any(
                    _typed_key(row["source_node_id"]) == _typed_key(selected_source)
                    for row in rows
                )
            if diagnostic_config["enabled"]:
                if prior is None or actions - 1 not in semantic_map_cache:
                    raise RuntimeError(
                        "failure diagnostic requires the cached pre-action evaluator state"
                    )
                failure_diagnostic = _failure_diagnostic(
                    graph=handle.latest_graph,
                    instance_map=handle.instance_map_cache[actions - 1],
                    semantic_map=semantic_map_cache[actions - 1],
                    private=prior.get("_evaluation_private") or {},
                    target_private_id=target_id,
                    selected_action=selected,
                    config=diagnostic_config,
                    state=diagnostic_state,
                )
        result = {
            "access_feasible": bool(details["access_feasible"]),
            "target_ever_visible": bool(details["target_ever_visible"]),
            "mechanism_consistent": mechanism_consistent,
            "collision": bool(handle.bridge.state.collision),
            "candidate_count": int(details["candidate_count"]),
            "eligible_candidate_count": int(details["eligible_candidate_count"]),
            "clean_candidate_count": int(details["clean_candidate_count"]),
            "endpoint": details["endpoint"],
            "read_only": True,
            "environment_state_restored": True,
        }
        if failure_diagnostic is not None:
            result["failure_diagnostics"] = failure_diagnostic
        return result

    def oracle_blocker_provider(*, episode_id: str, step: int):
        if episode_id != handle.episode_id or int(step) != int(
            handle.bridge.state.action_count
        ):
            raise ValueError("oracle provider step disagrees with live episode")
        if handle.latest_graph is None:
            raise RuntimeError("oracle provider requires the current runtime graph")
        details = evaluate(int(handle.bridge.state.action_count))
        return _oracle_rows(
            handle.latest_graph,
            details.get("_evaluation_private") or {},
            handle.instance_map_cache[int(handle.bridge.state.action_count)],
        )

    callbacks: Dict[str, Any] = {
        "access_evaluator": access_evaluator,
        "provenance": {
            "schema": "psg_mem_x3_evaluator_provenance_v1",
            "failure_diagnostics": copy.deepcopy(diagnostic_config),
            "read_only": True,
            "returns_simulator_ids": False,
            "returns_raw_gt_maps": False,
        },
    }
    if str(spec.get("arm_id", "")).lower() == "h":
        callbacks["oracle_blocker_provider"] = oracle_blocker_provider
    return callbacks


__all__ = [
    "DEFAULT_FAILURE_DIAGNOSTICS",
    "FAILURE_DIAGNOSTICS_SCHEMA",
    "FAILURE_DIAGNOSTIC_STEP_SCHEMA",
    "build_evaluator_adapter",
]
