#!/usr/bin/env python3
"""Audit deterministic CNABU candidate availability against frozen v1 causes.

The full frozen evidence pass validates the offline label contract without
PyBullet. An optional bounded live pass reconstructs learned CNABU support,
converts it to world AABBs, and queries current-state IK plus known fixed bodies
in PyBullet. GT masks/ids are used only after inference for audit matching.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from scene_graph_mem.relations.candidate_trajectory import (
    TRAJECTORY_CANDIDATE_IDS,
    validate_candidate_action_mask_result,
)
from scene_graph_mem.relations.path_aligned_features import (
    reconstruct_sparse_node_voxel_support,
)
from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.utils.action_conditioned_relation_oracle import (
    OracleActionFamilyConfig,
    build_cnabu_runtime_candidate_action_mask,
    merge_instance_stack,
)
from shelf_gym.utils.mapping_utils import HeightmapGeneration


DEFAULT_RECORDS_JSON = Path(
    "/data/manipulation_map_data/derived/action_conditioned_relation_oracle_v1/"
    "cnabu_training_pack_100_threshold_0p40_20260713/records.json"
)
DEFAULT_CNABU_DERIVED_ROOT = Path(
    "/data/manipulation_map_data/derived/cnabu_d3g/"
    "raw_cnabu_concat_1000_20260528_100559"
)
DEFAULT_SPLIT_JSON = DEFAULT_RECORDS_JSON.with_name("split_manifest.json")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _binary_metrics(predictions: Sequence[bool], labels: Sequence[bool]) -> Dict[str, Any]:
    prediction = np.asarray(predictions, dtype=bool)
    label = np.asarray(labels, dtype=bool)
    if prediction.shape != label.shape:
        raise ValueError("binary predictions and labels must align")
    tp = int(np.logical_and(prediction, label).sum())
    tn = int(np.logical_and(~prediction, ~label).sum())
    fp = int(np.logical_and(prediction, ~label).sum())
    fn = int(np.logical_and(~prediction, label).sum())
    count = int(len(label))
    return {
        "count": count,
        "positive": int(label.sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": float((tp + tn) / count) if count else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
    }


def _numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def _ordered_oracle_trajectories(evidence: Mapping[str, Any]) -> Dict[int, List[Mapping[str, Any]]]:
    result: Dict[int, List[Mapping[str, Any]]] = {}
    for target in evidence.get("targets", []):
        target_id = int(target["target_instance_id"])
        by_candidate = {
            str(trajectory["grasp_id"]): trajectory
            for trajectory in target.get("trajectories", [])
        }
        missing = [candidate for candidate in TRAJECTORY_CANDIDATE_IDS if candidate not in by_candidate]
        if missing:
            raise ValueError("oracle target {} is missing candidates {}".format(target_id, missing))
        result[target_id] = [by_candidate[candidate] for candidate in TRAJECTORY_CANDIDATE_IDS]
    return result


def audit_offline_contract(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    kinematic: List[bool] = []
    fixed_free: List[bool] = []
    eligible: List[bool] = []
    reconstructed: List[bool] = []
    cause_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        evidence = _load_json(Path(str(record["action_oracle_evidence_file"])))
        for trajectories in _ordered_oracle_trajectories(evidence).values():
            for trajectory in trajectories:
                is_kinematic = bool(trajectory.get("kinematically_feasible", False))
                is_fixed_free = not bool(trajectory.get("fixed_environment_collision", False))
                is_eligible = bool(trajectory.get("eligible_for_scoring", False))
                kinematic.append(is_kinematic)
                fixed_free.append(is_fixed_free)
                eligible.append(is_eligible)
                reconstructed.append(is_kinematic and is_fixed_free)
                reasons = list(trajectory.get("exclusion_reasons") or [])
                cause = "eligible" if is_eligible else "+".join(sorted(reasons)) or "other"
                cause_counts[cause] += 1
    return {
        "record_count": len(records),
        "candidate_count": len(eligible),
        "kinematically_feasible_count": int(sum(kinematic)),
        "fixed_environment_collision_free_count": int(sum(fixed_free)),
        "eligible_count": int(sum(eligible)),
        "cause_counts": dict(sorted(cause_counts.items())),
        "eligibility_reconstruction": _binary_metrics(reconstructed, eligible),
    }


def _select_scene_disjoint_records(
    records: Sequence[Mapping[str, Any]],
    limit: int,
) -> List[Mapping[str, Any]]:
    grouped: Dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    for record in records:
        grouped[str(record["sample_id"]).split("/", 1)[0]].append(record)
    names = sorted(
        grouped,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    selected: List[Mapping[str, Any]] = []
    while len(selected) < int(limit) and any(grouped.values()):
        for name in names:
            if grouped[name] and len(selected) < int(limit):
                selected.append(grouped[name].popleft())
    return selected


def _records_for_split(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    split_name: str,
) -> List[Mapping[str, Any]]:
    key = "val_sample_ids" if split_name == "val" else "{}_sample_ids".format(split_name)
    sample_ids = [str(value) for value in split_manifest.get(key, [])]
    by_id = {str(record["sample_id"]): record for record in records}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError("split manifest references missing sample ids: {}".format(missing))
    return [by_id[sample_id] for sample_id in sample_ids]


def _load_runtime_support(
    record: Mapping[str, Any],
    cnabu_derived_root: Path,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    sample_id = str(record["sample_id"])
    cnabu_path = cnabu_derived_root / "samples" / sample_id / "pre_action" / "cnabu_hms.npz"
    node_path = Path(str(record["cnabu_node_masks_path"]))
    with np.load(cnabu_path, allow_pickle=False) as data:
        alpha = np.asarray(data["occupancy_alpha"], dtype=np.float32)
        beta = np.asarray(data["occupancy_beta"], dtype=np.float32)
        semantic = np.asarray(data["semantic_concentration"], dtype=np.float32)
        occupancy_mean = alpha / np.maximum(alpha + beta, 1e-8)
        semantic_mean = semantic / np.maximum(semantic.sum(axis=0, keepdims=True), 1e-8)
        crop_rows = tuple(int(value) for value in np.asarray(data["crop_rows"]).tolist())
    with np.load(node_path, allow_pickle=False) as data:
        node_masks = np.asarray(data["node_masks"], dtype=bool)
        node_classes = np.asarray(data["node_semantic_labels"], dtype=np.int64)
        node_ids = np.asarray(data["component_ids"], dtype=np.int64)
        node_crop_rows = tuple(int(value) for value in np.asarray(data["crop_rows"]).tolist())
    if crop_rows != node_crop_rows:
        raise ValueError("CNABU and learned-node crop rows do not match")
    support = reconstruct_sparse_node_voxel_support(
        occupancy_mean,
        semantic_mean,
        node_masks,
        node_classes,
        crop_rows=crop_rows,
    )
    return support, node_masks, node_classes, node_ids, crop_rows


def _gt_masks_classes_ids(pre_action_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(pre_action_dir / "gt_hms.npz", allow_pickle=False) as data:
        merged = merge_instance_stack(data["instance_maps"])
        semantic = np.asarray(data["semantic_2d"])
    masks = []
    classes = []
    ids = []
    for raw_id in np.unique(merged):
        instance_id = int(raw_id)
        if instance_id in (-1, 0):
            continue
        mask = merged == raw_id
        values, counts = np.unique(semantic[mask], return_counts=True)
        class_id = int(values[int(np.argmax(counts))])
        if class_id < 0 or class_id >= 14:
            continue
        masks.append(mask)
        classes.append(class_id)
        ids.append(instance_id)
    return np.asarray(masks, dtype=bool), np.asarray(classes), np.asarray(ids)


def _match_learned_nodes_to_gt(
    node_masks: np.ndarray,
    node_classes: np.ndarray,
    gt_masks: np.ndarray,
    gt_classes: np.ndarray,
    gt_ids: np.ndarray,
    threshold: float,
) -> List[int]:
    intersections = np.logical_and(node_masks[:, None], gt_masks[None]).sum(axis=(2, 3))
    unions = np.logical_or(node_masks[:, None], gt_masks[None]).sum(axis=(2, 3))
    iou = intersections / np.maximum(unions, 1)
    iou[node_classes[:, None] != gt_classes[None]] = 0.0
    rows, cols = linear_sum_assignment(-iou)
    matched = [-1] * len(node_masks)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if float(iou[row, col]) >= float(threshold):
            matched[row] = int(gt_ids[col])
    return matched


def audit_live_runtime(
    records: Sequence[Mapping[str, Any]],
    *,
    cnabu_derived_root: Path,
    match_iou_threshold: float,
    support_boundary_quantile: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    environment = ShelfEnv(render=False, max_obj_num=25, use_ycb=True)
    heightmap_generation = HeightmapGeneration(
        height_resolution=0.005,
        mapping_version="MEM",
        n_classes=15,
    )
    runtime_kinematic: List[bool] = []
    oracle_kinematic: List[bool] = []
    runtime_fixed_free: List[bool] = []
    oracle_fixed_free: List[bool] = []
    runtime_eligible: List[bool] = []
    oracle_eligible: List[bool] = []
    runtime_hard_sample_counts: Dict[str, int] = defaultdict(int)
    runtime_hard_candidate_counts: Dict[str, int] = defaultdict(int)
    runtime_minimum_distances: Dict[str, List[float]] = defaultdict(list)
    scene_summaries = []
    try:
        for record in records:
            environment.reset_robot(environment.initial_parameters)
            environment.move_gripper(0.085)
            initial = np.asarray(environment.get_current_joint_config(), dtype=np.float64)
            support, node_masks, node_classes, node_ids, crop_rows = _load_runtime_support(
                record,
                cnabu_derived_root,
            )
            runtime = build_cnabu_runtime_candidate_action_mask(
                environment,
                heightmap_generation,
                support.indices_zyx,
                crop_rows=crop_rows,
                node_ids=node_ids.tolist(),
                initial_arm_config=initial,
                config=OracleActionFamilyConfig(),
                support_boundary_quantile=support_boundary_quantile,
            )
            gt_masks, gt_classes, gt_ids = _gt_masks_classes_ids(Path(str(record["sample_dir"])))
            matched_ids = _match_learned_nodes_to_gt(
                node_masks,
                node_classes,
                gt_masks,
                gt_classes,
                gt_ids,
                match_iou_threshold,
            )
            evidence = _load_json(Path(str(record["action_oracle_evidence_file"])))
            oracle = _ordered_oracle_trajectories(evidence)
            kinematic_mask = np.asarray(runtime["kinematic_mask"], dtype=bool)
            fixed_free_mask = np.asarray(
                runtime["fixed_environment_collision_free_mask"], dtype=bool
            )
            eligible_mask = validate_candidate_action_mask_result(
                runtime,
                node_ids=node_ids.tolist(),
            ).astype(bool)
            compared = 0
            fixed_compared = 0
            for node_index, instance_id in enumerate(matched_ids):
                if instance_id < 0 or instance_id not in oracle:
                    continue
                trajectories = oracle[instance_id]
                for candidate_index, trajectory in enumerate(trajectories):
                    runtime_k = bool(kinematic_mask[node_index, candidate_index])
                    oracle_k = bool(trajectory.get("kinematically_feasible", False))
                    candidate_detail = runtime["targets"][node_index]["candidates"][candidate_index]
                    for stage, stage_detail in candidate_detail.get("stage_evidence", {}).items():
                        for body, body_detail in stage_detail.get("fixed_bodies", {}).items():
                            for source, field in (
                                ("robot", "robot_hard_penetration_sample_count"),
                                (
                                    "carried_proxy",
                                    "carried_proxy_hard_penetration_sample_count",
                                ),
                            ):
                                count = int(body_detail.get(field, 0))
                                key = "/".join((str(stage), str(body), source))
                                runtime_hard_sample_counts[key] += count
                                runtime_hard_candidate_counts[key] += int(count > 0)
                                minimum = body_detail.get(
                                    "minimum_{}_signed_distance_m".format(source)
                                )
                                if minimum is not None:
                                    runtime_minimum_distances[key].append(float(minimum))
                    runtime_kinematic.append(runtime_k)
                    oracle_kinematic.append(oracle_k)
                    runtime_eligible.append(bool(eligible_mask[node_index, candidate_index]))
                    oracle_eligible.append(bool(trajectory.get("eligible_for_scoring", False)))
                    compared += 1
                    if runtime_k and oracle_k:
                        runtime_fixed_free.append(
                            bool(fixed_free_mask[node_index, candidate_index])
                        )
                        oracle_fixed_free.append(
                            not bool(trajectory.get("fixed_environment_collision", False))
                        )
                        fixed_compared += 1
            scene_summaries.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "learned_node_count": len(node_masks),
                    "matched_node_count": int(sum(value >= 0 for value in matched_ids)),
                    "candidate_comparison_count": compared,
                    "fixed_collision_comparison_count": fixed_compared,
                }
            )
    finally:
        environment.close()
    return {
        "scene_count": len(records),
        "scenes": scene_summaries,
        "kinematic_mask": _binary_metrics(runtime_kinematic, oracle_kinematic),
        "fixed_environment_collision_free_on_mutually_kinematic": _binary_metrics(
            runtime_fixed_free,
            oracle_fixed_free,
        ),
        "action_eligible_mask": _binary_metrics(runtime_eligible, oracle_eligible),
        "runtime_hard_penetration_diagnostics": {
            "sample_counts_by_stage_body_source": dict(
                sorted(runtime_hard_sample_counts.items())
            ),
            "candidate_counts_by_stage_body_source": dict(
                sorted(runtime_hard_candidate_counts.items())
            ),
            "minimum_signed_distance_m_by_stage_body_source": {
                key: _numeric_summary(values)
                for key, values in sorted(runtime_minimum_distances.items())
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "support_boundary_quantile": float(support_boundary_quantile),
        "runtime_inputs_use_gt_or_simulator_object_ids": False,
        "gt_used_only_for_post_inference_matching_and_metrics": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS_JSON)
    parser.add_argument("--cnabu-derived-root", type=Path, default=DEFAULT_CNABU_DERIVED_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--live-limit", type=int, default=0)
    parser.add_argument("--live-split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--match-iou-threshold", type=float, default=0.25)
    parser.add_argument("--support-boundary-quantile", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_json(args.records_json)
    if not isinstance(records, list):
        raise ValueError("records-json must contain a list")
    result: Dict[str, Any] = {
        "schema": "cnabu_runtime_candidate_action_mask_audit_v0",
        "records_json": str(args.records_json),
        "split_json": str(args.split_json),
        "cnabu_derived_root": str(args.cnabu_derived_root),
        "offline_contract": audit_offline_contract(records),
        "live_runtime": None,
        "safety": {
            "runs_training": False,
            "writes_checkpoints_models_or_datasets": False,
            "runtime_uses_gt_or_simulator_object_ids": False,
            "offline_audit_uses_oracle_causes": True,
        },
    }
    if int(args.live_limit) > 0:
        live_records = records
        if args.live_split != "all":
            split_manifest = _load_json(args.split_json)
            if not isinstance(split_manifest, Mapping):
                raise ValueError("split-json must contain a JSON object")
            live_records = _records_for_split(
                records,
                split_manifest,
                str(args.live_split),
            )
        selected = _select_scene_disjoint_records(live_records, int(args.live_limit))
        result["live_runtime"] = audit_live_runtime(
            selected,
            cnabu_derived_root=args.cnabu_derived_root,
            match_iou_threshold=float(args.match_iou_threshold),
            support_boundary_quantile=float(args.support_boundary_quantile),
        )
        result["live_runtime"]["split"] = str(args.live_split)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser()
        if output.exists():
            raise FileExistsError("refusing to overwrite {}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
