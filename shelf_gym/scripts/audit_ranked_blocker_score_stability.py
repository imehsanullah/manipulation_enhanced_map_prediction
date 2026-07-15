#!/usr/bin/env python3
"""Run the predeclared non-test ranked-blocker score-stability audit."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau

from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.utils.action_conditioned_relation_oracle import (
    StaticOraclePerturbationConfig,
    evaluate_saved_scene,
    rethreshold_static_oracle_contact_evidence,
)


WORKSPACE = Path("/home/user/ehsanullahm1/thesis")
PROJECT = WORKSPACE / "manipulation_enhanced_map_prediction"
DEFAULT_RECORDS = Path(
    "/data/manipulation_map_data/derived/action_conditioned_relation_oracle_v1/"
    "candidate_planner_swept_features_340_fresh_start100_q0p05_20260715/"
    "records_with_candidate_planner_swept_features.json"
)
DEFAULT_SPLIT = Path(
    "/data/manipulation_map_data/derived/action_conditioned_relation_oracle_v1/"
    "cnabu_training_pack_340_fresh_start100_threshold_0p40_20260714/"
    "split_manifest.json"
)
DEFAULT_GROUPS = ("0", "1", "2", "3", "10", "11", "12", "13")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_CONTACT_THRESHOLDS_M = (0.001, 0.002, 0.003)
STAGES = ("approach", "grasp", "extraction")
_WORKER_ENV: Optional[ShelfEnv] = None


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--scene-groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--records-per-group", type=int, default=4)
    parser.add_argument("--perturbation-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--contact-thresholds-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_CONTACT_THRESHOLDS_M),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--retain-compact-runs",
        action="store_true",
        help="Retain per-scene matrices; disabled by default to keep thesis records lightweight.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_group(sample_id: str) -> str:
    return str(sample_id).split("/", 1)[0]


def select_predeclared_stability_records(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_sample_ids: Sequence[str],
    scene_groups: Sequence[str],
    records_per_group: int,
) -> list[Dict[str, Any]]:
    """Select lexicographically first records per declared non-test group."""

    count = int(records_per_group)
    if count <= 0:
        raise ValueError("records_per_group must be positive")
    allowed = set(str(value) for value in allowed_sample_ids)
    by_group: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if sample_id in allowed:
            by_group[_scene_group(sample_id)].append(record)
    selected: list[Dict[str, Any]] = []
    for group in [str(value) for value in scene_groups]:
        options = sorted(by_group.get(group, []), key=lambda item: str(item["sample_id"]))
        if len(options) < count:
            raise ValueError(
                "scene group {} has only {} available non-test records; need {}".format(
                    group, len(options), count
                )
            )
        selected.extend(dict(item) for item in options[:count])
    if len({str(item["sample_id"]) for item in selected}) != len(selected):
        raise AssertionError("stability selection contains duplicate sample ids")
    return selected


def _threshold_key(value: float) -> str:
    return "{:.6f}".format(float(value)).rstrip("0").rstrip(".")


def _compact_scene_variant(record: Mapping[str, Any]) -> Dict[str, Any]:
    targets = []
    for target in record.get("targets", []):
        targets.append(
            {
                "target_index": int(target["target_index"]),
                "target_instance_id": int(target["target_instance_id"]),
                "eligible_trajectory_count": int(target["eligible_trajectory_count"]),
                "eligible_trajectory_weight": float(target["eligible_trajectory_weight"]),
                "has_defined_pair_scores": bool(target["has_defined_pair_scores"]),
                "trajectories": [
                    {
                        "trajectory_id": str(trajectory["trajectory_id"]),
                        "grasp_id": str(trajectory.get("grasp_id") or ""),
                        "eligible_for_scoring": bool(
                            trajectory.get("eligible_for_scoring", False)
                        ),
                        "blocked_by": [int(value) for value in trajectory.get("blocked_by", [])],
                        "blocked_by_stage": {
                            stage: [
                                int(value)
                                for value in dict(
                                    trajectory.get("blocked_by_stage") or {}
                                ).get(stage, [])
                            ]
                            for stage in STAGES
                        },
                    }
                    for trajectory in target.get("trajectories", [])
                ],
            }
        )
    return {
        "node_order_instance_ids": [
            int(value) for value in record["node_order_instance_ids"]
        ],
        "score_matrix": record["score_matrix"],
        "score_valid_mask": record["score_valid_mask"],
        "binary_adjacency_matrix": record["binary_adjacency_matrix"],
        "stage_score_matrices": record["stage_score_matrices"],
        "targets": targets,
    }


def compact_stability_scene(
    record: Mapping[str, Any],
    *,
    contact_thresholds_m: Sequence[float],
) -> Dict[str, Any]:
    """Keep matrices, target/candidate sets, and object-size diagnostics only."""

    thresholds = [float(value) for value in contact_thresholds_m]
    variants: Dict[str, Any] = {}
    for threshold in thresholds:
        if math.isclose(threshold, 0.002, rel_tol=0.0, abs_tol=1.0e-12):
            variant_record = record
        else:
            variant_record = rethreshold_static_oracle_contact_evidence(
                record, hard_penetration_m=threshold
            )
        variants[_threshold_key(threshold)] = _compact_scene_variant(variant_record)
    rebuilt_frozen = rethreshold_static_oracle_contact_evidence(
        record, hard_penetration_m=0.002
    )
    for name in (
        "score_matrix",
        "score_valid_mask",
        "binary_adjacency_matrix",
        "stage_score_matrices",
    ):
        if rebuilt_frozen[name] != record[name]:
            raise AssertionError(
                "stored contact evidence does not exactly reconstruct frozen {}".format(name)
            )
    object_sizes = {}
    for item in record.get("object_records", []):
        lower = np.asarray(item["world_aabb"][0], dtype=np.float64)
        upper = np.asarray(item["world_aabb"][1], dtype=np.float64)
        object_sizes[str(int(item["instance_id"]))] = float(np.prod(upper - lower))
    return {
        "sample_id": str(record["sample_id"]),
        "scene_group": _scene_group(str(record["sample_id"])),
        "object_aabb_volumes_m3": object_sizes,
        "variants_by_hard_penetration_m": variants,
        "perturbation": dict(record.get("metadata") or {}).get(
            "static_pose_yaw_perturbation"
        ),
        "runtime_seconds": float(dict(record.get("scene_summary") or {}).get("runtime_seconds", 0.0)),
        "friction_varied": False,
    }


def _worker_init() -> None:
    global _WORKER_ENV
    _WORKER_ENV = ShelfEnv(render=False, max_obj_num=25, use_ycb=True)
    atexit.register(_WORKER_ENV.close)


def _run_perturbation_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    if _WORKER_ENV is None:
        raise RuntimeError("stability worker environment was not initialized")
    record, _ = evaluate_saved_scene(
        _WORKER_ENV,
        pre_action_dir=Path(str(task["pre_action_dir"])),
        static_pose_yaw_perturbation=StaticOraclePerturbationConfig(
            seed=int(task["seed"]),
            xy_position_jitter_m=0.001,
            yaw_jitter_degrees=0.5,
        ),
    )
    if str(record["sample_id"]) != str(task["sample_id"]):
        raise RuntimeError("worker replayed the wrong saved scene")
    compact = compact_stability_scene(
        record,
        contact_thresholds_m=task["contact_thresholds_m"],
    )
    compact["seed"] = int(task["seed"])
    compact["run_id"] = str(task["run_id"])
    return compact


def validate_stability_run_matching(
    reference: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    *,
    expected_seed: int,
) -> None:
    """Guard pairing by scene, target, candidate, and declared seed."""

    if str(reference["sample_id"]) != str(perturbed["sample_id"]):
        raise ValueError("stability pair sample ids differ")
    if int(perturbed["seed"]) != int(expected_seed):
        raise ValueError("stability pair perturbation seed differs")
    frozen_key = _threshold_key(0.002)
    left = reference["variants_by_hard_penetration_m"][frozen_key]
    right = perturbed["variants_by_hard_penetration_m"][frozen_key]
    if left["node_order_instance_ids"] != right["node_order_instance_ids"]:
        raise ValueError("stability pair node orders differ")
    left_targets = {
        int(target["target_instance_id"]): {
            str(item["trajectory_id"]) for item in target["trajectories"]
        }
        for target in left["targets"]
    }
    right_targets = {
        int(target["target_instance_id"]): {
            str(item["trajectory_id"]) for item in target["trajectories"]
        }
        for target in right["targets"]
    }
    if left_targets != right_targets:
        raise ValueError("stability pair target/candidate identities differ")


def _matrix_arrays(variant: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(variant["score_valid_mask"], dtype=bool)
    scores = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in variant["score_matrix"]
        ],
        dtype=np.float64,
    )
    binary = np.asarray(variant["binary_adjacency_matrix"], dtype=bool)
    return scores, valid, binary


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _candidate_row(grasp_id: str) -> str:
    match = re.search(r"x(0\.\d+)_z(0\.\d+)", str(grasp_id))
    return "x{}_z{}".format(match.group(1), match.group(2)) if match else str(grasp_id)


def compare_stability_pair(
    reference: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    *,
    reference_threshold_m: float = 0.002,
    perturbed_threshold_m: float = 0.002,
) -> Dict[str, Any]:
    """Return pair/query/candidate stability observations for one paired scene."""

    left = reference["variants_by_hard_penetration_m"][_threshold_key(reference_threshold_m)]
    right = perturbed["variants_by_hard_penetration_m"][_threshold_key(perturbed_threshold_m)]
    if left["node_order_instance_ids"] != right["node_order_instance_ids"]:
        raise ValueError("cannot compare different node orders")
    scores_left, valid_left, binary_left = _matrix_arrays(left)
    scores_right, valid_right, binary_right = _matrix_arrays(right)
    comparable = valid_left & valid_right
    absolute_changes = np.abs(scores_left[comparable] - scores_right[comparable])
    edge_flips = binary_left[comparable] != binary_right[comparable]
    validity_changes = valid_left != valid_right
    node_ids = list(left["node_order_instance_ids"])
    query_rows = []
    source_change_rows = []
    for target_index, target_id in enumerate(node_ids):
        query_comparable = comparable[:, target_index]
        reference_defined = bool(np.any(valid_left[:, target_index]))
        perturbed_defined = bool(np.any(valid_right[:, target_index]))
        row: Dict[str, Any] = {
            "target_instance_id": int(target_id),
            "reference_defined": reference_defined,
            "perturbed_defined": perturbed_defined,
            "undefined_status_changed": reference_defined != perturbed_defined,
            "score_mae": None,
            "score_max_change": None,
            "edge_flip_count": 0,
            "comparable_pair_count": int(np.count_nonzero(query_comparable)),
            "top_max_set_jaccard": None,
            "kendall_tau_b": None,
        }
        if np.any(query_comparable):
            left_values = scores_left[query_comparable, target_index]
            right_values = scores_right[query_comparable, target_index]
            changes = np.abs(left_values - right_values)
            row["score_mae"] = float(changes.mean())
            row["score_max_change"] = float(changes.max())
            row["edge_flip_count"] = int(
                np.count_nonzero(
                    binary_left[query_comparable, target_index]
                    != binary_right[query_comparable, target_index]
                )
            )
            if float(left_values.max(initial=0.0)) > 0.0:
                source_indices = np.flatnonzero(query_comparable)
                left_top = set(
                    source_indices[np.isclose(left_values, float(left_values.max()))].tolist()
                )
                right_top = set(
                    source_indices[np.isclose(right_values, float(right_values.max()))].tolist()
                )
                row["top_max_set_jaccard"] = _jaccard(left_top, right_top)
            if np.unique(left_values).size >= 2 and np.unique(right_values).size >= 2:
                tau = kendalltau(left_values, right_values, variant="b").statistic
                row["kendall_tau_b"] = float(tau) if np.isfinite(tau) else None
            for source_index in np.flatnonzero(query_comparable):
                source_change_rows.append(
                    {
                        "source_instance_id": int(node_ids[source_index]),
                        "target_instance_id": int(target_id),
                        "absolute_score_change": float(
                            abs(
                                scores_left[source_index, target_index]
                                - scores_right[source_index, target_index]
                            )
                        ),
                        "edge_flipped": bool(
                            binary_left[source_index, target_index]
                            != binary_right[source_index, target_index]
                        ),
                    }
                )
        query_rows.append(row)

    left_targets = {int(item["target_instance_id"]): item for item in left["targets"]}
    right_targets = {int(item["target_instance_id"]): item for item in right["targets"]}
    candidate_rows = []
    for target_id in sorted(left_targets):
        left_candidates = {
            str(item["trajectory_id"]): item for item in left_targets[target_id]["trajectories"]
        }
        right_candidates = {
            str(item["trajectory_id"]): item for item in right_targets[target_id]["trajectories"]
        }
        if set(left_candidates) != set(right_candidates):
            raise ValueError("candidate ids differ inside paired target")
        for trajectory_id in sorted(left_candidates):
            left_candidate = left_candidates[trajectory_id]
            right_candidate = right_candidates[trajectory_id]
            candidate_rows.append(
                {
                    "target_instance_id": int(target_id),
                    "trajectory_id": trajectory_id,
                    "candidate_row": _candidate_row(left_candidate["grasp_id"]),
                    "eligibility_changed": bool(
                        left_candidate["eligible_for_scoring"]
                        != right_candidate["eligible_for_scoring"]
                    ),
                    "blocker_set_jaccard": _jaccard(
                        set(left_candidate["blocked_by"]),
                        set(right_candidate["blocked_by"]),
                    ),
                    "stage_blocker_set_jaccard": {
                        stage: _jaccard(
                            set(left_candidate["blocked_by_stage"][stage]),
                            set(right_candidate["blocked_by_stage"][stage]),
                        )
                        for stage in STAGES
                    },
                }
            )
    stage_changes = {}
    for stage in STAGES:
        stage_left = np.asarray(
            [
                [np.nan if value is None else float(value) for value in row]
                for row in left["stage_score_matrices"][stage]
            ]
        )
        stage_right = np.asarray(
            [
                [np.nan if value is None else float(value) for value in row]
                for row in right["stage_score_matrices"][stage]
            ]
        )
        changes = np.abs(stage_left[comparable] - stage_right[comparable])
        stage_changes[stage] = {
            "absolute_changes": [float(value) for value in changes],
        }
    return {
        "sample_id": str(reference["sample_id"]),
        "scene_group": str(reference["scene_group"]),
        "seed": perturbed.get("seed"),
        "reference_threshold_m": float(reference_threshold_m),
        "perturbed_threshold_m": float(perturbed_threshold_m),
        "absolute_score_changes": [float(value) for value in absolute_changes],
        "edge_flip_count": int(np.count_nonzero(edge_flips)),
        "comparable_pair_count": int(np.count_nonzero(comparable)),
        "validity_change_count": int(np.count_nonzero(validity_changes)),
        "query_rows": query_rows,
        "candidate_rows": candidate_rows,
        "source_change_rows": source_change_rows,
        "stage_changes": stage_changes,
    }


def _numeric(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
    }


def summarize_stability_comparisons(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    changes = [value for item in comparisons for value in item["absolute_score_changes"]]
    query_rows = [row for item in comparisons for row in item["query_rows"]]
    jaccards = [
        float(row["top_max_set_jaccard"])
        for row in query_rows
        if row["top_max_set_jaccard"] is not None
    ]
    taus = [
        float(row["kendall_tau_b"])
        for row in query_rows
        if row["kendall_tau_b"] is not None
    ]
    flip_count = int(sum(int(item["edge_flip_count"]) for item in comparisons))
    pair_count = int(sum(int(item["comparable_pair_count"]) for item in comparisons))
    candidate_rows = [row for item in comparisons for row in item["candidate_rows"]]
    by_candidate: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_candidate[str(row["candidate_row"])].append(row)
    by_stage = {
        stage: _numeric(
            [
                value
                for item in comparisons
                for value in item["stage_changes"][stage]["absolute_changes"]
            ]
        )
        for stage in STAGES
    }
    return {
        "comparison_count": int(len(comparisons)),
        "score_absolute_change": _numeric(changes),
        "binary_edge_flip_count": flip_count,
        "comparable_pair_count": pair_count,
        "binary_edge_flip_rate": float(flip_count / pair_count) if pair_count else None,
        "validity_change_count": int(
            sum(int(item["validity_change_count"]) for item in comparisons)
        ),
        "undefined_query_change_count": int(
            sum(bool(row["undefined_status_changed"]) for row in query_rows)
        ),
        "top_max_set_jaccard": _numeric(jaccards),
        "kendall_tau_b_defined_non_tied_queries": _numeric(taus),
        "candidate_rows": {
            name: {
                "candidate_count": int(len(rows)),
                "eligibility_change_count": int(
                    sum(bool(row["eligibility_changed"]) for row in rows)
                ),
                "eligibility_change_rate": float(
                    sum(bool(row["eligibility_changed"]) for row in rows) / len(rows)
                ),
                "blocker_set_jaccard": _numeric(
                    [float(row["blocker_set_jaccard"]) for row in rows]
                ),
            }
            for name, rows in sorted(by_candidate.items())
        },
        "stage_score_absolute_change": by_stage,
    }


def build_score_stability_report(
    references: Mapping[str, Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    contact_thresholds_m: Sequence[float],
) -> Dict[str, Any]:
    by_run = {(str(item["sample_id"]), int(item["seed"])): item for item in runs}
    expected = {
        (sample_id, int(seed)) for sample_id in references for seed in seeds
    }
    if set(by_run) != expected:
        raise ValueError("stability runs do not exactly match the predeclared scene/seed grid")
    comparisons = []
    for sample_id in sorted(references):
        for seed in seeds:
            run = by_run[(sample_id, int(seed))]
            validate_stability_run_matching(
                references[sample_id], run, expected_seed=int(seed)
            )
            comparisons.append(compare_stability_pair(references[sample_id], run))
    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in comparisons:
        grouped[str(item["scene_group"])].append(item)
    group_summaries = {
        group: summarize_stability_comparisons(items)
        for group, items in sorted(grouped.items())
    }
    overall = summarize_stability_comparisons(comparisons)
    volume_values = np.asarray(
        [
            float(value)
            for reference in references.values()
            for value in dict(reference.get("object_aabb_volumes_m3") or {}).values()
        ],
        dtype=np.float64,
    )
    volume_boundaries = (
        np.quantile(volume_values, [1.0 / 3.0, 2.0 / 3.0]).tolist()
        if volume_values.size
        else [0.0, 0.0]
    )

    def size_bin(volume: float) -> str:
        if volume <= float(volume_boundaries[0]):
            return "small"
        if volume <= float(volume_boundaries[1]):
            return "medium"
        return "large"

    source_pair_change_rows = []
    for item in comparisons:
        volumes = dict(
            references[str(item["sample_id"])].get("object_aabb_volumes_m3") or {}
        )
        for row in item["source_change_rows"]:
            volume = float(volumes[str(row["source_instance_id"])])
            source_pair_change_rows.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "scene_group": str(item["scene_group"]),
                    "seed": int(item["seed"]),
                    "source_instance_id": int(row["source_instance_id"]),
                    "target_instance_id": int(row["target_instance_id"]),
                    "absolute_score_change": float(row["absolute_score_change"]),
                    "edge_flipped": bool(row["edge_flipped"]),
                    "source_aabb_volume_m3": volume,
                    "source_object_size_bin": size_bin(volume),
                }
            )
    by_size: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_pair_change_rows:
        by_size[str(row["source_object_size_bin"])].append(row)
    object_size_summary = {
        name: {
            "pair_observation_count": int(len(rows)),
            "score_absolute_change": _numeric(
                [float(row["absolute_score_change"]) for row in rows]
            ),
            "binary_edge_flip_count": int(sum(bool(row["edge_flipped"]) for row in rows)),
            "binary_edge_flip_rate": float(
                sum(bool(row["edge_flipped"]) for row in rows) / len(rows)
            ),
        }
        for name, rows in sorted(by_size.items())
    }
    threshold_sensitivity: Dict[str, Any] = {}
    for threshold in contact_thresholds_m:
        if math.isclose(float(threshold), 0.002, abs_tol=1.0e-12):
            continue
        threshold_comparisons = [
            compare_stability_pair(
                run,
                run,
                reference_threshold_m=0.002,
                perturbed_threshold_m=float(threshold),
            )
            for run in runs
        ]
        threshold_sensitivity[_threshold_key(float(threshold))] = {
            "comparison_to_frozen_0.002_m": summarize_stability_comparisons(
                threshold_comparisons
            ),
            "selection_role": "sensitivity_only_not_selected_from_outcomes",
        }
    group_jaccard_gate = all(
        summary["top_max_set_jaccard"]["median"] is not None
        and float(summary["top_max_set_jaccard"]["median"]) >= 0.80
        for summary in group_summaries.values()
    )
    overall_tau = overall["kendall_tau_b_defined_non_tied_queries"]["median"]
    group_tau_gate = all(
        summary["kendall_tau_b_defined_non_tied_queries"]["median"] is None
        or float(summary["kendall_tau_b_defined_non_tied_queries"]["median"]) >= 0.0
        for summary in group_summaries.values()
    )
    edge_flip_rate = overall["binary_edge_flip_rate"]
    concentration_checks = {
        "candidate_row_max_eligibility_change_rate_at_most_0.10": all(
            value["eligibility_change_rate"] <= 0.10
            for value in overall["candidate_rows"].values()
        ),
        "stage_max_mean_absolute_change_at_most_0.10": all(
            value["mean"] is None or float(value["mean"]) <= 0.10
            for value in overall["stage_score_absolute_change"].values()
        ),
        "object_size_bin_max_edge_flip_rate_at_most_0.10": all(
            value["binary_edge_flip_rate"] <= 0.10
            for value in object_size_summary.values()
        ),
        "cnabu_matching_regime": "evaluated_in_projection_join_report",
    }
    gates = {
        "every_group_median_top_max_set_jaccard_at_least_0.80": group_jaccard_gate,
        "overall_median_kendall_tau_b_at_least_0.70": bool(
            overall_tau is not None and float(overall_tau) >= 0.70
        ),
        "no_group_negative_median_kendall_tau_b": group_tau_gate,
        "binary_edge_flip_rate_at_most_0.10": bool(
            edge_flip_rate is not None and float(edge_flip_rate) <= 0.10
        ),
        "candidate_and_stage_instability_not_concentrated": bool(
            all(
                value is True
                for value in concentration_checks.values()
                if isinstance(value, bool)
            )
        ),
        "projection_regime_check_pending": True,
    }
    paired_summaries = []
    for item in comparisons:
        query_rows = list(item["query_rows"])
        paired_summaries.append(
            {
                "sample_id": str(item["sample_id"]),
                "scene_group": str(item["scene_group"]),
                "seed": int(item["seed"]),
                "score_absolute_change": _numeric(item["absolute_score_changes"]),
                "binary_edge_flip_count": int(item["edge_flip_count"]),
                "comparable_pair_count": int(item["comparable_pair_count"]),
                "validity_change_count": int(item["validity_change_count"]),
                "top_max_set_jaccard": _numeric(
                    [
                        float(row["top_max_set_jaccard"])
                        for row in query_rows
                        if row["top_max_set_jaccard"] is not None
                    ]
                ),
                "kendall_tau_b": _numeric(
                    [
                        float(row["kendall_tau_b"])
                        for row in query_rows
                        if row["kendall_tau_b"] is not None
                    ]
                ),
                "undefined_query_change_count": int(
                    sum(bool(row["undefined_status_changed"]) for row in query_rows)
                ),
                "candidate_eligibility_change_count": int(
                    sum(bool(row["eligibility_changed"]) for row in item["candidate_rows"])
                ),
            }
        )
    return {
        "schema": "ranked_blocker_score_stability_report_v1",
        "selection_uses_test_groups": False,
        "perturbation": {
            "seeds": [int(value) for value in seeds],
            "xy_position_jitter_m": 0.001,
            "yaw_jitter_degrees": 0.5,
            "friction_varied": False,
        },
        "overall": overall,
        "scene_groups": group_summaries,
        "paired_scene_seed_summaries": paired_summaries,
        "source_object_size_bins": {
            "aabb_volume_tertile_boundaries_m3": [
                float(value) for value in volume_boundaries
            ],
            "summaries": object_size_summary,
        },
        "source_pair_change_rows_for_projection_join": source_pair_change_rows,
        "contact_threshold_sensitivity": threshold_sensitivity,
        "concentration_checks": concentration_checks,
        "gates": gates,
        "gate_complete": False,
        "gate_incomplete_reason": (
            "object-size and CNABU matching-regime concentration require the learned-node projection join"
        ),
    }


def _manifest(args: argparse.Namespace, selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    git = lambda *parts: subprocess.check_output(
        ["git", "-C", str(PROJECT), *parts], text=True
    ).strip()
    return {
        "schema": "ranked_blocker_score_stability_manifest_v1",
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_executable": sys.executable,
        "git_branch": git("branch", "--show-current"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short", "--branch"),
        "records_json": {
            "path": str(args.records_json.resolve()),
            "sha256": _sha256(args.records_json),
        },
        "split_json": {
            "path": str(args.split_json.resolve()),
            "sha256": _sha256(args.split_json),
        },
        "selected_samples": [str(item["sample_id"]) for item in selected],
        "scene_groups": [str(value) for value in args.scene_groups],
        "records_per_group": int(args.records_per_group),
        "perturbation_seeds": [int(value) for value in args.perturbation_seeds],
        "contact_thresholds_m": [float(value) for value in args.contact_thresholds_m],
        "workers": int(args.workers),
        "retain_compact_runs": bool(args.retain_compact_runs),
        "policy": {
            "runs_training": False,
            "writes_checkpoint": False,
            "writes_dataset": False,
            "varies_friction_for_static_score_stability": False,
            "uses_test_groups": False,
            "output_is_compact_audit_evidence_only": True,
        },
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite existing output directory: {}".format(args.output_dir))
    if int(args.workers) <= 0:
        raise SystemExit("--workers must be positive")
    seeds = [int(value) for value in args.perturbation_seeds]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise SystemExit("at least three unique perturbation seeds are required")
    thresholds = [float(value) for value in args.contact_thresholds_m]
    if not any(math.isclose(value, 0.002, abs_tol=1.0e-12) for value in thresholds):
        raise SystemExit("contact thresholds must include the frozen 0.002 m value")
    records = _read_json(args.records_json)
    split = _read_json(args.split_json)
    non_test_ids = [
        *[str(value) for value in split["train_sample_ids"]],
        *[str(value) for value in split["val_sample_ids"]],
    ]
    test_groups = set(str(value) for value in split["test_scene_groups"])
    if set(str(value) for value in args.scene_groups) & test_groups:
        raise SystemExit("stability target selection cannot include test scene groups")
    selected = select_predeclared_stability_records(
        records,
        allowed_sample_ids=non_test_ids,
        scene_groups=args.scene_groups,
        records_per_group=int(args.records_per_group),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = output_dir / "compact_runs"
    if args.retain_compact_runs:
        runs_dir.mkdir()
    manifest = _manifest(args, selected)
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        references = {}
        for record in selected:
            scene = _read_json(Path(record["action_oracle_evidence_file"]))
            references[str(record["sample_id"])] = compact_stability_scene(
                scene, contact_thresholds_m=thresholds
            )
        _write_json(
            output_dir / "selection.json",
            {
                "schema": "ranked_blocker_stability_selection_v1",
                "selection_rule": "lexicographically first N manifest-listed records per predeclared non-test group",
                "selected_samples": [
                    {
                        "sample_id": str(item["sample_id"]),
                        "scene_group": _scene_group(str(item["sample_id"])),
                        "pre_action_dir": str(item["sample_dir"]),
                    }
                    for item in selected
                ],
            },
        )
        tasks = [
            {
                "run_id": "{}__seed_{}".format(
                    str(record["sample_id"]).replace("/", "__"), seed
                ),
                "sample_id": str(record["sample_id"]),
                "pre_action_dir": str(record["sample_dir"]),
                "seed": int(seed),
                "contact_thresholds_m": thresholds,
            }
            for record in selected
            for seed in seeds
        ]
        runs = []
        with ProcessPoolExecutor(
            max_workers=int(args.workers), initializer=_worker_init
        ) as executor:
            futures = {executor.submit(_run_perturbation_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                runs.append(result)
                if args.retain_compact_runs:
                    _write_json(runs_dir / (str(task["run_id"]) + ".json"), result)
                print(
                    json.dumps(
                        {
                            "run_id": result["run_id"],
                            "runtime_seconds": result["runtime_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        report = build_score_stability_report(
            references,
            runs,
            seeds=seeds,
            contact_thresholds_m=thresholds,
        )
        _write_json(output_dir / "score_stability_report.json", report)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = float(time.perf_counter() - started)
        manifest["compact_run_count"] = int(len(runs))
        manifest["score_stability_report_sha256"] = _sha256(
            output_dir / "score_stability_report.json"
        )
        _write_json(manifest_path, manifest)
        print(json.dumps(report["gates"], indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = "{}: {}".format(type(error).__name__, error)
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
