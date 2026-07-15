#!/usr/bin/env python3
"""Run paired dynamic causal validation for target-conditioned blocker orders."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
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

from scene_graph_mem.relations.blocker_ranking import (
    compute_blocker_oracle_quantities,
    greedy_conditional_removal_plan,
    stable_node_id_sort_key,
)
from scene_graph_mem.relations.dynamic_blocker_causal import (
    aggregate_dynamic_causal_target,
    assert_dynamic_calibration_split_guard,
    assert_dynamic_frozen_test_split_guard,
    summarize_dynamic_causal_targets,
    summarize_dynamic_failure_modes,
)
from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.utils.action_conditioned_relation_oracle import (
    CounterfactualRandomizationConfig,
    evaluate_saved_scene,
    execute_forced_attachment_extraction_trial,
)


WORKSPACE = Path("/home/user/ehsanullahm1/thesis")
PROJECT = WORKSPACE / "manipulation_enhanced_map_prediction"
SCENE_GRAPH_PROJECT = WORKSPACE / "scene_graph_mem"
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
DEFAULT_CALIBRATION_GROUPS = ("0", "1", "2", "3", "10", "11", "12", "13")
POLICY_NAMES = (
    "raw_salience",
    "single_removal_gain",
    "shared_path_credit",
    "greedy_conditional_gain",
    "random_positive_control",
    "geometry_positive_control",
)
STAGES = ("approach", "grasp", "extraction")
_WORKER_ENV: Optional[ShelfEnv] = None


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--run-type",
        choices=("plumbing_smoke", "calibration", "frozen_test"),
        required=True,
    )
    parser.add_argument("--scene-groups", nargs="+", default=None)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--perturbation-seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen-predictions-json", type=Path)
    parser.add_argument("--ranking-target-decision", type=Path)
    parser.add_argument("--predeclare-only", action="store_true")
    parser.add_argument("--predeclared-selection-json", type=Path)
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


def _stable_score_order(
    source_ids: Sequence[Any], scores: Mapping[Any, float]
) -> list[Any]:
    return sorted(
        source_ids,
        key=lambda source_id: (
            -float(scores[source_id]),
            stable_node_id_sort_key(source_id),
        ),
    )


def _hashed_order(sample_id: str, target_id: Any, source_ids: Sequence[Any]) -> list[Any]:
    return sorted(
        source_ids,
        key=lambda source_id: (
            hashlib.sha256(
                "{}\0{}\0{}".format(sample_id, target_id, source_id).encode("utf-8")
            ).hexdigest(),
            stable_node_id_sort_key(source_id),
        ),
    )


def _aabb_distance(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    left_lower, left_upper = np.asarray(left[0]), np.asarray(left[1])
    right_lower, right_upper = np.asarray(right[0]), np.asarray(right[1])
    separation = np.maximum(0.0, np.maximum(left_lower - right_upper, right_lower - left_upper))
    return float(np.linalg.norm(separation))


def _candidate_height_bucket(grasp_id: str) -> str:
    for value in ("0.35", "0.55", "0.75"):
        if "z{}".format(value) in str(grasp_id):
            return value
    return "unknown"


def _path_contract(trajectory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "trajectory_id": str(trajectory["trajectory_id"]),
        "eligible_for_scoring": bool(trajectory.get("eligible_for_scoring", False)),
        "weight": float(trajectory.get("weight", 1.0)),
        "blocked_by": [int(value) for value in trajectory.get("blocked_by", [])],
        "blocked_by_stage": {
            stage: [
                int(value)
                for value in dict(trajectory.get("blocked_by_stage") or {}).get(stage, [])
            ]
            for stage in STAGES
        },
    }


def build_dynamic_target_contracts(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_sample_ids: Sequence[str],
    scene_groups: Sequence[str],
) -> list[Dict[str, Any]]:
    """Build model-free target contracts from frozen non-test oracle evidence."""

    allowed = set(str(value) for value in allowed_sample_ids)
    groups = set(str(value) for value in scene_groups)
    contracts: list[Dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if sample_id not in allowed or _scene_group(sample_id) not in groups:
            continue
        scene = _read_json(Path(str(record["action_oracle_evidence_file"])))
        source_ids = [int(value) for value in scene["node_order_instance_ids"]]
        index_by_id = {source_id: index for index, source_id in enumerate(source_ids)}
        object_by_id = {
            int(item["instance_id"]): item for item in scene.get("object_records", [])
        }
        valid_matrix = scene["score_valid_mask"]
        target_by_index = {
            int(item["target_index"]): item for item in scene.get("targets", [])
        }
        for target_index, target_id in enumerate(source_ids):
            target = target_by_index[target_index]
            paths = [_path_contract(item) for item in target.get("trajectories", [])]
            quantities = compute_blocker_oracle_quantities(paths, source_ids=source_ids)
            if not quantities["defined"]:
                continue
            valid_source_ids = [
                source_ids[source_index]
                for source_index in range(len(source_ids))
                if bool(valid_matrix[source_index][target_index])
            ]
            score_maps = {
                name: {
                    source_id: float(quantities[name][index_by_id[source_id]])
                    for source_id in valid_source_ids
                }
                for name in (
                    "raw_blocker_salience",
                    "single_removal_gain",
                    "shared_path_credit",
                )
            }
            positive_source_ids = [
                source_id
                for source_id in valid_source_ids
                if score_maps["raw_blocker_salience"][source_id] > 0.0
            ]
            if not positive_source_ids:
                continue
            raw_order = _stable_score_order(
                positive_source_ids, score_maps["raw_blocker_salience"]
            )
            single_order = _stable_score_order(
                positive_source_ids, score_maps["single_removal_gain"]
            )
            shared_order = _stable_score_order(
                positive_source_ids, score_maps["shared_path_credit"]
            )
            greedy = greedy_conditional_removal_plan(
                paths,
                source_ids=source_ids,
                candidate_source_ids=positive_source_ids,
            )["removed_source_ids"]
            random_order = _hashed_order(sample_id, target_id, positive_source_ids)
            random_all_order = _hashed_order(sample_id, target_id, valid_source_ids)
            target_aabb = object_by_id[target_id]["world_aabb"]
            geometry_all_order = sorted(
                valid_source_ids,
                key=lambda source_id: (
                    _aabb_distance(target_aabb, object_by_id[source_id]["world_aabb"]),
                    stable_node_id_sort_key(source_id),
                ),
            )
            geometry_positive_order = [
                source_id for source_id in geometry_all_order if source_id in positive_source_ids
            ]
            policy_orders = {
                "raw_salience": raw_order,
                "single_removal_gain": single_order,
                "shared_path_credit": shared_order,
                "greedy_conditional_gain": list(greedy),
                "random_positive_control": random_order,
                "geometry_positive_control": geometry_positive_order,
            }
            if any(set(order) != set(positive_source_ids) for order in policy_orders.values()):
                raise AssertionError("every dynamic policy must order every oracle-positive source")

            removed_sets: Dict[tuple[int, ...], set[str]] = {(): {"intact"}}
            for source_id in positive_source_ids:
                removed_sets.setdefault(tuple(sorted([source_id])), set()).add(
                    "every_oracle_positive_singleton"
                )
            for policy_name, order in policy_orders.items():
                for k in range(1, len(order) + 1):
                    removed = tuple(sorted(int(value) for value in order[:k]))
                    removed_sets.setdefault(removed, set()).add(
                        "{}_prefix_at_{}".format(policy_name, k)
                    )
            negative_ids = [
                source_id
                for source_id in valid_source_ids
                if score_maps["raw_blocker_salience"][source_id] == 0.0
            ]
            random_negative = (
                _hashed_order(sample_id + "/negative", target_id, negative_ids)[0]
                if negative_ids
                else None
            )
            if random_negative is not None:
                removed_sets.setdefault((int(random_negative),), set()).add(
                    "random_negative_control"
                )
            geometry_all_top1 = geometry_all_order[0] if geometry_all_order else None
            if geometry_all_top1 is not None:
                removed_sets.setdefault((int(geometry_all_top1),), set()).add(
                    "geometry_all_source_control"
                )
            conditions = [
                {
                    "condition_id": "remove_none_intact" if not removed else "remove_{}".format(
                        "_".join(str(value) for value in removed)
                    ),
                    "removed_source_ids": list(removed),
                    "roles": sorted(roles),
                }
                for removed, roles in sorted(
                    removed_sets.items(), key=lambda item: (len(item[0]), item[0])
                )
            ]
            eligible_candidates = [
                str(path["trajectory_id"])
                for path in paths
                if bool(path["eligible_for_scoring"])
            ]
            if not eligible_candidates:
                continue
            raw_max = max(score_maps["raw_blocker_salience"].values())
            distinct_positive_raw_levels = len(
                {
                    round(float(score_maps["raw_blocker_salience"][source_id]), 12)
                    for source_id in positive_source_ids
                }
            )
            tied_max = sum(
                math.isclose(score_maps["raw_blocker_salience"][source_id], raw_max)
                for source_id in positive_source_ids
            ) > 1
            single_positive = max(score_maps["single_removal_gain"].values()) > 0.0
            stage_mass = {
                stage: float(
                    sum(
                        quantities["stage_quantities"][stage]["raw_blocker_salience"]
                        [index_by_id[source_id]]
                        for source_id in positive_source_ids
                    )
                )
                for stage in STAGES
            }
            dominant_stage = min(
                STAGES, key=lambda stage: (-stage_mass[stage], STAGES.index(stage))
            )
            height_counts = Counter(
                _candidate_height_bucket(path["trajectory_id"])
                for path in paths
                if path["eligible_for_scoring"] and path["blocked_by"]
            )
            dominant_height = min(
                height_counts or {"unknown": 0},
                key=lambda height: (-height_counts[height], height),
            )
            contracts.append(
                {
                    "sample_id": sample_id,
                    "scene_group": _scene_group(sample_id),
                    "pre_action_dir": str(record["sample_dir"]),
                    "target_instance_id": int(target_id),
                    "node_order_instance_ids": source_ids,
                    "eligible_candidate_ids": sorted(eligible_candidates),
                    "positive_source_ids": positive_source_ids,
                    "valid_source_ids": valid_source_ids,
                    "random_all_source_order": random_all_order,
                    "geometry_all_source_order": geometry_all_order,
                    "policy_orders": policy_orders,
                    "policy_static_scores": {
                        "raw_salience": score_maps["raw_blocker_salience"],
                        "single_removal_gain": score_maps["single_removal_gain"],
                        "shared_path_credit": score_maps["shared_path_credit"],
                    },
                    "conditions": conditions,
                    "random_negative_control_source_id": random_negative,
                    "geometry_all_source_control_source_id": geometry_all_top1,
                    "static_structure": (
                        "single_removal_positive" if single_positive else "cooperative_zero_single_gain"
                    ),
                    "raw_maximum_tied": bool(tied_max),
                    "distinct_positive_raw_level_count": int(
                        distinct_positive_raw_levels
                    ),
                    "rank_correlation_eligible_structure": bool(
                        len(positive_source_ids) >= 2
                        and distinct_positive_raw_levels >= 2
                    ),
                    "dominant_blocker_stage": dominant_stage,
                    "dominant_candidate_height": dominant_height,
                    "stratum": "{}__{}__{}__z{}".format(
                        "single" if single_positive else "cooperative",
                        "tied" if tied_max else "untied",
                        dominant_stage,
                        dominant_height,
                    ),
                }
            )
    return contracts


def select_stratified_dynamic_targets(
    contracts: Sequence[Mapping[str, Any]],
    *,
    scene_groups: Sequence[str],
    target_count: int,
) -> list[Dict[str, Any]]:
    """Select distinct saved scenes while balancing causal/tie/stage/height strata."""

    count = int(target_count)
    groups = [str(value) for value in scene_groups]
    if count <= 0 or count < len(groups):
        raise ValueError("target_count must be positive and cover every scene group")
    base = count // len(groups)
    remainder = count % len(groups)
    selected: list[Dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        quota = base + int(group_index < remainder)
        options = sorted(
            [dict(item) for item in contracts if str(item["scene_group"]) == group],
            key=lambda item: (
                str(item["sample_id"]),
                stable_node_id_sort_key(item["target_instance_id"]),
            ),
        )
        chosen: list[Dict[str, Any]] = []
        used_samples = set()
        used_structures = Counter()
        used_structure_rankability = Counter()
        used_ties = Counter()
        used_stages = Counter()
        used_heights = Counter()
        while len(chosen) < quota:
            candidates = [item for item in options if item["sample_id"] not in used_samples]
            if not candidates:
                raise ValueError("scene group {} lacks {} distinct target scenes".format(group, quota))
            best = min(
                candidates,
                key=lambda item: (
                    used_structures[item["static_structure"]],
                    used_structure_rankability[
                        (
                            item["static_structure"],
                            bool(item["rank_correlation_eligible_structure"]),
                        )
                    ],
                    used_ties[bool(item["raw_maximum_tied"])],
                    used_stages[item["dominant_blocker_stage"]],
                    used_heights[item["dominant_candidate_height"]],
                    len(item["positive_source_ids"]),
                    str(item["sample_id"]),
                    stable_node_id_sort_key(item["target_instance_id"]),
                ),
            )
            chosen.append(best)
            used_samples.add(best["sample_id"])
            used_structures[best["static_structure"]] += 1
            used_structure_rankability[
                (
                    best["static_structure"],
                    bool(best["rank_correlation_eligible_structure"]),
                )
            ] += 1
            used_ties[bool(best["raw_maximum_tied"])] += 1
            used_stages[best["dominant_blocker_stage"]] += 1
            used_heights[best["dominant_candidate_height"]] += 1
        selected.extend(chosen)
    if len(selected) != count or len({item["sample_id"] for item in selected}) != count:
        raise AssertionError("dynamic selection must contain exactly one target per saved scene")
    return selected


def build_frozen_test_policy_contract(
    contract: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    """Join a no-GT model prediction to offline executable test interventions."""

    if (
        str(contract["sample_id"]) != str(prediction["sample_id"])
        or int(contract["target_instance_id"])
        != int(prediction["matched_gt_target_instance_id"])
    ):
        raise ValueError("frozen prediction and oracle target keys do not align")
    valid_sources = [int(value) for value in contract["valid_source_ids"]]
    positive_sources = [int(value) for value in contract["positive_source_ids"]]
    horizon = len(positive_sources)
    if horizon <= 0:
        raise ValueError("frozen test target must have an oracle-positive source")

    def executable_steps(
        entries: Sequence[Mapping[str, Any]],
        *,
        policy_name: str,
    ) -> list[Dict[str, Any]]:
        padded_entries = [dict(entry) for entry in entries]
        padded_entries.extend({} for _ in range(max(0, horizon - len(padded_entries))))
        removed: set[int] = set()
        steps = []
        for step_index, entry in enumerate(padded_entries[:horizon], start=1):
            source = entry.get("matched_gt_source_instance_id")
            executable = source is not None and int(source) in set(valid_sources)
            if executable:
                removed.add(int(source))
            steps.append(
                {
                    "step": step_index,
                    "learned_source_node_id": entry.get("learned_source_node_id"),
                    "intervention_source_id": int(source) if executable else None,
                    "intervention_executable": bool(executable),
                    "unavailable_reason": (
                        None
                        if executable
                        else (
                            "unmatched_learned_source_node"
                            if entry
                            else "no_learned_source_at_horizon_step"
                        )
                    ),
                    "removed_source_ids_after_step": sorted(removed),
                }
            )
        return steps

    def oracle_steps(order: Sequence[Any]) -> list[Dict[str, Any]]:
        if len(order) < horizon:
            raise ValueError("oracle/control order lacks the intervention horizon")
        removed: set[int] = set()
        steps = []
        for step_index, source in enumerate(order[:horizon], start=1):
            source_id = int(source)
            if source_id not in set(valid_sources):
                raise ValueError("oracle/control policy names an invalid source")
            removed.add(source_id)
            steps.append(
                {
                    "step": step_index,
                    "learned_source_node_id": None,
                    "intervention_source_id": source_id,
                    "intervention_executable": True,
                    "unavailable_reason": None,
                    "removed_source_ids_after_step": sorted(removed),
                }
            )
        return steps

    policies = {
        "accepted_absolute_probability": executable_steps(
            prediction["accepted_absolute_probability_order"],
            policy_name="accepted_absolute_probability",
        ),
        "accepted_conditional_union_gain": executable_steps(
            prediction["accepted_conditional_union_gain_order"],
            policy_name="accepted_conditional_union_gain",
        ),
        "raw_salience": oracle_steps(contract["policy_orders"]["raw_salience"]),
        "oracle_static_single_removal_gain": oracle_steps(
            contract["policy_orders"]["single_removal_gain"]
        ),
        "oracle_greedy_conditional_gain": oracle_steps(
            contract["policy_orders"]["greedy_conditional_gain"]
        ),
        "random_visible_source": executable_steps(
            prediction["random_visible_source_order"],
            policy_name="random_visible_source",
        ),
        "deterministic_geometry": executable_steps(
            prediction["deterministic_learned_geometry_order"],
            policy_name="deterministic_geometry",
        ),
    }
    removed_roles: Dict[tuple[int, ...], set[str]] = {(): {"intact"}}
    for source_id in positive_sources:
        removed_roles.setdefault((int(source_id),), set()).add(
            "every_oracle_positive_singleton"
        )
    for policy_name in (
        "raw_salience",
        "single_removal_gain",
        "greedy_conditional_gain",
    ):
        order = [int(value) for value in contract["policy_orders"][policy_name]]
        for k in range(1, horizon + 1):
            key = tuple(sorted(order[:k]))
            removed_roles.setdefault(key, set()).add(
                "oracle_{}_prefix_at_{}".format(policy_name, k)
            )
    for policy_name, steps in policies.items():
        for step in steps:
            key = tuple(int(value) for value in step["removed_source_ids_after_step"])
            removed_roles.setdefault(key, set()).add(
                "frozen_{}_prefix_at_{}".format(policy_name, step["step"])
            )
    conditions = [
        {
            "condition_id": (
                "remove_none_intact"
                if not removed
                else "remove_{}".format("_".join(str(value) for value in removed))
            ),
            "removed_source_ids": list(removed),
            "roles": sorted(roles),
        }
        for removed, roles in sorted(
            removed_roles.items(), key=lambda item: (len(item[0]), item[0])
        )
    ]
    result = dict(contract)
    result["conditions"] = conditions
    result["frozen_test_intervention_horizon"] = int(horizon)
    result["frozen_test_policies"] = policies
    result["frozen_oracle_policy_orders"] = {
        policy_name: list(contract["policy_orders"][policy_name])
        for policy_name in (
            "raw_salience",
            "single_removal_gain",
            "greedy_conditional_gain",
        )
    }
    result["prediction_alignment"] = {
        "learned_target_node_id": int(prediction["learned_target_node_id"]),
        "rank_query_valid": bool(prediction["rank_query_valid"]),
        "target_blockage_defined": bool(prediction["target_blockage_defined"]),
        "explainability_fraction": prediction.get("explainability_fraction"),
        "matched_executable_source_count": int(
            prediction["matched_executable_source_count"]
        ),
        "unmatched_learned_source_count": int(
            prediction["unmatched_learned_source_count"]
        ),
        "non_oracle_baseline_inputs": prediction["non_oracle_baseline_inputs"],
        "non_oracle_geometry_candidate_mask_source": prediction[
            "non_oracle_geometry_candidate_mask_source"
        ],
        "gt_alignment_is_offline_intervention_harness_only": True,
        "model_runtime_inputs_use_gt_or_oracle": False,
    }
    return result


def aggregate_frozen_test_policy_results(
    dynamic_result: Mapping[str, Any],
    policy_contracts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate fixed-horizon policies, retaining unmatched model steps as no-ops."""

    condition_by_removed = {
        tuple(sorted(int(value) for value in summary["removed_source_ids"])): summary
        for summary in dict(dynamic_result["condition_summaries"]).values()
    }
    intact = float(dynamic_result["intact_dynamic_accessibility"])
    best_singleton = dynamic_result.get("best_dynamic_singleton_access_gain")
    results: Dict[str, Any] = {}
    for policy_name, steps in policy_contracts.items():
        curve = [intact]
        executable = []
        for step in steps:
            key = tuple(
                sorted(int(value) for value in step["removed_source_ids_after_step"])
            )
            if key not in condition_by_removed:
                raise ValueError("frozen policy prefix lacks a dynamic condition")
            curve.append(float(condition_by_removed[key]["dynamic_accessibility"]))
            executable.append(bool(step["intervention_executable"]))
        gains = [float(value - intact) for value in curve]
        top1_gain = gains[1] if len(gains) > 1 else None
        top1_regret = (
            float(best_singleton - top1_gain)
            if best_singleton is not None and top1_gain is not None
            else None
        )
        first_new = next(
            (index for index, value in enumerate(curve[1:], start=1) if value > intact),
            None,
        )
        auc = (
            float(np.trapezoid(np.asarray(curve), dx=1.0) / (len(curve) - 1))
            if len(curve) > 1
            else intact
        )
        results[str(policy_name)] = {
            "intervention_horizon": int(len(steps)),
            "step_source_ids": [step["intervention_source_id"] for step in steps],
            "step_executable": executable,
            "unmatched_noop_step_count": int(sum(not value for value in executable)),
            "dynamic_accessibility_at_k": curve,
            "dynamic_accessibility_gain_at_k": gains,
            "top1_dynamic_access_gain": top1_gain,
            "top1_regret": top1_regret,
            "removals_to_first_new_access": first_new,
            "area_under_accessibility_curve": auc,
            "final_dynamic_accessibility": float(curve[-1]),
        }
    if best_singleton is not None:
        singleton_gains = {
            str(source_id): float(value)
            for source_id, value in dict(
                dynamic_result.get("dynamic_singleton_access_gain") or {}
            ).items()
            if value is not None
        }
        best_sources = sorted(
            source_id
            for source_id, value in singleton_gains.items()
            if math.isclose(value, float(best_singleton), abs_tol=1.0e-12)
        )
        oracle_final = float(intact + float(best_singleton))
        results["oracle_dynamic_best_single"] = {
            "intervention_horizon": 1,
            "step_source_ids": best_sources[:1],
            "best_source_ids_including_ties": best_sources,
            "step_executable": [True],
            "unmatched_noop_step_count": 0,
            "dynamic_accessibility_at_k": [intact, oracle_final],
            "dynamic_accessibility_gain_at_k": [0.0, float(best_singleton)],
            "top1_dynamic_access_gain": float(best_singleton),
            "top1_regret": 0.0,
            "removals_to_first_new_access": (
                1 if float(best_singleton) > 0.0 else None
            ),
            "area_under_accessibility_curve": float(
                np.trapezoid(np.asarray([intact, oracle_final]), dx=1.0)
            ),
            "final_dynamic_accessibility": oracle_final,
            "uses_dynamic_outcomes_as_oracle_upper_bound": True,
        }
    oracle_curve = results["oracle_greedy_conditional_gain"][
        "dynamic_accessibility_at_k"
    ]
    for result in results.values():
        result["sequence_regret_at_k"] = [
            float(oracle - observed)
            for oracle, observed in zip(
                oracle_curve, result["dynamic_accessibility_at_k"]
            )
        ]
    return results


def _worker_init() -> None:
    global _WORKER_ENV
    _WORKER_ENV = ShelfEnv(render=False, max_obj_num=25, use_ycb=True)
    atexit.register(_WORKER_ENV.close)


def _failure_causes(trial: Mapping[str, Any], clean_success: bool) -> list[str]:
    if clean_success:
        return []
    causes = set()
    joint_errors = dict(trial.get("stage_max_joint_error_rad") or {})
    for stage in STAGES:
        if float(joint_errors.get(stage, float("inf"))) > 0.08:
            causes.add("{}_joint_tracking".format(stage))
    for stage, names in dict(trial.get("fixed_hard_penetrations_by_stage") or {}).items():
        if names:
            causes.add("{}_fixed_environment_penetration".format(stage))
    for stage, ids in dict(trial.get("removed_blocker_hard_penetrations_by_stage") or {}).items():
        if ids:
            causes.add("{}_dynamic_object_penetration".format(stage))
    if not bool(dict(trial.get("extraction_progress") or {}).get("target_extracted", False)):
        causes.add("insufficient_extraction_progress")
    if not bool(
        dict(trial.get("monitored_displacement") or {}).get("monitored_objects_stable", False)
    ):
        causes.add("monitored_object_displacement")
    if not causes:
        causes.add("forced_attachment_protocol_failure")
    return sorted(causes)


def _run_target_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    if _WORKER_ENV is None:
        raise RuntimeError("dynamic worker environment was not initialized")
    replayed, debug = evaluate_saved_scene(
        _WORKER_ENV, pre_action_dir=Path(str(task["pre_action_dir"]))
    )
    if str(replayed["sample_id"]) != str(task["sample_id"]):
        raise RuntimeError("dynamic worker replayed the wrong scene")
    if [int(value) for value in replayed["node_order_instance_ids"]] != [
        int(value) for value in task["node_order_instance_ids"]
    ]:
        raise RuntimeError("dynamic replay node identity differs from frozen selection")
    target_id = int(task["target_instance_id"])
    trials = []
    started = time.perf_counter()
    for condition in task["active_conditions"]:
        removed = [int(value) for value in condition["removed_source_ids"]]
        monitored = [
            int(value)
            for value in task["node_order_instance_ids"]
            if int(value) != target_id and int(value) not in set(removed)
        ]
        for candidate_id in task["active_candidate_ids"]:
            if candidate_id not in debug:
                raise RuntimeError("selected candidate lacks replay debug trajectory")
            for seed in task["active_seeds"]:
                trial = execute_forced_attachment_extraction_trial(
                    _WORKER_ENV,
                    pre_action_dir=Path(str(task["pre_action_dir"])),
                    target_instance_id=target_id,
                    removed_instance_ids=removed,
                    monitored_instance_ids=monitored,
                    candidate_debug=debug[candidate_id],
                    randomization=CounterfactualRandomizationConfig(seed=int(seed)),
                    hard_penetration_m=0.002,
                )
                dynamic_hard = any(
                    bool(values)
                    for values in dict(
                        trial.get("removed_blocker_hard_penetrations_by_stage") or {}
                    ).values()
                )
                clean_success = bool(trial["success"] and not dynamic_hard)
                trials.append(
                    {
                        "run_id": "{}__target_{}__{}__{}__seed_{}".format(
                            str(task["sample_id"]).replace("/", "__"),
                            target_id,
                            condition["condition_id"],
                            candidate_id.replace("/", "__"),
                            int(seed),
                        ),
                        "condition_id": str(condition["condition_id"]),
                        "removed_source_ids": removed,
                        "candidate_id": str(candidate_id),
                        "seed": int(seed),
                        "protocol_success": bool(trial["success"]),
                        "clean_extraction_success": clean_success,
                        "failure_causes": _failure_causes(trial, clean_success),
                        "stage_max_joint_error_rad": {
                            key: (float(value) if np.isfinite(value) else None)
                            for key, value in dict(
                                trial.get("stage_max_joint_error_rad") or {}
                            ).items()
                        },
                        "fixed_hard_penetrations_by_stage": dict(
                            trial.get("fixed_hard_penetrations_by_stage") or {}
                        ),
                        "dynamic_object_hard_penetrations_by_stage": dict(
                            trial.get("removed_blocker_hard_penetrations_by_stage") or {}
                        ),
                        "extraction_progress_fraction": dict(
                            trial.get("extraction_progress") or {}
                        ).get("progress_fraction"),
                        "monitored_objects_stable": bool(
                            dict(trial.get("monitored_displacement") or {}).get(
                                "monitored_objects_stable", False
                            )
                        ),
                        "forced_attachment": True,
                    }
                )
    return {
        "sample_id": str(task["sample_id"]),
        "scene_group": str(task["scene_group"]),
        "target_instance_id": target_id,
        "stratum": str(task["stratum"]),
        "static_structure": str(task["static_structure"]),
        "trials": trials,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _policy_admissibility(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    single_rows = [row for row in rows if row["static_structure"] == "single_removal_positive"]
    cooperative_rows = [
        row for row in rows if row["static_structure"] == "cooperative_zero_single_gain"
    ]

    def assess(
        selected_rows: Sequence[Mapping[str, Any]],
        policy: str,
        metric: str,
    ) -> Dict[str, Any]:
        grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected_rows:
            grouped[str(row["scene_group"])].append(row)

        def mean(rows_for_mean: Sequence[Mapping[str, Any]], name: str) -> Optional[float]:
            values = [
                row["dynamic_result"]["policies"][name].get(metric)
                for row in rows_for_mean
            ]
            finite = [float(value) for value in values if value is not None]
            return float(np.mean(finite)) if finite else None

        group_results = {}
        wins = 0
        for group, group_rows in sorted(grouped.items()):
            policy_value = mean(group_rows, policy)
            random_value = mean(group_rows, "random_positive_control")
            geometry_value = mean(group_rows, "geometry_positive_control")
            beats = bool(
                policy_value is not None
                and random_value is not None
                and geometry_value is not None
                and policy_value > random_value
                and policy_value > geometry_value
            )
            wins += int(beats)
            group_results[group] = {
                "target_query_count": int(len(group_rows)),
                "policy": policy_value,
                "random_positive_control": random_value,
                "geometry_positive_control": geometry_value,
                "beats_both_controls": beats,
            }
        required = max(3, int(math.ceil(0.75 * len(group_results)))) if group_results else 0
        overall_policy = mean(selected_rows, policy)
        overall_random = mean(selected_rows, "random_positive_control")
        overall_geometry = mean(selected_rows, "geometry_positive_control")
        overall_beats = bool(
            overall_policy is not None
            and overall_random is not None
            and overall_geometry is not None
            and overall_policy > overall_random
            and overall_policy > overall_geometry
        )
        return {
            "intended_metric": metric,
            "target_query_count": int(len(selected_rows)),
            "overall": {
                "policy": overall_policy,
                "random_positive_control": overall_random,
                "geometry_positive_control": overall_geometry,
                "beats_both_controls": overall_beats,
            },
            "scene_groups": group_results,
            "group_win_count": int(wins),
            "required_group_win_count": int(required),
            "admissible": bool(overall_beats and wins >= required),
        }

    return {
        "single_removal_queries": {
            policy: assess(single_rows, policy, "top1_dynamic_access_gain")
            for policy in (
                "raw_salience",
                "single_removal_gain",
                "shared_path_credit",
            )
        },
        "cooperative_queries": {
            policy: assess(cooperative_rows, policy, "area_under_accessibility_curve")
            for policy in (
                "raw_salience",
                "single_removal_gain",
                "shared_path_credit",
                "greedy_conditional_gain",
            )
        },
        "small_scene_group_count_warning": (
            "group comparisons are descriptive; no population-significance claim is made"
        ),
    }


def build_dynamic_causal_report(
    selections: Sequence[Mapping[str, Any]],
    target_outputs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    selection_by_key = {
        (str(item["sample_id"]), int(item["target_instance_id"])): item
        for item in selections
    }
    output_by_key = {
        (str(item["sample_id"]), int(item["target_instance_id"])): item
        for item in target_outputs
    }
    if set(selection_by_key) != set(output_by_key):
        raise ValueError("dynamic outputs do not exactly match the predeclared targets")
    rows = []
    for key in sorted(selection_by_key):
        selection = selection_by_key[key]
        output = output_by_key[key]
        result = aggregate_dynamic_causal_target(
            output["trials"],
            condition_contracts=selection["conditions"],
            policy_orders=selection["policy_orders"],
            policy_static_scores=selection["policy_static_scores"],
            positive_source_ids=selection["positive_source_ids"],
            expected_candidate_ids=selection["eligible_candidate_ids"],
            expected_seeds=seeds,
        )
        rows.append(
            {
                "sample_id": key[0],
                "target_instance_id": key[1],
                "scene_group": str(selection["scene_group"]),
                "stratum": str(selection["stratum"]),
                "static_structure": str(selection["static_structure"]),
                "dynamic_result": result,
            }
        )
    return {
        "schema": "ranked_blocker_dynamic_causal_report_v1",
        "forced_attachment": True,
        "limitation": "validates access/extraction after forced attachment, not autonomous grasp closure",
        "dynamic_accessibility_definition": (
            "mean clean-extraction success over the fixed eligible candidate family and paired seeds"
        ),
        "target_results": rows,
        "summary": summarize_dynamic_causal_targets(rows),
        "policy_admissibility": _policy_admissibility(rows),
        "test_outcomes_used_for_target_or_route_selection": False,
    }


def build_frozen_test_causal_report(
    selections: Sequence[Mapping[str, Any]],
    target_outputs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    ranking_target_decision: Mapping[str, Any],
    frozen_predictions_identity: Mapping[str, Any],
    frozen_test_coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build held-out paired evidence without using it to redesign the route."""

    selection_by_key = {
        (str(item["sample_id"]), int(item["target_instance_id"])): item
        for item in selections
    }
    output_by_key = {
        (str(item["sample_id"]), int(item["target_instance_id"])): item
        for item in target_outputs
    }
    if set(selection_by_key) != set(output_by_key):
        raise ValueError("frozen outputs do not exactly match predeclared targets")
    base_rows = []
    for key in sorted(selection_by_key):
        selection = selection_by_key[key]
        output = output_by_key[key]
        result = aggregate_dynamic_causal_target(
            output["trials"],
            condition_contracts=selection["conditions"],
            policy_orders=selection["frozen_oracle_policy_orders"],
            policy_static_scores=selection["policy_static_scores"],
            positive_source_ids=selection["positive_source_ids"],
            expected_candidate_ids=selection["eligible_candidate_ids"],
            expected_seeds=seeds,
        )
        base_rows.append(
            {
                "sample_id": key[0],
                "target_instance_id": key[1],
                "scene_group": str(selection["scene_group"]),
                "stratum": str(selection["stratum"]),
                "static_structure": str(selection["static_structure"]),
                "dynamic_result": result,
            }
        )
    base = {
        "limitation": (
            "validates access/extraction after forced attachment, not autonomous grasp closure"
        ),
        "dynamic_accessibility_definition": (
            "mean clean-extraction success over the fixed eligible candidate family and paired seeds"
        ),
        "target_results": base_rows,
    }
    rows = []
    for row in base["target_results"]:
        key = (str(row["sample_id"]), int(row["target_instance_id"]))
        selection = selection_by_key[key]
        enriched = dict(row)
        enriched["frozen_test_intervention_horizon"] = int(
            selection["frozen_test_intervention_horizon"]
        )
        enriched["prediction_alignment"] = dict(selection["prediction_alignment"])
        enriched["frozen_policy_results"] = aggregate_frozen_test_policy_results(
            row["dynamic_result"], selection["frozen_test_policies"]
        )
        rows.append(enriched)

    policy_names = sorted(
        {
            name for row in rows for name in row["frozen_policy_results"]
        }
    )

    def summarize_rows(selected_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "target_query_count": int(len(selected_rows)),
            "policy_metrics": {},
        }
        for policy in policy_names:
            policy_values = [row["frozen_policy_results"][policy] for row in selected_rows]
            output["policy_metrics"][policy] = {}
            for metric in (
                "top1_dynamic_access_gain",
                "top1_regret",
                "removals_to_first_new_access",
                "area_under_accessibility_curve",
                "final_dynamic_accessibility",
                "unmatched_noop_step_count",
            ):
                values = [
                    float(value[metric])
                    for value in policy_values
                    if value.get(metric) is not None
                ]
                output["policy_metrics"][policy][metric] = {
                    "count": int(len(values)),
                    "mean": float(np.mean(values)) if values else None,
                    "minimum": float(np.min(values)) if values else None,
                    "maximum": float(np.max(values)) if values else None,
                }
            maximum_k = max(
                (len(value["dynamic_accessibility_at_k"]) for value in policy_values),
                default=0,
            )
            output["policy_metrics"][policy]["dynamic_accessibility_at_k"] = {
                str(k): {
                    "count": int(
                        sum(len(value["dynamic_accessibility_at_k"]) > k for value in policy_values)
                    ),
                    "mean": (
                        float(
                            np.mean(
                                [
                                    value["dynamic_accessibility_at_k"][k]
                                    for value in policy_values
                                    if len(value["dynamic_accessibility_at_k"]) > k
                                ]
                            )
                        )
                        if any(
                            len(value["dynamic_accessibility_at_k"]) > k
                            for value in policy_values
                        )
                        else None
                    ),
                }
                for k in range(maximum_k)
            }
        return output

    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    structured: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_group"])].append(row)
        structured[str(row["static_structure"])].append(row)

    def comparison(
        selected_rows: Sequence[Mapping[str, Any]],
        *,
        policy: str,
        baselines: Sequence[str],
        metric: str,
    ) -> Dict[str, Any]:
        def mean(name: str) -> Optional[float]:
            values = [
                row["frozen_policy_results"][name].get(metric)
                for row in selected_rows
            ]
            finite = [float(value) for value in values if value is not None]
            return float(np.mean(finite)) if finite else None

        selected_value = mean(policy)
        baseline_values = {name: mean(name) for name in baselines}
        return {
            "metric": metric,
            "target_query_count": int(len(selected_rows)),
            "selected_policy": policy,
            "selected_value": selected_value,
            "baselines": baseline_values,
            "selected_minus_baseline": {
                name: (
                    float(selected_value - value)
                    if selected_value is not None and value is not None
                    else None
                )
                for name, value in baseline_values.items()
            },
        }

    single_rows = structured.get("single_removal_positive", [])
    cooperative_rows = structured.get("cooperative_zero_single_gain", [])
    single_baselines = (
        "random_visible_source",
        "deterministic_geometry",
        "raw_salience",
        "oracle_static_single_removal_gain",
        "oracle_dynamic_best_single",
    )
    cooperative_baselines = (
        "random_visible_source",
        "deterministic_geometry",
        "accepted_absolute_probability",
        "raw_salience",
        "oracle_greedy_conditional_gain",
    )
    structure_counts = Counter(str(row["static_structure"]) for row in rows)
    stratum_counts = Counter(str(row["stratum"]) for row in rows)
    selected_group_counts = Counter(str(row["scene_group"]) for row in rows)
    report = {
        "schema": "ranked_blocker_frozen_test_causal_report_v1",
        "forced_attachment": True,
        "limitation": base["limitation"],
        "dynamic_accessibility_definition": base["dynamic_accessibility_definition"],
        "ranking_target_decision": dict(ranking_target_decision),
        "frozen_predictions": dict(frozen_predictions_identity),
        "query_coverage": {
            **dict(frozen_test_coverage),
            "executed_target_query_count": int(len(rows)),
            "selected_structure_counts": dict(sorted(structure_counts.items())),
            "selected_stratum_counts": dict(sorted(stratum_counts.items())),
            "selected_scene_group_counts": dict(
                sorted(selected_group_counts.items())
            ),
            "executed_undefined_target_query_count": int(
                sum(
                    not bool(row["prediction_alignment"]["target_blockage_defined"])
                    for row in rows
                )
            ),
            "single_removal_positive_query_count": int(
                structure_counts.get("single_removal_positive", 0)
            ),
            "cooperative_zero_single_gain_query_count": int(
                structure_counts.get("cooperative_zero_single_gain", 0)
            ),
        },
        "selection_and_policies_frozen_before_dynamic_outcomes": True,
        "test_outcomes_used_for_target_or_route_selection": False,
        "accepted_model_policy_semantics": (
            "accepted absolute graph probability; no listwise treatment was substituted"
        ),
        "non_oracle_baseline_semantics": {
            "random_visible_source": (
                "deterministic hash order over every learned CNABU source; unmatched "
                "sources remain fixed-horizon no-op interventions"
            ),
            "deterministic_geometry": (
                "descending planner-swept overlap, then clearance proximity, then "
                "2D box distance using learned CNABU runtime nodes only"
            ),
        },
        "oracle_baseline_semantics": {
            "oracle_dynamic_best_single": (
                "post-outcome upper bound over every predeclared positive singleton; "
                "used only for regret/comparison, never policy selection"
            ),
            "oracle_greedy_conditional_gain": (
                "predeclared greedy order from frozen oracle blocker sets"
            ),
        },
        "target_results": rows,
        "summary": {
            "all_test_targets": summarize_rows(rows),
            "scene_groups": {
                group: summarize_rows(group_rows)
                for group, group_rows in sorted(grouped.items())
            },
            "static_structures": {
                structure: summarize_rows(structure_rows)
                for structure, structure_rows in sorted(structured.items())
            },
            "failure_modes": summarize_dynamic_failure_modes(rows),
            "failure_modes_by_scene_group": {
                group: summarize_dynamic_failure_modes(group_rows)
                for group, group_rows in sorted(grouped.items())
            },
            "independent_evaluation_unit": "scene_group",
        },
        "primary_comparisons": {
            "single_removal_top1": comparison(
                single_rows,
                policy="accepted_absolute_probability",
                baselines=single_baselines,
                metric="top1_dynamic_access_gain",
            ),
            "cooperative_sequence_auc": comparison(
                cooperative_rows,
                policy="accepted_conditional_union_gain",
                baselines=cooperative_baselines,
                metric="area_under_accessibility_curve",
            ),
            "by_scene_group": {
                group: {
                    "single_removal_top1": comparison(
                        [
                            row
                            for row in group_rows
                            if row["static_structure"] == "single_removal_positive"
                        ],
                        policy="accepted_absolute_probability",
                        baselines=single_baselines,
                        metric="top1_dynamic_access_gain",
                    ),
                    "cooperative_sequence_auc": comparison(
                        [
                            row
                            for row in group_rows
                            if row["static_structure"]
                            == "cooperative_zero_single_gain"
                        ],
                        policy="accepted_conditional_union_gain",
                        baselines=cooperative_baselines,
                        metric="area_under_accessibility_curve",
                    ),
                }
                for group, group_rows in sorted(grouped.items())
            },
        },
        "bounded_inference_statement": (
            "Results are paired descriptive evidence on three held-out scene groups, "
            "not a population-significance claim."
        ),
    }
    single_comparison = report["primary_comparisons"]["single_removal_top1"]
    cooperative_comparison = report["primary_comparisons"][
        "cooperative_sequence_auc"
    ]

    def positive_against(comparison_row: Mapping[str, Any], names: Sequence[str]) -> bool:
        deltas = comparison_row["selected_minus_baseline"]
        return bool(
            int(comparison_row["target_query_count"]) > 0
            and all(
                deltas.get(name) is not None and float(deltas[name]) > 0.0
                for name in names
            )
        )

    single_gate = positive_against(
        single_comparison, ("random_visible_source", "deterministic_geometry")
    )
    cooperative_gate = positive_against(
        cooperative_comparison,
        (
            "random_visible_source",
            "deterministic_geometry",
            "accepted_absolute_probability",
        ),
    )
    report["thesis_level_causal_acceptance"] = {
        "single_removal_gate_passed": single_gate,
        "single_removal_gate": (
            "accepted absolute ranking has strictly higher aggregate top-1 dynamic "
            "access gain than learned-node random and learned-node geometry"
        ),
        "cooperative_sequence_gate_passed": cooperative_gate,
        "cooperative_sequence_gate": (
            "accepted conditional set ordering has strictly higher aggregate access-curve "
            "AUC than learned-node random, learned-node geometry, and static absolute order"
        ),
        "both_applicable_strata_present": bool(single_rows and cooperative_rows),
        "ranked_graph_is_supported_as_manipulation_policy": bool(
            single_gate and cooperative_gate
        ),
        "gate_is_descriptive_not_population_significance": True,
    }
    return report


def _manifest(
    args: argparse.Namespace,
    selections: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    git = lambda *parts: subprocess.check_output(
        ["git", "-C", str(PROJECT), *parts], text=True
    ).strip()
    full_protocol = args.run_type in ("calibration", "frozen_test")
    manifest = {
        "schema": "ranked_blocker_dynamic_causal_manifest_v1",
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_executable": sys.executable,
        "run_type": str(args.run_type),
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
        "source_identities": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "action_oracle": {
                "path": str(
                    (
                        PROJECT
                        / "shelf_gym/utils/action_conditioned_relation_oracle.py"
                    ).resolve()
                ),
                "sha256": _sha256(
                    PROJECT
                    / "shelf_gym/utils/action_conditioned_relation_oracle.py"
                ),
            },
            "dynamic_aggregation": {
                "path": str(
                    (
                        SCENE_GRAPH_PROJECT
                        / "src/scene_graph_mem/relations/dynamic_blocker_causal.py"
                    ).resolve()
                ),
                "sha256": _sha256(
                    SCENE_GRAPH_PROJECT
                    / "src/scene_graph_mem/relations/dynamic_blocker_causal.py"
                ),
            },
        },
        "selected_target_count": int(len(selections)),
        "selected_scene_groups": sorted({str(item["scene_group"]) for item in selections}),
        "perturbation_seeds": [int(value) for value in seeds],
        "workers": int(args.workers),
        "protocol": {
            "candidate_family": "all frozen eligible 3x3 candidates" if full_protocol else "first frozen eligible candidate only",
            "conditions": "all predeclared policy prefixes, every positive singleton, and controls" if full_protocol else "intact and raw-salience top1 only",
            "dynamic_friction_randomized": True,
            "xy_position_jitter_m": 0.001,
            "yaw_jitter_degrees": 0.5,
            "hard_penetration_m": 0.002,
            "forced_attachment": True,
        },
        "policy": {
            "runs_training": False,
            "writes_checkpoint": False,
            "writes_dataset": False,
            "uses_test_groups": bool(args.run_type == "frozen_test"),
            "test_outcomes_select_target": False,
            "compact_trial_records_only": True,
        },
    }
    if args.frozen_predictions_json is not None:
        manifest["frozen_predictions_json"] = {
            "path": str(args.frozen_predictions_json.resolve()),
            "sha256": _sha256(args.frozen_predictions_json),
        }
    if args.ranking_target_decision is not None:
        manifest["ranking_target_decision"] = {
            "path": str(args.ranking_target_decision.resolve()),
            "sha256": _sha256(args.ranking_target_decision),
        }
    if args.predeclared_selection_json is not None:
        manifest["predeclared_selection_json"] = {
            "path": str(args.predeclared_selection_json.resolve()),
            "sha256": _sha256(args.predeclared_selection_json),
        }
    return manifest


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite existing output directory: {}".format(args.output_dir))
    if int(args.workers) <= 0:
        raise SystemExit("--workers must be positive")
    if bool(args.predeclare_only) and args.run_type != "frozen_test":
        raise SystemExit("--predeclare-only is valid only for frozen_test")
    records = _read_json(args.records_json)
    split = _read_json(args.split_json)
    scene_groups = [
        str(value)
        for value in (
            args.scene_groups
            if args.scene_groups is not None
            else (
                split["test_scene_groups"]
                if args.run_type == "frozen_test"
                else DEFAULT_CALIBRATION_GROUPS
            )
        )
    ]
    args.scene_groups = scene_groups
    target_count = int(
        args.target_count
        if args.target_count is not None
        else (
            10
            if args.run_type == "plumbing_smoke"
            else (32 if args.run_type == "calibration" else 6)
        )
    )
    seeds = [
        int(value)
        for value in (
            args.perturbation_seeds
            if args.perturbation_seeds is not None
            else ([0] if args.run_type == "plumbing_smoke" else [0, 1, 2])
        )
    ]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise SystemExit("perturbation seeds must be non-empty and unique")
    if args.run_type == "calibration":
        if target_count < 30 or len(set(scene_groups)) < 4:
            raise SystemExit("calibration requires >=30 targets from >=4 scene groups")
        if seeds != [0, 1, 2]:
            raise SystemExit("calibration requires paired perturbation seeds 0 1 2")
    if args.run_type == "frozen_test":
        if (
            args.frozen_predictions_json is None
            or args.ranking_target_decision is None
            or args.predeclared_selection_json is None
        ):
            raise SystemExit(
                "frozen_test requires predictions, route decision, and predeclared selection path"
            )
        if set(scene_groups) != {str(value) for value in split["test_scene_groups"]}:
            raise SystemExit("frozen_test must use exactly the three held-out scene groups")
        if target_count < 2 * len(scene_groups):
            raise SystemExit("frozen_test requires at least two targets per test scene group")
        if seeds != [0, 1, 2]:
            raise SystemExit("frozen_test requires paired perturbation seeds 0 1 2")
    allowed_ids = (
        list(split["test_sample_ids"])
        if args.run_type == "frozen_test"
        else [*split["train_sample_ids"], *split["val_sample_ids"]]
    )
    contracts = build_dynamic_target_contracts(
        records,
        allowed_sample_ids=allowed_ids,
        scene_groups=scene_groups,
    )
    ranking_target_decision: Optional[Mapping[str, Any]] = None
    frozen_predictions_identity: Optional[Mapping[str, Any]] = None
    frozen_coverage: Optional[Mapping[str, Any]] = None
    if args.run_type == "frozen_test":
        ranking_target_decision = _read_json(args.ranking_target_decision)
        if (
            ranking_target_decision.get("schema")
            != "cnabu_ranking_target_decision_v1"
            or ranking_target_decision.get("status") != "frozen"
            or bool(ranking_target_decision.get("uses_test_dynamic_outcomes", True))
        ):
            raise SystemExit("frozen_test requires a frozen non-test route decision")
        frozen_predictions = _read_json(args.frozen_predictions_json)
        if (
            frozen_predictions.get("schema")
            != "cnabu_ranked_blocker_frozen_test_predictions_v1"
            or not bool(
                frozen_predictions.get(
                    "prediction_frozen_before_test_dynamic_outcomes", False
                )
            )
            or bool(frozen_predictions.get("test_dynamic_outcomes_used", True))
        ):
            raise SystemExit("frozen predictions are not predeclared test policies")
        prediction_by_key = {
            (str(row["sample_id"]), int(row["matched_gt_target_instance_id"])): row
            for row in frozen_predictions["rows"]
        }
        joined = []
        matched_contract_count = 0
        for contract in contracts:
            key = (str(contract["sample_id"]), int(contract["target_instance_id"]))
            prediction = prediction_by_key.get(key)
            if prediction is None:
                continue
            matched_contract_count += 1
            explainability = prediction.get("explainability_fraction")
            if (
                not bool(prediction.get("rank_query_valid", False))
                or not bool(prediction.get("target_blockage_defined", False))
                or explainability is None
                or float(explainability) < 0.5
            ):
                continue
            joined.append(build_frozen_test_policy_contract(contract, prediction))
        frozen_coverage = {
            "oracle_positive_test_contract_count": int(len(contracts)),
            "matched_checkpoint_target_contract_count": int(matched_contract_count),
            "high_coverage_executable_contract_count": int(len(joined)),
            "selection_explainability_floor": 0.5,
        }
        contracts_for_selection = joined
        frozen_predictions_identity = {
            "path": str(args.frozen_predictions_json.resolve()),
            "sha256": _sha256(args.frozen_predictions_json),
        }
    else:
        contracts_for_selection = contracts
    selected = select_stratified_dynamic_targets(
        contracts_for_selection,
        scene_groups=scene_groups,
        target_count=target_count,
    )
    if args.run_type == "frozen_test":
        assert_dynamic_frozen_test_split_guard(
            selected,
            test_scene_groups=split["test_scene_groups"],
            non_test_scene_groups=[
                *split["train_scene_groups"], *split["val_scene_groups"]
            ],
        )
        selected_structures_by_group = {
            group: {
                str(item["static_structure"])
                for item in selected
                if str(item["scene_group"]) == group
            }
            for group in scene_groups
        }
        required_structures = {
            "single_removal_positive",
            "cooperative_zero_single_gain",
        }
        if any(
            not required_structures.issubset(structures)
            for structures in selected_structures_by_group.values()
        ):
            raise SystemExit(
                "frozen test selection must cover single and cooperative strata in every group"
            )
        normalised_selected = json.loads(
            json.dumps(selected, sort_keys=True, allow_nan=False)
        )
        selection_path = args.predeclared_selection_json.resolve()
        if bool(args.predeclare_only):
            if selection_path.exists():
                raise SystemExit(
                    "refusing to overwrite predeclared selection: {}".format(
                        selection_path
                    )
                )
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            predeclaration = {
                "schema": "ranked_blocker_frozen_test_predeclaration_v1",
                "status": "frozen_before_dynamic_outcomes",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "selection_rule": (
                    "two distinct high-explainability targets per held-out group, "
                    "covering single-removal and cooperative-zero-gain strata"
                ),
                "test_static_labels_used_for_stratification": True,
                "test_model_outputs_used_only_for_runtime_coverage_filter": True,
                "test_dynamic_outcomes_inspected": False,
                "test_dynamic_outcomes_may_change_selection_or_policy": False,
                "ranking_target_decision": {
                    "path": str(args.ranking_target_decision.resolve()),
                    "sha256": _sha256(args.ranking_target_decision),
                },
                "frozen_predictions": {
                    "path": str(args.frozen_predictions_json.resolve()),
                    "sha256": _sha256(args.frozen_predictions_json),
                },
                "frozen_test_coverage": frozen_coverage,
                "targets": normalised_selected,
                "runner_source": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _sha256(Path(__file__).resolve()),
                },
                "dynamic_aggregation_source": {
                    "path": str(
                        (
                            SCENE_GRAPH_PROJECT
                            / "src/scene_graph_mem/relations/dynamic_blocker_causal.py"
                        ).resolve()
                    ),
                    "sha256": _sha256(
                        SCENE_GRAPH_PROJECT
                        / "src/scene_graph_mem/relations/dynamic_blocker_causal.py"
                    ),
                },
                "action_oracle_source": {
                    "path": str(
                        (
                            PROJECT
                            / "shelf_gym/utils/action_conditioned_relation_oracle.py"
                        ).resolve()
                    ),
                    "sha256": _sha256(
                        PROJECT
                        / "shelf_gym/utils/action_conditioned_relation_oracle.py"
                    ),
                },
            }
            _write_json(selection_path, predeclaration)
            print(json.dumps(predeclaration, indent=2, sort_keys=True))
            return 0
        predeclaration = _read_json(selection_path)
        if (
            predeclaration.get("schema")
            != "ranked_blocker_frozen_test_predeclaration_v1"
            or predeclaration.get("status")
            != "frozen_before_dynamic_outcomes"
            or bool(predeclaration.get("test_dynamic_outcomes_inspected", True))
            or bool(
                predeclaration.get(
                    "test_dynamic_outcomes_may_change_selection_or_policy", True
                )
            )
        ):
            raise SystemExit("held-out selection was not safely predeclared")
        if predeclaration.get("targets") != normalised_selected:
            raise SystemExit("current held-out selection differs from predeclared targets")
        if (
            predeclaration["ranking_target_decision"]["sha256"]
            != _sha256(args.ranking_target_decision)
            or predeclaration["frozen_predictions"]["sha256"]
            != _sha256(args.frozen_predictions_json)
        ):
            raise SystemExit("predeclared selection input identities changed")
        if (
            predeclaration["runner_source"]["sha256"]
            != _sha256(Path(__file__).resolve())
            or predeclaration["dynamic_aggregation_source"]["sha256"]
            != _sha256(
                SCENE_GRAPH_PROJECT
                / "src/scene_graph_mem/relations/dynamic_blocker_causal.py"
            )
            or predeclaration["action_oracle_source"]["sha256"]
            != _sha256(
                PROJECT / "shelf_gym/utils/action_conditioned_relation_oracle.py"
            )
        ):
            raise SystemExit("predeclared causal runner source identities changed")
    else:
        assert_dynamic_calibration_split_guard(
            selected,
            allowed_scene_groups=scene_groups,
            test_scene_groups=split["test_scene_groups"],
        )
    active_tasks = []
    for selection in selected:
        task = dict(selection)
        if args.run_type == "plumbing_smoke":
            raw_top1 = int(selection["policy_orders"]["raw_salience"][0])
            active_removed = {(), (raw_top1,)}
            task["active_conditions"] = [
                item
                for item in selection["conditions"]
                if tuple(sorted(item["removed_source_ids"])) in active_removed
            ]
            task["active_candidate_ids"] = [selection["eligible_candidate_ids"][0]]
        else:
            task["active_conditions"] = list(selection["conditions"])
            task["active_candidate_ids"] = list(selection["eligible_candidate_ids"])
        task["active_seeds"] = seeds
        active_tasks.append(task)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    target_dir = output_dir / "target_trials"
    target_dir.mkdir()
    manifest = _manifest(args, selected, seeds=seeds)
    if frozen_coverage is not None:
        manifest["frozen_test_coverage"] = dict(frozen_coverage)
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        output_dir / "selection.json",
        {
            "schema": "ranked_blocker_dynamic_selection_v1",
            "selection_frozen_before_dynamic_outcomes": True,
            "selection_uses_test_groups": bool(args.run_type == "frozen_test"),
            "test_outcomes_select_target_or_policy": False,
            "frozen_test_coverage": frozen_coverage,
            "targets": selected,
        },
    )
    started = time.perf_counter()
    try:
        outputs = []
        with ProcessPoolExecutor(
            max_workers=int(args.workers), initializer=_worker_init
        ) as executor:
            futures = {executor.submit(_run_target_task, task): task for task in active_tasks}
            for future in as_completed(futures):
                output = future.result()
                outputs.append(output)
                filename = "{}__target_{}.json".format(
                    output["sample_id"].replace("/", "__"),
                    output["target_instance_id"],
                )
                _write_json(target_dir / filename, output)
                print(
                    json.dumps(
                        {
                            "sample_id": output["sample_id"],
                            "target_instance_id": output["target_instance_id"],
                            "trial_count": len(output["trials"]),
                            "elapsed_seconds": output["elapsed_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if args.run_type == "calibration":
            report = build_dynamic_causal_report(selected, outputs, seeds=seeds)
        elif args.run_type == "frozen_test":
            report = build_frozen_test_causal_report(
                selected,
                outputs,
                seeds=seeds,
                ranking_target_decision=ranking_target_decision or {},
                frozen_predictions_identity=frozen_predictions_identity or {},
                frozen_test_coverage=frozen_coverage or {},
            )
        else:
            report = {
                "schema": "ranked_blocker_dynamic_plumbing_smoke_v1",
                "target_count": int(len(outputs)),
                "trial_count": int(sum(len(item["trials"]) for item in outputs)),
                "all_targets_completed": bool(len(outputs) == len(selected)),
                "all_trials_have_boolean_clean_success": bool(
                    all(
                        isinstance(trial["clean_extraction_success"], bool)
                        for output in outputs
                        for trial in output["trials"]
                    )
                ),
                "failure_cause_counts": dict(
                    sorted(
                        Counter(
                            cause
                            for output in outputs
                            for trial in output["trials"]
                            for cause in trial["failure_causes"]
                        ).items()
                    )
                ),
                "resource_smoke_only": True,
                "does_not_select_ranking_target": True,
            }
        report_path = output_dir / "dynamic_causal_report.json"
        _write_json(report_path, report)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = float(time.perf_counter() - started)
        manifest["target_output_count"] = int(len(outputs))
        manifest["trial_count"] = int(sum(len(item["trials"]) for item in outputs))
        manifest["report_sha256"] = _sha256(report_path)
        _write_json(manifest_path, manifest)
        if args.run_type == "plumbing_smoke":
            terminal_summary = report
        elif args.run_type == "calibration":
            terminal_summary = report["policy_admissibility"]
        else:
            terminal_summary = report["primary_comparisons"]
        print(json.dumps(terminal_summary, indent=2, sort_keys=True))
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
