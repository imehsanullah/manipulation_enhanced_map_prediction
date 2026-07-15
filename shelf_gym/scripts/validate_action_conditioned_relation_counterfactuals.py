#!/usr/bin/env python3
"""Run a small paired PyBullet counterfactual validation pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.utils.action_conditioned_relation_oracle import (
    evaluate_counterfactual_candidate,
    evaluate_saved_scene,
    list_counterfactual_candidates,
)


DEFAULT_ORACLE_DIR = Path(
    "/data/manipulation_map_data/derived/action_conditioned_relation_oracle_v1/"
    "prototype_30_20260713"
)
STRATUM_CYCLE = (
    "single_contains_action_only",
    "single_all_geometry_positive",
    "multiple_contains_action_only",
    "multiple_all_geometry_positive",
)
SELECTION_SCHEDULE = (
    ("single_all_geometry_positive", "low"),
    ("single_contains_action_only", "low"),
    ("single_contains_action_only", "mid"),
    ("single_all_geometry_positive", "mid"),
    ("single_contains_action_only", "high"),
    ("single_all_geometry_positive", "high"),
    ("multiple_contains_action_only", "low"),
    ("multiple_all_geometry_positive", "low"),
    ("multiple_contains_action_only", "mid"),
    ("multiple_all_geometry_positive", "high"),
)


def _candidate_score_bucket(candidate: Mapping[str, Any]) -> str:
    scores = [float(value) for value in (candidate.get("pair_scores") or {}).values()]
    if not scores:
        return "unknown"
    score = min(scores)
    return "low" if score < 0.5 else "mid" if score < 1.0 else "high"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_stratified_counterfactuals(
    scene_records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> List[Tuple[Mapping[str, Any], Dict[str, Any]]]:
    """Choose at most one candidate per scene while cycling key strata."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    by_stratum: Dict[str, List[Tuple[Mapping[str, Any], Dict[str, Any]]]] = {
        name: [] for name in STRATUM_CYCLE
    }
    for scene in scene_records:
        for candidate in list_counterfactual_candidates(scene):
            by_stratum.setdefault(candidate["stratum"], []).append((scene, candidate))

    selected: List[Tuple[Mapping[str, Any], Dict[str, Any]]] = []
    used_samples = set()
    used_grasps = set()
    used_candidates = set()
    schedule_index = 0
    while len(selected) < limit:
        progress = False
        schedule = list(SELECTION_SCHEDULE)
        for offset in range(len(schedule)):
            stratum, score_bucket = schedule[(schedule_index + offset) % len(schedule)]
            options = by_stratum.get(stratum, [])
            bucket_options = [
                item for item in options if _candidate_score_bucket(item[1]) == score_bucket
            ]
            if bucket_options:
                options = bucket_options
            options = sorted(
                options,
                key=lambda item: (
                    item[1].get("grasp_id") in used_grasps,
                    item[0]["sample_id"],
                    item[1]["trajectory_id"],
                ),
            )
            for scene, candidate in options:
                key = (scene["sample_id"], candidate["trajectory_id"])
                if scene["sample_id"] in used_samples or key in used_candidates:
                    continue
                selected.append((scene, candidate))
                used_samples.add(scene["sample_id"])
                used_grasps.add(candidate.get("grasp_id"))
                used_candidates.add(key)
                progress = True
                schedule_index = (schedule_index + offset + 1) % len(schedule)
                break
            if progress:
                break
        if not progress:
            break
    return selected


def aggregate_counterfactual_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    trials = [trial for record in records for trial in record["trials"]]
    interventions = [intervention for record in records for intervention in record["interventions"]]
    strata = Counter(trial["metadata"]["stratum"] for trial in trials)
    deltas = [float(item["success_delta"]) for item in interventions]
    outcomes = Counter(
        (
            "failure_to_success"
            if not trial["intact_success"] and trial["intervention_success"]
            else "success_to_success"
            if trial["intact_success"] and trial["intervention_success"]
            else "success_to_failure"
            if trial["intact_success"] and not trial["intervention_success"]
            else "failure_to_failure"
        )
        for trial in trials
    )
    intervention_fixed_contact_count = sum(
        any(
            trial["metadata"]["intervention_execution"].get(
                "fixed_hard_penetrations_by_stage",
                trial["metadata"]["intervention_execution"]["robot_fixed_contacts_by_stage"],
            ).values()
        )
        for trial in trials
    )
    contact_outcomes = Counter(
        trial.get("metadata", {}).get("contact_outcome", "unclassified") for trial in trials
    )
    threshold_selection = select_relation_score_threshold(trials)
    penetration_selection = select_hard_penetration_threshold(trials)
    return {
        "schema": "counterfactual_access_validation_pilot_summary_v1",
        "intervention_count": len(interventions),
        "paired_trial_count": len(trials),
        "intact_success_count": sum(bool(item["intact_success"]) for item in trials),
        "intervention_success_count": sum(bool(item["intervention_success"]) for item in trials),
        "positive_delta_count": sum(value > 0.0 for value in deltas),
        "zero_delta_count": sum(value == 0.0 for value in deltas),
        "negative_delta_count": sum(value < 0.0 for value in deltas),
        "mean_success_delta": float(sum(deltas) / len(deltas)) if deltas else None,
        "paired_outcome_counts": dict(sorted(outcomes.items())),
        "intervention_fixed_environment_contact_count": intervention_fixed_contact_count,
        "contact_outcome_counts": dict(sorted(contact_outcomes.items())),
        "stratum_counts": dict(sorted(strata.items())),
        "threshold_selection": threshold_selection,
        "hard_penetration_threshold_selection": penetration_selection,
        "interpretation": (
            "positive delta supports causal access blocking; zero/negative delta is not support for the "
            "predicted blocker set under this tested path"
        ),
        "limitations": [
            "small randomized saved-pose validation subset",
            "target is attached with a fixed constraint after the grasp waypoint",
            "validates access/extraction dynamics, not autonomous grasp closure",
        ],
    }


def select_relation_score_threshold(trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Select a single-blocker score threshold on causally evaluable paired trials."""

    examples = []
    for trial in trials:
        removed = [int(value) for value in trial.get("removed_instance_ids", [])]
        metadata = trial.get("metadata", {})
        outcome = metadata.get("contact_outcome")
        if len(removed) != 1 or outcome not in {"hard_blockage_supported", "contact_tolerated"}:
            continue
        scores = metadata.get("pair_scores") or {}
        score = scores.get(str(removed[0]))
        if score is None:
            continue
        examples.append((float(score), outcome == "hard_blockage_supported"))
    candidates = [value / 20.0 for value in range(21)]
    rows = []
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for score, label in examples:
            prediction = score >= threshold
            tp += int(prediction and label)
            fp += int(prediction and not label)
            fn += int(not prediction and label)
            tn += int(not prediction and not label)
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    best = max(rows, key=lambda item: (item["f1"], item["precision"], item["threshold"])) if rows else None
    return {
        "eligible_single_blocker_trial_count": len(examples),
        "positive_trial_count": sum(label for _, label in examples),
        "negative_trial_count": sum(not label for _, label in examples),
        "best": best,
        "grid": rows,
    }


def select_hard_penetration_threshold(trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Check whether deeper nominal penetration better predicts causal blockage."""

    examples = []
    for trial in trials:
        metadata = trial.get("metadata", {})
        outcome = metadata.get("contact_outcome")
        signed_distance = metadata.get("minimum_blocker_signed_distance_m")
        if outcome not in {"hard_blockage_supported", "contact_tolerated"} or signed_distance is None:
            continue
        examples.append((max(0.0, -float(signed_distance)), outcome == "hard_blockage_supported"))
    candidates = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
    rows = []
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for penetration, label in examples:
            prediction = penetration >= threshold
            tp += int(prediction and label)
            fp += int(prediction and not label)
            fn += int(not prediction and label)
            tn += int(not prediction and not label)
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append(
            {
                "threshold_m": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    best = max(rows, key=lambda item: (item["f1"], item["precision"], item["threshold_m"])) if rows else None
    return {
        "evaluable_trial_count": len(examples),
        "positive_trial_count": sum(label for _, label in examples),
        "negative_trial_count": sum(not label for _, label in examples),
        "selection_bias": "candidates were preselected at the configured hard-penetration threshold",
        "best": best,
        "grid": rows,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--limit", type=int, default=10, help="Number of distinct candidate interventions")
    parser.add_argument("--randomization-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite existing output directory: {}".format(args.output_dir))
    scene_paths = sorted(args.oracle_dir.glob("scene_*.json"))
    scene_records = [_read_json(path) for path in scene_paths]
    selected = select_stratified_counterfactuals(scene_records, limit=args.limit)
    if len(selected) < args.limit:
        raise SystemExit("only {} distinct-scene candidates available".format(len(selected)))
    args.output_dir.mkdir(parents=True)

    records: List[Dict[str, Any]] = []
    environment = ShelfEnv(render=False, max_obj_num=25, use_ycb=True)
    try:
        for index, (exported_scene, candidate) in enumerate(selected):
            pre_action_dir = Path(exported_scene["metadata"]["pre_action_dir"])
            replayed_scene, debug = evaluate_saved_scene(environment, pre_action_dir=pre_action_dir)
            if replayed_scene["sample_id"] != exported_scene["sample_id"]:
                raise RuntimeError("replayed scene does not match selected export")
            result = evaluate_counterfactual_candidate(
                environment,
                pre_action_dir=pre_action_dir,
                scene_record=replayed_scene,
                candidate=candidate,
                candidate_debug=debug[candidate["trajectory_id"]],
                randomization_seeds=args.randomization_seeds,
            )
            records.append(result)
            _write_json(args.output_dir / "pair_{:03d}.json".format(index), result)
            intervention = result["interventions"][0]
            print(
                json.dumps(
                    {
                        "sample_id": result["sample_id"],
                        "trajectory_id": candidate["trajectory_id"],
                        "stratum": candidate["stratum"],
                        "removed_instance_ids": candidate["removed_instance_ids"],
                        "paired_trial_count": len(result["trials"]),
                        "intact_success_probability": intervention["intact_success_probability"],
                        "intervention_success_probability": intervention[
                            "intervention_success_probability"
                        ],
                        "success_delta": intervention["success_delta"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        environment.close()

    summary = aggregate_counterfactual_records(records)
    summary["selected_pairs"] = [
        {
            "sample_id": scene["sample_id"],
            "trajectory_id": candidate["trajectory_id"],
            "stratum": candidate["stratum"],
        }
        for scene, candidate in selected
    ]
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": "counterfactual_access_validation_pilot_manifest_v1",
            "pair_files": ["pair_{:03d}.json".format(index) for index in range(len(records))],
            "summary_file": "summary.json",
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
