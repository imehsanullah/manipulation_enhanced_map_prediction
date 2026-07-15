#!/usr/bin/env python3
"""Export non-privileged planner action masks for explicit CNABU records.

This exporter never reads GT maps or oracle evidence. It reconstructs ordered
learned CNABU support, queries current-state IK and known fixed bodies, writes a
small mask JSON per record, and writes an enriched records JSON containing
``candidate_action_mask_file``. When requested, it also writes compact
planner-link/carried-target swept summaries without dynamic-object queries. It
refuses every overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from scene_graph_mem.relations.candidate_trajectory import (
    validate_candidate_action_mask_result,
    validate_candidate_planner_swept_features_result,
)
from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.scripts.audit_cnabu_runtime_candidate_action_mask import (
    DEFAULT_CNABU_DERIVED_ROOT,
    DEFAULT_RECORDS_JSON,
    _load_json,
    _load_runtime_support,
)
from shelf_gym.utils.action_conditioned_relation_oracle import (
    OracleActionFamilyConfig,
    build_cnabu_runtime_candidate_action_mask,
)
from shelf_gym.utils.mapping_utils import HeightmapGeneration


def _slim_mask_result(
    result: Mapping[str, Any],
    *,
    sample_id: str,
    node_ids: Sequence[Any],
    planner_swept_features_file: Path | None = None,
) -> Dict[str, Any]:
    validate_candidate_action_mask_result(result, node_ids=node_ids)
    payload = {
        "schema": "cnabu_runtime_candidate_action_mask_record_v0",
        "sample_id": str(sample_id),
        "source": str(result["source"]),
        "candidate_ids": list(result["candidate_ids"]),
        "node_ids": list(result["node_ids"]),
        "kinematic_mask": result["kinematic_mask"],
        "fixed_environment_collision_free_mask": result[
            "fixed_environment_collision_free_mask"
        ],
        "action_eligible_mask": result["action_eligible_mask"],
        "fixed_body_names": list(result["fixed_body_names"]),
        "support_boundary_quantile": float(result["support_boundary_quantile"]),
        "safety": dict(result["safety"]),
    }
    if planner_swept_features_file is not None:
        payload["candidate_planner_swept_features_file"] = str(
            planner_swept_features_file
        )
        payload["candidate_planner_swept_features_schema"] = (
            "cnabu_runtime_candidate_planner_swept_features_v1"
        )
    return payload


def _write_planner_swept_features(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    safety_json = json.dumps(dict(payload["safety"]), sort_keys=True)
    np.savez_compressed(
        path,
        schema=np.asarray(str(payload["schema"])),
        candidate_ids=np.asarray(payload["candidate_ids"], dtype=np.str_),
        node_ids_json=np.asarray(json.dumps(list(payload["node_ids"]))),
        pair_feature_names=np.asarray(
            payload["pair_feature_names"], dtype=np.str_
        ),
        pair_features=np.asarray(payload["pair_features"], dtype=np.float32),
        progress_bin_count=np.asarray(
            int(payload["progress_bin_count"]), dtype=np.int32
        ),
        clearance_normalization_m=np.asarray(
            float(payload["clearance_normalization_m"]), dtype=np.float32
        ),
        safety_json=np.asarray(safety_json),
    )


def export_candidate_action_masks(
    records: Sequence[Mapping[str, Any]],
    *,
    cnabu_derived_root: Path,
    output_root: Path,
    output_records_json: Path,
    support_boundary_quantile: float,
    include_planner_swept_features: bool,
    validate_only: bool,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("at least one explicit record is required")
    output_paths = [
        output_root / str(record["sample_id"]) / "candidate_action_mask.json"
        for record in records
    ]
    planner_feature_paths = [
        output_root
        / str(record["sample_id"])
        / "candidate_planner_swept_features.npz"
        for record in records
    ]
    conflicts = [path for path in output_paths if path.exists()]
    if include_planner_swept_features:
        conflicts.extend(path for path in planner_feature_paths if path.exists())
    if output_records_json.exists():
        conflicts.append(output_records_json)
    if conflicts:
        raise FileExistsError("refusing to overwrite: {}".format(conflicts))

    started = time.perf_counter()
    environment = ShelfEnv(render=False, max_obj_num=25, use_ycb=True)
    heightmap_generation = HeightmapGeneration(0.005, "MEM", n_classes=15)
    pending: List[
        tuple[Path, Dict[str, Any], Path | None, Dict[str, Any] | None]
    ] = []
    enriched_records: List[Dict[str, Any]] = []
    eligible_count = 0
    candidate_count = 0
    try:
        for record, output_path, planner_feature_path in zip(
            records,
            output_paths,
            planner_feature_paths,
        ):
            environment.reset_robot(environment.initial_parameters)
            environment.move_gripper(0.085)
            initial = np.asarray(
                environment.get_current_joint_config(),
                dtype=np.float64,
            )
            support, _masks, _classes, node_ids, crop_rows = _load_runtime_support(
                record,
                cnabu_derived_root,
            )
            result = build_cnabu_runtime_candidate_action_mask(
                environment,
                heightmap_generation,
                support.indices_zyx,
                crop_rows=crop_rows,
                node_ids=node_ids.tolist(),
                initial_arm_config=initial,
                config=OracleActionFamilyConfig(),
                support_boundary_quantile=support_boundary_quantile,
                include_planner_swept_features=bool(
                    include_planner_swept_features
                ),
            )
            planner_payload = None
            if include_planner_swept_features:
                planner_payload = dict(result.get("planner_swept_features") or {})
                validate_candidate_planner_swept_features_result(
                    planner_payload,
                    node_ids=node_ids.tolist(),
                )
            slim = _slim_mask_result(
                result,
                sample_id=str(record["sample_id"]),
                node_ids=node_ids.tolist(),
                planner_swept_features_file=(
                    planner_feature_path
                    if include_planner_swept_features
                    else None
                ),
            )
            pending.append(
                (
                    output_path,
                    slim,
                    planner_feature_path if include_planner_swept_features else None,
                    planner_payload,
                )
            )
            enriched = dict(record)
            enriched["candidate_action_mask_file"] = str(output_path)
            if include_planner_swept_features:
                enriched["candidate_planner_swept_features_file"] = str(
                    planner_feature_path
                )
            enriched_records.append(enriched)
            mask = np.asarray(slim["action_eligible_mask"], dtype=bool)
            eligible_count += int(mask.sum())
            candidate_count += int(mask.size)
            print(
                "prepared planner metadata {}/{} {}".format(
                    len(enriched_records),
                    len(records),
                    record["sample_id"],
                ),
                file=sys.stderr,
                flush=True,
            )
    finally:
        environment.close()

    if not validate_only:
        for output_path, payload, planner_feature_path, planner_payload in pending:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if planner_feature_path is not None and planner_payload is not None:
                _write_planner_swept_features(
                    planner_feature_path,
                    planner_payload,
                )
        output_records_json.parent.mkdir(parents=True, exist_ok=True)
        output_records_json.write_text(
            json.dumps(enriched_records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return {
        "schema": "cnabu_runtime_candidate_action_mask_export_summary_v1",
        "record_count": len(records),
        "candidate_count": candidate_count,
        "eligible_count": eligible_count,
        "eligible_fraction": float(eligible_count / candidate_count),
        "support_boundary_quantile": float(support_boundary_quantile),
        "includes_planner_swept_features": bool(
            include_planner_swept_features
        ),
        "output_root": str(output_root),
        "output_records_json": str(output_records_json),
        "validate_only": bool(validate_only),
        "runtime_seconds": float(time.perf_counter() - started),
        "safety": {
            "reads_gt_or_oracle_evidence": False,
            "queries_dynamic_scene_objects": False,
            "writes_checkpoints_models_or_hdf5": False,
            "refuses_overwrites": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS_JSON)
    parser.add_argument(
        "--cnabu-derived-root",
        type=Path,
        default=DEFAULT_CNABU_DERIVED_ROOT,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-records-json", type=Path, required=True)
    parser.add_argument("--support-boundary-quantile", type=float, default=0.05)
    parser.add_argument("--include-planner-swept-features", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_json(args.records_json)
    if not isinstance(records, list):
        raise ValueError("records-json must contain a list")
    if int(args.limit) < 0:
        raise ValueError("limit must be non-negative")
    if int(args.limit) > 0:
        records = records[: int(args.limit)]
    summary = export_candidate_action_masks(
        records,
        cnabu_derived_root=args.cnabu_derived_root,
        output_root=args.output_root.expanduser(),
        output_records_json=args.output_records_json.expanduser(),
        support_boundary_quantile=float(args.support_boundary_quantile),
        include_planner_swept_features=bool(
            args.include_planner_swept_features
        ),
        validate_only=bool(args.validate_only),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
