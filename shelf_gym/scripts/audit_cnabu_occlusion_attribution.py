"""Audit GT-versus-CNABU visual occlusion over executable MEM cameras.

Ray generation and GT adapters live in the MEM project.  Portable relation
math and audit metrics are imported from ``scene_graph_mem``.  The script is
read-only with respect to datasets and writes one compact strict-JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from scene_graph_mem.relations.belief_occlusion import (
    build_unresolved_uncertainty_field,
    compute_hard_occlusion_masses,
    hidden_uncertainty_components,
    hidden_uncertainty_naive,
)
from scene_graph_mem.relations.path_aligned_features import (
    reconstruct_sparse_node_voxel_support,
)
from scene_graph_mem.training.belief_occlusion_audit import (
    pair_regression_metrics,
    query_ranking_metrics,
    summarize_label_continuity,
    tie_invariant_average_precision,
)
from shelf_gym.utils.cnabu_occlusion_attribution import (
    align_oracle_supports_to_nodes,
    build_gt_object_voxel_supports,
    build_runtime_support_partition,
    dense_supports_from_sparse_indices,
    info_gain_raycast_to_canonical_zyx,
    match_nodes_to_gt_objects,
)
from shelf_gym.utils.information_gain_utils import InfoGainEval


REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = REPO_ROOT.parent
SCENE_GRAPH_ROOT = THESIS_ROOT / "scene_graph_mem"
DEFAULT_RAW_ROOT = Path("/data/manipulation_map_data/raw/map_data")
DEFAULT_CNABU_ROOT = Path(
    "/data/manipulation_map_data/derived/cnabu_d3g/"
    "raw_cnabu_concat_1000_20260528_100559"
)
DEFAULT_NODE_ROOT = Path(
    "/data/manipulation_map_data/derived/cnabu_scene_graph/"
    "learned_splitter_nodes_1000_20260712"
)
DEFAULT_SPLIT_MANIFEST = Path(
    "/data/manipulation_map_data/derived/cnabu_belief_occlusion_v1/"
    "stage0_split_340_scene_group_v1_20260716/split_manifest.json"
)
DEFAULT_CAMERA_PATH = Path(__file__).resolve().parent / "model" / "camera_matrices.npz"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()

    return {
        "path": str(repo),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def _parse_indices(value: str, *, upper_bound: int) -> Tuple[int, ...]:
    values: List[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(item) if item else None for item in chunk.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError("camera range must use start:stop[:step]")
            start = 0 if parts[0] is None else int(parts[0])
            stop = upper_bound if parts[1] is None else int(parts[1])
            step = 1 if len(parts) == 2 or parts[2] is None else int(parts[2])
            values.extend(range(start, stop, step))
        else:
            values.append(int(chunk))
    if not values:
        raise ValueError("at least one camera index is required")
    if len(set(values)) != len(values):
        raise ValueError("camera indices must be unique")
    if any(index < 0 or index >= int(upper_bound) for index in values):
        raise ValueError("camera index is outside the executable camera set")
    return tuple(values)


def _load_sample_ids(
    split_manifest: Path,
    *,
    split_role: str,
    explicit_ids: Optional[str],
    max_samples: Optional[int],
    allow_protected_test: bool,
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    key_by_role = {
        "train": "train_sample_ids",
        "validation": "validation_sample_ids",
        "protected_test": "protected_test_sample_ids",
    }
    if split_role == "protected_test" and not bool(allow_protected_test):
        raise PermissionError(
            "protected-test outcomes are sealed; pass --allow-protected-test only after method freeze"
        )
    declared = tuple(str(value) for value in manifest[key_by_role[split_role]])
    if explicit_ids:
        selected = tuple(value.strip() for value in explicit_ids.split(",") if value.strip())
        unknown = sorted(set(selected) - set(declared))
        if unknown:
            raise ValueError(
                "explicit sample ids are outside split role {}: {}".format(
                    split_role, unknown
                )
            )
    else:
        selected = declared
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("--max-samples must be positive")
        selected = selected[: int(max_samples)]
    if not selected:
        raise ValueError("sample selection is empty")
    return selected, manifest


def _quantile_bin(value: float, boundaries: Sequence[float], labels: Sequence[str]) -> str:
    for boundary, label in zip(boundaries, labels):
        if float(value) <= float(boundary):
            return str(label)
    return str(labels[-1])


def _camera_contexts(info_gain: InfoGainEval) -> Dict[int, Dict[str, Any]]:
    centers = np.stack(
        [np.asarray(rays[0, :3], dtype=np.float64) for rays in info_gain.all_rays]
    )
    x_edges = np.quantile(centers[:, 0], [1.0 / 3.0, 2.0 / 3.0])
    z_edges = np.quantile(centers[:, 2], [1.0 / 3.0, 2.0 / 3.0])
    radial = np.linalg.norm(centers[:, [0, 2]], axis=1)
    radial_edge = float(np.median(radial))
    result = {}
    for index, center in enumerate(centers):
        x_bin = _quantile_bin(center[0], x_edges, ("left", "center", "right"))
        z_bin = _quantile_bin(center[2], z_edges, ("low", "middle", "high"))
        distance_bin = "near" if radial[index] <= radial_edge else "far"
        result[index] = {
            "camera_index": int(index),
            "origin_world_xyz": center.tolist(),
            "x_bin": x_bin,
            "height_bin": z_bin,
            "distance_bin": distance_bin,
            "pose_bin": "{}_{}_{}".format(x_bin, z_bin, distance_bin),
        }
    return result


def _support_mean(values: np.ndarray, support: np.ndarray) -> Optional[float]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(support, dtype=bool)]
    return float(selected.mean()) if selected.size else None


def _level_bin(value: Optional[float]) -> str:
    if value is None:
        return "undefined"
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "medium"
    return "high"


def _match_bin(value: float) -> str:
    if value < 0.5:
        return "iou_[0.25,0.5)"
    if value < 0.75:
        return "iou_[0.5,0.75)"
    return "iou_[0.75,1.0]"


def _target_geometry_context(support: np.ndarray) -> Dict[str, Any]:
    indices = np.argwhere(support)
    if not len(indices):
        return {
            "target_depth_bin": "undefined",
            "target_height_bin": "undefined",
            "target_centroid_zyx": None,
        }
    centroid = indices.mean(axis=0)
    depth_fraction = float(centroid[1] / max(support.shape[1] - 1, 1))
    height_fraction = float(centroid[0] / max(support.shape[0] - 1, 1))
    return {
        "target_depth_bin": _quantile_bin(
            depth_fraction, (1.0 / 3.0, 2.0 / 3.0), ("front", "middle", "back")
        ),
        "target_height_bin": _quantile_bin(
            height_fraction, (1.0 / 3.0, 2.0 / 3.0), ("low", "middle", "high")
        ),
        "target_centroid_zyx": centroid.tolist(),
    }


def _optional_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(clean)) if clean else None


def _optional_median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.median(clean)) if clean else None


def _summarize_ranking_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "average_precision_nonzero",
        "dominant_recall_at_1",
        "spearman_no_ties",
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "mass_at_1",
        "mass_at_3",
        "mass_at_5",
    )
    return {
        "query_count": int(len(records)),
        "rankable_query_count": int(sum(record["positive_count"] >= 2 for record in records)),
        **{key: _optional_mean(record.get(key) for record in records) for key in keys},
    }


def _stratified_query_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key in keys:
        groups: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(key, "undefined"))].append(record)
        output[key] = {
            group: {
                **_summarize_ranking_records(values),
                "union_mae": _optional_mean(value.get("union_absolute_error") for value in values),
                "pair_mae": _optional_mean(value.get("pair_mae") for value in values),
            }
            for group, values in sorted(groups.items())
        }
    return output


def _scene_macro_ap(
    pair_records: Sequence[Mapping[str, Any]],
    *,
    oracle_key: str,
    predicted_key: str,
) -> Dict[str, Any]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in pair_records:
        grouped[str(record["sample_id"])].append(record)
    by_scene = {}
    for sample_id, values in sorted(grouped.items()):
        labels = np.asarray([float(value[oracle_key]) > 0.0 for value in values])
        scores = np.asarray([float(value[predicted_key]) for value in values])
        by_scene[sample_id] = tie_invariant_average_precision(labels, scores)
    return {
        "scene_count": int(len(by_scene)),
        "macro_average_precision_nonzero": _optional_mean(by_scene.values()),
        "by_scene": by_scene,
    }


def _pair_stratifications(
    pair_records: Sequence[Mapping[str, Any]],
    *,
    oracle_key: str,
    predicted_key: str,
) -> Dict[str, Any]:
    keys = (
        "camera_pose_bin",
        "target_depth_bin",
        "target_height_bin",
        "target_visible_fraction_bin",
        "occlusion_layer_bin",
        "match_quality_bin",
        "same_parent",
        "occupancy_epistemic_bin",
        "semantic_vacuity_bin",
        "scene_group",
    )
    output: Dict[str, Any] = {}
    for key in keys:
        groups: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in pair_records:
            groups[str(record.get(key, "undefined"))].append(record)
        output[key] = {}
        for group, values in sorted(groups.items()):
            oracle = np.asarray([value[oracle_key] for value in values], dtype=np.float64)
            predicted = np.asarray([value[predicted_key] for value in values], dtype=np.float64)
            output[key][group] = {
                **pair_regression_metrics(oracle, predicted),
                "positive_rate": float((oracle > 0.0).mean()) if len(oracle) else None,
                "average_precision_nonzero": tie_invariant_average_precision(
                    oracle > 0.0, predicted
                ),
            }
    return output


def _run_sample(
    *,
    sample_id: str,
    camera_indices: Sequence[int],
    info_gain: InfoGainEval,
    camera_contexts: Mapping[int, Mapping[str, Any]],
    raw_root: Path,
    cnabu_root: Path,
    node_root: Path,
    top_z: int,
    source_batch_size: int,
    occupancy_threshold: float,
    match_iou_threshold: float,
    include_naive_reference: bool,
    skip_source_attribution: bool,
) -> Dict[str, Any]:
    gt_path = raw_root / sample_id / "pre_action" / "gt_hms.npz"
    cnabu_path = cnabu_root / "samples" / sample_id / "pre_action" / "cnabu_hms.npz"
    node_path = node_root / "samples" / sample_id / "pre_action" / "node_masks.npz"
    for path in (gt_path, cnabu_path, node_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    load_started = time.perf_counter()
    with np.load(cnabu_path, allow_pickle=False) as data:
        occupancy_mean = np.asarray(data["occupancy_mean"], dtype=np.float32)
        occupancy_epistemic = np.asarray(data["occupancy_epistemic"], dtype=np.float32)
        semantic_mean = np.asarray(data["semantic_mean"], dtype=np.float32)
        semantic_vacuity = np.asarray(data["semantic_vacuity"], dtype=np.float32)
        crop_rows = tuple(int(value) for value in np.asarray(data["crop_rows"]).tolist())
    with np.load(node_path, allow_pickle=False) as data:
        node_masks = np.asarray(data["node_masks"], dtype=bool)
        node_classes = np.asarray(data["node_semantic_labels"], dtype=np.int64)
        node_scores = np.asarray(data["node_scores"], dtype=np.float64)
        node_crop_rows = tuple(
            int(value) for value in np.asarray(data["crop_rows"]).tolist()
        )
        node_source = str(np.asarray(data["node_source"]).item())
    if crop_rows != node_crop_rows:
        raise ValueError("CNABU and learned-node crop rows differ")
    if node_source != "learned_component_splitter":
        raise ValueError("Stage-1 audit requires learned_component_splitter nodes")

    sparse = reconstruct_sparse_node_voxel_support(
        occupancy_mean,
        semantic_mean,
        node_masks,
        node_classes,
        crop_rows=crop_rows,
        occupancy_threshold=float(occupancy_threshold),
    )
    deterministic_supports = dense_supports_from_sparse_indices(
        sparse.indices_zyx,
        grid_shape_zyx=sparse.grid_shape_zyx,
    )
    runtime_partition = build_runtime_support_partition(
        occupancy_mean=occupancy_mean,
        semantic_mean=semantic_mean,
        source_supports_zyx=deterministic_supports,
        occupancy_threshold=float(occupancy_threshold),
    )
    with np.load(gt_path, allow_pickle=False) as data:
        gt_supports = build_gt_object_voxel_supports(
            hm3d=data["hm3d"],
            semantic_2d=data["semantic_2d"],
            semantic_3d=data["semantic_3d"],
            instance_maps=data["instance_maps"],
            crop_rows=crop_rows,
            occupancy_threshold=float(occupancy_threshold),
        )
    matching = match_nodes_to_gt_objects(
        node_masks_raw_hw=node_masks,
        node_class_ids=node_classes,
        gt_masks_raw_hw=gt_supports.masks_raw_hw,
        gt_class_ids=gt_supports.class_ids,
        iou_threshold=float(match_iou_threshold),
    )
    oracle_alignment = align_oracle_supports_to_nodes(gt_supports, matching)
    uncertainty = build_unresolved_uncertainty_field(
        occupancy_mean,
        occupancy_epistemic,
        semantic_vacuity,
    )
    load_seconds = float(time.perf_counter() - load_started)

    node_matched = np.asarray(matching.node_to_gt_index) >= 0
    match_ious = np.asarray(matching.matched_iou, dtype=np.float64)
    target_contexts = []
    for node_index in range(len(node_masks)):
        support = oracle_alignment.target_supports_zyx[node_index]
        context = _target_geometry_context(support)
        context.update(
            {
                "node_index": int(node_index),
                "node_class_id": int(node_classes[node_index]),
                "node_score": float(node_scores[node_index]),
                "matched": bool(node_matched[node_index]),
                "match_iou": (
                    float(match_ious[node_index]) if node_matched[node_index] else None
                ),
                "match_quality_bin": (
                    _match_bin(float(match_ious[node_index]))
                    if node_matched[node_index]
                    else "unmatched"
                ),
                "occupancy_epistemic_mean": _support_mean(
                    uncertainty.occupancy_epistemic, support
                ),
                "semantic_vacuity_mean": _support_mean(
                    uncertainty.semantic_vacuity, support
                ),
            }
        )
        context["occupancy_epistemic_bin"] = _level_bin(
            context["occupancy_epistemic_mean"]
        )
        context["semantic_vacuity_bin"] = _level_bin(
            context["semantic_vacuity_mean"]
        )
        target_contexts.append(context)

    query_records = {"immediate": [], "potential": []}
    pair_records: List[Dict[str, Any]] = []
    attribution_records: Dict[str, List[Dict[str, Any]]] = {
        "occupancy_epistemic": [],
        "semantic_vacuity": [],
        "total": [],
    }
    runtime_records = []
    naive_records = []
    group = str(sample_id.split("/", 1)[0])
    for camera_index in camera_indices:
        runtime: Dict[str, Any] = {"camera_index": int(camera_index)}
        started = time.perf_counter()
        raw_raycast = info_gain.get_raycast(camera_idx=int(camera_index))
        runtime["raycast_generation_seconds"] = float(time.perf_counter() - started)
        started = time.perf_counter()
        rays = info_gain_raycast_to_canonical_zyx(
            raw_raycast,
            grid_shape_zyx=occupancy_mean.shape,
            crop_rows=crop_rows,
            raw_shape_hw=gt_supports.masks_raw_hw.shape[1:],
        )
        runtime["ray_conversion_seconds"] = float(time.perf_counter() - started)
        valid_ray_samples = np.all(rays >= 0, axis=-1)
        runtime["valid_ray_sample_count"] = int(valid_ray_samples.sum())
        runtime["rays_with_valid_sample_count"] = int(
            valid_ray_samples.any(axis=1).sum()
        )

        started = time.perf_counter()
        oracle = compute_hard_occlusion_masses(
            rays,
            oracle_alignment.source_supports_zyx,
            oracle_alignment.target_supports_zyx,
            target_source_indices=oracle_alignment.target_source_indices,
            unrepresented_support=oracle_alignment.unrepresented_support_zyx,
            fixed_environment_support=oracle_alignment.fixed_environment_support_zyx,
            top_z=int(top_z),
        )
        runtime["oracle_hard_seconds"] = float(time.perf_counter() - started)
        started = time.perf_counter()
        deterministic = compute_hard_occlusion_masses(
            rays,
            deterministic_supports,
            deterministic_supports,
            target_source_indices=np.arange(len(deterministic_supports)),
            unrepresented_support=runtime_partition.unrepresented_support_zyx,
            fixed_environment_support=runtime_partition.fixed_environment_support_zyx,
            top_z=int(top_z),
        )
        runtime["deterministic_hard_seconds"] = float(time.perf_counter() - started)

        oracle_attribution = None
        deterministic_attribution = None
        if not skip_source_attribution:
            started = time.perf_counter()
            oracle_attribution = hidden_uncertainty_components(
                rays,
                occupancy_mean,
                uncertainty,
                oracle_alignment.source_supports_zyx,
                source_batch_size=int(source_batch_size),
            )
            runtime["oracle_attribution_seconds"] = float(
                time.perf_counter() - started
            )
            started = time.perf_counter()
            deterministic_attribution = hidden_uncertainty_components(
                rays,
                occupancy_mean,
                uncertainty,
                deterministic_supports,
                source_batch_size=int(source_batch_size),
            )
            runtime["deterministic_attribution_seconds"] = float(
                time.perf_counter() - started
            )
        else:
            runtime["oracle_attribution_seconds"] = 0.0
            runtime["deterministic_attribution_seconds"] = 0.0

        if include_naive_reference:
            if oracle_attribution is None or deterministic_attribution is None:
                raise RuntimeError("naive reference requires source attribution")
            for source_name, source_supports, vector_values in (
                (
                    "oracle",
                    oracle_alignment.source_supports_zyx,
                    oracle_attribution["total"],
                ),
                ("deterministic", deterministic_supports, deterministic_attribution["total"]),
            ):
                started = time.perf_counter()
                naive = hidden_uncertainty_naive(
                    rays,
                    occupancy_mean,
                    uncertainty.total,
                    source_supports,
                )
                naive_records.append(
                    {
                        "camera_index": int(camera_index),
                        "support_source": source_name,
                        "wall_seconds": float(time.perf_counter() - started),
                        "max_absolute_error": float(
                            np.max(np.abs(naive - vector_values), initial=0.0)
                        ),
                    }
                )

        camera = camera_contexts[int(camera_index)]
        if oracle_attribution is not None and deterministic_attribution is not None:
            for component in attribution_records:
                valid_sources = node_matched.copy()
                metrics = query_ranking_metrics(
                    oracle_attribution[component],
                    deterministic_attribution[component],
                    valid_mask=valid_sources,
                )
                regression = pair_regression_metrics(
                    oracle_attribution[component],
                    deterministic_attribution[component],
                    valid_mask=valid_sources,
                )
                attribution_records[component].append(
                    {
                        "sample_id": sample_id,
                        "scene_group": group,
                        "camera_index": int(camera_index),
                        "camera_pose_bin": camera["pose_bin"],
                        **metrics,
                        "mae": regression["mae"],
                        "rmse": regression["rmse"],
                        "oracle_sum": float(
                            oracle_attribution[component][valid_sources].sum()
                        ),
                        "deterministic_sum": float(
                            deterministic_attribution[component][valid_sources].sum()
                        ),
                        "oracle_values": oracle_attribution[component][
                            valid_sources
                        ].tolist(),
                        "deterministic_values": deterministic_attribution[component][
                            valid_sources
                        ].tolist(),
                    }
                )

        common_target = (
            node_matched
            & oracle.target_defined_mask
            & deterministic.target_defined_mask
        )
        for target_index in np.flatnonzero(common_target):
            valid_sources = node_matched.copy()
            valid_sources[int(target_index)] = False
            target_context = target_contexts[int(target_index)]
            visible_fraction = float(oracle.target_voxel_projection_fraction[target_index])
            visible_bin = _quantile_bin(
                visible_fraction,
                (0.25, 0.75),
                ("low_[0,0.25]", "medium_(0.25,0.75]", "high_(0.75,1]"),
            )
            layer_count = int(
                (oracle.potential_pair_mass[:, target_index][valid_sources] > 0.0).sum()
            )
            layer_bin = "0" if layer_count == 0 else ("1" if layer_count == 1 else ("2" if layer_count == 2 else "3+"))
            for relation_name, oracle_matrix, deterministic_matrix in (
                ("immediate", oracle.immediate_pair_mass, deterministic.immediate_pair_mass),
                ("potential", oracle.potential_pair_mass, deterministic.potential_pair_mass),
            ):
                oracle_values = oracle_matrix[:, target_index][valid_sources]
                predicted_values = deterministic_matrix[:, target_index][valid_sources]
                metrics = query_ranking_metrics(oracle_values, predicted_values)
                regression = pair_regression_metrics(oracle_values, predicted_values)
                query_records[relation_name].append(
                    {
                        "sample_id": sample_id,
                        "scene_group": group,
                        "camera_index": int(camera_index),
                        "camera_pose_bin": camera["pose_bin"],
                        "target_index": int(target_index),
                        "target_depth_bin": target_context["target_depth_bin"],
                        "target_height_bin": target_context["target_height_bin"],
                        "target_visible_fraction": visible_fraction,
                        "target_visible_fraction_bin": visible_bin,
                        "occlusion_layer_count": layer_count,
                        "occlusion_layer_bin": layer_bin,
                        "match_quality_bin": target_context["match_quality_bin"],
                        "occupancy_epistemic_bin": target_context[
                            "occupancy_epistemic_bin"
                        ],
                        "semantic_vacuity_bin": target_context["semantic_vacuity_bin"],
                        "pair_mae": regression["mae"],
                        "pair_rmse": regression["rmse"],
                        "union_absolute_error": float(
                            abs(
                                deterministic.union_occlusion_mass[target_index]
                                - oracle.union_occlusion_mass[target_index]
                            )
                        ),
                        "oracle_union_mass": float(
                            oracle.union_occlusion_mass[target_index]
                        ),
                        "deterministic_union_mass": float(
                            deterministic.union_occlusion_mass[target_index]
                        ),
                        "oracle_represented_union_mass": float(
                            oracle.represented_union_mass[target_index]
                        ),
                        "oracle_unrepresented_union_mass": float(
                            oracle.unrepresented_union_mass[target_index]
                        ),
                        "oracle_fixed_union_mass": float(
                            oracle.fixed_environment_union_mass[target_index]
                        ),
                        "target_explainability": (
                            float(
                                oracle.represented_union_mass[target_index]
                                / oracle.union_occlusion_mass[target_index]
                            )
                            if oracle.union_occlusion_mass[target_index] > 0.0
                            else 1.0
                        ),
                        **metrics,
                        "oracle_values": oracle_values.tolist(),
                        "deterministic_values": predicted_values.tolist(),
                    }
                )

                source_indices = np.flatnonzero(valid_sources)
                for local_index, source_index in enumerate(source_indices):
                    pair_records.append(
                        {
                            "sample_id": sample_id,
                            "scene_group": group,
                            "camera_pose_bin": camera["pose_bin"],
                            "target_depth_bin": target_context["target_depth_bin"],
                            "target_height_bin": target_context["target_height_bin"],
                            "target_visible_fraction_bin": visible_bin,
                            "occlusion_layer_bin": layer_bin,
                            "match_quality_bin": _match_bin(
                                min(
                                    float(match_ious[source_index]),
                                    float(match_ious[target_index]),
                                )
                            ),
                            "same_parent": str(
                                sparse.parent_component_ids[source_index]
                                == sparse.parent_component_ids[target_index]
                            ).lower(),
                            "occupancy_epistemic_bin": target_context[
                                "occupancy_epistemic_bin"
                            ],
                            "semantic_vacuity_bin": target_context[
                                "semantic_vacuity_bin"
                            ],
                            "relation": relation_name,
                            "oracle_mass": float(oracle_values[local_index]),
                            "deterministic_mass": float(predicted_values[local_index]),
                        }
                    )

        runtime["total_measured_seconds"] = float(
            sum(value for key, value in runtime.items() if key.endswith("_seconds"))
        )
        runtime_records.append(runtime)

    return {
        "sample_id": sample_id,
        "scene_group": group,
        "paths": {
            "gt": str(gt_path),
            "gt_sha256": sha256_file(gt_path),
            "cnabu": str(cnabu_path),
            "cnabu_sha256": sha256_file(cnabu_path),
            "learned_nodes": str(node_path),
            "learned_nodes_sha256": sha256_file(node_path),
        },
        "load_and_support_seconds": load_seconds,
        "node_count": int(len(node_masks)),
        "node_source": node_source,
        "node_voxel_counts": list(sparse.voxel_counts),
        "node_parent_ids": list(sparse.parent_component_ids),
        "node_parent_match_fractions": list(sparse.parent_match_fractions),
        "matching": matching.to_dict(gt_instance_ids=gt_supports.instance_ids),
        "gt_coverage": gt_supports.coverage_summary(),
        "runtime_coverage": runtime_partition.coverage_summary(),
        "common_defined_target_queries": int(
            sum(len(values) for values in query_records.values())
            / max(len(query_records), 1)
        ),
        "query_records": query_records,
        "pair_records": pair_records,
        "attribution_records": attribution_records,
        "runtime_records": runtime_records,
        "naive_reference": naive_records,
        "target_contexts": target_contexts,
    }


def _aggregate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    query_records: Dict[str, List[Mapping[str, Any]]] = {
        "immediate": [],
        "potential": [],
    }
    pair_records: List[Mapping[str, Any]] = []
    attribution_records: Dict[str, List[Mapping[str, Any]]] = {
        "occupancy_epistemic": [],
        "semantic_vacuity": [],
        "total": [],
    }
    runtime_records: List[Mapping[str, Any]] = []
    naive_records: List[Mapping[str, Any]] = []
    for sample in samples:
        for relation in query_records:
            query_records[relation].extend(sample["query_records"][relation])
        pair_records.extend(sample["pair_records"])
        for component in attribution_records:
            attribution_records[component].extend(
                sample["attribution_records"][component]
            )
        runtime_records.extend(sample["runtime_records"])
        naive_records.extend(sample["naive_reference"])

    relation_output = {}
    for relation, records in query_records.items():
        relation_pairs = [
            record for record in pair_records if record["relation"] == relation
        ]
        oracle_flat = np.asarray(
            [record["oracle_mass"] for record in relation_pairs], dtype=np.float64
        )
        deterministic_flat = np.asarray(
            [record["deterministic_mass"] for record in relation_pairs],
            dtype=np.float64,
        )
        relation_output[relation] = {
            "pair_regression": pair_regression_metrics(
                oracle_flat, deterministic_flat
            ),
            "query_macro_ranking": _summarize_ranking_records(records),
            "scene_macro_ap": _scene_macro_ap(
                relation_pairs,
                oracle_key="oracle_mass",
                predicted_key="deterministic_mass",
            ),
            "union_mae": _optional_mean(
                record["union_absolute_error"] for record in records
            ),
            "mean_oracle_unrepresented_union_mass": _optional_mean(
                record["oracle_unrepresented_union_mass"] for record in records
            ),
            "mean_target_explainability": _optional_mean(
                record["target_explainability"] for record in records
            ),
            "continuity": summarize_label_continuity(
                [record["oracle_values"] for record in records]
            ),
            "stratified_query": _stratified_query_summary(
                records,
                keys=(
                    "camera_pose_bin",
                    "target_depth_bin",
                    "target_height_bin",
                    "target_visible_fraction_bin",
                    "occlusion_layer_bin",
                    "match_quality_bin",
                    "occupancy_epistemic_bin",
                    "semantic_vacuity_bin",
                    "scene_group",
                ),
            ),
            "stratified_pair": _pair_stratifications(
                relation_pairs,
                oracle_key="oracle_mass",
                predicted_key="deterministic_mass",
            ),
        }

    attribution_output = {}
    for component, records in attribution_records.items():
        oracle_flat = np.concatenate(
            [np.asarray(record["oracle_values"], dtype=np.float64) for record in records]
        ) if records else np.zeros(0, dtype=np.float64)
        deterministic_flat = np.concatenate(
            [
                np.asarray(record["deterministic_values"], dtype=np.float64)
                for record in records
            ]
        ) if records else np.zeros(0, dtype=np.float64)
        attribution_output[component] = {
            "source_regression": pair_regression_metrics(
                oracle_flat, deterministic_flat
            ),
            "query_macro_ranking": _summarize_ranking_records(records),
            "continuity": summarize_label_continuity(
                [record["oracle_values"] for record in records]
            ),
            "oracle_sum_mean": _optional_mean(record["oracle_sum"] for record in records),
            "deterministic_sum_mean": _optional_mean(
                record["deterministic_sum"] for record in records
            ),
            "stratified": _stratified_query_summary(
                records, keys=("camera_pose_bin", "scene_group")
            ),
        }

    runtime_keys = (
        "raycast_generation_seconds",
        "ray_conversion_seconds",
        "oracle_hard_seconds",
        "deterministic_hard_seconds",
        "oracle_attribution_seconds",
        "deterministic_attribution_seconds",
        "total_measured_seconds",
    )
    return {
        "successful_sample_count": int(len(samples)),
        "relation": relation_output,
        "source_attribution": attribution_output,
        "coverage": {
            "matched_node_fraction_mean": _optional_mean(
                sample["matching"]["matched_node_count"] / sample["node_count"]
                for sample in samples
            ),
            "gt_object_support_coverage_mean": _optional_mean(
                sample["gt_coverage"]["object_support_coverage"] for sample in samples
            ),
            "runtime_partition_coverage_mean": _optional_mean(
                sample["runtime_coverage"]["partition_coverage"] for sample in samples
            ),
            "common_defined_target_queries": int(
                sum(sample["common_defined_target_queries"] for sample in samples)
            ),
        },
        "runtime": {
            "camera_query_count": int(len(runtime_records)),
            "mean_seconds": {
                key: _optional_mean(record[key] for record in runtime_records)
                for key in runtime_keys
            },
            "median_seconds": {
                key: _optional_median(record[key] for record in runtime_records)
                for key in runtime_keys
            },
            "estimated_all_300_camera_seconds_from_mean": {
                key: (
                    300.0 * _optional_mean(record[key] for record in runtime_records)
                    if runtime_records
                    else None
                )
                for key in runtime_keys
            },
            "naive_reference": naive_records,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--split-role",
        choices=("train", "validation", "protected_test"),
        default="train",
    )
    parser.add_argument("--sample-ids", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-protected-test", action="store_true")
    parser.add_argument("--camera-indices", type=str, default="0")
    parser.add_argument("--ray-subsample", type=int, default=8)
    parser.add_argument("--top-z", type=int, default=3)
    parser.add_argument("--source-batch-size", type=int, default=4)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.25)
    parser.add_argument("--include-naive-reference", action="store_true")
    parser.add_argument("--skip-source-attribution", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cnabu-root", type=Path, default=DEFAULT_CNABU_ROOT)
    parser.add_argument("--node-root", type=Path, default=DEFAULT_NODE_ROOT)
    parser.add_argument("--camera-path", type=Path, default=DEFAULT_CAMERA_PATH)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))
    split_manifest_path = args.split_manifest.resolve()
    camera_path = args.camera_path.resolve()
    for path in (split_manifest_path, camera_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if int(args.ray_subsample) <= 0 or int(args.top_z) <= 0:
        raise ValueError("ray subsample and top-Z must be positive")
    sample_ids, split_manifest = _load_sample_ids(
        split_manifest_path,
        split_role=str(args.split_role),
        explicit_ids=args.sample_ids,
        max_samples=args.max_samples,
        allow_protected_test=bool(args.allow_protected_test),
    )
    if bool(args.include_naive_reference) and (
        len(sample_ids) != 1
        or len(_parse_indices(args.camera_indices, upper_bound=300)) != 1
    ):
        raise ValueError("naive reference is restricted to one sample and one camera")
    if bool(args.include_naive_reference) and bool(args.skip_source_attribution):
        raise ValueError("naive reference cannot be combined with --skip-source-attribution")
    started = time.perf_counter()
    initialization_started = time.perf_counter()
    info_gain = InfoGainEval(
        str(camera_path),
        subsample=int(args.ray_subsample),
        occupancy_thold=0.95,
        cached=False,
    )
    camera_count = int(len(info_gain.camera_matrices))
    camera_indices = _parse_indices(args.camera_indices, upper_bound=camera_count)
    camera_contexts = _camera_contexts(info_gain)
    initialization_seconds = float(time.perf_counter() - initialization_started)

    samples = []
    failures = []
    for sample_id in sample_ids:
        try:
            samples.append(
                _run_sample(
                    sample_id=sample_id,
                    camera_indices=camera_indices,
                    info_gain=info_gain,
                    camera_contexts=camera_contexts,
                    raw_root=args.raw_root.resolve(),
                    cnabu_root=args.cnabu_root.resolve(),
                    node_root=args.node_root.resolve(),
                    top_z=int(args.top_z),
                    source_batch_size=int(args.source_batch_size),
                    occupancy_threshold=float(args.occupancy_threshold),
                    match_iou_threshold=float(args.match_iou_threshold),
                    include_naive_reference=bool(args.include_naive_reference),
                    skip_source_attribution=bool(args.skip_source_attribution),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if not samples:
        raise RuntimeError("all selected samples failed: {}".format(failures))

    # Retain per-sample provenance/coverage/runtime, while keeping pair/query
    # arrays out of the durable report after their stratified aggregates exist.
    aggregate = _aggregate(samples)
    for relation in aggregate["relation"].values():
        relation["continuity"].pop("per_query", None)
    for component in aggregate["source_attribution"].values():
        component["continuity"].pop("per_query", None)
    for sample in samples:
        sample["record_counts"] = {
            "immediate_queries": len(sample["query_records"]["immediate"]),
            "potential_queries": len(sample["query_records"]["potential"]),
            "pair_records": len(sample["pair_records"]),
            "attribution_camera_queries": len(
                sample["attribution_records"]["total"]
            ),
        }
        sample.pop("pair_records")
        sample.pop("query_records")
        sample.pop("attribution_records")

    report = {
        "schema": "cnabu_belief_occlusion_headroom_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "repositories": {
            "manipulation_enhanced_map_prediction": _git_state(REPO_ROOT),
            "scene_graph_mem": _git_state(SCENE_GRAPH_ROOT),
        },
        "split": {
            "role": str(args.split_role),
            "manifest_path": str(split_manifest_path),
            "manifest_sha256": sha256_file(split_manifest_path),
            "split_content_sha256": split_manifest.get("split_content_sha256"),
            "protected_outcomes_read": bool(args.split_role == "protected_test"),
            "selected_sample_ids": list(sample_ids),
        },
        "camera": {
            "path": str(camera_path),
            "sha256": sha256_file(camera_path),
            "executable_camera_count": camera_count,
            "selected_camera_indices": list(camera_indices),
            "ray_subsample": int(args.ray_subsample),
            "selected_contexts": {
                str(index): camera_contexts[index] for index in camera_indices
            },
        },
        "contract": {
            "relation_schema": "cnabu_visual_occlusion_v1",
            "top_z": int(args.top_z),
            "occupancy_threshold": float(args.occupancy_threshold),
            "match_iou_threshold": float(args.match_iou_threshold),
            "lambda_occ": 1.0,
            "lambda_sem": 1.0,
            "occupancy_epistemic_scale": 1.0 / 12.0,
            "normalization": "none",
            "source_attribution_computed": not bool(args.skip_source_attribution),
        },
        "roots": {
            "raw": str(args.raw_root.resolve()),
            "cnabu": str(args.cnabu_root.resolve()),
            "learned_nodes": str(args.node_root.resolve()),
        },
        "timing": {
            "initialization_seconds": initialization_seconds,
            "total_seconds": float(time.perf_counter() - started),
        },
        "memory": {
            "process_max_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
        },
        "aggregate": aggregate,
        "samples": samples,
        "failures": failures,
        "safety": {
            "training_run": False,
            "checkpoint_written": False,
            "dataset_export_written": False,
            "gt_used_only_for_offline_oracle_and_evaluation": True,
            "deterministic_runtime_inputs_use_gt": False,
            "protected_test_outcomes_read": bool(args.split_role == "protected_test"),
            "physical_relation_assets_or_records_loaded": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sample_count": len(samples),
                "failure_count": len(failures),
                "camera_count": len(camera_indices),
                "headline": {
                    "immediate_pair_mae": aggregate["relation"]["immediate"][
                        "pair_regression"
                    ]["mae"],
                    "immediate_ndcg_at_3": aggregate["relation"]["immediate"][
                        "query_macro_ranking"
                    ]["ndcg_at_3"],
                    "potential_pair_mae": aggregate["relation"]["potential"][
                        "pair_regression"
                    ]["mae"],
                    "source_total_ndcg_at_3": aggregate["source_attribution"][
                        "total"
                    ]["query_macro_ranking"]["ndcg_at_3"],
                    "mean_deterministic_attribution_seconds": aggregate["runtime"][
                        "mean_seconds"
                    ]["deterministic_attribution_seconds"],
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
