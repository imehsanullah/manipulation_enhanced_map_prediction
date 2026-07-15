"""PyBullet adapter for action-conditioned scene-graph relation targets.

The adapter replays saved MEM scenes and supplies direct robot/object collision
observations to the portable scoring contract in ``scene_graph_mem``.  It does
not train a model or depend on Detectron2.
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scene_graph_mem.relations.action_conditioned_oracle import (
    ACTION_CONDITIONED_TARGET_METHOD_V1,
    build_action_conditioned_oracle_record,
    build_counterfactual_validation_record,
    evaluate_collision_trajectory,
)
from scene_graph_mem.relations.candidate_trajectory import (
    TRAJECTORY_CANDIDATE_IDS,
    TRAJECTORY_PLANNER_SWEPT_CLEARANCE_NORMALIZATION_M,
    TRAJECTORY_PLANNER_SWEPT_PAIR_FEATURE_NAMES,
    TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT,
)


@dataclass(frozen=True)
class OracleActionFamilyConfig:
    """Frontal grid action family for low-y shelf extraction."""

    opening_y: float = 0.68
    pregrasp_clearance_m: float = 0.05
    grasp_penetration_m: float = 0.015
    lift_distance_m: float = 0.01
    candidate_lateral_fractions: Tuple[float, ...] = (0.35, 0.50, 0.65)
    candidate_height_fractions: Tuple[float, ...] = (0.35, 0.55, 0.75)
    approach_samples: int = 16
    grasp_samples: int = 6
    lift_samples: int = 6
    extraction_samples: int = 12
    collision_distance_m: float = 0.0
    hard_penetration_m: float = 0.002
    max_ik_position_residual_m: float = 0.015
    # Frozen from the 10-candidate/30-trial clean-extraction validation pilot.
    binary_threshold: float = 0.4


@dataclass(frozen=True)
class CounterfactualRandomizationConfig:
    """Small deterministic perturbations shared by an intact/removal pair."""

    seed: int = 0
    xy_position_jitter_m: float = 0.001
    yaw_jitter_degrees: float = 0.5
    friction_scale_min: float = 0.9
    friction_scale_max: float = 1.1


@dataclass(frozen=True)
class StaticOraclePerturbationConfig:
    """Pose/yaw-only perturbation for relation-score stability audits."""

    seed: int = 0
    xy_position_jitter_m: float = 0.001
    yaw_jitter_degrees: float = 0.5


def merge_instance_stack(instance_stack: Any, background_value: int = -1) -> np.ndarray:
    stack = np.asarray(instance_stack)
    if stack.ndim == 2:
        return stack.copy()
    if stack.ndim != 3:
        raise ValueError("instance_maps must have shape [H,W] or [V,H,W]")
    merged = np.zeros_like(stack[0])
    for layer in stack:
        np.copyto(merged, layer, where=layer != background_value)
    return merged


def extract_gt_object_records(gt_hms_path: Path | str, object_class_max_exclusive: int = 14) -> List[Dict[str, Any]]:
    """Extract exact GT object masks and ordering information from one saved scene."""

    path = Path(gt_hms_path)
    with np.load(path, allow_pickle=False) as data:
        if "instance_maps" not in data.files or "semantic_2d" not in data.files:
            raise KeyError("{} must contain instance_maps and semantic_2d".format(path))
        merged = merge_instance_stack(data["instance_maps"])
        semantics = np.asarray(data["semantic_2d"])
    if merged.shape != semantics.shape:
        raise ValueError("merged instance map and semantic_2d must have the same shape")

    records: List[Dict[str, Any]] = []
    for raw_instance_id in np.unique(merged):
        instance_id = int(raw_instance_id)
        if instance_id in (-1, 0):
            continue
        mask = merged == raw_instance_id
        semantic_values, semantic_counts = np.unique(semantics[mask], return_counts=True)
        class_id = int(semantic_values[int(np.argmax(semantic_counts))])
        if class_id < 0 or class_id >= int(object_class_max_exclusive):
            continue
        ys, xs = np.nonzero(mask)
        records.append(
            {
                "instance_id": instance_id,
                "semantic_class_id": class_id,
                "pixel_count": int(mask.sum()),
                "bbox_yx_minmax": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
                "centroid_yx": [float(ys.mean()), float(xs.mean())],
            }
        )
    records.sort(key=lambda item: int(item["instance_id"]))
    return records


def front_extraction_waypoints(
    *,
    world_aabb: Sequence[Sequence[float]],
    height_fraction: float,
    config: OracleActionFamilyConfig,
    lateral_fraction: float = 0.5,
) -> Dict[str, List[float]]:
    """Create one frontal grid grasp followed by lift and low-y extraction."""

    lower = np.asarray(world_aabb[0], dtype=np.float64)
    upper = np.asarray(world_aabb[1], dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or bool(np.any(upper <= lower)):
        raise ValueError("world_aabb must contain increasing 3D lower/upper corners")
    fraction = float(height_fraction)
    lateral = float(lateral_fraction)
    if fraction <= 0.0 or fraction >= 1.0 or lateral <= 0.0 or lateral >= 1.0:
        raise ValueError("lateral_fraction and height_fraction must be in (0,1)")

    grasp_x = float(lower[0] + lateral * (upper[0] - lower[0]))
    grasp_z = float(lower[2] + fraction * (upper[2] - lower[2]))
    pregrasp_y = float(lower[1] - config.pregrasp_clearance_m)
    grasp_y = float(lower[1] + min(config.grasp_penetration_m, 0.25 * (upper[1] - lower[1])))
    extraction_y = float(min(config.opening_y, pregrasp_y - 0.02))
    lifted_z = float(grasp_z + config.lift_distance_m)
    return {
        "pregrasp": [grasp_x, pregrasp_y, grasp_z],
        "grasp": [grasp_x, grasp_y, grasp_z],
        "lift": [grasp_x, grasp_y, lifted_z],
        "extraction": [grasp_x, extraction_y, lifted_z],
    }


def interpolate_joint_configs(
    start: Sequence[float],
    stop: Sequence[float],
    count: int,
    *,
    include_start: bool = True,
) -> List[np.ndarray]:
    start_array = np.asarray(start, dtype=np.float64)
    stop_array = np.asarray(stop, dtype=np.float64)
    if start_array.shape != stop_array.shape or start_array.ndim != 1:
        raise ValueError("joint interpolation endpoints must be equal-length vectors")
    if int(count) < 2:
        raise ValueError("joint interpolation requires at least two samples")
    samples = list(np.linspace(start_array, stop_array, int(count)))
    return samples if include_start else samples[1:]


def build_geometry_pseudo_gt_adjacency(object_records: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    """Reproduce the retained low-y, positive-x-overlap geometry baseline."""

    count = len(object_records)
    adjacency = [[0 for _ in range(count)] for _ in range(count)]
    for source_index, source in enumerate(object_records):
        source_y = float(source["centroid_yx"][0])
        source_box = [int(value) for value in source["bbox_yx_minmax"]]
        for target_index, target in enumerate(object_records):
            if source_index == target_index or source_y >= float(target["centroid_yx"][0]):
                continue
            target_box = [int(value) for value in target["bbox_yx_minmax"]]
            overlap = min(source_box[3], target_box[3]) - max(source_box[1], target_box[1]) + 1
            if overlap > 0:
                adjacency[source_index][target_index] = 1
    return adjacency


def compare_relation_targets(
    *,
    node_order: Sequence[int],
    geometry_adjacency: Sequence[Sequence[int]],
    action_adjacency: Sequence[Sequence[int]],
    action_score_valid_mask: Sequence[Sequence[bool]],
) -> Dict[str, Any]:
    count = len(node_order)
    geometry = np.asarray(geometry_adjacency, dtype=np.int64)
    action = np.asarray(action_adjacency, dtype=np.int64)
    valid = np.asarray(action_score_valid_mask, dtype=bool)
    if geometry.shape != (count, count) or action.shape != (count, count) or valid.shape != (count, count):
        raise ValueError("relation matrices must match node_order")

    both: List[List[int]] = []
    geometry_only: List[List[int]] = []
    action_only: List[List[int]] = []
    neither = 0
    undefined = 0
    for source_index, source_id in enumerate(node_order):
        for target_index, target_id in enumerate(node_order):
            if source_index == target_index:
                continue
            if not bool(valid[source_index, target_index]):
                undefined += 1
                continue
            pair = [int(source_id), int(target_id)]
            if geometry[source_index, target_index] and action[source_index, target_index]:
                both.append(pair)
            elif geometry[source_index, target_index]:
                geometry_only.append(pair)
            elif action[source_index, target_index]:
                action_only.append(pair)
            else:
                neither += 1
    comparable = len(both) + len(geometry_only) + len(action_only) + neither
    return {
        "comparable_pair_count": comparable,
        "undefined_action_pair_count": undefined,
        "both_positive_pairs": both,
        "geometry_only_pairs": geometry_only,
        "action_only_pairs": action_only,
        "both_negative_pair_count": neither,
        "agreement_count": len(both) + neither,
        "agreement_rate": float((len(both) + neither) / comparable) if comparable else None,
    }


def _reset_arm_config(env: Any, arm_config: Sequence[float]) -> None:
    for link_index, value in zip(env.arm_joint_indices, arm_config):
        env._p.resetJointState(env.robot_id, int(link_index), float(value), physicsClientId=env.client_id)


def _solve_ik_position(
    env: Any,
    position: Sequence[float],
    seed_config: Sequence[float],
    max_residual: float,
) -> Tuple[Optional[np.ndarray], float]:
    _reset_arm_config(env, seed_config)
    joints = np.asarray(env.get_ik_joints(position, env.init_ori, link=env.tool_tip_id), dtype=np.float64)
    if joints.shape != (6,) or not bool(np.isfinite(joints).all()):
        return None, float("inf")
    _reset_arm_config(env, joints)
    actual = np.asarray(
        env._p.getLinkState(env.robot_id, env.tool_tip_id, physicsClientId=env.client_id)[0],
        dtype=np.float64,
    )
    residual = float(np.linalg.norm(actual - np.asarray(position, dtype=np.float64)))
    if not math.isfinite(residual) or residual > float(max_residual):
        return None, residual
    return joints, residual


def _host_xyz(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not bool(np.isfinite(array).all()):
        raise ValueError("map/world conversion must produce one finite xyz point")
    return array


def cnabu_sparse_support_world_aabbs(
    heightmap_generation: Any,
    indices_zyx: Sequence[Any],
    *,
    crop_rows: Sequence[int],
    boundary_quantile: float = 0.0,
) -> List[List[List[float]]]:
    """Convert ordered CNABU sparse support to runtime world AABBs.

    CNABU support is ``[z, cropped-y, image-x]``. The MEM/D3G export's image-x
    axis is mirrored relative to PyBullet world x, while raw-y keeps its
    direction. ``boundary_quantile`` can trim a symmetric, low-mass support
    tail before forming the box so a single uncertain boundary voxel does not
    become exact collision geometry. This conversion uses the mapper bounds
    and calibration only; it does not consume GT masks, simulator instance
    ids, or object poses.
    """

    rows = tuple(int(value) for value in crop_rows)
    if len(rows) != 2 or rows[1] <= rows[0]:
        raise ValueError("crop_rows must contain increasing raw-map bounds")
    quantile = float(boundary_quantile)
    if not math.isfinite(quantile) or quantile < 0.0 or quantile > 0.25:
        raise ValueError("boundary_quantile must be finite and in [0, 0.25]")
    result: List[List[List[float]]] = []
    for raw_indices in indices_zyx:
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices):
            raise ValueError("each CNABU node support must be non-empty [K,3] zyx indices")
        minimum = np.asarray(
            np.quantile(indices, quantile, axis=0, method="lower"),
            dtype=np.int64,
        )
        maximum_exclusive = (
            np.asarray(
                np.quantile(indices, 1.0 - quantile, axis=0, method="higher"),
                dtype=np.int64,
            )
            + 1
        )
        lower_map_xyz = np.asarray(
            [minimum[2], minimum[1] + rows[0], minimum[0]], dtype=np.float64
        )
        upper_map_xyz = np.asarray(
            [
                maximum_exclusive[2],
                maximum_exclusive[1] + rows[0],
                maximum_exclusive[0],
            ],
            dtype=np.float64,
        )
        unmirrored_lower_world = _host_xyz(
            heightmap_generation.map_point_to_world_point(lower_map_xyz)
        )
        unmirrored_upper_world = _host_xyz(
            heightmap_generation.map_point_to_world_point(upper_map_xyz)
        )
        bounds = np.asarray(getattr(heightmap_generation, "bounds", None), dtype=np.float64)
        if bounds.shape != (3, 2) or not bool(np.isfinite(bounds).all()):
            raise ValueError("heightmap generation must expose finite [3,2] world bounds")
        x_reflection_sum = float(bounds[0, 0] + bounds[0, 1])
        lower_world = unmirrored_lower_world.copy()
        upper_world = unmirrored_upper_world.copy()
        lower_world[0] = x_reflection_sum - unmirrored_upper_world[0]
        upper_world[0] = x_reflection_sum - unmirrored_lower_world[0]
        if bool(np.any(upper_world <= lower_world)):
            raise ValueError("CNABU support conversion produced a non-increasing world AABB")
        result.append([lower_world.tolist(), upper_world.tolist()])
    return result


def cnabu_sparse_support_world_voxels(
    heightmap_generation: Any,
    indices_zyx: Sequence[Any],
    *,
    crop_rows: Sequence[int],
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Convert ordered CNABU voxels to world centers and one cell half extent.

    The conversion follows the same mirrored image-x convention as
    :func:`cnabu_sparse_support_world_aabbs`. It uses only the affine MEM map
    calibration and ordered CNABU support; no simulator object state enters.
    """

    rows = tuple(int(value) for value in crop_rows)
    if len(rows) != 2 or rows[1] <= rows[0]:
        raise ValueError("crop_rows must contain increasing raw-map bounds")
    origin = _host_xyz(
        heightmap_generation.map_point_to_world_point(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        )
    )
    basis = np.stack(
        [
            _host_xyz(
                heightmap_generation.map_point_to_world_point(
                    np.eye(3, dtype=np.float64)[axis]
                )
            )
            - origin
            for axis in range(3)
        ],
        axis=1,
    )
    if not bool(np.isfinite(basis).all()) or abs(float(np.linalg.det(basis))) <= 0.0:
        raise ValueError("heightmap calibration must be a finite invertible affine map")
    half_extents = 0.5 * np.sum(np.abs(basis), axis=1)
    if bool(np.any(half_extents <= 0.0)):
        raise ValueError("heightmap calibration must produce positive voxel extents")
    bounds = np.asarray(getattr(heightmap_generation, "bounds", None), dtype=np.float64)
    if bounds.shape != (3, 2) or not bool(np.isfinite(bounds).all()):
        raise ValueError("heightmap generation must expose finite [3,2] world bounds")
    x_reflection_sum = float(bounds[0, 0] + bounds[0, 1])

    result: List[np.ndarray] = []
    for raw_indices in indices_zyx:
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices):
            raise ValueError("each CNABU node support must be non-empty [K,3] zyx indices")
        map_centers = np.stack(
            (
                indices[:, 2].astype(np.float64) + 0.5,
                indices[:, 1].astype(np.float64) + float(rows[0]) + 0.5,
                indices[:, 0].astype(np.float64) + 0.5,
            ),
            axis=1,
        )
        centers = origin[None, :] + map_centers @ basis.T
        centers[:, 0] = x_reflection_sum - centers[:, 0]
        if not bool(np.isfinite(centers).all()):
            raise ValueError("CNABU support conversion produced non-finite world centers")
        result.append(centers.astype(np.float64, copy=False))
    return tuple(result), half_extents.astype(np.float64, copy=False)


def _first_contact_progress(values: Sequence[float]) -> float:
    for index, value in enumerate(values):
        if float(value) > 0.0:
            return float((index + 1) / max(len(values), 1))
    return 0.0


def _longest_contact_run_fraction(values: Sequence[float]) -> float:
    longest = 0
    current = 0
    for value in values:
        if float(value) > 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest / max(len(values), 1))


def _source_swept_aabb_summary(
    source_world_aabb: Sequence[Sequence[float]],
    aabb_bins: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> Tuple[float, float, float, float, float]:
    if len(aabb_bins) != TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT:
        raise ValueError("planner swept geometry must use four progress bins")
    source = np.asarray(source_world_aabb, dtype=np.float64)
    if (
        source.shape != (2, 3)
        or not bool(np.isfinite(source).all())
        or bool(np.any(source[1] <= source[0]))
    ):
        raise ValueError("source_world_aabb must be finite and increasing [2,3]")
    source_volume = float(np.prod(source[1] - source[0]))

    bin_minimum_distances: List[float] = []
    bin_overlap_fractions: List[float] = []
    for raw_boxes in aabb_bins:
        boxes = np.asarray(raw_boxes, dtype=np.float64)
        if boxes.size == 0:
            minimum_distance = float("inf")
            overlap_fraction = 0.0
        else:
            if boxes.ndim != 3 or tuple(boxes.shape[1:]) != (2, 3):
                raise ValueError("each planner progress bin must contain [B,2,3] AABBs")
            if not bool(np.isfinite(boxes).all()) or bool(
                np.any(boxes[:, 1] <= boxes[:, 0])
            ):
                raise ValueError("planner swept AABBs must be finite and increasing")
            axis_gap = np.maximum(
                np.maximum(
                    boxes[:, 0, :] - source[1][None, :],
                    source[0][None, :] - boxes[:, 1, :],
                ),
                0.0,
            )
            minimum_distance = float(np.linalg.norm(axis_gap, axis=1).min())
            intersection_extent = np.maximum(
                np.minimum(source[1][None, :], boxes[:, 1, :])
                - np.maximum(source[0][None, :], boxes[:, 0, :]),
                0.0,
            )
            overlap_fraction = min(
                float(np.prod(intersection_extent, axis=1).sum() / source_volume),
                1.0,
            )
        bin_minimum_distances.append(minimum_distance)
        bin_overlap_fractions.append(overlap_fraction)

    finite_minimum = [
        value for value in bin_minimum_distances if math.isfinite(value)
    ]
    minimum_clearance = min(finite_minimum) if finite_minimum else float("inf")
    clearance_norm = (
        1.0
        if not math.isfinite(minimum_clearance)
        else min(
            minimum_clearance
            / TRAJECTORY_PLANNER_SWEPT_CLEARANCE_NORMALIZATION_M,
            1.0,
        )
    )
    return (
        min(float(sum(bin_overlap_fractions)), 1.0),
        _first_contact_progress(bin_overlap_fractions),
        _longest_contact_run_fraction(bin_overlap_fractions),
        max(bin_overlap_fractions, default=0.0),
        float(clearance_norm),
    )


def build_candidate_planner_swept_features(
    *,
    source_world_aabbs: Sequence[Sequence[Sequence[float]]],
    node_ids: Sequence[Any],
    targets: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize solved robot-link/carried-target sweeps against CNABU support."""

    source_aabbs = tuple(
        np.asarray(value, dtype=np.float64) for value in source_world_aabbs
    )
    ordered_node_ids = list(node_ids)
    if len(source_aabbs) != len(ordered_node_ids):
        raise ValueError("source_world_aabbs must align with node_ids")
    ordered_targets = list(targets)
    if [target.get("node_id") for target in ordered_targets] != ordered_node_ids:
        raise ValueError("planner swept targets must preserve CNABU node order")
    n = len(source_aabbs)
    features = np.zeros(
        (
            n,
            n,
            len(TRAJECTORY_CANDIDATE_IDS),
            len(TRAJECTORY_PLANNER_SWEPT_PAIR_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    clearance_indices = tuple(
        index
        for index, name in enumerate(TRAJECTORY_PLANNER_SWEPT_PAIR_FEATURE_NAMES)
        if name.endswith("_minimum_clearance_norm")
    )
    for source_index in range(n):
        for target_index in range(n):
            if source_index != target_index:
                features[source_index, target_index, :, clearance_indices] = 1.0
    component_routes = (
        ("approach", "robot_link_aabbs_by_stage"),
        ("grasp", "robot_link_aabbs_by_stage"),
        ("extraction", "robot_link_aabbs_by_stage"),
        ("extraction", "carried_target_aabbs_by_stage"),
    )
    for target_index, target in enumerate(ordered_targets):
        candidates = list(target.get("candidates") or [])
        observed_order = tuple(str(value.get("candidate_id")) for value in candidates)
        if observed_order != TRAJECTORY_CANDIDATE_IDS:
            raise ValueError("planner swept candidates must use the frozen 3x3 order")
        for candidate_index, candidate in enumerate(candidates):
            if not bool(candidate.get("kinematically_feasible", False)):
                continue
            geometry = dict(candidate.get("planner_swept_geometry") or {})
            if int(geometry.get("progress_bin_count", -1)) != (
                TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT
            ):
                raise ValueError("feasible planner candidate is missing four-bin geometry")
            for source_index, source_world_aabb in enumerate(source_aabbs):
                if source_index == target_index:
                    continue
                values: List[float] = []
                for stage, route in component_routes:
                    stage_geometry = dict(geometry.get(route) or {})
                    raw_bins = stage_geometry.get(stage)
                    if raw_bins is None:
                        raw_bins = [
                            []
                            for _ in range(
                                TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT
                            )
                        ]
                    values.extend(
                        _source_swept_aabb_summary(
                            source_world_aabb,
                            raw_bins,
                        )
                    )
                features[
                    source_index, target_index, candidate_index
                ] = np.asarray(values, dtype=np.float32)

    return {
        "schema": "cnabu_runtime_candidate_planner_swept_features_v1",
        "candidate_ids": list(TRAJECTORY_CANDIDATE_IDS),
        "node_ids": ordered_node_ids,
        "pair_feature_names": list(TRAJECTORY_PLANNER_SWEPT_PAIR_FEATURE_NAMES),
        "pair_features": features,
        "progress_bin_count": TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT,
        "clearance_normalization_m": (
            TRAJECTORY_PLANNER_SWEPT_CLEARANCE_NORMALIZATION_M
        ),
        "safety": {
            "uses_gt_or_simulator_instance_ids": False,
            "uses_simulator_object_poses": False,
            "queries_dynamic_scene_objects": False,
            "uses_current_robot_state": True,
            "uses_cnabu_ordered_node_geometry": True,
            "uses_robot_link_geometry": True,
        },
    }


def build_runtime_candidate_kinematic_mask(
    env: Any,
    *,
    target_world_aabbs: Sequence[Sequence[Sequence[float]]],
    node_ids: Optional[Sequence[Any]] = None,
    initial_arm_config: Optional[Sequence[float]] = None,
    config: Optional[OracleActionFamilyConfig] = None,
) -> Dict[str, Any]:
    """Solve the frozen candidate waypoints and return an ordered runtime mask.

    This is the action-generator side of the factorized scene-graph contract.
    It checks only deterministic IK availability from the current robot state;
    fixed-environment collision remains the CNABU relation head's learned term.
    """

    cfg = config or OracleActionFamilyConfig()
    candidate_specs = tuple(
        (float(lateral), float(height))
        for lateral in cfg.candidate_lateral_fractions
        for height in cfg.candidate_height_fractions
    )
    candidate_ids = tuple(
        "front_x{:.2f}_z{:.2f}".format(lateral, height)
        for lateral, height in candidate_specs
    )
    if candidate_ids != TRAJECTORY_CANDIDATE_IDS:
        raise ValueError("runtime action family must match the frozen scene-graph candidates")
    aabbs = [
        np.asarray(world_aabb, dtype=np.float64)
        for world_aabb in target_world_aabbs
    ]
    if any(aabb.shape != (2, 3) or bool(np.any(aabb[1] <= aabb[0])) for aabb in aabbs):
        raise ValueError("target_world_aabbs must contain increasing [2,3] boxes")
    ordered_node_ids = list(range(len(aabbs))) if node_ids is None else list(node_ids)
    if len(ordered_node_ids) != len(aabbs):
        raise ValueError("node_ids must align with target_world_aabbs")
    if initial_arm_config is None:
        states = env._p.getJointStates(
            env.robot_id,
            list(env.arm_joint_indices),
            physicsClientId=env.client_id,
        )
        initial = np.asarray([state[0] for state in states], dtype=np.float64)
    else:
        initial = np.asarray(initial_arm_config, dtype=np.float64)
    if initial.shape != (len(env.arm_joint_indices),) or not bool(np.isfinite(initial).all()):
        raise ValueError("initial_arm_config must align with finite arm joints")

    mask = np.zeros((len(aabbs), len(candidate_ids)), dtype=bool)
    target_diagnostics: List[Dict[str, Any]] = []
    try:
        for target_index, (node_id, world_aabb) in enumerate(zip(ordered_node_ids, aabbs)):
            candidate_diagnostics: List[Dict[str, Any]] = []
            for candidate_index, (lateral, height) in enumerate(candidate_specs):
                waypoints = front_extraction_waypoints(
                    world_aabb=world_aabb,
                    lateral_fraction=lateral,
                    height_fraction=height,
                    config=cfg,
                )
                seed = initial.copy()
                solved: Dict[str, np.ndarray] = {}
                residuals: Dict[str, float] = {}
                failed_waypoint: Optional[str] = None
                for waypoint_name in ("pregrasp", "grasp", "lift", "extraction"):
                    joint_config, residual = _solve_ik_position(
                        env,
                        waypoints[waypoint_name],
                        seed,
                        cfg.max_ik_position_residual_m,
                    )
                    residuals[waypoint_name] = float(residual)
                    if joint_config is None:
                        failed_waypoint = waypoint_name
                        break
                    solved[waypoint_name] = joint_config
                    seed = joint_config
                mask[target_index, candidate_index] = failed_waypoint is None
                stage_configs: Dict[str, List[List[float]]] = {}
                if failed_waypoint is None:
                    stage_configs = {
                        "approach": [
                            config.tolist()
                            for config in interpolate_joint_configs(
                                initial,
                                solved["pregrasp"],
                                cfg.approach_samples,
                            )
                        ],
                        "grasp": [
                            config.tolist()
                            for config in interpolate_joint_configs(
                                solved["pregrasp"],
                                solved["grasp"],
                                cfg.grasp_samples,
                            )
                        ],
                        "extraction": [
                            config.tolist()
                            for config in (
                                interpolate_joint_configs(
                                    solved["grasp"],
                                    solved["lift"],
                                    cfg.lift_samples,
                                )
                                + interpolate_joint_configs(
                                    solved["lift"],
                                    solved["extraction"],
                                    cfg.extraction_samples,
                                    include_start=False,
                                )
                            )
                        ],
                    }
                candidate_diagnostics.append(
                    {
                        "candidate_id": candidate_ids[candidate_index],
                        "kinematically_feasible": bool(failed_waypoint is None),
                        "failed_waypoint": failed_waypoint,
                        "ik_position_residuals_m": residuals,
                        "waypoints": waypoints,
                        "stage_joint_configs": stage_configs,
                    }
                )
                _reset_arm_config(env, initial)
            target_diagnostics.append(
                {
                    "node_id": node_id,
                    "world_aabb": world_aabb.tolist(),
                    "candidates": candidate_diagnostics,
                }
            )
    finally:
        _reset_arm_config(env, initial)

    return {
        "schema": "cnabu_runtime_candidate_kinematic_mask_v0",
        "candidate_ids": list(candidate_ids),
        "node_ids": ordered_node_ids,
        "kinematic_mask": mask.tolist(),
        "targets": target_diagnostics,
        "source": "deterministic_current_robot_state_pybullet_ik",
        "safety": {
            "uses_gt_or_simulator_instance_ids": False,
            "uses_cnabu_ordered_node_geometry": True,
            "runs_collision_or_contact_queries": False,
        },
    }


def _runtime_initial_arm_config(
    env: Any,
    initial_arm_config: Optional[Sequence[float]],
) -> np.ndarray:
    if initial_arm_config is None:
        states = env._p.getJointStates(
            env.robot_id,
            list(env.arm_joint_indices),
            physicsClientId=env.client_id,
        )
        initial = np.asarray([state[0] for state in states], dtype=np.float64)
    else:
        initial = np.asarray(initial_arm_config, dtype=np.float64)
    if initial.shape != (len(env.arm_joint_indices),) or not bool(np.isfinite(initial).all()):
        raise ValueError("initial_arm_config must align with finite arm joints")
    return initial


def _runtime_fixed_environment_candidate_collision(
    env: Any,
    *,
    target_world_aabb: Sequence[Sequence[float]],
    stage_joint_configs: Mapping[str, Sequence[Sequence[float]]],
    fixed_body_ids: Mapping[str, int],
    config: OracleActionFamilyConfig,
    capture_planner_swept_geometry: bool = False,
) -> Dict[str, Any]:
    """Check one solved candidate against fixed bodies using a CNABU box proxy."""

    aabb = np.asarray(target_world_aabb, dtype=np.float64)
    half_extents = 0.5 * (aabb[1] - aabb[0])
    initial_center = 0.5 * (aabb[0] + aabb[1])
    if bool(np.any(half_extents <= 0.0)):
        raise ValueError("target_world_aabb must have positive extents")
    collision_shape_id = env._p.createCollisionShape(
        env._p.GEOM_BOX,
        halfExtents=half_extents.tolist(),
        physicsClientId=env.client_id,
    )
    proxy_body_id = env._p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=int(collision_shape_id),
        basePosition=initial_center.tolist(),
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=env.client_id,
    )
    allowed_links = _collision_link_indices(env)
    grasp_link_id = int(getattr(env, "grasp_link_id", env.tool_tip_id))
    grasp_configs = list(stage_joint_configs.get("grasp") or [])
    if not grasp_configs:
        env._p.removeBody(int(proxy_body_id), physicsClientId=env.client_id)
        raise ValueError("fixed-environment checking requires a solved grasp stage")
    _reset_arm_config(env, grasp_configs[-1])
    grasp_link_position = np.asarray(
        env._p.getLinkState(
            env.robot_id,
            grasp_link_id,
            physicsClientId=env.client_id,
        )[0],
        dtype=np.float64,
    )
    stage_evidence: Dict[str, Any] = {}
    robot_link_aabbs_by_stage: Dict[str, List[List[List[List[float]]]]] = {}
    carried_target_aabbs_by_stage: Dict[str, List[List[List[List[float]]]]] = {}
    hard_collision = False
    try:
        for stage in ("approach", "grasp", "extraction"):
            configs = list(stage_joint_configs.get(stage) or [])
            robot_bin_bounds: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = [
                {} for _ in range(TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT)
            ]
            carried_bin_bounds: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
                None for _ in range(TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT)
            ]
            hard_sample_count = 0
            contact_sample_count = 0
            body_evidence: Dict[str, Dict[str, Any]] = {}
            for sample_index, joint_config in enumerate(configs):
                _reset_arm_config(env, joint_config)
                progress_bin = min(
                    int(
                        sample_index
                        * TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT
                        / max(len(configs), 1)
                    ),
                    TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT - 1,
                )
                if stage == "extraction":
                    link_position = np.asarray(
                        env._p.getLinkState(
                            env.robot_id,
                            grasp_link_id,
                            physicsClientId=env.client_id,
                        )[0],
                        dtype=np.float64,
                    )
                    proxy_center = initial_center + (link_position - grasp_link_position)
                else:
                    proxy_center = initial_center
                if capture_planner_swept_geometry:
                    for link_index in sorted(allowed_links):
                        raw_lower, raw_upper = env._p.getAABB(
                            env.robot_id,
                            linkIndex=int(link_index),
                            physicsClientId=env.client_id,
                        )
                        lower = np.asarray(raw_lower, dtype=np.float64)
                        upper = np.asarray(raw_upper, dtype=np.float64)
                        if (
                            lower.shape != (3,)
                            or upper.shape != (3,)
                            or not bool(np.isfinite(lower).all())
                            or not bool(np.isfinite(upper).all())
                            or bool(np.any(upper <= lower))
                        ):
                            raise ValueError("robot link AABB must be finite and increasing")
                        previous = robot_bin_bounds[progress_bin].get(
                            int(link_index)
                        )
                        if previous is None:
                            robot_bin_bounds[progress_bin][int(link_index)] = (
                                lower,
                                upper,
                            )
                        else:
                            robot_bin_bounds[progress_bin][int(link_index)] = (
                                np.minimum(previous[0], lower),
                                np.maximum(previous[1], upper),
                            )
                    if stage == "extraction":
                        proxy_lower = proxy_center - half_extents
                        proxy_upper = proxy_center + half_extents
                        previous_carried = carried_bin_bounds[progress_bin]
                        if previous_carried is None:
                            carried_bin_bounds[progress_bin] = (
                                proxy_lower,
                                proxy_upper,
                            )
                        else:
                            carried_bin_bounds[progress_bin] = (
                                np.minimum(previous_carried[0], proxy_lower),
                                np.maximum(previous_carried[1], proxy_upper),
                            )
                env._p.resetBasePositionAndOrientation(
                    int(proxy_body_id),
                    proxy_center.tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=env.client_id,
                )
                env._p.performCollisionDetection(physicsClientId=env.client_id)
                sample_contact = False
                sample_hard = False
                for name, body_id in fixed_body_ids.items():
                    robot_summary = summarize_signed_distances(
                        _closest_distances(
                            env,
                            body_a=env.robot_id,
                            body_b=int(body_id),
                            query_distance_m=config.collision_distance_m,
                            allowed_link_indices_a=allowed_links,
                        ),
                        hard_penetration_m=config.hard_penetration_m,
                    )
                    proxy_summary = summarize_signed_distances([], hard_penetration_m=config.hard_penetration_m)
                    if stage == "extraction":
                        proxy_summary = summarize_signed_distances(
                            _closest_distances(
                                env,
                                body_a=int(proxy_body_id),
                                body_b=int(body_id),
                                query_distance_m=config.collision_distance_m,
                            ),
                            hard_penetration_m=config.hard_penetration_m,
                        )
                    has_contact = bool(
                        robot_summary["has_contact"] or proxy_summary["has_contact"]
                    )
                    has_hard = bool(
                        robot_summary["has_hard_penetration"]
                        or proxy_summary["has_hard_penetration"]
                    )
                    sample_contact = sample_contact or has_contact
                    sample_hard = sample_hard or has_hard
                    if has_contact:
                        minimums = [
                            value
                            for value in (
                                robot_summary["minimum_signed_distance_m"],
                                proxy_summary["minimum_signed_distance_m"],
                            )
                            if value is not None
                        ]
                        current = body_evidence.setdefault(
                            str(name),
                            {
                                "contact_sample_count": 0,
                                "hard_penetration_sample_count": 0,
                                "robot_contact_sample_count": 0,
                                "robot_hard_penetration_sample_count": 0,
                                "carried_proxy_contact_sample_count": 0,
                                "carried_proxy_hard_penetration_sample_count": 0,
                                "minimum_signed_distance_m": None,
                                "minimum_robot_signed_distance_m": None,
                                "minimum_carried_proxy_signed_distance_m": None,
                            },
                        )
                        current["contact_sample_count"] += 1
                        current["hard_penetration_sample_count"] += int(has_hard)
                        current["robot_contact_sample_count"] += int(
                            robot_summary["has_contact"]
                        )
                        current["robot_hard_penetration_sample_count"] += int(
                            robot_summary["has_hard_penetration"]
                        )
                        current["carried_proxy_contact_sample_count"] += int(
                            proxy_summary["has_contact"]
                        )
                        current["carried_proxy_hard_penetration_sample_count"] += int(
                            proxy_summary["has_hard_penetration"]
                        )
                        for key, value in (
                            (
                                "minimum_robot_signed_distance_m",
                                robot_summary["minimum_signed_distance_m"],
                            ),
                            (
                                "minimum_carried_proxy_signed_distance_m",
                                proxy_summary["minimum_signed_distance_m"],
                            ),
                        ):
                            if value is not None and (
                                current[key] is None or float(value) < current[key]
                            ):
                                current[key] = float(value)
                        minimum = min(minimums)
                        if (
                            current["minimum_signed_distance_m"] is None
                            or minimum < current["minimum_signed_distance_m"]
                        ):
                            current["minimum_signed_distance_m"] = minimum
                contact_sample_count += int(sample_contact)
                hard_sample_count += int(sample_hard)
                hard_collision = hard_collision or sample_hard
            stage_evidence[stage] = {
                "sample_count": len(configs),
                "contact_sample_count": contact_sample_count,
                "hard_penetration_sample_count": hard_sample_count,
                "fixed_bodies": body_evidence,
            }
            if capture_planner_swept_geometry:
                robot_link_aabbs_by_stage[stage] = [
                    [
                        [lower.tolist(), upper.tolist()]
                        for _link_index, (lower, upper) in sorted(bounds.items())
                    ]
                    for bounds in robot_bin_bounds
                ]
                if stage == "extraction":
                    carried_target_aabbs_by_stage[stage] = [
                        []
                        if bounds is None
                        else [[bounds[0].tolist(), bounds[1].tolist()]]
                        for bounds in carried_bin_bounds
                    ]
    finally:
        env._p.removeBody(int(proxy_body_id), physicsClientId=env.client_id)
    result = {
        "fixed_environment_collision_free": not hard_collision,
        "fixed_environment_collision": hard_collision,
        "stage_evidence": stage_evidence,
        "carried_target_geometry": {
            "method": "axis_aligned_cnabu_world_aabb_translation_proxy_v0",
            "half_extents_m": half_extents.tolist(),
            "initial_center_m": initial_center.tolist(),
        },
    }
    if capture_planner_swept_geometry:
        result["planner_swept_geometry"] = {
            "schema": "runtime_robot_link_and_carried_proxy_swept_aabbs_v1",
            "progress_bin_count": TRAJECTORY_PLANNER_SWEPT_PROGRESS_BIN_COUNT,
            "robot_link_aabbs_by_stage": robot_link_aabbs_by_stage,
            "carried_target_aabbs_by_stage": carried_target_aabbs_by_stage,
        }
    return result


def build_runtime_candidate_action_mask(
    env: Any,
    *,
    target_world_aabbs: Sequence[Sequence[Sequence[float]]],
    node_ids: Optional[Sequence[Any]] = None,
    initial_arm_config: Optional[Sequence[float]] = None,
    fixed_body_ids: Optional[Mapping[str, int]] = None,
    config: Optional[OracleActionFamilyConfig] = None,
    include_planner_swept_geometry: bool = False,
) -> Dict[str, Any]:
    """Combine current-state IK and known-fixed-body collision availability.

    Target geometry is supplied by ordered CNABU world AABBs. The carried
    target is represented by a temporary axis-aligned box proxy, never by a
    simulator object id or pose. Dynamic scene objects are deliberately not
    queried here; their blocker evidence remains the learned relation term.
    """

    cfg = config or OracleActionFamilyConfig()
    initial = _runtime_initial_arm_config(env, initial_arm_config)
    fixed_ids = dict(_fixed_body_ids(env) if fixed_body_ids is None else fixed_body_ids)
    if not fixed_ids:
        raise ValueError("fixed_body_ids must contain at least one known environment body")
    kinematic = build_runtime_candidate_kinematic_mask(
        env,
        target_world_aabbs=target_world_aabbs,
        node_ids=node_ids,
        initial_arm_config=initial,
        config=cfg,
    )
    kinematic_mask = np.asarray(kinematic["kinematic_mask"], dtype=bool)
    fixed_free = np.zeros_like(kinematic_mask, dtype=bool)
    targets: List[Dict[str, Any]] = []
    try:
        for target_index, target in enumerate(kinematic["targets"]):
            candidate_results: List[Dict[str, Any]] = []
            for candidate_index, candidate in enumerate(target["candidates"]):
                if not bool(kinematic_mask[target_index, candidate_index]):
                    candidate_results.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "kinematically_feasible": False,
                            "fixed_environment_evaluated": False,
                            "fixed_environment_collision_free": False,
                            "action_eligible": False,
                            "failed_waypoint": candidate["failed_waypoint"],
                            "stage_evidence": {},
                        }
                    )
                    continue
                collision = _runtime_fixed_environment_candidate_collision(
                    env,
                    target_world_aabb=target["world_aabb"],
                    stage_joint_configs=candidate["stage_joint_configs"],
                    fixed_body_ids=fixed_ids,
                    config=cfg,
                    capture_planner_swept_geometry=bool(
                        include_planner_swept_geometry
                    ),
                )
                fixed_free[target_index, candidate_index] = bool(
                    collision["fixed_environment_collision_free"]
                )
                candidate_result = {
                        "candidate_id": candidate["candidate_id"],
                        "kinematically_feasible": True,
                        "fixed_environment_evaluated": True,
                        "fixed_environment_collision_free": bool(
                            collision["fixed_environment_collision_free"]
                        ),
                        "action_eligible": bool(
                            collision["fixed_environment_collision_free"]
                        ),
                        "failed_waypoint": None,
                        "stage_evidence": collision["stage_evidence"],
                        "carried_target_geometry": collision["carried_target_geometry"],
                    }
                if include_planner_swept_geometry:
                    candidate_result["planner_swept_geometry"] = collision[
                        "planner_swept_geometry"
                    ]
                candidate_results.append(candidate_result)
            targets.append(
                {
                    "node_id": target["node_id"],
                    "world_aabb": target["world_aabb"],
                    "candidates": candidate_results,
                }
            )
    finally:
        _reset_arm_config(env, initial)
    action_mask = kinematic_mask & fixed_free
    return {
        "schema": "cnabu_runtime_candidate_action_mask_v0",
        "candidate_ids": list(kinematic["candidate_ids"]),
        "node_ids": list(kinematic["node_ids"]),
        "kinematic_mask": kinematic_mask.tolist(),
        "fixed_environment_collision_free_mask": fixed_free.tolist(),
        "action_eligible_mask": action_mask.tolist(),
        "targets": targets,
        "source": "current_robot_ik_plus_known_fixed_bodies_plus_cnabu_aabb_proxy",
        "fixed_body_names": [str(name) for name in fixed_ids],
        "safety": {
            "uses_gt_or_simulator_instance_ids": False,
            "uses_simulator_object_poses": False,
            "uses_cnabu_ordered_node_geometry": True,
            "queries_dynamic_scene_objects": False,
            "queries_known_fixed_environment_bodies": True,
            "creates_temporary_cnabu_aabb_collision_proxy": True,
            "exports_planner_swept_geometry": bool(
                include_planner_swept_geometry
            ),
        },
    }


def build_cnabu_runtime_candidate_kinematic_mask(
    env: Any,
    heightmap_generation: Any,
    indices_zyx: Sequence[Any],
    *,
    crop_rows: Sequence[int],
    node_ids: Optional[Sequence[Any]] = None,
    initial_arm_config: Optional[Sequence[float]] = None,
    config: Optional[OracleActionFamilyConfig] = None,
    support_boundary_quantile: float = 0.0,
) -> Dict[str, Any]:
    """Compose CNABU support calibration and current-state IK masking.

    This is the single runtime adapter used after ``scene_graph_mem`` has
    reconstructed ordered sparse node support. It preserves that node order
    through world calibration and the frozen nine-candidate action family.
    """

    target_world_aabbs = cnabu_sparse_support_world_aabbs(
        heightmap_generation,
        indices_zyx,
        crop_rows=crop_rows,
        boundary_quantile=support_boundary_quantile,
    )
    result = build_runtime_candidate_kinematic_mask(
        env,
        target_world_aabbs=target_world_aabbs,
        node_ids=node_ids,
        initial_arm_config=initial_arm_config,
        config=config,
    )
    result["source"] = (
        "cnabu_sparse_support_world_calibration_plus_"
        "deterministic_current_robot_state_pybullet_ik"
    )
    return result


def build_cnabu_runtime_candidate_action_mask(
    env: Any,
    heightmap_generation: Any,
    indices_zyx: Sequence[Any],
    *,
    crop_rows: Sequence[int],
    node_ids: Optional[Sequence[Any]] = None,
    initial_arm_config: Optional[Sequence[float]] = None,
    fixed_body_ids: Optional[Mapping[str, int]] = None,
    config: Optional[OracleActionFamilyConfig] = None,
    support_boundary_quantile: float = 0.05,
    include_planner_swept_features: bool = False,
) -> Dict[str, Any]:
    """Build planner-owned candidate eligibility from live CNABU support.

    The 5% support envelope is the bounded-audit default that removes sparse
    uncertain boundary tails and the systematic one-voxel shelf penetration
    of a literal min/max envelope. It is computed before all PyBullet queries
    and never uses GT or simulator object state.
    """

    target_world_aabbs = cnabu_sparse_support_world_aabbs(
        heightmap_generation,
        indices_zyx,
        crop_rows=crop_rows,
        boundary_quantile=support_boundary_quantile,
    )
    result = build_runtime_candidate_action_mask(
        env,
        target_world_aabbs=target_world_aabbs,
        node_ids=node_ids,
        initial_arm_config=initial_arm_config,
        fixed_body_ids=fixed_body_ids,
        config=config,
        include_planner_swept_geometry=bool(include_planner_swept_features),
    )
    if include_planner_swept_features:
        source_world_aabbs = cnabu_sparse_support_world_aabbs(
            heightmap_generation,
            indices_zyx,
            crop_rows=crop_rows,
            boundary_quantile=0.0,
        )
        result["planner_swept_features"] = build_candidate_planner_swept_features(
            source_world_aabbs=source_world_aabbs,
            node_ids=result["node_ids"],
            targets=result["targets"],
        )
        for target in result["targets"]:
            for candidate in target["candidates"]:
                candidate.pop("planner_swept_geometry", None)
    result["source"] = (
        "cnabu_robust_support_world_calibration_plus_current_robot_ik_plus_"
        "known_fixed_bodies"
    )
    result["support_boundary_quantile"] = float(support_boundary_quantile)
    result["safety"]["exports_planner_swept_features"] = bool(
        include_planner_swept_features
    )
    return result


def _collision_link_indices(env: Any) -> set[int]:
    return set(range(int(env._p.getNumJoints(env.robot_id, physicsClientId=env.client_id))))


def summarize_signed_distances(
    distances_m: Sequence[float],
    *,
    hard_penetration_m: float,
) -> Dict[str, Any]:
    """Classify PyBullet signed distances without equating all contact to blockage."""

    threshold = float(hard_penetration_m)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("hard_penetration_m must be finite and non-negative")
    finite = [float(value) for value in distances_m if math.isfinite(float(value))]
    minimum = min(finite) if finite else None
    return {
        "has_contact": bool(minimum is not None and minimum <= 0.0),
        "has_hard_penetration": bool(minimum is not None and minimum < -threshold),
        "minimum_signed_distance_m": minimum,
    }


def summarize_extraction_progress(
    *,
    actual_displacement: Sequence[float],
    planned_carried_positions: Sequence[Sequence[float]],
    minimum_progress_fraction: float,
) -> Dict[str, Any]:
    """Measure carried-object motion relative to this candidate's planned extraction."""

    positions = np.asarray(planned_carried_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
        raise ValueError("planned_carried_positions must contain at least two xyz positions")
    threshold = float(minimum_progress_fraction)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_progress_fraction must be in (0, 1]")
    planned = positions[-1] - positions[0]
    planned_distance = float(np.linalg.norm(planned))
    if planned_distance <= 1e-9:
        raise ValueError("planned carried-object extraction displacement must be non-zero")
    actual = np.asarray(actual_displacement, dtype=np.float64)
    if actual.shape != (3,):
        raise ValueError("actual_displacement must be xyz")
    projected = float(np.dot(actual, planned / planned_distance))
    progress_fraction = projected / planned_distance
    return {
        "target_extracted": bool(progress_fraction >= threshold),
        "progress_fraction": float(progress_fraction),
        "projected_displacement_m": projected,
        "planned_displacement_m": planned_distance,
        "planned_displacement": planned.tolist(),
        "minimum_progress_fraction": threshold,
    }


def summarize_monitored_displacement(
    *,
    object_displacements_m: Mapping[str, float],
    monitored_instance_ids: Sequence[int],
    maximum_displacement_m: float,
) -> Dict[str, Any]:
    """Classify blocker motion without treating unrelated scene settling as failure."""

    threshold = float(maximum_displacement_m)
    if threshold <= 0.0:
        raise ValueError("maximum_displacement_m must be positive")
    selected = {
        str(int(instance_id)): float(object_displacements_m.get(str(int(instance_id)), 0.0))
        for instance_id in monitored_instance_ids
    }
    maximum = max(selected.values(), default=0.0)
    return {
        "monitored_object_displacements_m": selected,
        "maximum_monitored_displacement_m": maximum,
        "maximum_allowed_displacement_m": threshold,
        "monitored_objects_stable": bool(maximum <= threshold),
    }


def _closest_distances(
    env: Any,
    *,
    body_a: int,
    body_b: int,
    query_distance_m: float,
    allowed_link_indices_a: Optional[set[int]] = None,
) -> List[float]:
    points = env._p.getClosestPoints(
        bodyA=int(body_a),
        bodyB=int(body_b),
        distance=float(query_distance_m),
        physicsClientId=env.client_id,
    )
    return [
        float(point[8])
        for point in points
        if allowed_link_indices_a is None or int(point[3]) in allowed_link_indices_a
    ]


def _target_pose_in_grasp_frame(
    env: Any,
    *,
    target_id: int,
    grasp_config: Sequence[float],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    _reset_arm_config(env, grasp_config)
    link_state = env._p.getLinkState(
        env.robot_id,
        env.grasp_link_id,
        physicsClientId=env.client_id,
    )
    inverse_position, inverse_orientation = env._p.invertTransform(link_state[4], link_state[5])
    target_position, target_orientation = env._p.getBasePositionAndOrientation(
        int(target_id),
        physicsClientId=env.client_id,
    )
    local_position, local_orientation = env._p.multiplyTransforms(
        inverse_position,
        inverse_orientation,
        target_position,
        target_orientation,
    )
    return tuple(local_position), tuple(local_orientation)


def _carried_target_world_pose(
    env: Any,
    *,
    local_position: Sequence[float],
    local_orientation: Sequence[float],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    link_state = env._p.getLinkState(
        env.robot_id,
        env.grasp_link_id,
        physicsClientId=env.client_id,
    )
    position, orientation = env._p.multiplyTransforms(
        link_state[4],
        link_state[5],
        local_position,
        local_orientation,
    )
    return tuple(position), tuple(orientation)


def _sample_stage_collisions(
    env: Any,
    *,
    stage_configs: Mapping[str, Sequence[Sequence[float]]],
    target_id: int,
    grasp_config: Sequence[float],
    object_ids: Sequence[int],
    fixed_body_ids: Mapping[str, int],
    collision_distance_m: float,
    hard_penetration_m: float,
) -> Dict[str, Any]:
    allowed_links = _collision_link_indices(env)
    hard_instance_counts: Dict[str, Dict[int, int]] = {}
    contact_instance_counts: Dict[str, Dict[int, int]] = {}
    hard_fixed_counts: Dict[str, int] = {}
    contact_fixed_counts: Dict[str, int] = {}
    stage_counts: Dict[str, int] = {}
    tool_paths: Dict[str, List[List[float]]] = {}
    carried_target_paths: Dict[str, List[List[float]]] = {}
    contact_evidence: Dict[str, Dict[str, Any]] = {}
    target_original_pose = env._p.getBasePositionAndOrientation(
        int(target_id),
        physicsClientId=env.client_id,
    )
    local_target_pose = _target_pose_in_grasp_frame(
        env,
        target_id=int(target_id),
        grasp_config=grasp_config,
    )
    try:
        for raw_stage, configs in stage_configs.items():
            stage = str(raw_stage)
            per_instance_hard = {int(instance_id): 0 for instance_id in object_ids}
            per_instance_contact = {int(instance_id): 0 for instance_id in object_ids}
            fixed_hard_samples = 0
            fixed_contact_samples = 0
            tips: List[List[float]] = []
            target_path: List[List[float]] = []
            instance_evidence: Dict[str, Dict[str, Any]] = {}
            fixed_evidence: Dict[str, Dict[str, Any]] = {}
            for joint_config in configs:
                _reset_arm_config(env, joint_config)
                carrying_target = stage == "extraction"
                if carrying_target:
                    target_position, target_orientation = _carried_target_world_pose(
                        env,
                        local_position=local_target_pose[0],
                        local_orientation=local_target_pose[1],
                    )
                    env._p.resetBasePositionAndOrientation(
                        int(target_id),
                        target_position,
                        target_orientation,
                        physicsClientId=env.client_id,
                    )
                    target_path.append([float(value) for value in target_position])
                else:
                    env._p.resetBasePositionAndOrientation(
                        int(target_id),
                        target_original_pose[0],
                        target_original_pose[1],
                        physicsClientId=env.client_id,
                    )
                env._p.performCollisionDetection(physicsClientId=env.client_id)
                tips.append(
                    [
                        float(value)
                        for value in env._p.getLinkState(
                            env.robot_id,
                            env.tool_tip_id,
                            physicsClientId=env.client_id,
                        )[0]
                    ]
                )

                for instance_id in object_ids:
                    instance_id = int(instance_id)
                    robot_summary = summarize_signed_distances(
                        _closest_distances(
                            env,
                            body_a=env.robot_id,
                            body_b=instance_id,
                            query_distance_m=collision_distance_m,
                            allowed_link_indices_a=allowed_links,
                        ),
                        hard_penetration_m=hard_penetration_m,
                    )
                    carried_summary = summarize_signed_distances([], hard_penetration_m=hard_penetration_m)
                    if carrying_target and instance_id != int(target_id):
                        carried_summary = summarize_signed_distances(
                            _closest_distances(
                                env,
                                body_a=int(target_id),
                                body_b=instance_id,
                                query_distance_m=collision_distance_m,
                            ),
                            hard_penetration_m=hard_penetration_m,
                        )
                    has_contact = bool(robot_summary["has_contact"] or carried_summary["has_contact"])
                    has_hard = bool(
                        robot_summary["has_hard_penetration"]
                        or carried_summary["has_hard_penetration"]
                    )
                    per_instance_contact[instance_id] += int(has_contact)
                    per_instance_hard[instance_id] += int(has_hard)
                    if has_contact:
                        evidence = instance_evidence.setdefault(
                            str(instance_id),
                            {
                                "contact_sample_count": 0,
                                "hard_penetration_sample_count": 0,
                                "robot_contact_sample_count": 0,
                                "carried_target_contact_sample_count": 0,
                                "minimum_signed_distance_m": None,
                            },
                        )
                        evidence["contact_sample_count"] += 1
                        evidence["hard_penetration_sample_count"] += int(has_hard)
                        evidence["robot_contact_sample_count"] += int(robot_summary["has_contact"])
                        evidence["carried_target_contact_sample_count"] += int(
                            carried_summary["has_contact"]
                        )
                        distances = [
                            value
                            for value in (
                                robot_summary["minimum_signed_distance_m"],
                                carried_summary["minimum_signed_distance_m"],
                            )
                            if value is not None
                        ]
                        minimum = min(distances)
                        if (
                            evidence["minimum_signed_distance_m"] is None
                            or minimum < evidence["minimum_signed_distance_m"]
                        ):
                            evidence["minimum_signed_distance_m"] = minimum

                sample_fixed_contact = False
                sample_fixed_hard = False
                for name, body_id in fixed_body_ids.items():
                    robot_summary = summarize_signed_distances(
                        _closest_distances(
                            env,
                            body_a=env.robot_id,
                            body_b=int(body_id),
                            query_distance_m=collision_distance_m,
                            allowed_link_indices_a=allowed_links,
                        ),
                        hard_penetration_m=hard_penetration_m,
                    )
                    carried_summary = summarize_signed_distances([], hard_penetration_m=hard_penetration_m)
                    if carrying_target:
                        carried_summary = summarize_signed_distances(
                            _closest_distances(
                                env,
                                body_a=int(target_id),
                                body_b=int(body_id),
                                query_distance_m=collision_distance_m,
                            ),
                            hard_penetration_m=hard_penetration_m,
                        )
                    has_contact = bool(robot_summary["has_contact"] or carried_summary["has_contact"])
                    has_hard = bool(
                        robot_summary["has_hard_penetration"]
                        or carried_summary["has_hard_penetration"]
                    )
                    sample_fixed_contact = sample_fixed_contact or has_contact
                    sample_fixed_hard = sample_fixed_hard or has_hard
                    if has_contact:
                        evidence = fixed_evidence.setdefault(
                            str(name),
                            {
                                "contact_sample_count": 0,
                                "hard_penetration_sample_count": 0,
                                "robot_contact_sample_count": 0,
                                "carried_target_contact_sample_count": 0,
                                "minimum_signed_distance_m": None,
                            },
                        )
                        evidence["contact_sample_count"] += 1
                        evidence["hard_penetration_sample_count"] += int(has_hard)
                        evidence["robot_contact_sample_count"] += int(robot_summary["has_contact"])
                        evidence["carried_target_contact_sample_count"] += int(
                            carried_summary["has_contact"]
                        )
                        distances = [
                            value
                            for value in (
                                robot_summary["minimum_signed_distance_m"],
                                carried_summary["minimum_signed_distance_m"],
                            )
                            if value is not None
                        ]
                        minimum = min(distances)
                        if (
                            evidence["minimum_signed_distance_m"] is None
                            or minimum < evidence["minimum_signed_distance_m"]
                        ):
                            evidence["minimum_signed_distance_m"] = minimum
                fixed_contact_samples += int(sample_fixed_contact)
                fixed_hard_samples += int(sample_fixed_hard)

            hard_instance_counts[stage] = per_instance_hard
            contact_instance_counts[stage] = per_instance_contact
            hard_fixed_counts[stage] = fixed_hard_samples
            contact_fixed_counts[stage] = fixed_contact_samples
            stage_counts[stage] = len(configs)
            tool_paths[stage] = tips
            carried_target_paths[stage] = target_path
            contact_evidence[stage] = {
                "instances": instance_evidence,
                "fixed_environment": fixed_evidence,
            }
    finally:
        env._p.resetBasePositionAndOrientation(
            int(target_id),
            target_original_pose[0],
            target_original_pose[1],
            physicsClientId=env.client_id,
        )
    return {
        "hard_instance_counts": hard_instance_counts,
        "contact_instance_counts": contact_instance_counts,
        "hard_fixed_counts": hard_fixed_counts,
        "contact_fixed_counts": contact_fixed_counts,
        "stage_counts": stage_counts,
        "tool_paths": tool_paths,
        "carried_target_paths": carried_target_paths,
        "contact_evidence": contact_evidence,
    }


def restore_saved_scene(env: Any, pre_action_dir: Path | str) -> List[Dict[str, Any]]:
    pre_action = Path(pre_action_dir)
    with (pre_action / "placed_objects.pkl").open("rb") as handle:
        arrangement = pickle.load(handle)
    object_records = extract_gt_object_records(pre_action / "gt_hms.npz")

    env.initial_reset()
    env.current_obj_ids, env.current_obj_classes = env.obj.physically_place_objects(arrangement)
    env.initialize_object_info()
    expected_ids = {int(item["instance_id"]) for item in object_records}
    replayed_ids = {int(value) for value in env.current_obj_ids}
    if expected_ids != replayed_ids:
        raise ValueError(
            "replayed PyBullet ids do not match GT object ids: missing={}, extra={}".format(
                sorted(expected_ids - replayed_ids),
                sorted(replayed_ids - expected_ids),
            )
        )
    class_by_id = {int(key): int(value) for key, value in env.obj.get_id_to_class_dict().items()}
    for record in object_records:
        instance_id = int(record["instance_id"])
        if class_by_id.get(instance_id) != int(record["semantic_class_id"]):
            raise ValueError("replayed class does not match GT semantic class for id {}".format(instance_id))
        lower, upper = env._p.getAABB(instance_id, physicsClientId=env.client_id)
        record["world_aabb"] = [list(map(float, lower)), list(map(float, upper))]
        record["object_name"] = str(env.obj.obj_urdf_names[int(record["semantic_class_id"])])
    return object_records


def _build_candidate(
    env: Any,
    *,
    target_record: Mapping[str, Any],
    lateral_fraction: float,
    height_fraction: float,
    object_ids: Sequence[int],
    fixed_body_ids: Mapping[str, int],
    initial_arm_config: Sequence[float],
    config: OracleActionFamilyConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    target_id = int(target_record["instance_id"])
    grasp_id = "front_x{:.2f}_z{:.2f}".format(float(lateral_fraction), float(height_fraction))
    trajectory_id = "{}/{}".format(target_id, grasp_id)
    waypoints = front_extraction_waypoints(
        world_aabb=target_record["world_aabb"],
        height_fraction=height_fraction,
        config=config,
        lateral_fraction=lateral_fraction,
    )
    solved: Dict[str, np.ndarray] = {}
    residuals: Dict[str, float] = {}
    seed = np.asarray(initial_arm_config, dtype=np.float64)
    for name in ("pregrasp", "grasp", "lift", "extraction"):
        joint_config, residual = _solve_ik_position(
            env,
            waypoints[name],
            seed,
            config.max_ik_position_residual_m,
        )
        residuals[name] = residual
        if joint_config is None:
            observation = evaluate_collision_trajectory(
                trajectory_id=trajectory_id,
                grasp_id=grasp_id,
                target_instance_id=target_id,
                stage_collision_sample_counts={},
                stage_sample_counts={},
                kinematically_feasible=False,
                metadata={
                    "failed_waypoint": name,
                    "ik_position_residual_m": residual,
                    "waypoints": waypoints,
                },
            )
            _reset_arm_config(env, initial_arm_config)
            return observation, {"stage_configs": {}, "tool_paths": {}}
        solved[name] = joint_config
        seed = joint_config

    stage_configs = {
        "approach": interpolate_joint_configs(
            initial_arm_config,
            solved["pregrasp"],
            config.approach_samples,
        ),
        "grasp": interpolate_joint_configs(
            solved["pregrasp"],
            solved["grasp"],
            config.grasp_samples,
        ),
        "extraction": (
            interpolate_joint_configs(
                solved["grasp"],
                solved["lift"],
                config.lift_samples,
            )
            + interpolate_joint_configs(
                solved["lift"],
                solved["extraction"],
                config.extraction_samples,
                include_start=False,
            )
        ),
    }
    sampled = _sample_stage_collisions(
        env,
        stage_configs=stage_configs,
        target_id=target_id,
        grasp_config=solved["grasp"],
        object_ids=object_ids,
        fixed_body_ids=fixed_body_ids,
        collision_distance_m=config.collision_distance_m,
        hard_penetration_m=config.hard_penetration_m,
    )
    contact_only_ids = sorted(
        {
            int(instance_id)
            for stage, counts in sampled["contact_instance_counts"].items()
            for instance_id, count in counts.items()
            if int(instance_id) != target_id
            and int(count) > 0
            and int(sampled["hard_instance_counts"][stage].get(instance_id, 0)) == 0
        }
    )
    observation = evaluate_collision_trajectory(
        trajectory_id=trajectory_id,
        grasp_id=grasp_id,
        target_instance_id=target_id,
        stage_collision_sample_counts=sampled["hard_instance_counts"],
        stage_sample_counts=sampled["stage_counts"],
        fixed_environment_collision_sample_counts=sampled["hard_fixed_counts"],
        metadata={
            "waypoints": waypoints,
            "ik_position_residuals_m": residuals,
            "tool_tip_positions_by_stage": sampled["tool_paths"],
            "carried_target_positions_by_stage": sampled["carried_target_paths"],
            "collision_distance_m": config.collision_distance_m,
            "hard_penetration_m": config.hard_penetration_m,
            "contact_only_instance_ids": contact_only_ids,
            "contact_sample_counts_by_stage": sampled["contact_instance_counts"],
            "fixed_contact_sample_counts_by_stage": sampled["contact_fixed_counts"],
            "contact_evidence_by_stage": sampled["contact_evidence"],
            "blocking_evidence": "hard_penetration_beyond_threshold",
            "extraction_envelope": "robot_arm_open_gripper_plus_rigidly_carried_target_geometry_v1",
        },
    )
    _reset_arm_config(env, initial_arm_config)
    return observation, {
        "stage_configs": stage_configs,
        "tool_paths": sampled["tool_paths"],
        "carried_target_paths": sampled["carried_target_paths"],
    }


def evaluate_saved_scene(
    env: Any,
    *,
    pre_action_dir: Path | str,
    config: Optional[OracleActionFamilyConfig] = None,
    static_pose_yaw_perturbation: Optional[StaticOraclePerturbationConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Replay and evaluate one saved scene without executing or writing actions."""

    cfg = config or OracleActionFamilyConfig()
    started = time.perf_counter()
    pre_action = Path(pre_action_dir)
    object_records = restore_saved_scene(env, pre_action)
    perturbation_record = None
    if static_pose_yaw_perturbation is not None:
        perturbation_record = apply_static_oracle_pose_yaw_perturbation(
            env,
            object_records=object_records,
            config=static_pose_yaw_perturbation,
        )
    env.reset_robot(env.initial_parameters)
    env.move_gripper(0.085)
    initial_arm_config = np.asarray(env.get_current_joint_config(), dtype=np.float64)
    object_ids = [int(item["instance_id"]) for item in object_records]
    fixed_body_ids = {
        "plane": int(env.planeID),
        "table": int(env.UR5Stand_id),
        "shelf": int(env.shelf_id),
        "wall": int(env.wall_id),
        **{"rack_{}".format(index): int(body_id) for index, body_id in enumerate(env.rack_ids)},
    }

    observations: List[Dict[str, Any]] = []
    debug_candidates: Dict[str, Dict[str, Any]] = {}
    for target_record in object_records:
        for lateral_fraction in cfg.candidate_lateral_fractions:
            for height_fraction in cfg.candidate_height_fractions:
                observation, debug = _build_candidate(
                    env,
                    target_record=target_record,
                    lateral_fraction=float(lateral_fraction),
                    height_fraction=float(height_fraction),
                    object_ids=object_ids,
                    fixed_body_ids=fixed_body_ids,
                    initial_arm_config=initial_arm_config,
                    config=cfg,
                )
                observations.append(observation)
                debug_candidates[str(observation["trajectory_id"])] = debug

    sample_id = "/".join(pre_action.parts[-3:-1])
    record = build_action_conditioned_oracle_record(
        sample_id=sample_id,
        instance_ids=object_ids,
        trajectories=observations,
        binary_threshold=cfg.binary_threshold,
        target_method=ACTION_CONDITIONED_TARGET_METHOD_V1,
        score_definition=(
            "weighted fraction of otherwise feasible target trajectories with robot or rigidly "
            "carried-target penetration into the source object beyond hard_penetration_m"
        ),
        robot={"name": "UR5", "simulator": "PyBullet"},
        gripper={"name": "Robotiq 85", "collision_state": "open_0.085m"},
        shelf_opening={"axis": "world_y", "side": "low", "extraction_y": cfg.opening_y},
        action_family={**asdict(cfg), "orientation_euler_xyz": [float(v) for v in env.init_ori]},
        metadata={
            "pre_action_dir": str(pre_action),
            "collision_query": (
                "PyBullet getClosestPoints over sampled robot configurations and rigidly carried "
                "target poses"
            ),
            "scene_replay_id_alignment": "exact_set_match",
            "static_pose_yaw_perturbation": perturbation_record,
            "static_score_friction_varied": False,
        },
    )
    geometry = build_geometry_pseudo_gt_adjacency(object_records)
    action_adjacency = record["binary_adjacency_matrix"]
    comparison = compare_relation_targets(
        node_order=object_ids,
        geometry_adjacency=geometry,
        action_adjacency=action_adjacency,
        action_score_valid_mask=record["score_valid_mask"],
    )
    record["object_records"] = object_records
    record["geometry_pseudo_gt_v0"] = {
        "adjacency_matrix": geometry,
        "edge_count": int(np.asarray(geometry).sum()),
    }
    record["geometry_vs_action_comparison"] = comparison
    record["scene_summary"] = {
        "object_count": len(object_records),
        "trajectory_candidate_count": len(observations),
        "kinematically_feasible_count": sum(item["kinematically_feasible"] for item in observations),
        "eligible_trajectory_count": sum(item["eligible_for_scoring"] for item in observations),
        "fixed_environment_collision_count": sum(item["fixed_environment_collision"] for item in observations),
        "contact_only_trajectory_count": sum(
            bool(item.get("metadata", {}).get("contact_only_instance_ids")) for item in observations
        ),
        "action_edge_count": int(np.asarray(action_adjacency).sum()),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _reset_arm_config(env, initial_arm_config)
    return record, debug_candidates


def render_scene_oracle_debug(
    scene_record: Mapping[str, Any],
    output_path: Path | str,
    *,
    trajectory_id: Optional[str] = None,
) -> Path:
    """Render one 3D trajectory/object diagnostic from serialized evidence."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations = [
        trajectory
        for target in scene_record.get("targets", [])
        for trajectory in target.get("trajectories", [])
    ]
    if trajectory_id is not None:
        selected = next(item for item in observations if item["trajectory_id"] == trajectory_id)
    else:
        selected = next(
            (
                item
                for item in observations
                if item["blocked_by"] and bool(item.get("eligible_for_scoring"))
            ),
            next((item for item in observations if item["blocked_by"]), observations[0]),
        )
    target_id = int(selected["target_instance_id"])
    blockers = {int(value) for value in selected["blocked_by"]}

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    for item in scene_record.get("object_records", []):
        instance_id = int(item["instance_id"])
        lower = np.asarray(item["world_aabb"][0], dtype=float)
        upper = np.asarray(item["world_aabb"][1], dtype=float)
        size = upper - lower
        color = "#2ca02c" if instance_id == target_id else "#d62728" if instance_id in blockers else "#9aa0a6"
        alpha = 0.55 if instance_id == target_id or instance_id in blockers else 0.12
        axis.bar3d(lower[0], lower[1], lower[2], size[0], size[1], size[2], color=color, alpha=alpha)
        if instance_id == target_id or instance_id in blockers:
            axis.text(*(lower + size / 2.0), str(instance_id), fontsize=8)

    colors = {"approach": "#1f77b4", "grasp": "#ff7f0e", "extraction": "#9467bd"}
    paths = selected.get("metadata", {}).get("tool_tip_positions_by_stage", {})
    for stage, points in paths.items():
        values = np.asarray(points, dtype=float)
        if values.ndim == 2 and len(values):
            axis.plot(values[:, 0], values[:, 1], values[:, 2], color=colors.get(stage, "black"), label=stage)
    axis.set_xlabel("world x")
    axis.set_ylabel("world y (opening is low y)")
    axis.set_zlabel("world z")
    axis.set_title(
        "{} | target {} | blocked_by {} | eligible {}".format(
            scene_record.get("sample_id"),
            target_id,
            sorted(blockers),
            selected["eligible_for_scoring"],
        )
    )
    axis.legend(loc="upper left")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def aggregate_prototype_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    comparisons = [item["geometry_vs_action_comparison"] for item in records]
    summaries = [item["scene_summary"] for item in records]
    methods = sorted(
        {str((item.get("relation_target") or {}).get("method", "")) for item in records}
    )
    if len(methods) != 1 or not methods[0]:
        raise ValueError("prototype records must use one non-empty relation target method")
    comparable = sum(int(item["comparable_pair_count"]) for item in comparisons)
    agreement = sum(int(item["agreement_count"]) for item in comparisons)
    return {
        "schema": "action_conditioned_oracle_prototype_summary_v1",
        "relation_target_method": methods[0],
        "scene_count": len(records),
        "object_count": sum(int(item["object_count"]) for item in summaries),
        "trajectory_candidate_count": sum(int(item["trajectory_candidate_count"]) for item in summaries),
        "kinematically_feasible_count": sum(int(item["kinematically_feasible_count"]) for item in summaries),
        "eligible_trajectory_count": sum(int(item["eligible_trajectory_count"]) for item in summaries),
        "fixed_environment_collision_count": sum(
            int(item["fixed_environment_collision_count"]) for item in summaries
        ),
        "contact_only_trajectory_count": sum(
            int(item.get("contact_only_trajectory_count", 0)) for item in summaries
        ),
        "geometry_edge_count": sum(int(item["geometry_pseudo_gt_v0"]["edge_count"]) for item in records),
        "action_edge_count": sum(int(item["action_edge_count"]) for item in summaries),
        "comparable_pair_count": comparable,
        "undefined_action_pair_count": sum(int(item["undefined_action_pair_count"]) for item in comparisons),
        "both_positive_count": sum(len(item["both_positive_pairs"]) for item in comparisons),
        "geometry_only_count": sum(len(item["geometry_only_pairs"]) for item in comparisons),
        "action_only_count": sum(len(item["action_only_pairs"]) for item in comparisons),
        "agreement_rate": float(agreement / comparable) if comparable else None,
        "runtime_seconds": sum(float(item["runtime_seconds"]) for item in summaries),
        "limitations": [
            "v1 uses a fixed 3x3 frontal grasp grid rather than a learned grasp sampler",
            "carried-target pose is rigidly attached to the grasp frame during static extraction checks",
            "hard penetration and binary relation thresholds remain validation candidates until the paired pilot",
        ],
    }


def counterfactual_candidate_descriptor(
    scene_record: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe one blocked path for stratified paired intervention selection."""

    if not bool(trajectory.get("eligible_for_scoring")) or not trajectory.get("blocked_by"):
        raise ValueError("counterfactual candidates must be eligible blocked trajectories")
    node_order = [int(value) for value in scene_record["node_order_instance_ids"]]
    index_by_id = {instance_id: index for index, instance_id in enumerate(node_order)}
    geometry = np.asarray(scene_record["geometry_pseudo_gt_v0"]["adjacency_matrix"], dtype=np.int64)
    target_id = int(trajectory["target_instance_id"])
    blocker_ids = [int(value) for value in trajectory["blocked_by"]]
    geometry_flags = [bool(geometry[index_by_id[source_id], index_by_id[target_id]]) for source_id in blocker_ids]
    blocker_order = "single" if len(blocker_ids) == 1 else "multiple"
    agreement = "all_geometry_positive" if all(geometry_flags) else "contains_action_only"
    score_matrix = scene_record.get("score_matrix") or []
    pair_scores = {}
    if len(score_matrix) == len(node_order) and all(len(row) == len(node_order) for row in score_matrix):
        pair_scores = {
            str(source_id): float(score_matrix[index_by_id[source_id]][index_by_id[target_id]])
            for source_id in blocker_ids
        }
    minimum_distances = []
    for stage_evidence in trajectory.get("metadata", {}).get("contact_evidence_by_stage", {}).values():
        instances = stage_evidence.get("instances", {})
        for source_id in blocker_ids:
            evidence = instances.get(str(source_id))
            if evidence and evidence.get("minimum_signed_distance_m") is not None:
                minimum_distances.append(float(evidence["minimum_signed_distance_m"]))
    return {
        "sample_id": str(scene_record["sample_id"]),
        "trajectory_id": str(trajectory["trajectory_id"]),
        "grasp_id": trajectory.get("grasp_id"),
        "target_instance_id": target_id,
        "removed_instance_ids": blocker_ids,
        "blocker_order": blocker_order,
        "geometry_agreement": agreement,
        "stratum": "{}_{}".format(blocker_order, agreement),
        "blocked_by_stage": dict(trajectory.get("blocked_by_stage") or {}),
        "pair_scores": pair_scores,
        "minimum_blocker_signed_distance_m": min(minimum_distances) if minimum_distances else None,
    }


def list_counterfactual_candidates(scene_record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for target in scene_record.get("targets", []):
        for trajectory in target.get("trajectories", []):
            if bool(trajectory.get("eligible_for_scoring")) and trajectory.get("blocked_by"):
                candidates.append(counterfactual_candidate_descriptor(scene_record, trajectory))
    candidates.sort(
        key=lambda item: (
            item["stratum"],
            len(item["removed_instance_ids"]),
            item["trajectory_id"],
        )
    )
    return candidates


def _fixed_body_ids(env: Any) -> Dict[str, int]:
    return {
        "plane": int(env.planeID),
        "table": int(env.UR5Stand_id),
        "shelf": int(env.shelf_id),
        "wall": int(env.wall_id),
        **{"rack_{}".format(index): int(body_id) for index, body_id in enumerate(env.rack_ids)},
    }


def _execution_subsequence(configs: Sequence[Sequence[float]], stride: int = 3) -> List[np.ndarray]:
    if not configs:
        return []
    selected = [np.asarray(configs[index], dtype=np.float64) for index in range(0, len(configs), int(stride))]
    final = np.asarray(configs[-1], dtype=np.float64)
    if not np.array_equal(selected[-1], final):
        selected.append(final)
    return selected


def _combined_dynamic_contact_summary(
    env: Any,
    *,
    target_id: int,
    other_body_id: int,
    hard_penetration_m: float,
    include_carried_target: bool,
) -> Dict[str, Any]:
    robot_points = env._p.getContactPoints(
        bodyA=env.robot_id,
        bodyB=int(other_body_id),
        physicsClientId=env.client_id,
    )
    target_points = (
        env._p.getContactPoints(
            bodyA=int(target_id),
            bodyB=int(other_body_id),
            physicsClientId=env.client_id,
        )
        if include_carried_target
        else []
    )
    robot = summarize_signed_distances(
        [float(point[8]) for point in robot_points],
        hard_penetration_m=hard_penetration_m,
    )
    target = summarize_signed_distances(
        [float(point[8]) for point in target_points],
        hard_penetration_m=hard_penetration_m,
    )
    distances = [
        value
        for value in (robot["minimum_signed_distance_m"], target["minimum_signed_distance_m"])
        if value is not None
    ]
    return {
        "has_contact": bool(robot["has_contact"] or target["has_contact"]),
        "has_hard_penetration": bool(
            robot["has_hard_penetration"] or target["has_hard_penetration"]
        ),
        "robot_contact": bool(robot["has_contact"]),
        "carried_target_contact": bool(target["has_contact"]),
        "minimum_signed_distance_m": min(distances) if distances else None,
    }


def apply_counterfactual_randomization(
    env: Any,
    *,
    object_ids: Sequence[int],
    config: CounterfactualRandomizationConfig,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(config.seed))
    if config.friction_scale_min <= 0.0 or config.friction_scale_max < config.friction_scale_min:
        raise ValueError("counterfactual friction scale range must be positive and increasing")
    records: List[Dict[str, Any]] = []
    for instance_id in sorted(int(value) for value in object_ids):
        position, orientation = env._p.getBasePositionAndOrientation(
            instance_id,
            physicsClientId=env.client_id,
        )
        euler = list(env._p.getEulerFromQuaternion(orientation))
        delta_xy = rng.uniform(
            -float(config.xy_position_jitter_m),
            float(config.xy_position_jitter_m),
            size=2,
        )
        delta_yaw = math.radians(
            float(rng.uniform(-config.yaw_jitter_degrees, config.yaw_jitter_degrees))
        )
        new_position = [
            float(position[0] + delta_xy[0]),
            float(position[1] + delta_xy[1]),
            float(position[2]),
        ]
        euler[2] = float(euler[2] + delta_yaw)
        new_orientation = env._p.getQuaternionFromEuler(euler)
        dynamics = env._p.getDynamicsInfo(instance_id, -1, physicsClientId=env.client_id)
        base_friction = float(dynamics[1])
        friction_scale = float(rng.uniform(config.friction_scale_min, config.friction_scale_max))
        env._p.resetBasePositionAndOrientation(
            instance_id,
            new_position,
            new_orientation,
            physicsClientId=env.client_id,
        )
        env._p.changeDynamics(
            instance_id,
            -1,
            lateralFriction=base_friction * friction_scale,
            physicsClientId=env.client_id,
        )
        records.append(
            {
                "instance_id": instance_id,
                "delta_xy_m": [float(value) for value in delta_xy],
                "delta_yaw_degrees": math.degrees(delta_yaw),
                "friction_scale": friction_scale,
            }
        )
    env._p.performCollisionDetection(physicsClientId=env.client_id)
    return {
        "seed": int(config.seed),
        "xy_position_jitter_m": float(config.xy_position_jitter_m),
        "yaw_jitter_degrees": float(config.yaw_jitter_degrees),
        "friction_scale_range": [
            float(config.friction_scale_min),
            float(config.friction_scale_max),
        ],
        "objects": records,
    }


def apply_static_oracle_pose_yaw_perturbation(
    env: Any,
    *,
    object_records: Sequence[Dict[str, Any]],
    config: StaticOraclePerturbationConfig,
) -> Dict[str, Any]:
    """Apply deterministic score-audit pose/yaw jitter without touching dynamics.

    The records' world AABBs are refreshed after the perturbation so target
    candidate waypoints follow the perturbed target rather than its saved pose.
    """

    xy_jitter = float(config.xy_position_jitter_m)
    yaw_jitter = float(config.yaw_jitter_degrees)
    if (
        not math.isfinite(xy_jitter)
        or not math.isfinite(yaw_jitter)
        or xy_jitter < 0.0
        or yaw_jitter < 0.0
    ):
        raise ValueError("static score pose/yaw jitter ranges must be finite and non-negative")
    by_id = {int(item["instance_id"]): item for item in object_records}
    if len(by_id) != len(object_records):
        raise ValueError("static score perturbation object ids must be unique")
    rng = np.random.default_rng(int(config.seed))
    perturbations: List[Dict[str, Any]] = []
    for instance_id in sorted(by_id):
        position, orientation = env._p.getBasePositionAndOrientation(
            instance_id,
            physicsClientId=env.client_id,
        )
        euler = list(env._p.getEulerFromQuaternion(orientation))
        delta_xy = rng.uniform(-xy_jitter, xy_jitter, size=2)
        delta_yaw_degrees = float(rng.uniform(-yaw_jitter, yaw_jitter))
        new_position = [
            float(position[0] + delta_xy[0]),
            float(position[1] + delta_xy[1]),
            float(position[2]),
        ]
        euler[2] = float(euler[2] + math.radians(delta_yaw_degrees))
        new_orientation = env._p.getQuaternionFromEuler(euler)
        env._p.resetBasePositionAndOrientation(
            instance_id,
            new_position,
            new_orientation,
            physicsClientId=env.client_id,
        )
        perturbations.append(
            {
                "instance_id": int(instance_id),
                "delta_xy_m": [float(value) for value in delta_xy],
                "delta_yaw_degrees": delta_yaw_degrees,
            }
        )
    env._p.performCollisionDetection(physicsClientId=env.client_id)
    for instance_id, record in by_id.items():
        lower, upper = env._p.getAABB(
            instance_id,
            physicsClientId=env.client_id,
        )
        record["world_aabb"] = [
            [float(value) for value in lower],
            [float(value) for value in upper],
        ]
    return {
        "schema": "static_oracle_pose_yaw_perturbation_v1",
        "seed": int(config.seed),
        "xy_position_jitter_m": xy_jitter,
        "yaw_jitter_degrees": yaw_jitter,
        "friction_varied": False,
        "objects": perturbations,
    }


def rethreshold_static_oracle_contact_evidence(
    scene_record: Mapping[str, Any],
    *,
    hard_penetration_m: float,
) -> Dict[str, Any]:
    """Rebuild relation scores from stored minimum distances at one threshold.

    This is a read-only sensitivity calculation over already sampled static
    collision configurations.  It does not rerun PyBullet and must not be
    interpreted as dynamic counterfactual evidence.
    """

    threshold = float(hard_penetration_m)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("hard_penetration_m must be finite and non-negative")
    instance_ids = [int(value) for value in scene_record.get("node_order_instance_ids", [])]
    if not instance_ids:
        raise ValueError("scene_record must contain node_order_instance_ids")
    observations: List[Dict[str, Any]] = []
    for target in scene_record.get("targets", []):
        for trajectory in target.get("trajectories", []):
            metadata = dict(trajectory.get("metadata") or {})
            contact_by_stage = dict(metadata.get("contact_evidence_by_stage") or {})
            stage_records = dict(trajectory.get("stages") or {})
            stage_names = list(trajectory.get("action_stages") or stage_records)
            collision_counts: Dict[str, Dict[int, int]] = {}
            fixed_counts: Dict[str, int] = {}
            sample_counts: Dict[str, int] = {}
            for stage in stage_names:
                stage_evidence = dict(contact_by_stage.get(stage) or {})
                instance_evidence = dict(stage_evidence.get("instances") or {})
                fixed_evidence = dict(stage_evidence.get("fixed_environment") or {})
                collision_counts[str(stage)] = {
                    int(instance_id): 1
                    for instance_id, evidence in instance_evidence.items()
                    if evidence.get("minimum_signed_distance_m") is not None
                    and float(evidence["minimum_signed_distance_m"]) < -threshold
                }
                fixed_counts[str(stage)] = int(
                    any(
                        evidence.get("minimum_signed_distance_m") is not None
                        and float(evidence["minimum_signed_distance_m"]) < -threshold
                        for evidence in fixed_evidence.values()
                    )
                )
                original_sample_count = int(
                    dict(stage_records.get(stage) or {}).get("sample_count", 0)
                )
                sample_counts[str(stage)] = max(
                    original_sample_count,
                    fixed_counts[str(stage)],
                    max(collision_counts[str(stage)].values(), default=0),
                )
            observations.append(
                evaluate_collision_trajectory(
                    trajectory_id=str(trajectory["trajectory_id"]),
                    grasp_id=trajectory.get("grasp_id"),
                    target_instance_id=int(trajectory["target_instance_id"]),
                    stage_collision_sample_counts=collision_counts,
                    stage_sample_counts=sample_counts,
                    fixed_environment_collision_sample_counts=fixed_counts,
                    weight=float(trajectory.get("weight", 1.0)),
                    kinematically_feasible=bool(
                        trajectory.get("kinematically_feasible", False)
                    ),
                    metadata={
                        **metadata,
                        "rethresholded_hard_penetration_m": threshold,
                        "rethresholded_from_minimum_signed_distances": True,
                    },
                )
            )
    context = dict(scene_record.get("oracle_context") or {})
    action_family = dict(context.get("action_family") or {})
    action_family["hard_penetration_m"] = threshold
    return build_action_conditioned_oracle_record(
        sample_id=str(scene_record.get("sample_id") or ""),
        instance_ids=instance_ids,
        trajectories=observations,
        binary_threshold=float(scene_record.get("binary_threshold", 0.4)),
        target_method=str(
            dict(scene_record.get("relation_target") or {}).get(
                "method", ACTION_CONDITIONED_TARGET_METHOD_V1
            )
        ),
        score_definition=str(
            dict(scene_record.get("relation_target") or {}).get(
                "score_definition",
                "weighted fraction of eligible paths with hard penetration",
            )
        ),
        robot=dict(context.get("robot") or {}),
        gripper=dict(context.get("gripper") or {}),
        shelf_opening=dict(context.get("shelf_opening") or {}),
        action_family=action_family,
        voxel_grid=dict(context.get("voxel_grid") or {}),
        metadata={
            "source_scene_schema": scene_record.get("schema"),
            "rethresholded_hard_penetration_m": threshold,
            "rethresholded_from_stored_static_contact_evidence": True,
            "runs_pybullet_replay": False,
        },
    )


def execute_forced_attachment_extraction_trial(
    env: Any,
    *,
    pre_action_dir: Path | str,
    target_instance_id: int,
    removed_instance_ids: Sequence[int],
    monitored_instance_ids: Sequence[int] = (),
    candidate_debug: Mapping[str, Any],
    randomization: Optional[CounterfactualRandomizationConfig] = None,
    minimum_extraction_progress_fraction: float = 0.8,
    max_final_joint_error_rad: float = 0.08,
    maximum_monitored_object_displacement_m: float = 0.01,
    hard_penetration_m: float = 0.002,
) -> Dict[str, Any]:
    """Dynamically execute one saved candidate, optionally removing blockers.

    The target is attached with a PyBullet fixed constraint after the grasp
    waypoint. This isolates access/extraction validation from grasp-closure
    quality; the resulting pilot must not be reported as autonomous grasp
    success.
    """

    object_records = restore_saved_scene(env, pre_action_dir)
    object_ids = {int(item["instance_id"]) for item in object_records}
    target_id = int(target_instance_id)
    removed_ids = sorted({int(value) for value in removed_instance_ids})
    monitored_ids = sorted({int(value) for value in monitored_instance_ids})
    if (
        target_id not in object_ids
        or not set(removed_ids).issubset(object_ids - {target_id})
        or not set(monitored_ids).issubset(object_ids - {target_id})
    ):
        raise ValueError("target, removed, and monitored ids must refer to distinct replayed objects")

    randomization_record = apply_counterfactual_randomization(
        env,
        object_ids=sorted(object_ids),
        config=randomization or CounterfactualRandomizationConfig(),
    )

    for offset, instance_id in enumerate(removed_ids):
        env._p.changeDynamics(instance_id, -1, mass=0, physicsClientId=env.client_id)
        env._p.setCollisionFilterGroupMask(instance_id, -1, 0, 0, physicsClientId=env.client_id)
        env._p.resetBasePositionAndOrientation(
            instance_id,
            [3.0 + 0.2 * offset, -3.0, 0.25],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.client_id,
        )

    env.reset_robot(env.initial_parameters)
    env.move_gripper(0.085)
    initial_position = np.asarray(
        env._p.getBasePositionAndOrientation(target_id, physicsClientId=env.client_id)[0],
        dtype=np.float64,
    )
    initial_other_positions = {
        instance_id: np.asarray(
            env._p.getBasePositionAndOrientation(instance_id, physicsClientId=env.client_id)[0],
            dtype=np.float64,
        )
        for instance_id in sorted(object_ids - set(removed_ids) - {target_id})
    }
    fixed_ids = _fixed_body_ids(env)
    stage_configs = dict(candidate_debug.get("stage_configs") or {})
    if not all(stage_configs.get(stage) for stage in ("approach", "grasp", "extraction")):
        raise ValueError("counterfactual execution requires solved approach/grasp/extraction configs")
    carried_target_paths = dict(candidate_debug.get("carried_target_paths") or {})
    if not carried_target_paths.get("extraction"):
        raise ValueError("counterfactual execution requires the planned carried-target path")

    stage_joint_errors: Dict[str, float] = {}
    blocker_contact_stages: Dict[str, List[int]] = {}
    blocker_hard_stages: Dict[str, List[int]] = {}
    fixed_contact_stages: Dict[str, List[str]] = {}
    fixed_hard_stages: Dict[str, List[str]] = {}
    constraint_id: Optional[int] = None
    grasp_reached = False
    try:
        for stage in ("approach", "grasp"):
            stage_blocker_contacts = set()
            stage_blocker_hard = set()
            stage_fixed_contacts = set()
            stage_fixed_hard = set()
            for joint_config in _execution_subsequence(stage_configs[stage]):
                env.execute_joint_states(joint_config, absolute=True)
                for instance_id in monitored_ids:
                    summary = _combined_dynamic_contact_summary(
                        env,
                        target_id=target_id,
                        other_body_id=instance_id,
                        hard_penetration_m=hard_penetration_m,
                        include_carried_target=False,
                    )
                    if summary["has_contact"]:
                        stage_blocker_contacts.add(instance_id)
                    if summary["has_hard_penetration"]:
                        stage_blocker_hard.add(instance_id)
                for name, body_id in fixed_ids.items():
                    summary = _combined_dynamic_contact_summary(
                        env,
                        target_id=target_id,
                        other_body_id=body_id,
                        hard_penetration_m=hard_penetration_m,
                        include_carried_target=False,
                    )
                    if summary["has_contact"]:
                        stage_fixed_contacts.add(name)
                    if summary["has_hard_penetration"]:
                        stage_fixed_hard.add(name)
            final_config = np.asarray(stage_configs[stage][-1], dtype=np.float64)
            actual = np.asarray(env.get_current_joint_config(), dtype=np.float64)
            stage_joint_errors[stage] = float(np.max(np.abs(actual - final_config)))
            blocker_contact_stages[stage] = sorted(stage_blocker_contacts)
            blocker_hard_stages[stage] = sorted(stage_blocker_hard)
            fixed_contact_stages[stage] = sorted(stage_fixed_contacts)
            fixed_hard_stages[stage] = sorted(stage_fixed_hard)

        grasp_reached = stage_joint_errors["grasp"] <= float(max_final_joint_error_rad)
        if grasp_reached:
            constraint_id = int(env.add_constraint_to_gripper(target_id))
            env.step_simulation(env.per_step_iterations)
            stage_blocker_contacts = set()
            stage_blocker_hard = set()
            stage_fixed_contacts = set()
            stage_fixed_hard = set()
            for joint_config in _execution_subsequence(stage_configs["extraction"]):
                env.execute_joint_states(joint_config, absolute=True)
                for instance_id in monitored_ids:
                    summary = _combined_dynamic_contact_summary(
                        env,
                        target_id=target_id,
                        other_body_id=instance_id,
                        hard_penetration_m=hard_penetration_m,
                        include_carried_target=True,
                    )
                    if summary["has_contact"]:
                        stage_blocker_contacts.add(instance_id)
                    if summary["has_hard_penetration"]:
                        stage_blocker_hard.add(instance_id)
                for name, body_id in fixed_ids.items():
                    summary = _combined_dynamic_contact_summary(
                        env,
                        target_id=target_id,
                        other_body_id=body_id,
                        hard_penetration_m=hard_penetration_m,
                        include_carried_target=True,
                    )
                    if summary["has_contact"]:
                        stage_fixed_contacts.add(name)
                    if summary["has_hard_penetration"]:
                        stage_fixed_hard.add(name)
            final_config = np.asarray(stage_configs["extraction"][-1], dtype=np.float64)
            actual = np.asarray(env.get_current_joint_config(), dtype=np.float64)
            stage_joint_errors["extraction"] = float(np.max(np.abs(actual - final_config)))
            blocker_contact_stages["extraction"] = sorted(stage_blocker_contacts)
            blocker_hard_stages["extraction"] = sorted(stage_blocker_hard)
            fixed_contact_stages["extraction"] = sorted(stage_fixed_contacts)
            fixed_hard_stages["extraction"] = sorted(stage_fixed_hard)
        else:
            stage_joint_errors["extraction"] = float("inf")
            blocker_contact_stages["extraction"] = []
            blocker_hard_stages["extraction"] = []
            fixed_contact_stages["extraction"] = []
            fixed_hard_stages["extraction"] = []

        final_position = np.asarray(
            env._p.getBasePositionAndOrientation(target_id, physicsClientId=env.client_id)[0],
            dtype=np.float64,
        )
        displacement = final_position - initial_position
        final_reached = stage_joint_errors["extraction"] <= float(max_final_joint_error_rad)
        fixed_collision = any(fixed_hard_stages.values())
        extraction_progress = summarize_extraction_progress(
            actual_displacement=displacement,
            planned_carried_positions=carried_target_paths["extraction"],
            minimum_progress_fraction=minimum_extraction_progress_fraction,
        )
        other_displacements = {
            str(instance_id): float(
                np.linalg.norm(
                    np.asarray(
                        env._p.getBasePositionAndOrientation(instance_id, physicsClientId=env.client_id)[0],
                        dtype=np.float64,
                    )
                    - position
                )
            )
            for instance_id, position in initial_other_positions.items()
        }
        monitored_displacement = summarize_monitored_displacement(
            object_displacements_m=other_displacements,
            monitored_instance_ids=monitored_ids,
            maximum_displacement_m=maximum_monitored_object_displacement_m,
        )
        success = bool(
            grasp_reached
            and final_reached
            and extraction_progress["target_extracted"]
            and monitored_displacement["monitored_objects_stable"]
            and not fixed_collision
        )
        return {
            "success": success,
            "success_definition": (
                "grasp/final joint errors <= {:.3f} rad, carried-target progress along the "
                "candidate extraction >= {:.3f}, "
                "monitored blocker displacement <= {:.3f} m, "
                "and no robot/carried-target fixed-environment penetration deeper than {:.3f} m at "
                "sampled execution configurations"
            ).format(
                max_final_joint_error_rad,
                minimum_extraction_progress_fraction,
                maximum_monitored_object_displacement_m,
                hard_penetration_m,
            ),
            "forced_attachment": True,
            "target_initial_position": initial_position.tolist(),
            "target_final_position": final_position.tolist(),
            "target_displacement": displacement.tolist(),
            "extraction_progress": extraction_progress,
            "monitored_displacement": monitored_displacement,
            "stage_max_joint_error_rad": stage_joint_errors,
            "robot_fixed_contacts_by_stage": fixed_contact_stages,
            "fixed_hard_penetrations_by_stage": fixed_hard_stages,
            "robot_removed_blocker_contacts_by_stage": blocker_contact_stages,
            "removed_blocker_hard_penetrations_by_stage": blocker_hard_stages,
            "hard_penetration_m": float(hard_penetration_m),
            "randomization": randomization_record,
            "max_other_object_displacement_m": max(other_displacements.values(), default=0.0),
            "other_object_displacements_m": other_displacements,
        }
    finally:
        if constraint_id is not None:
            env._p.removeConstraint(constraint_id, physicsClientId=env.client_id)


def evaluate_counterfactual_candidate(
    env: Any,
    *,
    pre_action_dir: Path | str,
    scene_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_debug: Mapping[str, Any],
    randomization_seeds: Sequence[int] = (0,),
) -> Dict[str, Any]:
    """Execute paired intact/removal trials across small deterministic perturbations."""

    seeds = [int(value) for value in randomization_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("randomization_seeds must be non-empty and unique")
    hard_penetration_m = float(
        scene_record.get("oracle_context", {})
        .get("action_family", {})
        .get("hard_penetration_m", 0.002)
    )
    trials: List[Dict[str, Any]] = []
    for seed in seeds:
        randomization = CounterfactualRandomizationConfig(seed=seed)
        intact = execute_forced_attachment_extraction_trial(
            env,
            pre_action_dir=pre_action_dir,
            target_instance_id=int(candidate["target_instance_id"]),
            removed_instance_ids=[],
            monitored_instance_ids=candidate["removed_instance_ids"],
            candidate_debug=candidate_debug,
            randomization=randomization,
            hard_penetration_m=hard_penetration_m,
        )
        intervention = execute_forced_attachment_extraction_trial(
            env,
            pre_action_dir=pre_action_dir,
            target_instance_id=int(candidate["target_instance_id"]),
            removed_instance_ids=candidate["removed_instance_ids"],
            monitored_instance_ids=candidate["removed_instance_ids"],
            candidate_debug=candidate_debug,
            randomization=randomization,
            hard_penetration_m=hard_penetration_m,
        )
        if not intact["success"] and intervention["success"]:
            outcome = "hard_blockage_supported"
        elif intact["success"]:
            outcome = "contact_tolerated"
        elif any(intervention["fixed_hard_penetrations_by_stage"].values()):
            outcome = "invalid_fixed_environment_path"
        else:
            outcome = "unresolved_failure"
        trials.append(
            {
                "target_instance_id": int(candidate["target_instance_id"]),
                "removed_instance_ids": candidate["removed_instance_ids"],
                "grasp_id": candidate.get("grasp_id"),
                "randomization_id": "seed_{}".format(seed),
                "intact_success": bool(intact["success"]),
                "intervention_success": bool(intervention["success"]),
                "metadata": {
                    "trajectory_id": candidate["trajectory_id"],
                    "stratum": candidate["stratum"],
                    "pair_scores": dict(candidate.get("pair_scores") or {}),
                    "minimum_blocker_signed_distance_m": candidate.get(
                        "minimum_blocker_signed_distance_m"
                    ),
                    "contact_outcome": outcome,
                    "intact_execution": intact,
                    "intervention_execution": intervention,
                },
            }
        )
    return build_counterfactual_validation_record(
        sample_id=str(scene_record["sample_id"]),
        trials=trials,
        target_method=ACTION_CONDITIONED_TARGET_METHOD_V1,
        metadata={
            "validation_kind": "dynamic_pybullet_forced_attachment_randomized_v1",
            "pre_action_dir": str(pre_action_dir),
            "randomization_seeds": seeds,
            "limitation": "tests access/extraction dynamics after forced attachment, not autonomous grasp closure",
        },
    )


__all__ = [
    "OracleActionFamilyConfig",
    "CounterfactualRandomizationConfig",
    "StaticOraclePerturbationConfig",
    "apply_counterfactual_randomization",
    "apply_static_oracle_pose_yaw_perturbation",
    "aggregate_prototype_records",
    "build_candidate_planner_swept_features",
    "build_geometry_pseudo_gt_adjacency",
    "build_cnabu_runtime_candidate_action_mask",
    "build_cnabu_runtime_candidate_kinematic_mask",
    "build_runtime_candidate_action_mask",
    "build_runtime_candidate_kinematic_mask",
    "cnabu_sparse_support_world_aabbs",
    "cnabu_sparse_support_world_voxels",
    "compare_relation_targets",
    "counterfactual_candidate_descriptor",
    "evaluate_counterfactual_candidate",
    "evaluate_saved_scene",
    "execute_forced_attachment_extraction_trial",
    "extract_gt_object_records",
    "front_extraction_waypoints",
    "interpolate_joint_configs",
    "list_counterfactual_candidates",
    "merge_instance_stack",
    "render_scene_oracle_debug",
    "rethreshold_static_oracle_contact_evidence",
    "restore_saved_scene",
    "summarize_extraction_progress",
    "summarize_monitored_displacement",
    "summarize_signed_distances",
]
