"""MEM adapters for CNABU visual-occlusion attribution.

This module owns coordinate conversion and privileged offline GT support
construction.  Portable ray-ordered relation math remains in
``scene_graph_mem.relations.belief_occlusion``.  Runtime helpers in this file
use only CNABU beliefs and learned-node supports; functions whose names begin
with ``build_gt`` or ``align_oracle`` are evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def info_gain_raycast_to_canonical_zyx(
    raycast: Any,
    *,
    grid_shape_zyx: Sequence[int],
    crop_rows: Sequence[int],
    raw_shape_hw: Sequence[int],
) -> np.ndarray:
    """Convert planner ``InfoGainEval`` indices to CNABU ``[z,ycrop,x]``.

    The existing planner samples ``voxels[ray[...,1], W-ray[...,0],
    ray[...,2]]`` and uses a full zero triplet as padding.  This function
    applies that exact convention, subtracts the CNABU raw-y crop offset, and
    replaces padding or out-of-crop samples with ``[-1,-1,-1]``.
    Camera-near to camera-far order is preserved.
    """

    raw = _host_array(raycast)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("raycast must have shape [R,L,3]")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("raycast indices must be integer-valued")
    shape = tuple(int(value) for value in grid_shape_zyx)
    rows = tuple(int(value) for value in crop_rows)
    raw_shape = tuple(int(value) for value in raw_shape_hw)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("grid_shape_zyx must contain three positive values")
    if len(rows) != 2 or rows[0] < 0 or rows[1] <= rows[0]:
        raise ValueError("crop_rows must contain increasing non-negative bounds")
    if len(raw_shape) != 2 or any(value <= 0 for value in raw_shape):
        raise ValueError("raw_shape_hw must contain two positive values")
    if rows[1] > raw_shape[0]:
        raise ValueError("crop_rows exceed raw_shape_hw")
    if shape[1] != rows[1] - rows[0] or shape[2] != raw_shape[1]:
        raise ValueError("CNABU grid y/x dimensions do not match crop/raw width")

    values = raw.astype(np.int64, copy=False)
    result = np.full(values.shape, -1, dtype=np.int32)
    padding = np.all(values == 0, axis=-1)
    z = values[..., 2]
    y = values[..., 1] - rows[0]
    # This intentionally has no ``-1``: it mirrors the planner's established
    # ``width - ray_x`` indexing expression exactly.
    x = raw_shape[1] - values[..., 0]
    valid = (
        ~padding
        & (z >= 0)
        & (z < shape[0])
        & (y >= 0)
        & (y < shape[1])
        & (x >= 0)
        & (x < shape[2])
    )
    result[..., 0][valid] = z[valid].astype(np.int32)
    result[..., 1][valid] = y[valid].astype(np.int32)
    result[..., 2][valid] = x[valid].astype(np.int32)
    return result


def merge_last_valid_instance_stack(
    instance_maps: Any,
    *,
    invalid_value: int = -1,
) -> np.ndarray:
    """Merge MEM instance-map views using the project's last-valid convention."""

    stack = _host_array(instance_maps)
    if stack.ndim == 2:
        return stack.copy()
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError("instance_maps must have shape [H,W] or non-empty [V,H,W]")
    merged = np.full(stack.shape[1:], invalid_value, dtype=stack.dtype)
    for layer in stack:
        np.copyto(merged, layer, where=layer != invalid_value)
    return merged


@dataclass(frozen=True)
class GTObjectVoxelSupports:
    """Privileged object/fixed supports in canonical CNABU coordinates."""

    supports_zyx: np.ndarray
    masks_raw_hw: np.ndarray
    instance_ids: Tuple[int, ...]
    class_ids: Tuple[int, ...]
    fixed_environment_support_zyx: np.ndarray
    unrepresented_object_support_zyx: np.ndarray
    occupied_support_zyx: np.ndarray
    crop_rows: Tuple[int, int]
    occupancy_threshold: float

    def __post_init__(self) -> None:
        supports = np.asarray(self.supports_zyx, dtype=bool)
        masks = np.asarray(self.masks_raw_hw, dtype=bool)
        fixed = np.asarray(self.fixed_environment_support_zyx, dtype=bool)
        unrepresented = np.asarray(self.unrepresented_object_support_zyx, dtype=bool)
        occupied = np.asarray(self.occupied_support_zyx, dtype=bool)
        if supports.ndim != 4 or fixed.shape != supports.shape[1:]:
            raise ValueError("GT supports must use aligned [N,Z,Ycrop,X] geometry")
        if masks.ndim != 3 or masks.shape[0] != supports.shape[0]:
            raise ValueError("GT raw masks must align with object supports")
        if not (
            len(self.instance_ids) == len(self.class_ids) == supports.shape[0]
        ):
            raise ValueError("GT instance/class identifiers must align with supports")
        if unrepresented.shape != fixed.shape or occupied.shape != fixed.shape:
            raise ValueError("GT residual supports must share the canonical grid")
        if supports.shape[0] and np.any(supports.sum(axis=0) > 1):
            raise ValueError("GT object supports must be disjoint")
        if np.any(fixed & (supports.any(axis=0) if supports.shape[0] else False)):
            raise ValueError("fixed environment and GT object supports must be disjoint")
        if np.any(unrepresented & fixed):
            raise ValueError("unrepresented object and fixed supports must be disjoint")
        object_union = supports.any(axis=0) if supports.shape[0] else np.zeros_like(fixed)
        if np.any((object_union | fixed | unrepresented) & ~occupied):
            raise ValueError("GT partitions must be subsets of occupied support")

    @property
    def num_objects(self) -> int:
        return int(self.supports_zyx.shape[0])

    def coverage_summary(self) -> Dict[str, Any]:
        object_union = (
            self.supports_zyx.any(axis=0)
            if self.num_objects
            else np.zeros_like(self.occupied_support_zyx)
        )
        object_voxels = int(object_union.sum())
        missing = int(self.unrepresented_object_support_zyx.sum())
        return {
            "object_count": self.num_objects,
            "occupied_voxel_count": int(self.occupied_support_zyx.sum()),
            "represented_object_voxel_count": object_voxels,
            "unrepresented_object_voxel_count": missing,
            "fixed_environment_voxel_count": int(
                self.fixed_environment_support_zyx.sum()
            ),
            "object_support_coverage": (
                float(object_voxels / (object_voxels + missing))
                if object_voxels + missing
                else None
            ),
        }


def _majority_object_class(
    semantic_2d: np.ndarray,
    mask: np.ndarray,
    *,
    object_class_max_exclusive: int,
) -> Optional[int]:
    values = semantic_2d[mask]
    if values.size == 0:
        return None
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    if values.size == 0:
        return None
    labels, counts = np.unique(values, return_counts=True)
    majority = int(labels[int(np.argmax(counts))])
    return majority if majority < int(object_class_max_exclusive) else None


def build_gt_object_voxel_supports(
    *,
    hm3d: Any,
    semantic_2d: Any,
    semantic_3d: Any,
    instance_maps: Any,
    crop_rows: Sequence[int],
    occupancy_threshold: float = 0.5,
    object_class_max_exclusive: int = 14,
    min_mask_pixels: int = 1,
) -> GTObjectVoxelSupports:
    """Build evaluation-only object supports from saved MEM GT arrays.

    Object identity comes only from the merged simulator instance masks.  A
    voxel belongs to an object when its raw column has that instance id and
    its semantic-3D class agrees with the mask's majority object class.  Any
    occupied object-class voxel not represented by those supports is retained
    as explicit unrepresented mass.
    """

    occupancy_values = _host_array(hm3d).astype(np.float64, copy=False)
    labels_2d = _host_array(semantic_2d)
    labels_3d = _host_array(semantic_3d)
    if occupancy_values.ndim != 3:
        raise ValueError("hm3d must have shape [Hraw,W,Z]")
    if labels_2d.shape != occupancy_values.shape[:2]:
        raise ValueError("semantic_2d must align with hm3d H,W")
    if labels_3d.shape != occupancy_values.shape:
        raise ValueError("semantic_3d must align with hm3d")
    if not np.isfinite(occupancy_values).all():
        raise ValueError("hm3d contains NaN or Inf")
    if not 0.0 <= float(occupancy_threshold) <= 1.0:
        raise ValueError("occupancy_threshold must be in [0,1]")
    if int(object_class_max_exclusive) <= 0 or int(min_mask_pixels) <= 0:
        raise ValueError("class boundary and minimum mask size must be positive")
    rows = tuple(int(value) for value in crop_rows)
    if (
        len(rows) != 2
        or rows[0] < 0
        or rows[1] > occupancy_values.shape[0]
        or rows[1] <= rows[0]
    ):
        raise ValueError("crop_rows are invalid for hm3d")

    merged = merge_last_valid_instance_stack(instance_maps)
    if merged.shape != labels_2d.shape:
        raise ValueError("merged instance maps must align with semantic_2d")
    occupied_raw = occupancy_values > float(occupancy_threshold)
    object_records = []
    for raw_id in np.unique(merged):
        instance_id = int(round(float(raw_id)))
        if instance_id <= 0 or not np.isclose(float(raw_id), instance_id):
            continue
        mask = merged == raw_id
        if int(mask.sum()) < int(min_mask_pixels):
            continue
        class_id = _majority_object_class(
            labels_2d,
            mask,
            object_class_max_exclusive=int(object_class_max_exclusive),
        )
        if class_id is None:
            continue
        object_records.append((instance_id, class_id, mask))
    object_records.sort(key=lambda item: int(item[0]))

    supports_raw = []
    for _instance_id, class_id, mask in object_records:
        supports_raw.append(
            occupied_raw
            & mask[:, :, None]
            & (labels_3d == int(class_id))
        )
    if supports_raw:
        support_stack_raw = np.stack(supports_raw).astype(bool, copy=False)
        if np.any(support_stack_raw.sum(axis=0) > 1):
            raise ValueError("merged GT instance masks produced overlapping voxel supports")
    else:
        support_stack_raw = np.zeros((0,) + occupied_raw.shape, dtype=bool)

    object_semantic_raw = labels_3d < int(object_class_max_exclusive)
    represented_raw = (
        support_stack_raw.any(axis=0)
        if len(support_stack_raw)
        else np.zeros_like(occupied_raw)
    )
    unrepresented_raw = occupied_raw & object_semantic_raw & ~represented_raw
    fixed_raw = occupied_raw & ~object_semantic_raw

    def canonical(volume: np.ndarray) -> np.ndarray:
        return np.transpose(volume[rows[0] : rows[1], :, :], (2, 0, 1))

    supports = np.stack([canonical(value) for value in support_stack_raw]) if len(
        support_stack_raw
    ) else np.zeros(
        (0, occupancy_values.shape[2], rows[1] - rows[0], occupancy_values.shape[1]),
        dtype=bool,
    )
    return GTObjectVoxelSupports(
        supports_zyx=supports,
        masks_raw_hw=np.stack([value[2] for value in object_records])
        if object_records
        else np.zeros((0,) + labels_2d.shape, dtype=bool),
        instance_ids=tuple(int(value[0]) for value in object_records),
        class_ids=tuple(int(value[1]) for value in object_records),
        fixed_environment_support_zyx=canonical(fixed_raw),
        unrepresented_object_support_zyx=canonical(unrepresented_raw),
        occupied_support_zyx=canonical(occupied_raw),
        crop_rows=rows,
        occupancy_threshold=float(occupancy_threshold),
    )


@dataclass(frozen=True)
class NodeGTMatching:
    """One-to-one, class-aware learned-node to GT-object matching."""

    node_to_gt_index: Tuple[int, ...]
    matched_iou: Tuple[float, ...]
    iou_threshold: float
    gt_object_count: int

    def __post_init__(self) -> None:
        if len(self.node_to_gt_index) != len(self.matched_iou):
            raise ValueError("matching arrays must have the same node count")
        if not 0.0 <= float(self.iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in [0,1]")
        seen = []
        for index, iou in zip(self.node_to_gt_index, self.matched_iou):
            if int(index) >= int(self.gt_object_count) or int(index) < -1:
                raise ValueError("node_to_gt_index contains an invalid GT index")
            if int(index) >= 0:
                seen.append(int(index))
                if float(iou) < float(self.iou_threshold):
                    raise ValueError("matched IoU is below the matching threshold")
            elif float(iou) != 0.0:
                raise ValueError("unmatched nodes must have zero matched_iou")
        if len(seen) != len(set(seen)):
            raise ValueError("GT matching must be one-to-one")

    @property
    def matched_node_count(self) -> int:
        return int(sum(index >= 0 for index in self.node_to_gt_index))

    @property
    def unmatched_gt_indices(self) -> Tuple[int, ...]:
        matched = {int(index) for index in self.node_to_gt_index if index >= 0}
        return tuple(index for index in range(self.gt_object_count) if index not in matched)

    def to_dict(self, *, gt_instance_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        instance_ids = None if gt_instance_ids is None else tuple(int(v) for v in gt_instance_ids)
        if instance_ids is not None and len(instance_ids) != int(self.gt_object_count):
            raise ValueError("gt_instance_ids must align with the GT object count")
        records = []
        for node_index, (gt_index, iou) in enumerate(
            zip(self.node_to_gt_index, self.matched_iou)
        ):
            records.append(
                {
                    "node_index": int(node_index),
                    "gt_index": int(gt_index) if gt_index >= 0 else None,
                    "gt_instance_id": (
                        int(instance_ids[gt_index])
                        if instance_ids is not None and gt_index >= 0
                        else None
                    ),
                    "iou": float(iou) if gt_index >= 0 else None,
                }
            )
        return {
            "method": "class_aware_hungarian_mask_iou_one_to_one",
            "iou_threshold": float(self.iou_threshold),
            "node_count": len(self.node_to_gt_index),
            "gt_object_count": int(self.gt_object_count),
            "matched_node_count": self.matched_node_count,
            "unmatched_gt_indices": list(self.unmatched_gt_indices),
            "records": records,
        }


def match_nodes_to_gt_objects(
    *,
    node_masks_raw_hw: Any,
    node_class_ids: Sequence[int],
    gt_masks_raw_hw: Any,
    gt_class_ids: Sequence[int],
    iou_threshold: float = 0.25,
) -> NodeGTMatching:
    """Maximize total class-compatible mask IoU, then reject weak pairs."""

    nodes = _host_array(node_masks_raw_hw).astype(bool, copy=False)
    gt = _host_array(gt_masks_raw_hw).astype(bool, copy=False)
    node_classes = np.asarray(node_class_ids, dtype=np.int64)
    gt_classes = np.asarray(gt_class_ids, dtype=np.int64)
    if nodes.ndim != 3 or gt.ndim != 3 or nodes.shape[1:] != gt.shape[1:]:
        raise ValueError("node and GT masks must be aligned [N,H,W] arrays")
    if node_classes.shape != (len(nodes),) or gt_classes.shape != (len(gt),):
        raise ValueError("class identifiers must align with node/GT masks")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in [0,1]")
    node_to_gt = np.full(len(nodes), -1, dtype=np.int64)
    matched_iou = np.zeros(len(nodes), dtype=np.float64)
    if len(nodes) and len(gt):
        intersection = np.logical_and(nodes[:, None], gt[None]).sum(axis=(2, 3))
        union = np.logical_or(nodes[:, None], gt[None]).sum(axis=(2, 3))
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union > 0,
        )
        compatible = node_classes[:, None] == gt_classes[None, :]
        # Incompatible assignments are more expensive than any valid IoU loss.
        costs = np.where(compatible, 1.0 - iou, 2.0)
        row_indices, column_indices = linear_sum_assignment(costs)
        for node_index, gt_index in zip(row_indices.tolist(), column_indices.tolist()):
            value = float(iou[node_index, gt_index])
            if bool(compatible[node_index, gt_index]) and value >= float(iou_threshold):
                node_to_gt[node_index] = int(gt_index)
                matched_iou[node_index] = value
    return NodeGTMatching(
        node_to_gt_index=tuple(int(value) for value in node_to_gt.tolist()),
        matched_iou=tuple(float(value) for value in matched_iou.tolist()),
        iou_threshold=float(iou_threshold),
        gt_object_count=int(len(gt)),
    )


@dataclass(frozen=True)
class OracleNodeAlignment:
    source_supports_zyx: np.ndarray
    target_supports_zyx: np.ndarray
    target_source_indices: np.ndarray
    unrepresented_support_zyx: np.ndarray
    fixed_environment_support_zyx: np.ndarray
    target_defined_by_matching: np.ndarray


def align_oracle_supports_to_nodes(
    gt_supports: GTObjectVoxelSupports,
    matching: NodeGTMatching,
) -> OracleNodeAlignment:
    """Align privileged GT sources/targets to runtime node order explicitly."""

    if int(matching.gt_object_count) != int(gt_supports.num_objects):
        raise ValueError("matching and GT supports use different object counts")
    node_count = len(matching.node_to_gt_index)
    shape = gt_supports.supports_zyx.shape[1:]
    aligned = np.zeros((node_count,) + shape, dtype=bool)
    target_source = np.full(node_count, -1, dtype=np.int32)
    defined = np.zeros(node_count, dtype=bool)
    for node_index, gt_index in enumerate(matching.node_to_gt_index):
        if int(gt_index) < 0:
            continue
        aligned[node_index] = gt_supports.supports_zyx[int(gt_index)]
        target_source[node_index] = int(node_index)
        defined[node_index] = bool(aligned[node_index].any())

    unmatched_support = np.zeros(shape, dtype=bool)
    for gt_index in matching.unmatched_gt_indices:
        unmatched_support |= gt_supports.supports_zyx[int(gt_index)]
    unrepresented = gt_supports.unrepresented_object_support_zyx | unmatched_support
    represented = aligned.any(axis=0) if node_count else np.zeros(shape, dtype=bool)
    unrepresented &= ~represented
    return OracleNodeAlignment(
        source_supports_zyx=aligned.copy(),
        target_supports_zyx=aligned.copy(),
        target_source_indices=target_source,
        unrepresented_support_zyx=unrepresented,
        fixed_environment_support_zyx=gt_supports.fixed_environment_support_zyx.copy(),
        target_defined_by_matching=defined,
    )


def dense_supports_from_sparse_indices(
    indices_zyx: Sequence[Any],
    *,
    grid_shape_zyx: Sequence[int],
) -> np.ndarray:
    shape = tuple(int(value) for value in grid_shape_zyx)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("grid_shape_zyx must contain three positive dimensions")
    dense = np.zeros((len(indices_zyx),) + shape, dtype=bool)
    for node_index, raw_indices in enumerate(indices_zyx):
        indices = _host_array(raw_indices).astype(np.int64, copy=False)
        if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0:
            raise ValueError("each sparse node support must be non-empty [M,3]")
        if np.any(indices < 0):
            raise ValueError("sparse support indices must be non-negative")
        for axis, limit in enumerate(shape):
            if np.any(indices[:, axis] >= limit):
                raise ValueError("sparse support indices exceed grid_shape_zyx")
        dense[node_index, indices[:, 0], indices[:, 1], indices[:, 2]] = True
    if len(dense) and np.any(dense.sum(axis=0) > 1):
        raise ValueError("learned-node supports must be disjoint")
    return dense


@dataclass(frozen=True)
class RuntimeSupportPartition:
    source_supports_zyx: np.ndarray
    unrepresented_support_zyx: np.ndarray
    fixed_environment_support_zyx: np.ndarray
    occupied_support_zyx: np.ndarray

    def coverage_summary(self) -> Dict[str, Any]:
        represented = (
            self.source_supports_zyx.any(axis=0)
            if len(self.source_supports_zyx)
            else np.zeros_like(self.occupied_support_zyx)
        )
        occupied_count = int(self.occupied_support_zyx.sum())
        return {
            "occupied_voxel_count": occupied_count,
            "represented_node_voxel_count": int(represented.sum()),
            "unrepresented_object_voxel_count": int(
                self.unrepresented_support_zyx.sum()
            ),
            "fixed_environment_voxel_count": int(
                self.fixed_environment_support_zyx.sum()
            ),
            "partition_coverage": (
                float(
                    (
                        represented
                        | self.unrepresented_support_zyx
                        | self.fixed_environment_support_zyx
                    ).sum()
                    / occupied_count
                )
                if occupied_count
                else None
            ),
        }


def build_runtime_support_partition(
    *,
    occupancy_mean: Any,
    semantic_mean: Any,
    source_supports_zyx: Any,
    occupancy_threshold: float = 0.5,
    object_class_max_exclusive: int = 14,
) -> RuntimeSupportPartition:
    """Partition CNABU occupancy without GT or simulator instance ids."""

    occupancy = _host_array(occupancy_mean).astype(np.float64, copy=False)
    semantic = _host_array(semantic_mean).astype(np.float64, copy=False)
    sources = _host_array(source_supports_zyx).astype(bool, copy=False)
    if occupancy.ndim != 3 or semantic.ndim != 3 or semantic.shape[1:] != occupancy.shape[1:]:
        raise ValueError("occupancy and semantic means must align as [Z,Y,X]/[K,Y,X]")
    if sources.ndim != 4 or sources.shape[1:] != occupancy.shape:
        raise ValueError("source supports must align as [N,Z,Y,X]")
    if len(sources) and np.any(sources.sum(axis=0) > 1):
        raise ValueError("source supports must be disjoint")
    if not 0.0 <= float(occupancy_threshold) <= 1.0:
        raise ValueError("occupancy_threshold must be in [0,1]")
    if int(object_class_max_exclusive) <= 0 or int(object_class_max_exclusive) > len(semantic):
        raise ValueError("object_class_max_exclusive must index semantic channels")
    if not np.isfinite(occupancy).all() or not np.isfinite(semantic).all():
        raise ValueError("CNABU beliefs contain NaN or Inf")

    occupied = occupancy >= float(occupancy_threshold)
    represented = sources.any(axis=0) if len(sources) else np.zeros_like(occupied)
    labels = semantic.argmax(axis=0)
    object_columns = labels < int(object_class_max_exclusive)
    unrepresented = occupied & object_columns[None, :, :] & ~represented
    fixed = occupied & ~object_columns[None, :, :] & ~represented
    return RuntimeSupportPartition(
        source_supports_zyx=sources.copy(),
        unrepresented_support_zyx=unrepresented,
        fixed_environment_support_zyx=fixed,
        occupied_support_zyx=occupied,
    )


__all__ = [
    "GTObjectVoxelSupports",
    "NodeGTMatching",
    "OracleNodeAlignment",
    "RuntimeSupportPartition",
    "align_oracle_supports_to_nodes",
    "build_gt_object_voxel_supports",
    "build_runtime_support_partition",
    "dense_supports_from_sparse_indices",
    "info_gain_raycast_to_canonical_zyx",
    "match_nodes_to_gt_objects",
    "merge_last_valid_instance_stack",
]
