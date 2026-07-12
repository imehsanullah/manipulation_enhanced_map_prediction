"""Runtime scene-graph extraction from CNABU map beliefs.

This module intentionally stays inference-only. It extracts CNABU 3D semantic
connected components as pseudo-object nodes, then applies a deterministic
front/lateral-overlap rule for directed ``blocks_access_to`` edges.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


DEFAULT_YCB_CLASS_NAMES = (
    "YcbTomatoSoupCan",
    "Ycbsuger1",
    "YcbPottedMeatCan",
    "YcbOrionPie",
    "YcbMustardBottle",
    "YcbMasterChefCan",
    "YcbGelatinBox",
    "YcbCrackerBox",
    "YcbChipsCan",
    "YcbBleachCleanser",
    "backmeel",
    "collezione",
    "muesli",
    "vollmilch",
)

DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS = (
    206,
    161,
    180,
    163,
    278,
    448,
    138,
    215,
    333,
    244,
    200,
    718,
    391,
    236,
)
DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS = (
    16,
    14,
    15,
    15,
    22,
    35,
    19,
    21,
    26,
    22,
    20,
    41,
    31,
    21,
)
DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS = (
    16,
    14,
    15,
    15,
    22,
    35,
    19,
    21,
    26,
    21,
    20,
    42,
    31,
    21,
)


@dataclass(frozen=True)
class ComponentExtractionConfig:
    occupancy_threshold: float = 0.50
    semantic_confidence_threshold: float = 0.0
    max_semantic_vacuity: Optional[float] = None
    max_occupancy_epistemic: Optional[float] = None
    min_voxels: int = 50
    min_pixels: int = 5
    connectivity: int = 1
    object_class_max_exclusive: int = 14


@dataclass(frozen=True)
class ComponentSplitConfig:
    enabled: bool = False
    method: str = "seeded_distance_watershed"
    core_occupancy_threshold: float = 0.70
    core_semantic_confidence_threshold: float = 0.0
    min_seed_voxels: int = 10
    min_seed_pixels: int = 3
    min_split_voxels: int = 50
    min_split_pixels: int = 5
    min_split_seeds: int = 2
    max_splits: int = 8
    candidate_area_multiplier: float = 1.65
    candidate_bbox_multiplier: float = 1.65
    candidate_min_area_pixels: int = 40
    footprint_erosion_iterations: int = 2
    min_child_area_fraction: float = 0.25
    min_neck_removed_fraction: float = 0.10
    class_area_prior_pixels: Sequence[int] = DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS
    class_width_prior_pixels: Sequence[int] = DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS
    class_height_prior_pixels: Sequence[int] = DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS


@dataclass(frozen=True)
class BlocksAccessRuleConfig:
    relation: str = "blocks_access_to"
    access_axis: str = "y"
    opening_side: str = "low"
    lateral_axis: Optional[str] = None
    min_front_gap: float = 0.0
    min_lateral_overlap: float = 0.0
    lateral_overlap_mode: str = "union"


def predict_scene_graph_from_cnabu(
    *,
    cnabu_path: Optional[str | Path] = None,
    occupancy_mean: Any = None,
    semantic_mean: Any = None,
    occupancy_alpha: Any = None,
    occupancy_beta: Any = None,
    semantic_concentration: Any = None,
    occupancy_epistemic: Any = None,
    semantic_vacuity: Any = None,
    occupancy_distribution: Any = None,
    raw_shape_hw: Optional[Sequence[int]] = None,
    crop_rows: Optional[Sequence[int]] = None,
    selected_view_indices: Optional[Sequence[int]] = None,
    component_config: ComponentExtractionConfig | Mapping[str, Any] | None = None,
    component_split_config: ComponentSplitConfig | Mapping[str, Any] | None = None,
    edge_config: BlocksAccessRuleConfig | Mapping[str, Any] | None = None,
    class_names: Sequence[str] = DEFAULT_YCB_CLASS_NAMES,
    sample_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    include_masks: bool = True,
) -> Dict[str, Any]:
    """Build a serializable scene graph from CNABU arrays or ``cnabu_hms.npz``.

    Runtime callers can pass CNABU means directly, evidential alpha/beta and
    Dirichlet concentration arrays, or the interleaved occupancy beta
    distribution used by ``run_cnabu_pipeline.py``. If ``cnabu_path`` is
    provided, missing arrays and shape metadata are read from that file.
    """

    loaded = _load_cnabu_npz(Path(cnabu_path)) if cnabu_path is not None else {}
    loaded_arrays = loaded.get("arrays", {})
    source_metadata = loaded.get("metadata", {})

    occupancy_distribution = _first_not_none(
        occupancy_distribution,
        loaded_arrays.get("occupancy_distribution"),
    )
    occupancy_alpha = _first_not_none(occupancy_alpha, loaded_arrays.get("occupancy_alpha"))
    occupancy_beta = _first_not_none(occupancy_beta, loaded_arrays.get("occupancy_beta"))
    occupancy_mean = _first_not_none(occupancy_mean, loaded_arrays.get("occupancy_mean"))
    occupancy_epistemic = _first_not_none(
        occupancy_epistemic,
        loaded_arrays.get("occupancy_epistemic"),
    )
    semantic_concentration = _first_not_none(
        semantic_concentration,
        loaded_arrays.get("semantic_concentration"),
    )
    semantic_mean = _first_not_none(semantic_mean, loaded_arrays.get("semantic_mean"))
    semantic_vacuity = _first_not_none(semantic_vacuity, loaded_arrays.get("semantic_vacuity"))
    crop_rows = _first_not_none(crop_rows, loaded_arrays.get("crop_rows"))
    selected_view_indices = _first_not_none(
        selected_view_indices,
        loaded_arrays.get("selected_view_indices"),
    )
    raw_shape_hw = _first_not_none(
        raw_shape_hw,
        source_metadata.get("raw_shape_hw"),
        loaded_arrays.get("raw_shape_hms"),
    )

    component_cfg = _coerce_config(component_config, ComponentExtractionConfig)
    split_cfg = _coerce_config(component_split_config, ComponentSplitConfig)
    edge_cfg = _coerce_config(edge_config, BlocksAccessRuleConfig)

    occupancy_mean_array, occupancy_epistemic_array = _derive_occupancy_arrays(
        occupancy_mean=occupancy_mean,
        occupancy_alpha=occupancy_alpha,
        occupancy_beta=occupancy_beta,
        occupancy_distribution=occupancy_distribution,
        occupancy_epistemic=occupancy_epistemic,
    )
    semantic_mean_array, semantic_vacuity_array = _derive_semantic_arrays(
        semantic_mean=semantic_mean,
        semantic_concentration=semantic_concentration,
        semantic_vacuity=semantic_vacuity,
    )

    raw_height, raw_width = _shape_hw_from_value(raw_shape_hw, fallback_hw=semantic_mean_array.shape[1:])
    crop_start, crop_stop = _crop_rows_from_value(
        crop_rows,
        crop_height=int(semantic_mean_array.shape[1]),
        raw_height=raw_height,
    )

    nodes, shape_info = extract_cnabu_component_nodes(
        occupancy_mean=occupancy_mean_array,
        semantic_mean=semantic_mean_array,
        occupancy_epistemic=occupancy_epistemic_array,
        semantic_vacuity=semantic_vacuity_array,
        raw_shape_hw=(raw_height, raw_width),
        crop_rows=(crop_start, crop_stop),
        config=component_cfg,
        split_config=split_cfg,
        class_names=class_names,
        include_masks=include_masks,
    )
    edges, edge_rule_info = build_blocks_access_edges(
        nodes,
        config=edge_cfg,
        image_shape_hw=(raw_height, raw_width),
    )
    adjacency = _adjacency_matrix(nodes, edges)

    graph_metadata: Dict[str, Any] = {
        "source": "cnabu_hms_npz" if cnabu_path is not None else "cnabu_arrays",
        "source_path": str(cnabu_path) if cnabu_path is not None else None,
        "sample_id": sample_id,
        "node_source": "cnabu_3d_components",
        "edge_source": "deterministic_front_lateral_overlap_rule",
        "requires_gt": False,
        "uses_gt": False,
        "uses_simulator_instance_labels": False,
        "uses_d3g": False,
        "uses_learned_graph_checkpoint": False,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "raw_shape_hw": [raw_height, raw_width],
        "crop_rows": [crop_start, crop_stop],
        "selected_view_indices": _int_list(selected_view_indices),
        "shape_info": shape_info,
        "source_metadata": source_metadata,
        "caller_metadata": dict(metadata or {}),
    }

    return {
        "schema": "mem_cnabu_rule_scene_graph_v0",
        "relation": edge_cfg.relation,
        "nodes": nodes,
        "edges": edges,
        "adjacency_matrix": adjacency,
        "thresholds": {
            "component_extraction": asdict(component_cfg),
            "component_splitting": asdict(split_cfg),
            "edge_rule": {
                **asdict(edge_cfg),
                "resolved_lateral_axis": edge_rule_info["resolved_lateral_axis"],
                "overlap_metric_note": (
                    "Edges require positive lateral pixel overlap and "
                    "lateral_overlap_mode >= min_lateral_overlap."
                ),
            },
        },
        "metadata": graph_metadata,
    }


def extract_cnabu_component_nodes(
    *,
    occupancy_mean: Any,
    semantic_mean: Any,
    occupancy_epistemic: Any = None,
    semantic_vacuity: Any = None,
    raw_shape_hw: Sequence[int],
    crop_rows: Sequence[int],
    config: ComponentExtractionConfig | Mapping[str, Any] | None = None,
    split_config: ComponentSplitConfig | Mapping[str, Any] | None = None,
    class_names: Sequence[str] = DEFAULT_YCB_CLASS_NAMES,
    include_masks: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract serializable pseudo-object nodes from CNABU belief arrays."""

    cfg = _coerce_config(config, ComponentExtractionConfig)
    split_cfg = _coerce_config(split_config, ComponentSplitConfig)
    _validate_component_config(cfg)
    _validate_split_config(split_cfg, cfg)

    occupancy = _squeeze_array(_to_numpy(occupancy_mean, "occupancy_mean"), 3, "occupancy_mean").astype(
        np.float32,
        copy=False,
    )
    semantic = _squeeze_array(_to_numpy(semantic_mean, "semantic_mean"), 3, "semantic_mean").astype(
        np.float32,
        copy=False,
    )
    occ_epi = _optional_array(occupancy_epistemic, 3, "occupancy_epistemic")
    sem_vac = _optional_array(semantic_vacuity, 2, "semantic_vacuity")

    if occupancy.shape[1:] != semantic.shape[1:]:
        raise ValueError(
            "occupancy_mean spatial shape {} must match semantic_mean {}".format(
                occupancy.shape[1:],
                semantic.shape[1:],
            )
        )
    if occ_epi is not None and occ_epi.shape != occupancy.shape:
        raise ValueError(f"occupancy_epistemic shape {occ_epi.shape} must match occupancy_mean {occupancy.shape}")
    if sem_vac is not None and sem_vac.shape != semantic.shape[1:]:
        raise ValueError(f"semantic_vacuity shape {sem_vac.shape} must match semantic spatial {semantic.shape[1:]}")

    raw_height, raw_width = _shape_hw_from_value(raw_shape_hw, fallback_hw=semantic.shape[1:])
    crop_start, crop_stop = _crop_rows_from_value(
        crop_rows,
        crop_height=int(semantic.shape[1]),
        raw_height=raw_height,
    )
    crop_height, crop_width = [int(dim) for dim in semantic.shape[1:]]
    if int(raw_width) != crop_width:
        raise ValueError(f"raw width {raw_width} must match CNABU crop width {crop_width}")

    semantic_labels = semantic.argmax(axis=0)
    semantic_confidence = semantic.max(axis=0)
    occupied = occupancy >= float(cfg.occupancy_threshold)
    if occ_epi is not None and cfg.max_occupancy_epistemic is not None:
        occupied = occupied & (occ_epi <= float(cfg.max_occupancy_epistemic))

    semantic_allowed = semantic_confidence >= float(cfg.semantic_confidence_threshold)
    if sem_vac is not None and cfg.max_semantic_vacuity is not None:
        semantic_allowed = semantic_allowed & (sem_vac <= float(cfg.max_semantic_vacuity))

    structure = ndimage.generate_binary_structure(3, int(cfg.connectivity))
    nodes: List[Dict[str, Any]] = []
    component_id = 1
    parent_component_count = 0
    split_parent_count = 0
    split_node_count = 0
    unsplit_node_count = 0
    split_candidate_count = 0
    split_child_region_count = 0

    for class_id in range(int(cfg.object_class_max_exclusive)):
        class_columns = (semantic_labels == int(class_id)) & semantic_allowed
        if not bool(class_columns.any()):
            continue
        class_volume = occupied & class_columns[None, :, :]
        if not bool(class_volume.any()):
            continue

        labels, _ = ndimage.label(class_volume, structure=structure)
        for label_id, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None:
                continue
            full_component = labels == label_id
            parent_voxel_count = int(full_component.sum())
            if parent_voxel_count < int(cfg.min_voxels):
                continue

            parent_crop_mask = np.any(full_component, axis=0)
            parent_pixel_count = int(parent_crop_mask.sum())
            if parent_pixel_count < int(cfg.min_pixels):
                continue

            parent_component_count += 1
            parent_component_id = int(component_id)
            split_regions = _split_component_regions(
                component_mask=full_component,
                occupancy=occupancy,
                semantic_confidence=semantic_confidence,
                parent_component_id=parent_component_id,
                parent_label_id=int(label_id),
                class_id=int(class_id),
                extraction_config=cfg,
                split_config=split_cfg,
                structure=structure,
            )
            if any(bool(item["metadata"].get("candidate_considered", False)) for item in split_regions):
                split_candidate_count += 1
            if any(bool(item["metadata"].get("was_split", False)) for item in split_regions):
                split_parent_count += 1
                split_child_region_count += int(len(split_regions))

            for split_region in split_regions:
                region_component = split_region["mask"]
                region_crop_mask = np.any(region_component, axis=0)
                region_pixel_count = int(region_crop_mask.sum())
                region_voxel_count = int(region_component.sum())
                if region_voxel_count < 1 or region_pixel_count < 1:
                    continue

                mask = np.zeros((raw_height, raw_width), dtype=bool)
                mask[crop_start:crop_stop, :] = region_crop_mask
                bbox_xyxy = _bbox_xyxy_from_mask(mask)
                centroid_yx = _centroid_yx(mask)
                component_occ = occupancy[region_component]
                component_epistemic = occ_epi[region_component] if occ_epi is not None else None
                component_confidence = semantic_confidence[region_crop_mask]
                component_vacuity = sem_vac[region_crop_mask] if sem_vac is not None else None
                mean_occupancy = float(component_occ.mean()) if component_occ.size else 0.0
                mean_semantic_confidence = (
                    float(component_confidence.mean()) if component_confidence.size else 0.0
                )
                z_range, crop_bbox_yx = _component_zyx_bounds(region_component)
                split_metadata = _node_split_metadata(
                    split_region["metadata"],
                    region_voxel_count=region_voxel_count,
                    region_pixel_count=region_pixel_count,
                    mean_occupancy=mean_occupancy,
                    mean_semantic_confidence=mean_semantic_confidence,
                )
                if bool(split_metadata.get("was_split", False)):
                    split_node_count += 1
                else:
                    unsplit_node_count += 1

                node: Dict[str, Any] = {
                    "id": int(component_id),
                    "node_index": int(len(nodes)),
                    "component_id": int(component_id),
                    "node_source": _node_source_from_split_metadata(split_metadata),
                    "parent_component_id": int(split_metadata["parent_component_id"]),
                    "split_id": int(split_metadata["split_id"]),
                    "split_method": split_metadata["method"],
                    "was_split": bool(split_metadata["was_split"]),
                    "class_id": int(class_id),
                    "class_name": _class_name(int(class_id), class_names),
                    "score": float(mean_occupancy * mean_semantic_confidence),
                    "bbox_xyxy_abs": bbox_xyxy,
                    "bbox_format": "xyxy_exclusive",
                    "centroid_yx": [float(centroid_yx[0]), float(centroid_yx[1])],
                    "centroid_xy": [float(centroid_yx[1]), float(centroid_yx[0])],
                    "area_pixels": int(region_pixel_count),
                    "voxel_count": int(region_voxel_count),
                    "z_range": z_range,
                    "crop_bbox_yx": crop_bbox_yx,
                    "confidence": {
                        "mean_occupancy": mean_occupancy,
                        "max_occupancy": float(component_occ.max()) if component_occ.size else 0.0,
                        "mean_semantic_confidence": mean_semantic_confidence,
                    },
                    "uncertainty": {
                        "mean_occupancy_epistemic": (
                            float(component_epistemic.mean())
                            if component_epistemic is not None and component_epistemic.size
                            else None
                        ),
                        "mean_semantic_vacuity": (
                            float(component_vacuity.mean())
                            if component_vacuity is not None and component_vacuity.size
                            else None
                        ),
                    },
                    "split": split_metadata,
                }
                if include_masks:
                    node["mask"] = encode_binary_mask_rle(mask)
                nodes.append(node)
                component_id += 1

    shape_info = {
        "occupancy_mean_shape_zhw": [int(dim) for dim in occupancy.shape],
        "semantic_mean_shape_chw": [int(dim) for dim in semantic.shape],
        "raw_shape_hw": [int(raw_height), int(raw_width)],
        "crop_rows": [int(crop_start), int(crop_stop)],
        "cnabu_shape_hw": [int(crop_height), int(crop_width)],
        "uncertainty_available": {
            "occupancy_epistemic": occ_epi is not None,
            "semantic_vacuity": sem_vac is not None,
        },
        "component_splitting": {
            "enabled": bool(split_cfg.enabled),
            "method": str(split_cfg.method),
            "num_parent_components": int(parent_component_count),
            "num_split_candidate_components": int(split_candidate_count),
            "num_split_parent_components": int(split_parent_count),
            "num_split_nodes": int(split_node_count),
            "num_unsplit_nodes": int(unsplit_node_count),
            "avg_children_per_split_component": (
                float(split_child_region_count / split_parent_count) if split_parent_count else 0.0
            ),
        },
    }
    return nodes, shape_info


def build_blocks_access_edges(
    nodes: Sequence[Mapping[str, Any]],
    *,
    config: BlocksAccessRuleConfig | Mapping[str, Any] | None = None,
    image_shape_hw: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate directed ``blocks_access_to`` edges from node geometry."""

    cfg = _coerce_config(config, BlocksAccessRuleConfig)
    _validate_edge_config(cfg)
    lateral_axis = _resolve_lateral_axis(cfg)
    image_height, image_width = _shape_hw_from_value(image_shape_hw, fallback_hw=(1, 1))

    edges: List[Dict[str, Any]] = []
    for source_index, source in enumerate(nodes):
        source_access = _node_axis_coordinate(source, cfg.access_axis)
        for target_index, target in enumerate(nodes):
            if source_index == target_index:
                continue
            target_access = _node_axis_coordinate(target, cfg.access_axis)
            if cfg.opening_side == "low":
                access_gap = float(target_access - source_access)
            else:
                access_gap = float(source_access - target_access)
            if access_gap <= float(cfg.min_front_gap):
                continue

            overlap = _lateral_overlap_stats(source, target, lateral_axis)
            if overlap["overlap_pixels"] <= 0.0:
                continue
            metric = float(overlap[cfg.lateral_overlap_mode])
            if metric < float(cfg.min_lateral_overlap):
                continue

            edge = {
                "source": int(source["id"]),
                "target": int(target["id"]),
                "source_index": int(source_index),
                "target_index": int(target_index),
                "predicate": cfg.relation,
                "relation": cfg.relation,
                "decision": True,
                "score": metric,
                "rule": "front_lateral_overlap",
                "access_axis": cfg.access_axis,
                "opening_side": cfg.opening_side,
                "lateral_axis": lateral_axis,
                "source_access_coordinate": float(source_access),
                "target_access_coordinate": float(target_access),
                "access_coordinate_gap": float(access_gap),
                "min_front_gap": float(cfg.min_front_gap),
                "min_lateral_overlap": float(cfg.min_lateral_overlap),
                "lateral_overlap_mode": cfg.lateral_overlap_mode,
                "lateral_overlap_pixels": float(overlap["overlap_pixels"]),
                "lateral_overlap_union": float(overlap["union"]),
                "lateral_overlap_min": float(overlap["min"]),
                "lateral_overlap_source": float(overlap["source"]),
                "lateral_overlap_target": float(overlap["target"]),
                "image_shape_hw": [int(image_height), int(image_width)],
            }
            edges.append(edge)

    edges.sort(key=lambda item: (item["source_index"], item["target_index"]))
    return edges, {"resolved_lateral_axis": lateral_axis}


def encode_binary_mask_rle(mask: Any) -> Dict[str, Any]:
    """Encode a 2D boolean mask as JSON-safe row-major run lengths."""

    mask_array = np.asarray(mask, dtype=np.uint8)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask_array.shape}")
    flat = mask_array.reshape(-1)
    counts: List[int] = []
    current_value = 0
    run_length = 0
    for value in flat.tolist():
        value = int(value)
        if value == current_value:
            run_length += 1
        else:
            counts.append(int(run_length))
            current_value = value
            run_length = 1
    counts.append(int(run_length))
    return {
        "encoding": "rle",
        "order": "C",
        "size": [int(mask_array.shape[0]), int(mask_array.shape[1])],
        "counts": counts,
    }


def decode_binary_mask_rle(encoded: Mapping[str, Any]) -> np.ndarray:
    """Decode masks produced by :func:`encode_binary_mask_rle`."""

    if encoded.get("encoding") != "rle":
        raise ValueError("only rle mask encoding is supported")
    height, width = [int(value) for value in encoded["size"]]
    values: List[int] = []
    current_value = 0
    for count in encoded["counts"]:
        values.extend([current_value] * int(count))
        current_value = 1 - current_value
    array = np.asarray(values, dtype=bool)
    if array.size != height * width:
        raise ValueError(f"decoded RLE has {array.size} pixels, expected {height * width}")
    return array.reshape((height, width))


def _load_cnabu_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing CNABU belief file: {path}")

    arrays: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "occupancy_alpha",
            "occupancy_beta",
            "semantic_concentration",
            "occupancy_mean",
            "occupancy_epistemic",
            "semantic_mean",
            "semantic_vacuity",
            "selected_view_indices",
            "crop_rows",
            "raw_shape_hms",
            "cnabu_shape_hw",
        ):
            if key in data.files:
                arrays[key] = np.asarray(data[key])
        if "metadata_json" in data.files:
            try:
                loaded_metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                if isinstance(loaded_metadata, Mapping):
                    metadata.update(dict(loaded_metadata))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata["metadata_json_parse_error"] = True

    if "raw_shape_hw" not in metadata and "raw_shape_hms" in arrays:
        metadata["raw_shape_hw"] = list(_shape_hw_from_value(arrays["raw_shape_hms"], fallback_hw=(0, 0)))
    return {"arrays": arrays, "metadata": metadata}


def _derive_occupancy_arrays(
    *,
    occupancy_mean: Any,
    occupancy_alpha: Any,
    occupancy_beta: Any,
    occupancy_distribution: Any,
    occupancy_epistemic: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if occupancy_distribution is not None and (occupancy_alpha is None or occupancy_beta is None):
        occupancy_alpha, occupancy_beta = _split_interleaved_occupancy_distribution(occupancy_distribution)

    alpha = _optional_array(occupancy_alpha, 3, "occupancy_alpha")
    beta = _optional_array(occupancy_beta, 3, "occupancy_beta")
    mean = _optional_array(occupancy_mean, 3, "occupancy_mean")
    epistemic = _optional_array(occupancy_epistemic, 3, "occupancy_epistemic")

    if mean is None:
        if alpha is None or beta is None:
            raise ValueError("occupancy_mean or occupancy alpha/beta evidence is required")
        denom = np.maximum(alpha + beta, 1e-8)
        mean = alpha / denom

    if epistemic is None and alpha is not None and beta is not None:
        denom = np.maximum((alpha + beta) ** 2 * (alpha + beta + 1.0), 1e-8)
        epistemic = (alpha * beta) / denom

    return mean.astype(np.float32, copy=False), (
        epistemic.astype(np.float32, copy=False) if epistemic is not None else None
    )


def _derive_semantic_arrays(
    *,
    semantic_mean: Any,
    semantic_concentration: Any,
    semantic_vacuity: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    concentration = _optional_array(semantic_concentration, 3, "semantic_concentration")
    mean = _optional_array(semantic_mean, 3, "semantic_mean")
    vacuity = _optional_array(semantic_vacuity, 2, "semantic_vacuity")

    if mean is None:
        if concentration is None:
            raise ValueError("semantic_mean or semantic_concentration is required")
        concentration_sum = np.maximum(concentration.sum(axis=0), 1e-8)
        mean = concentration / concentration_sum[None, :, :]

    if vacuity is None and concentration is not None:
        concentration_sum = np.maximum(concentration.sum(axis=0), 1e-8)
        vacuity = float(concentration.shape[0]) / concentration_sum

    return mean.astype(np.float32, copy=False), (
        vacuity.astype(np.float32, copy=False) if vacuity is not None else None
    )


def _split_interleaved_occupancy_distribution(occupancy_distribution: Any) -> Tuple[np.ndarray, np.ndarray]:
    distribution = _squeeze_array(
        _to_numpy(occupancy_distribution, "occupancy_distribution"),
        3,
        "occupancy_distribution",
    ).astype(np.float32, copy=False)
    if distribution.shape[0] % 2 != 0:
        raise ValueError(
            "occupancy_distribution first axis must interleave beta/alpha channels and have even length"
        )
    beta = distribution[0::2]
    alpha = distribution[1::2]
    return alpha, beta


def _coerce_config(value: Any, cls: Any) -> Any:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**dict(value))
    raise TypeError(f"expected {cls.__name__}, mapping, or None, got {type(value)!r}")


def _validate_component_config(config: ComponentExtractionConfig) -> None:
    if not (0.0 <= float(config.occupancy_threshold) <= 1.0):
        raise ValueError("occupancy_threshold must be in [0, 1]")
    if float(config.semantic_confidence_threshold) < 0.0:
        raise ValueError("semantic_confidence_threshold must be non-negative")
    if int(config.min_voxels) < 1:
        raise ValueError("min_voxels must be positive")
    if int(config.min_pixels) < 1:
        raise ValueError("min_pixels must be positive")
    if int(config.connectivity) not in {1, 2, 3}:
        raise ValueError("connectivity must be 1, 2, or 3")
    if int(config.object_class_max_exclusive) < 1:
        raise ValueError("object_class_max_exclusive must be positive")


def _validate_split_config(config: ComponentSplitConfig, component_config: ComponentExtractionConfig) -> None:
    if config.method not in {"seeded_distance_watershed", "candidate_gated_2d_footprint"}:
        raise ValueError(
            "component split method must be 'seeded_distance_watershed' or "
            "'candidate_gated_2d_footprint'"
        )
    if not (0.0 <= float(config.core_occupancy_threshold) <= 1.0):
        raise ValueError("core_occupancy_threshold must be in [0, 1]")
    if float(config.core_occupancy_threshold) < float(component_config.occupancy_threshold):
        raise ValueError("core_occupancy_threshold must be >= occupancy_threshold")
    if float(config.core_semantic_confidence_threshold) < 0.0:
        raise ValueError("core_semantic_confidence_threshold must be non-negative")
    if int(config.min_seed_voxels) < 1:
        raise ValueError("min_seed_voxels must be positive")
    if int(config.min_seed_pixels) < 1:
        raise ValueError("min_seed_pixels must be positive")
    if int(config.min_split_voxels) < 1:
        raise ValueError("min_split_voxels must be positive")
    if int(config.min_split_pixels) < 1:
        raise ValueError("min_split_pixels must be positive")
    if int(config.min_split_seeds) < 2:
        raise ValueError("min_split_seeds must be at least 2")
    if int(config.max_splits) < int(config.min_split_seeds):
        raise ValueError("max_splits must be >= min_split_seeds")
    if float(config.candidate_area_multiplier) <= 0.0:
        raise ValueError("candidate_area_multiplier must be positive")
    if float(config.candidate_bbox_multiplier) <= 0.0:
        raise ValueError("candidate_bbox_multiplier must be positive")
    if int(config.candidate_min_area_pixels) < 1:
        raise ValueError("candidate_min_area_pixels must be positive")
    if int(config.footprint_erosion_iterations) < 0:
        raise ValueError("footprint_erosion_iterations must be non-negative")
    if not (0.0 <= float(config.min_child_area_fraction) < 1.0):
        raise ValueError("min_child_area_fraction must be in [0, 1)")
    if not (0.0 <= float(config.min_neck_removed_fraction) < 1.0):
        raise ValueError("min_neck_removed_fraction must be in [0, 1)")


def _split_component_regions(
    *,
    component_mask: np.ndarray,
    occupancy: np.ndarray,
    semantic_confidence: np.ndarray,
    parent_component_id: int,
    parent_label_id: int,
    class_id: int,
    extraction_config: ComponentExtractionConfig,
    split_config: ComponentSplitConfig,
    structure: np.ndarray,
) -> List[Dict[str, Any]]:
    parent_voxel_count = int(component_mask.sum())
    parent_area_pixels = int(np.any(component_mask, axis=0).sum())
    parent_occ = occupancy[component_mask]
    parent_crop_mask = np.any(component_mask, axis=0)
    parent_conf = semantic_confidence[parent_crop_mask]
    parent_confidence = {
        "parent_mean_occupancy": float(parent_occ.mean()) if parent_occ.size else 0.0,
        "parent_max_occupancy": float(parent_occ.max()) if parent_occ.size else 0.0,
        "parent_mean_semantic_confidence": float(parent_conf.mean()) if parent_conf.size else 0.0,
    }
    common = {
        "enabled": bool(split_config.enabled),
        "method": str(split_config.method),
        "parent_component_id": int(parent_component_id),
        "parent_label_id": int(parent_label_id),
        "class_id": int(class_id),
        "parent_voxel_count": int(parent_voxel_count),
        "parent_area_pixels": int(parent_area_pixels),
        "occupancy_threshold": float(extraction_config.occupancy_threshold),
        "core_occupancy_threshold": float(split_config.core_occupancy_threshold),
        "core_semantic_confidence_threshold": float(split_config.core_semantic_confidence_threshold),
        "confidence": dict(parent_confidence),
    }

    if not bool(split_config.enabled):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **common,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "core_seed_count": 0,
                    "seed_voxel_count": None,
                    "seed_area_pixels": None,
                    "candidate_considered": False,
                    "candidate_gate_reasons": [],
                    "class_size_prior": None,
                    "split_confidence": 0.0,
                    "split_reason": "disabled",
                },
            }
        ]

    if split_config.method == "candidate_gated_2d_footprint":
        return _split_component_regions_candidate_gated_2d(
            component_mask=component_mask,
            occupancy=occupancy,
            semantic_confidence=semantic_confidence,
            common=common,
            split_config=split_config,
        )
    return _split_component_regions_seeded_distance(
        component_mask=component_mask,
        occupancy=occupancy,
        semantic_confidence=semantic_confidence,
        common=common,
        split_config=split_config,
        structure=structure,
    )


def _split_component_regions_seeded_distance(
    *,
    component_mask: np.ndarray,
    occupancy: np.ndarray,
    semantic_confidence: np.ndarray,
    common: Mapping[str, Any],
    split_config: ComponentSplitConfig,
    structure: np.ndarray,
) -> List[Dict[str, Any]]:
    semantic_core = semantic_confidence >= float(split_config.core_semantic_confidence_threshold)
    core_mask = (
        component_mask
        & (occupancy >= float(split_config.core_occupancy_threshold))
        & semantic_core[None, :, :]
    )
    seed_labels, _ = ndimage.label(core_mask, structure=structure)
    seeds = _valid_core_seeds(
        seed_labels,
        min_seed_voxels=int(split_config.min_seed_voxels),
        min_seed_pixels=int(split_config.min_seed_pixels),
        max_splits=int(split_config.max_splits),
    )
    if len(seeds) < int(split_config.min_split_seeds):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **common,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "core_seed_count": int(len(seeds)),
                    "seed_voxel_count": int(seeds[0]["voxel_count"]) if seeds else None,
                    "seed_area_pixels": int(seeds[0]["area_pixels"]) if seeds else None,
                    "candidate_considered": True,
                    "candidate_gate_reasons": ["global_3d_core_attempt"],
                    "class_size_prior": None,
                    "split_confidence": 0.0,
                    "split_reason": "insufficient_core_seeds",
                },
            }
        ]

    remapped_seed_labels = np.zeros_like(seed_labels, dtype=np.int32)
    for new_label, seed in enumerate(seeds, start=1):
        remapped_seed_labels[seed_labels == int(seed["label"])] = int(new_label)

    nearest_seed_indices = ndimage.distance_transform_edt(
        remapped_seed_labels == 0,
        return_distances=False,
        return_indices=True,
    )
    assigned_seed_labels = remapped_seed_labels[tuple(nearest_seed_indices)]

    split_regions: List[Dict[str, Any]] = []
    for split_id, seed in enumerate(seeds, start=1):
        region = component_mask & (assigned_seed_labels == int(split_id))
        region_voxels = int(region.sum())
        region_pixels = int(np.any(region, axis=0).sum())
        if region_voxels < int(split_config.min_split_voxels):
            continue
        if region_pixels < int(split_config.min_split_pixels):
            continue
        split_regions.append(
            {
                "mask": region,
                "metadata": {
                    **common,
                    "was_split": True,
                    "split_id": int(split_id),
                    "num_splits": 0,
                    "core_seed_count": int(len(seeds)),
                    "seed_voxel_count": int(seed["voxel_count"]),
                    "seed_area_pixels": int(seed["area_pixels"]),
                    "candidate_considered": True,
                    "candidate_gate_reasons": ["global_3d_core_attempt"],
                    "class_size_prior": None,
                    "split_confidence": 1.0,
                    "split_reason": "multiple_core_seeds",
                },
            }
        )

    if len(split_regions) < int(split_config.min_split_seeds):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **common,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "core_seed_count": int(len(seeds)),
                    "seed_voxel_count": None,
                    "seed_area_pixels": None,
                    "candidate_considered": True,
                    "candidate_gate_reasons": ["global_3d_core_attempt"],
                    "class_size_prior": None,
                    "split_confidence": 0.0,
                    "split_reason": "split_regions_too_small",
                },
            }
        ]

    for split_region in split_regions:
        split_region["metadata"]["num_splits"] = int(len(split_regions))
    return split_regions


def _split_component_regions_candidate_gated_2d(
    *,
    component_mask: np.ndarray,
    occupancy: np.ndarray,
    semantic_confidence: np.ndarray,
    common: Mapping[str, Any],
    split_config: ComponentSplitConfig,
) -> List[Dict[str, Any]]:
    footprint = np.any(component_mask, axis=0)
    gate = _footprint_candidate_gate(footprint=footprint, common=common, split_config=split_config)
    base_metadata = {
        **common,
        "candidate_considered": bool(gate["candidate_considered"]),
        "candidate_gate_reasons": list(gate["candidate_gate_reasons"]),
        "candidate_gate_reason": str(gate["candidate_gate_reason"]),
        "class_size_prior": gate["class_size_prior"],
        "core_seed_count": 0,
        "seed_voxel_count": None,
        "seed_area_pixels": None,
        "split_confidence": 0.0,
    }
    if not bool(gate["candidate_considered"]):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **base_metadata,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "split_reason": "not_candidate_by_class_size_or_shape",
                },
            }
        ]

    footprint_max_occupancy = np.max(np.where(component_mask, occupancy, 0.0), axis=0)
    semantic_core = semantic_confidence >= float(split_config.core_semantic_confidence_threshold)
    core_2d = (
        footprint
        & (footprint_max_occupancy >= float(split_config.core_occupancy_threshold))
        & semantic_core
    )
    erosion_iterations = int(split_config.footprint_erosion_iterations)
    seed_footprint = core_2d
    if erosion_iterations > 0:
        seed_footprint = ndimage.binary_erosion(
            seed_footprint,
            structure=ndimage.generate_binary_structure(2, 1),
            iterations=erosion_iterations,
            border_value=0,
        )
    seed_labels, _ = ndimage.label(seed_footprint, structure=ndimage.generate_binary_structure(2, 1))
    seeds = _valid_2d_seeds(
        seed_labels,
        min_seed_pixels=int(split_config.min_seed_pixels),
        max_splits=int(split_config.max_splits),
    )
    if len(seeds) < int(split_config.min_split_seeds):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **base_metadata,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "core_seed_count": int(len(seeds)),
                    "seed_area_pixels": int(seeds[0]["area_pixels"]) if seeds else None,
                    "split_reason": "candidate_no_separated_2d_lobes",
                },
            }
        ]

    parent_area = max(int(footprint.sum()), 1)
    seed_area = int(seed_footprint.sum())
    neck_removed_fraction = max(0.0, float(parent_area - seed_area) / float(parent_area))
    if neck_removed_fraction < float(split_config.min_neck_removed_fraction):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **base_metadata,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "core_seed_count": int(len(seeds)),
                    "seed_area_pixels": int(seeds[0]["area_pixels"]) if seeds else None,
                    "footprint_neck_removed_fraction": float(neck_removed_fraction),
                    "split_reason": "candidate_weak_neck_evidence",
                },
            }
        ]

    remapped_seed_labels = np.zeros_like(seed_labels, dtype=np.int32)
    for new_label, seed in enumerate(seeds, start=1):
        remapped_seed_labels[seed_labels == int(seed["label"])] = int(new_label)
    nearest_seed_indices = ndimage.distance_transform_edt(
        remapped_seed_labels == 0,
        return_distances=False,
        return_indices=True,
    )
    assigned_seed_labels = remapped_seed_labels[tuple(nearest_seed_indices)]
    min_child_pixels = max(
        int(split_config.min_split_pixels),
        int(np.ceil(parent_area * float(split_config.min_child_area_fraction))),
    )

    split_regions: List[Dict[str, Any]] = []
    for split_id, seed in enumerate(seeds, start=1):
        region_2d = footprint & (assigned_seed_labels == int(split_id))
        region = component_mask & region_2d[None, :, :]
        region_voxels = int(region.sum())
        region_pixels = int(region_2d.sum())
        if region_voxels < int(split_config.min_split_voxels):
            continue
        if region_pixels < int(min_child_pixels):
            continue
        split_regions.append(
            {
                "mask": region,
                "metadata": {
                    **base_metadata,
                    "was_split": True,
                    "split_id": int(split_id),
                    "num_splits": 0,
                    "core_seed_count": int(len(seeds)),
                    "seed_area_pixels": int(seed["area_pixels"]),
                    "footprint_neck_removed_fraction": float(neck_removed_fraction),
                    "min_child_area_pixels": int(min_child_pixels),
                    "split_confidence": float(
                        min(
                            1.0,
                            0.35
                            + min(0.35, 0.10 * float(len(seeds)))
                            + min(0.30, float(neck_removed_fraction)),
                        )
                    ),
                    "split_reason": "candidate_2d_separated_lobes",
                },
            }
        )

    if len(split_regions) < int(split_config.min_split_seeds):
        return [
            {
                "mask": component_mask,
                "metadata": {
                    **base_metadata,
                    "was_split": False,
                    "split_id": 1,
                    "num_splits": 1,
                    "core_seed_count": int(len(seeds)),
                    "seed_area_pixels": None,
                    "footprint_neck_removed_fraction": float(neck_removed_fraction),
                    "min_child_area_pixels": int(min_child_pixels),
                    "split_reason": "candidate_split_regions_too_small",
                },
            }
        ]

    for split_region in split_regions:
        split_region["metadata"]["num_splits"] = int(len(split_regions))
    return split_regions


def _valid_core_seeds(
    seed_labels: np.ndarray,
    *,
    min_seed_voxels: int,
    min_seed_pixels: int,
    max_splits: int,
) -> List[Dict[str, int]]:
    seeds: List[Dict[str, int]] = []
    for label_id, slices in enumerate(ndimage.find_objects(seed_labels), start=1):
        if slices is None:
            continue
        seed_mask = seed_labels[slices] == label_id
        voxel_count = int(seed_mask.sum())
        if voxel_count < int(min_seed_voxels):
            continue
        full_seed_mask = seed_labels == label_id
        area_pixels = int(np.any(full_seed_mask, axis=0).sum())
        if area_pixels < int(min_seed_pixels):
            continue
        seeds.append(
            {
                "label": int(label_id),
                "voxel_count": int(voxel_count),
                "area_pixels": int(area_pixels),
            }
        )
    seeds.sort(key=lambda item: (int(item["voxel_count"]), int(item["area_pixels"])), reverse=True)
    return seeds[: int(max_splits)]


def _valid_2d_seeds(
    seed_labels: np.ndarray,
    *,
    min_seed_pixels: int,
    max_splits: int,
) -> List[Dict[str, int]]:
    seeds: List[Dict[str, int]] = []
    for label_id, slices in enumerate(ndimage.find_objects(seed_labels), start=1):
        if slices is None:
            continue
        seed_mask = seed_labels[slices] == label_id
        area_pixels = int(seed_mask.sum())
        if area_pixels < int(min_seed_pixels):
            continue
        seeds.append({"label": int(label_id), "area_pixels": int(area_pixels)})
    seeds.sort(key=lambda item: int(item["area_pixels"]), reverse=True)
    return seeds[: int(max_splits)]


def _footprint_candidate_gate(
    *,
    footprint: np.ndarray,
    common: Mapping[str, Any],
    split_config: ComponentSplitConfig,
) -> Dict[str, Any]:
    class_id = int(common["class_id"])
    area = int(footprint.sum())
    bbox = _bbox_xyxy_from_mask(footprint)
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    area_prior = _class_prior_value(
        split_config.class_area_prior_pixels,
        class_id=class_id,
        fallback=max(area, 1),
    )
    width_prior = _class_prior_value(
        split_config.class_width_prior_pixels,
        class_id=class_id,
        fallback=max(width, 1),
    )
    height_prior = _class_prior_value(
        split_config.class_height_prior_pixels,
        class_id=class_id,
        fallback=max(height, 1),
    )
    reasons: List[str] = []
    if area >= int(split_config.candidate_min_area_pixels):
        if float(area) >= float(area_prior) * float(split_config.candidate_area_multiplier):
            reasons.append("area_exceeds_class_prior")
        if float(width) >= float(width_prior) * float(split_config.candidate_bbox_multiplier):
            reasons.append("width_exceeds_class_prior")
        if float(height) >= float(height_prior) * float(split_config.candidate_bbox_multiplier):
            reasons.append("height_exceeds_class_prior")

    probe = footprint
    if int(split_config.footprint_erosion_iterations) > 0:
        probe = ndimage.binary_erosion(
            probe,
            structure=ndimage.generate_binary_structure(2, 1),
            iterations=max(1, int(split_config.footprint_erosion_iterations)),
            border_value=0,
        )
    probe_labels, _ = ndimage.label(probe, structure=ndimage.generate_binary_structure(2, 1))
    probe_seeds = _valid_2d_seeds(
        probe_labels,
        min_seed_pixels=int(split_config.min_seed_pixels),
        max_splits=int(split_config.max_splits),
    )
    if len(probe_seeds) >= int(split_config.min_split_seeds):
        reasons.append("footprint_has_separated_lobes")

    return {
        "candidate_considered": bool(reasons) and area >= int(split_config.candidate_min_area_pixels),
        "candidate_gate_reasons": reasons,
        "candidate_gate_reason": ",".join(reasons) if reasons else "not_suspicious",
        "class_size_prior": {
            "class_id": int(class_id),
            "area_pixels": int(area_prior),
            "width_pixels": int(width_prior),
            "height_pixels": int(height_prior),
            "candidate_area_multiplier": float(split_config.candidate_area_multiplier),
            "candidate_bbox_multiplier": float(split_config.candidate_bbox_multiplier),
            "observed_area_pixels": int(area),
            "observed_width_pixels": int(width),
            "observed_height_pixels": int(height),
            "eroded_lobe_count": int(len(probe_seeds)),
        },
    }


def _class_prior_value(values: Sequence[int], *, class_id: int, fallback: int) -> int:
    array = np.asarray(values).reshape(-1)
    if 0 <= int(class_id) < int(array.size):
        value = int(round(float(array[int(class_id)])))
        if value > 0:
            return value
    return int(max(1, fallback))


def _node_split_metadata(
    metadata: Mapping[str, Any],
    *,
    region_voxel_count: int,
    region_pixel_count: int,
    mean_occupancy: float,
    mean_semantic_confidence: float,
) -> Dict[str, Any]:
    result = dict(metadata)
    confidence = dict(result.get("confidence", {}))
    confidence.update(
        {
            "region_mean_occupancy": float(mean_occupancy),
            "region_mean_semantic_confidence": float(mean_semantic_confidence),
        }
    )
    result["confidence"] = confidence
    result["region_voxel_count"] = int(region_voxel_count)
    result["region_area_pixels"] = int(region_pixel_count)
    return result


def _node_source_from_split_metadata(metadata: Mapping[str, Any]) -> str:
    if not bool(metadata.get("was_split", False)):
        return "cnabu_3d_component"
    if str(metadata.get("method")) == "candidate_gated_2d_footprint":
        return "cnabu_2d_footprint_split_component"
    return "cnabu_3d_split_component"


def _component_zyx_bounds(mask: np.ndarray) -> Tuple[List[int], List[int]]:
    zs, ys, xs = np.nonzero(mask)
    if zs.size == 0 or ys.size == 0 or xs.size == 0:
        return [0, 0], [0, 0, 0, 0]
    return (
        [int(zs.min()), int(zs.max()) + 1],
        [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1],
    )


def _validate_edge_config(config: BlocksAccessRuleConfig) -> None:
    if config.access_axis not in {"x", "y"}:
        raise ValueError("access_axis must be 'x' or 'y'")
    if config.opening_side not in {"low", "high"}:
        raise ValueError("opening_side must be 'low' or 'high'")
    if config.lateral_axis is not None and config.lateral_axis not in {"x", "y"}:
        raise ValueError("lateral_axis must be 'x', 'y', or None")
    if config.lateral_axis == config.access_axis:
        raise ValueError("lateral_axis must differ from access_axis")
    if float(config.min_front_gap) < 0.0:
        raise ValueError("min_front_gap must be non-negative")
    if float(config.min_lateral_overlap) < 0.0:
        raise ValueError("min_lateral_overlap must be non-negative")
    if config.lateral_overlap_mode not in {"union", "min", "source", "target", "pixels"}:
        raise ValueError("lateral_overlap_mode must be one of union, min, source, target, pixels")


def _resolve_lateral_axis(config: BlocksAccessRuleConfig) -> str:
    if config.lateral_axis is not None:
        return config.lateral_axis
    return "x" if config.access_axis == "y" else "y"


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _optional_array(value: Any, expected_ndim: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    return _squeeze_array(_to_numpy(value, name), expected_ndim, name).astype(np.float32, copy=False)


def _squeeze_array(array: np.ndarray, expected_ndim: int, name: str) -> np.ndarray:
    result = np.asarray(array)
    while result.ndim == expected_ndim + 1 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dims, got shape {result.shape}")
    return result


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _shape_hw_from_value(value: Optional[Sequence[int]], *, fallback_hw: Sequence[int]) -> Tuple[int, int]:
    if value is None:
        return int(fallback_hw[0]), int(fallback_hw[1])
    array = np.asarray(value).reshape(-1)
    if array.size == 2:
        return int(array[0]), int(array[1])
    if array.size >= 3:
        return int(array[1]), int(array[2])
    raise ValueError(f"shape value must have at least two entries, got {value!r}")


def _crop_rows_from_value(
    value: Optional[Sequence[int]],
    *,
    crop_height: int,
    raw_height: int,
) -> Tuple[int, int]:
    if value is None:
        return 0, int(crop_height)
    rows = np.asarray(value).reshape(-1)
    if rows.size != 2:
        raise ValueError(f"crop_rows must have two entries, got {value!r}")
    start, stop = int(rows[0]), int(rows[1])
    if stop - start != int(crop_height):
        raise ValueError(f"crop_rows {[start, stop]} do not match crop height {crop_height}")
    if start < 0 or stop > int(raw_height):
        raise ValueError(f"crop_rows {[start, stop]} outside raw height {raw_height}")
    return start, stop


def _bbox_xyxy_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _centroid_yx(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def _class_name(class_id: int, class_names: Sequence[str]) -> Optional[str]:
    if 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    return None


def _node_axis_coordinate(node: Mapping[str, Any], axis: str) -> float:
    centroid_xy = node.get("centroid_xy")
    if centroid_xy is None:
        bbox = node["bbox_xyxy_abs"]
        centroid_xy = [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]
    return float(centroid_xy[0 if axis == "x" else 1])


def _axis_interval(node: Mapping[str, Any], axis: str) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in node["bbox_xyxy_abs"]]
    return (x1, x2) if axis == "x" else (y1, y2)


def _lateral_overlap_stats(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    axis: str,
) -> Dict[str, float]:
    source_min, source_max = _axis_interval(source, axis)
    target_min, target_max = _axis_interval(target, axis)
    eps = 1e-6
    source_extent = max(source_max - source_min, eps)
    target_extent = max(target_max - target_min, eps)
    overlap = max(0.0, min(source_max, target_max) - max(source_min, target_min))
    span = max(max(source_max, target_max) - min(source_min, target_min), eps)
    return {
        "overlap_pixels": float(overlap),
        "pixels": float(overlap),
        "union": float(overlap / span),
        "min": float(overlap / max(min(source_extent, target_extent), eps)),
        "source": float(overlap / source_extent),
        "target": float(overlap / target_extent),
    }


def _adjacency_matrix(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    adjacency = [[0 for _ in nodes] for _ in nodes]
    for edge in edges:
        adjacency[int(edge["source_index"])][int(edge["target_index"])] = 1
    return adjacency


def _int_list(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(value) for value in np.asarray(values).reshape(-1).tolist()]


__all__ = [
    "BlocksAccessRuleConfig",
    "ComponentExtractionConfig",
    "ComponentSplitConfig",
    "DEFAULT_YCB_CLASS_NAMES",
    "build_blocks_access_edges",
    "decode_binary_mask_rle",
    "encode_binary_mask_rle",
    "extract_cnabu_component_nodes",
    "predict_scene_graph_from_cnabu",
]
