"""MEM runtime wrapper for the learned CNABU component splitter checkpoint.

The wrapper is inference-only: it loads the D3G checkpoint once, extracts
CNABU parent components with the existing MEM rule path, applies the learned
component splitter to those parents, then reuses the deterministic MEM
``blocks_access_to`` edge rule.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from shelf_gym.utils.cnabu_scene_graph import (
    BlocksAccessRuleConfig,
    ComponentExtractionConfig,
    DEFAULT_YCB_CLASS_NAMES,
    _crop_rows_from_value,
    _derive_occupancy_arrays,
    _derive_semantic_arrays,
    _first_not_none,
    _load_cnabu_npz,
    _shape_hw_from_value,
    _to_numpy,
    _adjacency_matrix,
    build_blocks_access_edges,
    encode_binary_mask_rle,
    predict_scene_graph_from_cnabu,
)


THESIS_ROOT = Path("/home/user/ehsanullahm1/thesis")
DEFAULT_D3G_TOOLS_DIR = THESIS_ROOT / "scene_graph_related_research_papers" / "D3G" / "tools"
DEFAULT_CHECKPOINT_PATH = (
    THESIS_ROOT
    / "scene_graph_related_research_papers"
    / "D3G"
    / "checkpoints"
    / "mem_cnabu_component_splitter_full1000_20260701_234841"
    / "model_best_validation.pth"
)


def _import_d3g_runtime_helpers(d3g_tools_dir: Path) -> Dict[str, Any]:
    tools_dir = Path(d3g_tools_dir)
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"missing D3G tools directory: {tools_dir}")
    tools_text = str(tools_dir)
    if tools_text not in sys.path:
        sys.path.insert(0, tools_text)

    from train_mem_cnabu_component_splitter import (  # noqa: WPS433
        CHECKPOINT_SCHEMA,
        load_runtime_checkpoint,
        split_parent_with_model,
    )
    from train_mem_cnabu_node_proposal import nodes_from_graph  # noqa: WPS433

    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "load_runtime_checkpoint": load_runtime_checkpoint,
        "nodes_from_graph": nodes_from_graph,
        "split_parent_with_model": split_parent_with_model,
    }


class LearnedCnabuComponentSplitter:
    """Load once and run the D3G learned component splitter on MEM CNABU arrays."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        *,
        d3g_tools_dir: str | Path = DEFAULT_D3G_TOOLS_DIR,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.d3g_tools_dir = Path(d3g_tools_dir)
        self.device = torch.device(device)
        self._helpers = _import_d3g_runtime_helpers(self.d3g_tools_dir)

        started = time.perf_counter()
        self.model, self.config, self.checkpoint_payload = self._helpers["load_runtime_checkpoint"](
            self.checkpoint_path,
            self.device,
        )
        self.load_seconds = time.perf_counter() - started
        self.checkpoint_schema = str(self.checkpoint_payload.get("schema"))

    @property
    def d3g_import_required(self) -> bool:
        return True

    def predict_scene_graph(
        self,
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
        edge_config: BlocksAccessRuleConfig | Mapping[str, Any] | None = None,
        class_names: Sequence[str] = DEFAULT_YCB_CLASS_NAMES,
        sample_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        include_masks: bool = True,
    ) -> Dict[str, Any]:
        """Predict a scene graph from CNABU arrays or a saved ``cnabu_hms.npz``."""

        prepared = _prepare_cnabu_arrays(
            cnabu_path=cnabu_path,
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            occupancy_alpha=occupancy_alpha,
            occupancy_beta=occupancy_beta,
            semantic_concentration=semantic_concentration,
            occupancy_epistemic=occupancy_epistemic,
            semantic_vacuity=semantic_vacuity,
            occupancy_distribution=occupancy_distribution,
            raw_shape_hw=raw_shape_hw,
            crop_rows=crop_rows,
            selected_view_indices=selected_view_indices,
        )
        parent_started = time.perf_counter()
        parent_graph = predict_scene_graph_from_cnabu(
            cnabu_path=cnabu_path,
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            occupancy_alpha=occupancy_alpha,
            occupancy_beta=occupancy_beta,
            semantic_concentration=semantic_concentration,
            occupancy_epistemic=occupancy_epistemic,
            semantic_vacuity=semantic_vacuity,
            occupancy_distribution=occupancy_distribution,
            raw_shape_hw=raw_shape_hw,
            crop_rows=crop_rows,
            selected_view_indices=selected_view_indices,
            component_config=component_config,
            component_split_config={"enabled": False},
            edge_config=edge_config,
            class_names=class_names,
            sample_id=sample_id,
            metadata=metadata,
            include_masks=True,
        )
        parent_seconds = time.perf_counter() - parent_started

        raw_features = _d3g_raw_features_from_prepared(prepared)
        parent_nodes = self._helpers["nodes_from_graph"](parent_graph)
        learned_started = time.perf_counter()
        nodes: List[Dict[str, Any]] = []
        split_parent_count = 0
        split_node_count = 0
        for parent_index, parent in enumerate(parent_nodes):
            children = self._helpers["split_parent_with_model"](
                model=self.model,
                raw_features=raw_features,
                parent=parent,
                config=self.config,
                device=self.device,
            )
            split_accepted = bool(len(children) > 1 or any(bool(child.was_split) for child in children))
            if split_accepted:
                split_parent_count += 1
                split_node_count += len(children)
            parent_payload = parent_graph["nodes"][parent_index]
            for child_index, child in enumerate(children, start=1):
                nodes.append(
                    _learned_node_payload(
                        node_id=len(nodes) + 1,
                        child=child,
                        parent=parent,
                        parent_payload=parent_payload,
                        split_accepted=split_accepted,
                        child_index=child_index,
                        num_children=len(children),
                        checkpoint_path=self.checkpoint_path,
                        include_mask=include_masks,
                    )
                )
        learned_seconds = time.perf_counter() - learned_started

        edge_started = time.perf_counter()
        edges, edge_rule_info = build_blocks_access_edges(
            nodes,
            config=edge_config,
            image_shape_hw=prepared["raw_shape_hw"],
        )
        edge_seconds = time.perf_counter() - edge_started
        metadata_payload = copy.deepcopy(parent_graph.get("metadata", {}))
        metadata_payload.update(
            {
                "node_source": "learned_component_splitter",
                "parent_node_source": "cnabu_3d_components",
                "requires_gt": False,
                "uses_gt": False,
                "uses_simulator_instance_labels": False,
                "uses_d3g": True,
                "d3g_import_required": True,
                "uses_learned_graph_checkpoint": True,
                "learned_checkpoint_path": str(self.checkpoint_path),
                "learned_checkpoint_schema": self.checkpoint_schema,
                "learned_checkpoint_load_seconds": float(self.load_seconds),
                "num_parent_components": int(len(parent_nodes)),
                "num_split_parent_components": int(split_parent_count),
                "num_split_nodes": int(split_node_count),
                "num_nodes": int(len(nodes)),
                "num_edges": int(len(edges)),
                "raw_shape_hw": [int(value) for value in prepared["raw_shape_hw"]],
                "crop_rows": [int(value) for value in prepared["crop_rows"]],
                "selected_view_indices": prepared["selected_view_indices"],
                "caller_metadata": dict(metadata or {}),
                "runtime_timing_seconds": {
                    "checkpoint_load": float(self.load_seconds),
                    "parent_component_extraction": float(parent_seconds),
                    "learned_splitter_inference": float(learned_seconds),
                    "edge_generation": float(edge_seconds),
                    "total_graph_generation": float(parent_seconds + learned_seconds + edge_seconds),
                },
            }
        )
        thresholds = copy.deepcopy(parent_graph.get("thresholds", {}))
        thresholds["component_splitting"] = {
            "enabled": True,
            "method": "learned_component_splitter",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_schema": self.checkpoint_schema,
            "split_prob_threshold": float(self.config["INFERENCE"]["SPLIT_PROB_THRESHOLD"]),
            "count_margin_threshold": float(self.config["INFERENCE"]["COUNT_MARGIN_THRESHOLD"]),
            "center_threshold": float(self.config["INFERENCE"]["CENTER_THRESHOLD"]),
        }
        thresholds["edge_rule"] = {
            **thresholds.get("edge_rule", {}),
            "resolved_lateral_axis": edge_rule_info["resolved_lateral_axis"],
        }

        return {
            "schema": "mem_cnabu_learned_component_splitter_scene_graph_v0",
            "relation": parent_graph.get("relation", "blocks_access_to"),
            "nodes": nodes,
            "edges": edges,
            "adjacency_matrix": _adjacency_matrix(nodes, edges),
            "thresholds": thresholds,
            "metadata": metadata_payload,
        }


def _prepare_cnabu_arrays(
    *,
    cnabu_path: Optional[str | Path],
    occupancy_mean: Any,
    semantic_mean: Any,
    occupancy_alpha: Any,
    occupancy_beta: Any,
    semantic_concentration: Any,
    occupancy_epistemic: Any,
    semantic_vacuity: Any,
    occupancy_distribution: Any,
    raw_shape_hw: Optional[Sequence[int]],
    crop_rows: Optional[Sequence[int]],
    selected_view_indices: Optional[Sequence[int]],
) -> Dict[str, Any]:
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
    selected = (
        [int(value) for value in np.asarray(selected_view_indices).reshape(-1).tolist()]
        if selected_view_indices is not None
        else None
    )
    return {
        "occupancy_mean": occupancy_mean_array,
        "occupancy_epistemic": occupancy_epistemic_array,
        "semantic_mean": semantic_mean_array,
        "semantic_vacuity": semantic_vacuity_array,
        "raw_shape_hw": (int(raw_height), int(raw_width)),
        "crop_rows": (int(crop_start), int(crop_stop)),
        "selected_view_indices": selected,
    }


def _d3g_raw_features_from_prepared(prepared: Mapping[str, Any]) -> np.ndarray:
    occupancy = np.asarray(prepared["occupancy_mean"], dtype=np.float32)
    semantic = np.asarray(prepared["semantic_mean"], dtype=np.float32)
    occ_epi = prepared.get("occupancy_epistemic")
    sem_vac = prepared.get("semantic_vacuity")
    raw_shape_hw = tuple(int(value) for value in prepared["raw_shape_hw"])
    crop_rows = tuple(int(value) for value in prepared["crop_rows"])

    occupied = occupancy >= 0.5
    any_occ = occupied.any(axis=0)
    z_indices = np.arange(occupancy.shape[0], dtype=np.float32)[:, None, None]
    top = np.where(occupied, z_indices, -1.0).max(axis=0) / max(1.0, occupancy.shape[0] - 1.0)
    bottom_raw = np.where(occupied, z_indices, occupancy.shape[0] + 1.0).min(axis=0)
    bottom = np.where(any_occ, bottom_raw, 0.0) / max(1.0, occupancy.shape[0] - 1.0)

    channels: List[np.ndarray] = []
    for channel in semantic[:14]:
        channels.append(_pad_crop(channel, raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
    channels.extend(
        [
            _pad_crop(occupancy.max(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            _pad_crop(occupancy.mean(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            _pad_crop(occupied.mean(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            _pad_crop(np.where(any_occ, top, 0.0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            _pad_crop(bottom, raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
        ]
    )
    if occ_epi is not None:
        occ_epi_array = np.asarray(occ_epi, dtype=np.float32)
        channels.append(_pad_crop(occ_epi_array.mean(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
        channels.append(_pad_crop(occ_epi_array.max(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
    else:
        channels.extend([np.zeros(raw_shape_hw, dtype=np.float32), np.zeros(raw_shape_hw, dtype=np.float32)])
    if sem_vac is not None:
        channels.append(_pad_crop(np.asarray(sem_vac, dtype=np.float32), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
    else:
        channels.append(np.zeros(raw_shape_hw, dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _pad_crop(array: Any, *, raw_shape_hw: Tuple[int, int], crop_rows: Tuple[int, int]) -> np.ndarray:
    crop = _to_numpy(array, "array").astype(np.float32, copy=False)
    result = np.zeros(raw_shape_hw, dtype=np.float32)
    result[int(crop_rows[0]):int(crop_rows[1]), :] = crop
    return result


def _learned_node_payload(
    *,
    node_id: int,
    child: Any,
    parent: Any,
    parent_payload: Mapping[str, Any],
    split_accepted: bool,
    child_index: int,
    num_children: int,
    checkpoint_path: Path,
    include_mask: bool,
) -> Dict[str, Any]:
    split_metadata = {
        "enabled": True,
        "method": "learned_component_splitter",
        "parent_component_id": int(parent.id),
        "parent_node_id": int(parent.id),
        "parent_area_pixels": int(parent.area),
        "parent_bbox_xyxy_abs": [int(value) for value in parent.bbox_xyxy_abs],
        "was_split": bool(split_accepted),
        "accepted": bool(split_accepted),
        "split_id": int(child_index if split_accepted else 1),
        "child_index": int(child_index if split_accepted else 0),
        "num_splits": int(num_children if split_accepted else 1),
        "num_children": int(num_children if split_accepted else 1),
        "split_reason": "learned_split_accepted" if split_accepted else "learned_split_not_accepted",
        "split_confidence": float(child.score),
        "checkpoint_path": str(checkpoint_path),
    }
    payload: Dict[str, Any] = {
        "id": int(node_id),
        "node_index": int(node_id - 1),
        "component_id": int(node_id),
        "node_source": (
            "cnabu_learned_component_split_component"
            if split_accepted
            else str(parent_payload.get("node_source", "cnabu_3d_component"))
        ),
        "parent_component_id": int(parent.id),
        "split_id": int(split_metadata["split_id"]),
        "split_method": "learned_component_splitter",
        "was_split": bool(split_accepted),
        "class_id": int(child.class_id),
        "class_name": str(child.class_name),
        "score": float(child.score),
        "bbox_xyxy_abs": [int(value) for value in child.bbox_xyxy_abs],
        "bbox_format": "xyxy_exclusive",
        "centroid_yx": [float(value) for value in child.centroid_yx],
        "centroid_xy": [float(child.centroid_yx[1]), float(child.centroid_yx[0])],
        "area_pixels": int(child.area),
        "voxel_count": None,
        "z_range": parent_payload.get("z_range"),
        "crop_bbox_yx": parent_payload.get("crop_bbox_yx"),
        "parent_component_bbox_xyxy_abs": [int(value) for value in parent.bbox_xyxy_abs],
        "parent_component_area_pixels": int(parent.area),
        "confidence": {
            "learned_splitter_score": float(child.score),
            "parent_score": float(parent.score),
        },
        "uncertainty": copy.deepcopy(parent_payload.get("uncertainty", {})),
        "split": split_metadata,
        "split_metadata": split_metadata,
    }
    if include_mask:
        payload["mask"] = encode_binary_mask_rle(child.mask)
    return payload


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_D3G_TOOLS_DIR",
    "LearnedCnabuComponentSplitter",
]
