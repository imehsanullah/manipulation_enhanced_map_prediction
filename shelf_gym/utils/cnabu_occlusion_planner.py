"""Fixed-budget frontier allocation for CNABU belief-occlusion planning."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def live_cnabu_belief_arrays(
    occupancy_distribution: Any,
    semantic_concentration: Any,
) -> Dict[str, np.ndarray]:
    """Derive the frozen CNABU mean/uncertainty arrays from live MEM tensors."""

    distribution = _host_array(occupancy_distribution).astype(np.float32, copy=False)
    concentration = _host_array(semantic_concentration).astype(np.float32, copy=False)
    if distribution.ndim != 4 or distribution.shape[0] != 1 or distribution.shape[1] % 2:
        raise ValueError("occupancy_distribution must have shape [1,2Z,H,W]")
    if concentration.ndim != 4 or concentration.shape[0] != 1:
        raise ValueError("semantic_concentration must have shape [1,C,H,W]")
    alpha = distribution[0, 1::2]
    beta = distribution[0, 0::2]
    if (
        alpha.shape[1:] != concentration.shape[2:]
        or not np.isfinite(alpha).all()
        or not np.isfinite(beta).all()
        or np.any(alpha <= 0.0)
        or np.any(beta <= 0.0)
    ):
        raise ValueError("live occupancy evidence is invalid or spatially misaligned")
    semantic = concentration[0]
    if (
        semantic.shape[0] <= 0
        or not np.isfinite(semantic).all()
        or np.any(semantic <= 0.0)
    ):
        raise ValueError("live semantic concentration must be finite and positive")
    occupancy_total = alpha + beta
    occupancy_mean = alpha / occupancy_total
    occupancy_epistemic = (
        alpha * beta
        / (occupancy_total * occupancy_total * (occupancy_total + 1.0))
    )
    semantic_total = semantic.sum(axis=0)
    semantic_mean = semantic / semantic_total[None]
    semantic_vacuity = float(semantic.shape[0]) / semantic_total
    return {
        "occupancy_mean": occupancy_mean.astype(np.float32, copy=False),
        "occupancy_epistemic": occupancy_epistemic.astype(np.float32, copy=False),
        "semantic_mean": semantic_mean.astype(np.float32, copy=False),
        "semantic_vacuity": semantic_vacuity.astype(np.float32, copy=False),
    }


@dataclass(frozen=True)
class FrontierSamplingDecision:
    """One probability decision over an already filtered frontier population."""

    probabilities: Optional[np.ndarray]
    frontier_source_indices: np.ndarray
    diagnostics: Dict[str, Any]


class OcclusionFrontierAllocator:
    """Mix object-balanced occlusion guidance with uniform exploration.

    The allocator never creates or filters frontier points.  It assigns the
    fixed population supplied by the existing MEM sampler to learned source
    masks, then returns an optional probability vector for the sampler's
    unchanged no-replacement draw.  ``None`` requests the exact historical
    uniform choice call.
    """

    def __init__(
        self,
        *,
        source_masks_raw_hw: Any,
        source_scores: Sequence[float] | np.ndarray,
        guidance_fraction: float = 0.75,
        score_provenance: str,
    ) -> None:
        masks = np.asarray(source_masks_raw_hw, dtype=bool)
        scores = np.asarray(source_scores, dtype=np.float64)
        if masks.ndim != 3 or masks.shape[1] <= 0 or masks.shape[2] <= 0:
            raise ValueError("source_masks_raw_hw must have shape [N,H,W]")
        if scores.ndim != 1 or scores.shape[0] != masks.shape[0]:
            raise ValueError("source_scores must align one-to-one with source masks")
        if not np.isfinite(scores).all() or np.any(scores < 0.0):
            raise ValueError("source_scores must be finite and non-negative")
        fraction = float(guidance_fraction)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("guidance_fraction must lie in [0,1]")
        if not str(score_provenance):
            raise ValueError("score_provenance must be explicit")
        self.source_masks_raw_hw = masks.copy()
        self.source_scores = scores.copy()
        self.guidance_fraction = fraction
        self.score_provenance = str(score_provenance)

    def sampling_probabilities(self, frontier_indices: Any) -> FrontierSamplingDecision:
        """Return mixed probabilities in the supplied frontier order."""

        frontiers = np.asarray(frontier_indices)
        if frontiers.ndim != 2 or frontiers.shape[1] < 2:
            raise ValueError("frontier_indices must have shape [M,>=2]")
        if not np.issubdtype(frontiers.dtype, np.integer):
            raise ValueError("frontier_indices must be integer-valued")
        frontiers = frontiers.astype(np.int64, copy=False)
        population_count = int(frontiers.shape[0])
        height, width = self.source_masks_raw_hw.shape[1:]
        if population_count and (
            np.any(frontiers[:, 0] < 0)
            or np.any(frontiers[:, 0] >= height)
            or np.any(frontiers[:, 1] < 0)
            or np.any(frontiers[:, 1] >= width)
        ):
            raise ValueError("frontier coordinate is outside the raw mask frame")

        frontier_sources = np.full(population_count, -1, dtype=np.int64)
        if population_count and len(self.source_scores):
            membership = self.source_masks_raw_hw[
                :, frontiers[:, 0], frontiers[:, 1]
            ].T
            candidate_scores = np.where(
                membership,
                self.source_scores[None, :],
                -np.inf,
            )
            has_source = membership.any(axis=1)
            # np.argmax is stable toward the lower source index for exact ties.
            frontier_sources[has_source] = np.argmax(
                candidate_scores[has_source], axis=1
            )

        source_frontier_counts = np.bincount(
            frontier_sources[frontier_sources >= 0],
            minlength=len(self.source_scores),
        ).astype(np.int64, copy=False)
        eligible = source_frontier_counts > 0
        positive_eligible = eligible & (self.source_scores > 0.0)
        positive_score_sum = float(self.source_scores[positive_eligible].sum())
        positive_without_frontier = int(
            ((self.source_scores > 0.0) & ~eligible).sum()
        )
        best_source_index: Optional[int] = None
        if bool(positive_eligible.any()):
            eligible_indices = np.flatnonzero(positive_eligible)
            best_source_index = int(
                eligible_indices[
                    np.argmax(self.source_scores[eligible_indices])
                ]
            )

        diagnostics: Dict[str, Any] = {
            "schema": "cnabu_occlusion_frontier_allocation_v1",
            "score_provenance": self.score_provenance,
            "guidance_fraction": self.guidance_fraction,
            "frontier_population_count": population_count,
            "source_count": int(len(self.source_scores)),
            "eligible_source_count": int(eligible.sum()),
            "positive_eligible_source_count": int(positive_eligible.sum()),
            "positive_source_without_frontier_count": positive_without_frontier,
            "source_frontier_counts": source_frontier_counts.tolist(),
            "best_source_index": best_source_index,
            "used_guidance": False,
            "fallback_reason": None,
            "probability_min": None,
            "probability_max": None,
        }
        if population_count == 0:
            diagnostics["fallback_reason"] = "empty_frontier_population"
            return FrontierSamplingDecision(None, frontier_sources, diagnostics)
        if positive_score_sum <= 0.0:
            diagnostics["fallback_reason"] = "no_positive_eligible_source"
            return FrontierSamplingDecision(None, frontier_sources, diagnostics)

        guided = np.zeros(population_count, dtype=np.float64)
        for source_index in np.flatnonzero(positive_eligible):
            source_probability = float(
                self.source_scores[source_index] / positive_score_sum
            )
            source_frontiers = frontier_sources == int(source_index)
            guided[source_frontiers] = source_probability / int(
                source_frontier_counts[source_index]
            )
        probabilities = (
            (1.0 - self.guidance_fraction) / population_count
            + self.guidance_fraction * guided
        )
        probabilities /= probabilities.sum()
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise RuntimeError("frontier allocation produced invalid probabilities")
        diagnostics.update(
            {
                "used_guidance": True,
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
            }
        )
        return FrontierSamplingDecision(
            probabilities.copy(), frontier_sources, diagnostics
        )

    def selection_diagnostics(
        self,
        decision: FrontierSamplingDecision,
        selected_population_indices: Any,
    ) -> Dict[str, Any]:
        """Summarize selected source inclusion without changing the draw."""

        selected = np.asarray(selected_population_indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("selected_population_indices must be a 1-D integer array")
        selected = selected.astype(np.int64, copy=False)
        population_count = len(decision.frontier_source_indices)
        if np.any(selected < 0) or np.any(selected >= population_count):
            raise ValueError("selected population index is out of range")
        selected_sources = decision.frontier_source_indices[selected]
        best_source = decision.diagnostics.get("best_source_index")
        return {
            **decision.diagnostics,
            "selected_population_indices": selected.tolist(),
            "selected_source_indices": selected_sources.tolist(),
            "selected_named_source_count": int((selected_sources >= 0).sum()),
            "selected_unique_source_indices": sorted(
                int(value) for value in np.unique(selected_sources) if value >= 0
            ),
            "best_source_included": (
                None
                if best_source is None
                else bool(np.any(selected_sources == int(best_source)))
            ),
        }


class BeliefOcclusionAllocationController:
    """Build one deterministic or privileged-oracle allocator per belief revision."""

    def __init__(
        self,
        *,
        arm: str,
        node_splitter: Any,
        info_gain: Any,
        device: str = "cuda:0",
        raw_shape_hw: Sequence[int] = (140, 200),
        crop_rows: Sequence[int] = (10, 130),
        occupancy_threshold: float = 0.5,
        match_iou_threshold: float = 0.25,
        source_batch_size: int = 4,
        guidance_fraction: float = 0.75,
    ) -> None:
        if str(arm) not in {"deterministic", "oracle"}:
            raise ValueError("arm must be deterministic or oracle")
        raw_shape = tuple(int(value) for value in raw_shape_hw)
        rows = tuple(int(value) for value in crop_rows)
        if len(raw_shape) != 2 or len(rows) != 2 or rows[1] - rows[0] <= 0:
            raise ValueError("raw_shape_hw and crop_rows are invalid")
        if rows[0] < 0 or rows[1] > raw_shape[0]:
            raise ValueError("crop_rows exceed raw_shape_hw")
        self.arm = str(arm)
        self.node_splitter = node_splitter
        self.info_gain = info_gain
        self.device = str(device)
        self.raw_shape_hw = raw_shape
        self.crop_rows = rows
        self.occupancy_threshold = float(occupancy_threshold)
        self.match_iou_threshold = float(match_iou_threshold)
        self.source_batch_size = int(source_batch_size)
        self.guidance_fraction = float(guidance_fraction)
        self._canonical_ray_cache: Dict[int, np.ndarray] = {}
        self._belief_revision = 0
        self.history: list[Dict[str, Any]] = []

    def _synchronize(self) -> None:
        import torch

        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))

    def _canonical_rays(
        self,
        camera_index: int,
        *,
        grid_shape_zyx: Sequence[int],
    ) -> Tuple[np.ndarray, bool, float]:
        from shelf_gym.utils.cnabu_occlusion_attribution import (
            info_gain_raycast_to_canonical_zyx,
        )

        index = int(camera_index)
        if index in self._canonical_ray_cache:
            return self._canonical_ray_cache[index], True, 0.0
        started = time.perf_counter()
        raw = self.info_gain.get_raycast(camera_idx=index)
        canonical = info_gain_raycast_to_canonical_zyx(
            raw,
            grid_shape_zyx=grid_shape_zyx,
            crop_rows=self.crop_rows,
            raw_shape_hw=self.raw_shape_hw,
        )
        elapsed = float(time.perf_counter() - started)
        canonical.setflags(write=False)
        self._canonical_ray_cache[index] = canonical
        return canonical, False, elapsed

    def build_allocator(
        self,
        *,
        occupancy_distribution: Any,
        semantic_concentration: Any,
        camera_index: int,
        environment: Any = None,
    ) -> OcclusionFrontierAllocator:
        """Build a fixed-budget allocator from the current live belief."""

        from scene_graph_mem.relations.belief_occlusion import (
            build_unresolved_uncertainty_field,
        )
        from scene_graph_mem.relations.occlusion_attribution import (
            TorchHiddenUncertaintyCache,
        )
        from scene_graph_mem.relations.path_aligned_features import (
            reconstruct_sparse_node_voxel_support,
        )
        from scene_graph_mem.runtime.cnabu_scene_graph import decode_binary_mask_rle
        from shelf_gym.utils.cnabu_occlusion_attribution import (
            align_oracle_supports_to_nodes,
            build_gt_object_voxel_supports,
            build_runtime_support_partition,
            dense_supports_from_sparse_indices,
            match_nodes_to_gt_objects,
        )

        total_started = time.perf_counter()
        belief = live_cnabu_belief_arrays(
            occupancy_distribution,
            semantic_concentration,
        )
        self._synchronize()
        nodes_started = time.perf_counter()
        node_result = self.node_splitter.predict_nodes(
            occupancy_distribution=occupancy_distribution,
            semantic_concentration=semantic_concentration,
            raw_shape_hw=self.raw_shape_hw,
            crop_rows=self.crop_rows,
            metadata={"consumer": "cnabu_belief_occlusion_v1"},
            include_masks=True,
        )
        self._synchronize()
        node_seconds = float(time.perf_counter() - nodes_started)
        node_metadata = dict(node_result.get("metadata", {}))
        if node_metadata.get("physical_relation_executed") is not False:
            raise RuntimeError("belief-occlusion planning requires the node-only splitter")
        if node_metadata.get("physical_relation_assets_or_records_loaded") is not False:
            raise RuntimeError("physical relation assets are forbidden in this planner")
        nodes = list(node_result.get("nodes", []))
        masks = (
            np.stack(
                [decode_binary_mask_rle(node["mask"]).astype(bool) for node in nodes]
            )
            if nodes
            else np.zeros((0,) + self.raw_shape_hw, dtype=bool)
        )
        classes = np.asarray([int(node["class_id"]) for node in nodes], dtype=np.int64)
        sparse = reconstruct_sparse_node_voxel_support(
            belief["occupancy_mean"],
            belief["semantic_mean"],
            masks,
            classes,
            crop_rows=self.crop_rows,
            occupancy_threshold=self.occupancy_threshold,
        )
        deterministic_supports = dense_supports_from_sparse_indices(
            sparse.indices_zyx,
            grid_shape_zyx=sparse.grid_shape_zyx,
        )
        runtime_partition = build_runtime_support_partition(
            occupancy_mean=belief["occupancy_mean"],
            semantic_mean=belief["semantic_mean"],
            source_supports_zyx=deterministic_supports,
            occupancy_threshold=self.occupancy_threshold,
        )

        oracle_record: Optional[Dict[str, Any]] = None
        source_supports = deterministic_supports
        if self.arm == "oracle":
            if environment is None:
                raise ValueError("oracle allocation requires the current environment")
            gt_raw = environment.get_gt_height_map(no_tqdm=True)
            required = {
                "voxel_height_map",
                "semantic_gt",
                "voxel_semantic_map",
                "instance_maps",
            }
            missing = sorted(required - set(gt_raw))
            if missing:
                raise KeyError("oracle GT data is missing {}".format(missing))
            gt_supports = build_gt_object_voxel_supports(
                hm3d=gt_raw["voxel_height_map"],
                semantic_2d=gt_raw["semantic_gt"],
                semantic_3d=gt_raw["voxel_semantic_map"],
                instance_maps=gt_raw["instance_maps"],
                crop_rows=self.crop_rows,
                occupancy_threshold=self.occupancy_threshold,
            )
            matching = match_nodes_to_gt_objects(
                node_masks_raw_hw=masks,
                node_class_ids=classes,
                gt_masks_raw_hw=gt_supports.masks_raw_hw,
                gt_class_ids=gt_supports.class_ids,
                iou_threshold=self.match_iou_threshold,
            )
            aligned = align_oracle_supports_to_nodes(gt_supports, matching)
            source_supports = aligned.source_supports_zyx
            oracle_record = {
                "matching": matching.to_dict(
                    gt_instance_ids=gt_supports.instance_ids
                ),
                "gt_coverage": gt_supports.coverage_summary(),
                "unrepresented_voxel_count": int(
                    aligned.unrepresented_support_zyx.sum()
                ),
            }

        field = build_unresolved_uncertainty_field(
            belief["occupancy_mean"],
            belief["occupancy_epistemic"],
            belief["semantic_vacuity"],
        )
        components = {
            "occupancy_epistemic": field.occupancy_epistemic * field.lambda_occ,
            "semantic_vacuity": field.semantic_vacuity * field.lambda_sem,
            "total": field.total,
        }
        rays, ray_cache_hit, ray_seconds = self._canonical_rays(
            int(camera_index),
            grid_shape_zyx=belief["occupancy_mean"].shape,
        )
        revision = "planner-belief-{}".format(self._belief_revision)
        self._belief_revision += 1
        self._synchronize()
        cache_started = time.perf_counter()
        cache = TorchHiddenUncertaintyCache(
            occupancy_mean=belief["occupancy_mean"],
            component_masses=components,
            source_supports=source_supports,
            belief_revision=revision,
            support_provenance=(
                "runtime_cnabu_learned_component_splitter"
                if self.arm == "deterministic"
                else "offline_gt_aligned_to_live_learned_nodes"
            ),
            device=self.device,
            source_batch_size=self.source_batch_size,
        )
        self._synchronize()
        cache_seconds = float(time.perf_counter() - cache_started)
        self._synchronize()
        score_started = time.perf_counter()
        scores = cache.score(rays, belief_revision=revision)
        self._synchronize()
        score_seconds = float(time.perf_counter() - score_started)
        allocator = OcclusionFrontierAllocator(
            source_masks_raw_hw=masks,
            source_scores=scores["total"],
            guidance_fraction=self.guidance_fraction,
            score_provenance=(
                "deterministic_cnabu_top1_vig_camera"
                if self.arm == "deterministic"
                else "oracle_gt_aligned_top1_vig_camera_offline_only"
            ),
        )
        record: Dict[str, Any] = {
            "schema": "cnabu_belief_occlusion_planner_query_v1",
            "arm": self.arm,
            "camera_index": int(camera_index),
            "belief_revision": revision,
            "uses_gt": self.arm == "oracle",
            "deployable": self.arm == "deterministic",
            "physical_relation_executed": False,
            "physical_relation_assets_or_records_loaded": False,
            "node_count": int(len(nodes)),
            "node_class_ids": classes.tolist(),
            "node_masks_raw_shape": list(masks.shape),
            "node_metadata": node_metadata,
            "runtime_partition": runtime_partition.coverage_summary(),
            "oracle": oracle_record,
            "source_scores": {
                name: np.asarray(values, dtype=np.float64).tolist()
                for name, values in scores.items()
            },
            "source_score_sums": {
                name: float(np.asarray(values, dtype=np.float64).sum())
                for name, values in scores.items()
            },
            "cache": cache.metadata.to_dict(),
            "ray_cache_hit": bool(ray_cache_hit),
            "timing_seconds": {
                "node_generation": node_seconds,
                "ray_generation_and_conversion": ray_seconds,
                "attribution_cache_construction": cache_seconds,
                "attribution_score": score_seconds,
                "attribution_incremental": float(
                    ray_seconds + cache_seconds + score_seconds
                ),
                "controller_total": float(time.perf_counter() - total_started),
            },
            "frontier_allocation": None,
            "feasible_path_count": None,
            "valuable_source_without_feasible_path": None,
            "selected_candidate": None,
            "execution": None,
        }
        self.history.append(record)
        return allocator

    def record_sampling_result(
        self,
        *,
        allocation_diagnostics: Optional[Mapping[str, Any]],
        feasible_path_count: int,
    ) -> None:
        if not self.history:
            raise RuntimeError("no attribution query exists for sampling diagnostics")
        count = int(feasible_path_count)
        if count < 0:
            raise ValueError("feasible_path_count must be non-negative")
        allocation = (
            None if allocation_diagnostics is None else dict(allocation_diagnostics)
        )
        self.history[-1]["frontier_allocation"] = allocation
        self.history[-1]["feasible_path_count"] = count
        self.history[-1]["valuable_source_without_feasible_path"] = bool(
            allocation is not None
            and allocation.get("best_source_index") is not None
            and not bool(allocation.get("best_source_has_feasible_path", False))
        )

    def record_scoring_result(self, *, selected_candidate_index: int) -> None:
        """Attach the unchanged VIG-selected path to its sampled source."""

        if not self.history:
            raise RuntimeError("no attribution query exists for scoring diagnostics")
        record = self.history[-1]
        count = record.get("feasible_path_count")
        index = int(selected_candidate_index)
        if count is None or index < 0 or index >= int(count):
            raise ValueError("selected candidate index is outside feasible paths")
        allocation = record.get("frontier_allocation")
        source_index = None
        best_source_index = None
        if allocation is not None:
            source_indices = allocation.get("feasible_path_source_indices", [])
            if len(source_indices) != int(count):
                raise RuntimeError("feasible-path source diagnostics are incomplete")
            source_index = int(source_indices[index])
            best_source_index = allocation.get("best_source_index")
        record["selected_candidate"] = {
            "candidate_index": index,
            "source_index": source_index,
            "source_is_named": (
                None if source_index is None else bool(source_index >= 0)
            ),
            "source_is_best_attribution_source": (
                None
                if source_index is None or best_source_index is None
                else bool(source_index == int(best_source_index))
            ),
            "selection_utility": "unchanged_mem_push_vig",
            "sampling_priority_provenance": self.arm,
        }

    def record_execution_result(
        self,
        *,
        executed: bool,
        push_return_code: Optional[int],
        object_drop: bool,
    ) -> None:
        """Record whether the VIG-selected path reached safe execution."""

        if not self.history:
            raise RuntimeError("no attribution query exists for execution diagnostics")
        if executed and self.history[-1].get("selected_candidate") is None:
            raise RuntimeError("executed push has no recorded selected candidate")
        code = None if push_return_code is None else int(push_return_code)
        self.history[-1]["execution"] = {
            "executed": bool(executed),
            "push_return_code": code,
            "tilted_object_failure": None if code is None else bool(code != 0),
            "object_drop": bool(object_drop),
            "accepted_without_tilt_or_drop": bool(
                executed and code == 0 and not object_drop
            ),
        }


__all__ = [
    "BeliefOcclusionAllocationController",
    "FrontierSamplingDecision",
    "OcclusionFrontierAllocator",
    "live_cnabu_belief_arrays",
]
