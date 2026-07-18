from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shelf_gym"
    / "utils"
    / "cnabu_occlusion_planner.py"
)
_SPEC = importlib.util.spec_from_file_location("cnabu_occlusion_planner", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

OcclusionFrontierAllocator = _MODULE.OcclusionFrontierAllocator
live_cnabu_belief_arrays = _MODULE.live_cnabu_belief_arrays


def test_object_balanced_mixture_does_not_reward_large_masks() -> None:
    masks = np.zeros((2, 6, 7), dtype=bool)
    masks[0, 1, 1:3] = True
    masks[1, 2, 1] = True
    allocator = OcclusionFrontierAllocator(
        source_masks_raw_hw=masks,
        source_scores=[3.0, 1.0],
        guidance_fraction=0.75,
        score_provenance="deterministic_cnabu",
    )
    frontiers = np.asarray(
        [[1, 1, 3], [1, 2, 4], [2, 1, 5], [4, 5, 6]], dtype=np.int64
    )

    decision = allocator.sampling_probabilities(frontiers)

    assert decision.frontier_source_indices.tolist() == [0, 0, 1, -1]
    assert decision.probabilities == pytest.approx(
        [0.34375, 0.34375, 0.25, 0.0625]
    )
    assert decision.diagnostics["eligible_source_count"] == 2
    assert decision.diagnostics["positive_eligible_source_count"] == 2
    assert decision.diagnostics["best_source_index"] == 0
    assert decision.diagnostics["used_guidance"] is True


def test_no_positive_eligible_source_requests_exact_original_uniform_path() -> None:
    masks = np.zeros((2, 4, 5), dtype=bool)
    masks[0, 1, 1] = True
    allocator = OcclusionFrontierAllocator(
        source_masks_raw_hw=masks,
        source_scores=[0.0, 7.0],
        guidance_fraction=0.75,
        score_provenance="deterministic_cnabu",
    )

    decision = allocator.sampling_probabilities(
        np.asarray([[1, 1, 2], [2, 2, 2]], dtype=np.int64)
    )

    assert decision.probabilities is None
    assert decision.diagnostics["used_guidance"] is False
    assert decision.diagnostics["fallback_reason"] == "no_positive_eligible_source"


def test_overlap_uses_highest_score_and_stable_lower_index_for_exact_ties() -> None:
    masks = np.ones((3, 3, 3), dtype=bool)
    frontiers = np.asarray([[1, 1, 1]], dtype=np.int64)
    high = OcclusionFrontierAllocator(
        source_masks_raw_hw=masks,
        source_scores=[1.0, 4.0, 2.0],
        score_provenance="oracle_offline",
    ).sampling_probabilities(frontiers)
    tied = OcclusionFrontierAllocator(
        source_masks_raw_hw=masks,
        source_scores=[4.0, 4.0, 2.0],
        score_provenance="oracle_offline",
    ).sampling_probabilities(frontiers)

    assert high.frontier_source_indices.tolist() == [1]
    assert tied.frontier_source_indices.tolist() == [0]
    assert high.probabilities.tolist() == [1.0]


def test_allocator_rejects_invalid_scores_fraction_and_frontier_coordinates() -> None:
    masks = np.zeros((1, 3, 4), dtype=bool)
    with pytest.raises(ValueError, match="finite and non-negative"):
        OcclusionFrontierAllocator(
            source_masks_raw_hw=masks,
            source_scores=[np.nan],
            score_provenance="bad",
        )
    with pytest.raises(ValueError, match="guidance_fraction"):
        OcclusionFrontierAllocator(
            source_masks_raw_hw=masks,
            source_scores=[1.0],
            guidance_fraction=1.1,
            score_provenance="bad",
        )
    allocator = OcclusionFrontierAllocator(
        source_masks_raw_hw=masks,
        source_scores=[1.0],
        score_provenance="test",
    )
    with pytest.raises(ValueError, match="outside the raw mask frame"):
        allocator.sampling_probabilities(np.asarray([[3, 0, 0]], dtype=np.int64))


def test_live_belief_conversion_matches_mem_interleaved_evidence_convention() -> None:
    # MEM stores beta evidence in even channels and alpha evidence in odd ones.
    distribution = np.asarray([[[[3.0]], [[9.0]], [[7.0]], [[1.0]]]])
    semantics = np.asarray([[[[2.0]], [[6.0]]]])

    result = live_cnabu_belief_arrays(distribution, semantics)

    assert result["occupancy_mean"][:, 0, 0].tolist() == pytest.approx(
        [9.0 / 12.0, 1.0 / 8.0]
    )
    assert result["occupancy_epistemic"][:, 0, 0].tolist() == pytest.approx(
        [27.0 / (12.0 * 12.0 * 13.0), 7.0 / (8.0 * 8.0 * 9.0)]
    )
    assert result["semantic_mean"][:, 0, 0].tolist() == pytest.approx([0.25, 0.75])
    assert result["semantic_vacuity"][0, 0] == pytest.approx(0.25)
