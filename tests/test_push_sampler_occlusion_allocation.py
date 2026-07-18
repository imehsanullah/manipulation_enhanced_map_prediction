from __future__ import annotations

import numpy as np
import unittest
from unittest.mock import patch

from shelf_gym.utils.cnabu_occlusion_planner import OcclusionFrontierAllocator
from shelf_gym.utils.pushing_utils import PushSampler


class PushSamplerOcclusionAllocationTest(unittest.TestCase):
    def test_default_frontier_draw_preserves_exact_choice_signature(self) -> None:
        sampler = PushSampler()
        population = np.asarray(
            [[1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=np.int64
        )
        calls = []

        def fake_choice(*args, **kwargs):
            calls.append((args, kwargs))
            return np.asarray([1, 3], dtype=np.int64)

        with patch("shelf_gym.utils.pushing_utils.cp.random.choice", fake_choice):
            selected = sampler._sample_frontier_indices(
                population, num_points=2, frontier_allocator=None
            )

        self.assertEqual(calls, [((4, 2), {"replace": False})])
        self.assertEqual(selected.tolist(), [[1, 2, 2], [2, 2, 2]])
        self.assertIsNone(sampler.last_frontier_allocation)

    def test_guided_draw_passes_one_probability_vector_and_records_selection(self) -> None:
        sampler = PushSampler()
        population = np.asarray(
            [[1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=np.int64
        )
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[0, 1, 1:3] = True
        allocator = OcclusionFrontierAllocator(
            source_masks_raw_hw=masks,
            source_scores=[2.0],
            score_provenance="deterministic_cnabu",
        )
        calls = []

        def fake_choice(*args, **kwargs):
            calls.append((args, kwargs))
            return np.asarray([0, 2], dtype=np.int64)

        with patch("shelf_gym.utils.pushing_utils.cp.random.choice", fake_choice):
            selected = sampler._sample_frontier_indices(
                population, num_points=2, frontier_allocator=allocator
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], (4, 2))
        self.assertFalse(calls[0][1]["replace"])
        probabilities = np.asarray(calls[0][1]["p"].get())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertEqual(selected.tolist(), [[1, 1, 2], [2, 1, 2]])
        self.assertTrue(sampler.last_frontier_allocation["used_guidance"])
        self.assertTrue(sampler.last_frontier_allocation["best_source_included"])
        self.assertEqual(
            sampler.last_frontier_allocation["selected_source_indices"], [0, -1]
        )

    def test_feasible_path_sources_are_diagnostic_only(self) -> None:
        sampler = PushSampler()
        sampler.last_frontier_allocation = {"best_source_index": 2}
        result = sampler._finalize_frontier_source_diagnostics(
            {
                "paths": ["a", "b", "c"],
                "path_annotations": [[], [], []],
                "motion_parametrization": [1, 2, 3],
                "_frontier_source_indices": [2, -1, 4],
            }
        )

        self.assertEqual(
            set(result), {"paths", "path_annotations", "motion_parametrization"}
        )
        self.assertEqual(
            sampler.last_frontier_allocation["feasible_path_source_indices"],
            [2, -1, 4],
        )
        self.assertEqual(
            sampler.last_frontier_allocation["feasible_unique_source_indices"],
            [2, 4],
        )
        self.assertTrue(
            sampler.last_frontier_allocation["best_source_has_feasible_path"]
        )


if __name__ == "__main__":
    unittest.main()
