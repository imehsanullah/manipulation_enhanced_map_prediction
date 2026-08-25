from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shelf_gym.scripts.prepare_psg_mem_x3_episode import (
    TARGET_CATALOG_SCHEMA,
    build_episode_spec,
    evaluator_target_catalog,
    load_evaluator_target_catalog,
)
from shelf_gym.scripts.psg_mem_x3_runtime_adapter import validate_runtime_spec


THESIS_ROOT = Path(__file__).resolve().parents[2]
SCENE_GRAPH_SRC = THESIS_ROOT / "scene_graph_mem" / "src"
X3_CONFIG = (
    THESIS_ROOT
    / "scene_graph_mem"
    / "configs"
    / "psg_mem"
    / "experiments"
    / "x3_system_retrieval.yaml"
)


class PrepareX3EpisodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.scene = self.root / "placed_objects.pkl"
        self.snapshot = self.root / "initial_state.json"
        self.scene.write_bytes(b"fixed scene fixture\n")
        self.snapshot.write_text("{}\n", encoding="utf-8")
        instances = np.full((140, 200), -1, dtype=np.int64)
        semantics = np.full((140, 200), 14, dtype=np.int64)
        instances[8:18, 30:40] = 101
        semantics[8:18, 30:40] = 2
        instances[108:122, 120:138] = 202
        semantics[108:122, 120:138] = 4
        self.gt_hms = self.root / "gt_hms.npz"
        np.savez_compressed(
            self.gt_hms,
            instance_maps=instances[None],
            semantic_2d=semantics,
        )
        self.target_catalog = self.root / "target_catalog.json"
        self.target_catalog.write_text(
            json.dumps(
                {
                    "schema": TARGET_CATALOG_SCHEMA,
                    "scene_group_id": "fixture",
                    "source": "live_gt_height_map_evaluator_only",
                    "raw_shape_hw": [140, 200],
                    "depth_partition": "raw_y_thirds_v1",
                    "targets": [
                        {
                            "evaluation_object_id": 101,
                            "class_id": 2,
                            "coarse_region": "front",
                        },
                        {
                            "evaluation_object_id": 202,
                            "class_id": 4,
                            "coarse_region": "back",
                        },
                    ],
                    "runtime_visible": False,
                    "contains_simulator_instance_ids": True,
                    "planner_may_read": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, *, arm_id: str = "d") -> dict:
        return build_episode_spec(
            episode_id=f"fixture-{arm_id}",
            arm_id=arm_id,
            scene_path=self.scene,
            initial_state_snapshot=self.snapshot,
            gt_hms_path=self.gt_hms,
            experiment_config_path=X3_CONFIG,
            scene_graph_mem_src=SCENE_GRAPH_SRC,
            target_seed=3,
            planner_seed=0,
            requested_region="back",
            action_budget=5,
            max_sampled_pushes=12,
            relation_device="cuda:0",
            render=False,
        )

    def test_catalog_uses_raw_map_depth_thirds(self) -> None:
        self.assertEqual(
            evaluator_target_catalog(self.gt_hms),
            [
                {
                    "evaluation_object_id": 101,
                    "class_id": 2,
                    "coarse_region": "front",
                },
                {
                    "evaluation_object_id": 202,
                    "class_id": 4,
                    "coarse_region": "back",
                },
            ],
        )

    def test_prepared_spec_redacts_runtime_and_hashes_every_input(self) -> None:
        spec = self._build()
        self.assertEqual(
            spec["task"]["planner_query"],
            {"class_id": 4, "coarse_region": "back"},
        )
        self.assertEqual(spec["task"]["evaluation_token"]["evaluation_object_id"], 202)
        self.assertNotIn("evaluation_object_id", json.dumps(spec["integration"]))
        self.assertEqual(len(spec["integration"]["scene"]["sha256"]), 64)
        self.assertEqual(set(spec["adapter_artifacts"]), {"runtime", "evaluator"})
        self.assertTrue(
            all(
                len(identity["sha256"]) == 64
                for identity in spec["adapter_artifacts"].values()
            )
        )
        self.assertEqual(spec["episode_config"]["budget"], 5)
        self.assertEqual(spec["planner_overrides"]["representation"], "graph")
        validated = validate_runtime_spec(
            {
                "episode_id": spec["episode_id"],
                "arm_id": spec["arm_id"],
                "planner_query": spec["task"]["planner_query"],
                "integration": spec["integration"],
            }
        )
        self.assertIs(validated["persistent"], True)

    def test_compact_evaluator_catalog_replaces_gt_map_without_runtime_leak(
        self,
    ) -> None:
        self.assertEqual(
            load_evaluator_target_catalog(self.target_catalog),
            evaluator_target_catalog(self.gt_hms),
        )
        spec = build_episode_spec(
            episode_id="catalog-fixture",
            arm_id="d",
            scene_path=self.scene,
            initial_state_snapshot=self.snapshot,
            gt_hms_path=None,
            target_catalog_path=self.target_catalog,
            experiment_config_path=X3_CONFIG,
            scene_graph_mem_src=SCENE_GRAPH_SRC,
            target_seed=3,
            planner_seed=0,
            requested_region="back",
            action_budget=5,
            max_sampled_pushes=12,
            relation_device="cuda:0",
            render=False,
        )
        self.assertNotIn("gt_hms", spec["evaluation"])
        self.assertEqual(
            spec["evaluation"]["target_catalog"]["path"],
            str(self.target_catalog),
        )
        self.assertNotIn("evaluation_object_id", json.dumps(spec["integration"]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_episode_spec(
                episode_id="bad-fixture",
                arm_id="d",
                scene_path=self.scene,
                initial_state_snapshot=self.snapshot,
                gt_hms_path=self.gt_hms,
                target_catalog_path=self.target_catalog,
                experiment_config_path=X3_CONFIG,
                scene_graph_mem_src=SCENE_GRAPH_SRC,
                target_seed=3,
                planner_seed=0,
                requested_region="back",
                action_budget=5,
                max_sampled_pushes=12,
                relation_device="cuda:0",
                render=False,
            )

    def test_arm_specific_flat_representation_is_materialized(self) -> None:
        spec = self._build(arm_id="f")
        self.assertEqual(spec["planner_overrides"]["representation"], "flat_marginal")


if __name__ == "__main__":
    unittest.main()
