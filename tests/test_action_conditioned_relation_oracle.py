from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from shelf_gym.utils.action_conditioned_relation_oracle import (
    OracleActionFamilyConfig,
    build_cnabu_runtime_candidate_kinematic_mask,
    build_candidate_planner_swept_features,
    build_geometry_pseudo_gt_adjacency,
    build_runtime_candidate_action_mask,
    build_runtime_candidate_kinematic_mask,
    cnabu_sparse_support_world_aabbs,
    cnabu_sparse_support_world_voxels,
    compare_relation_targets,
    counterfactual_candidate_descriptor,
    extract_gt_object_records,
    front_extraction_waypoints,
    interpolate_joint_configs,
    summarize_extraction_progress,
    summarize_monitored_displacement,
    summarize_signed_distances,
)
from shelf_gym.scripts.inspect_action_conditioned_relation_oracle import select_scene_disjoint_round_robin
from shelf_gym.scripts.validate_action_conditioned_relation_counterfactuals import (
    aggregate_counterfactual_records,
    select_relation_score_threshold,
    select_hard_penetration_threshold,
    select_stratified_counterfactuals,
)


class ActionConditionedRelationOracleAdapterTest(unittest.TestCase):
    def test_cnabu_sparse_support_converts_to_runtime_world_aabbs_without_gt(self) -> None:
        class FakeHeightmapGeneration:
            bounds = np.asarray([[-0.5, 0.5], [0.0, 2.0], [0.0, 2.0]])

            def map_point_to_world_point(self, point):
                point = np.asarray(point, dtype=np.float64)
                return np.asarray([-0.5 + point[0] * 0.1, point[1] * 0.1, point[2] * 0.1])

        boxes = cnabu_sparse_support_world_aabbs(
            FakeHeightmapGeneration(),
            [np.asarray([[3, 1, 2], [4, 2, 3]], dtype=np.int16)],
            crop_rows=(10, 12),
        )

        self.assertTrue(np.allclose(boxes[0][0], [0.1, 1.1, 0.3]))
        self.assertTrue(np.allclose(boxes[0][1], [0.3, 1.3, 0.5]))

    def test_cnabu_sparse_support_robust_box_trims_low_mass_boundary_voxel(self) -> None:
        class FakeHeightmapGeneration:
            bounds = np.asarray([[-0.5, 0.5], [0.0, 2.0], [0.0, 2.0]])

            def map_point_to_world_point(self, point):
                point = np.asarray(point, dtype=np.float64)
                return np.asarray([-0.5 + point[0] * 0.01, point[1] * 0.01, point[2] * 0.01])

        core = np.repeat(np.asarray([[3, 1, 2]], dtype=np.int16), 100, axis=0)
        support = np.concatenate((core, np.asarray([[3, 1, 50]], dtype=np.int16)), axis=0)
        full = cnabu_sparse_support_world_aabbs(
            FakeHeightmapGeneration(),
            [support],
            crop_rows=(10, 12),
        )
        robust = cnabu_sparse_support_world_aabbs(
            FakeHeightmapGeneration(),
            [support],
            crop_rows=(10, 12),
            boundary_quantile=0.05,
        )

        self.assertGreater(full[0][1][0] - full[0][0][0], 0.4)
        self.assertAlmostEqual(robust[0][1][0] - robust[0][0][0], 0.01)

    def test_cnabu_sparse_support_world_voxels_preserve_order_and_cell_extent(self) -> None:
        class FakeHeightmapGeneration:
            bounds = np.asarray([[-0.5, 0.5], [0.0, 2.0], [0.0, 2.0]])

            def map_point_to_world_point(self, point):
                point = np.asarray(point, dtype=np.float64)
                return np.asarray(
                    [-0.5 + point[0] * 0.1, point[1] * 0.1, point[2] * 0.1]
                )

        centers, half_extents = cnabu_sparse_support_world_voxels(
            FakeHeightmapGeneration(),
            [
                np.asarray([[3, 1, 2], [4, 2, 3]], dtype=np.int16),
                np.asarray([[5, 3, 4]], dtype=np.int16),
            ],
            crop_rows=(10, 14),
        )

        self.assertEqual(len(centers), 2)
        self.assertTrue(np.allclose(centers[0][0], [0.25, 1.15, 0.35]))
        self.assertTrue(np.allclose(centers[0][1], [0.15, 1.25, 0.45]))
        self.assertTrue(np.allclose(half_extents, [0.05, 0.05, 0.05]))

    def test_planner_swept_features_separate_stages_and_clearance(self) -> None:
        def bins(*occupied_boxes):
            values = [[] for _ in range(4)]
            for index, box in occupied_boxes:
                values[index].append(box)
            return values

        hit_box = [[-0.02, -0.02, -0.02], [0.02, 0.02, 0.02]]
        grasp_near_box = [[0.03, -0.02, -0.02], [0.05, 0.02, 0.02]]
        far_box = [[2.0, 2.0, 2.0], [2.1, 2.1, 2.1]]
        geometry = {
            "progress_bin_count": 4,
            "robot_link_aabbs_by_stage": {
                "approach": bins((0, hit_box), (1, far_box)),
                "grasp": bins((0, grasp_near_box)),
                "extraction": bins((2, hit_box)),
            },
            "carried_target_aabbs_by_stage": {
                "extraction": bins((3, hit_box)),
            },
        }

        def target(node_id, enabled):
            candidates = []
            for index, candidate_id in enumerate(
                (
                    "front_x0.35_z0.35",
                    "front_x0.35_z0.55",
                    "front_x0.35_z0.75",
                    "front_x0.50_z0.35",
                    "front_x0.50_z0.55",
                    "front_x0.50_z0.75",
                    "front_x0.65_z0.35",
                    "front_x0.65_z0.55",
                    "front_x0.65_z0.75",
                )
            ):
                feasible = bool(enabled and index == 0)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "kinematically_feasible": feasible,
                        "planner_swept_geometry": geometry if feasible else {},
                    }
                )
            return {"node_id": node_id, "candidates": candidates}

        result = build_candidate_planner_swept_features(
            source_world_aabbs=(
                np.asarray(
                    [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]],
                    dtype=np.float64,
                ),
                np.asarray(
                    [[0.99, 0.99, 0.99], [1.01, 1.01, 1.01]],
                    dtype=np.float64,
                ),
            ),
            node_ids=[101, 202],
            targets=[target(101, False), target(202, True)],
        )

        features = np.asarray(result["pair_features"], dtype=np.float32)
        names = list(result["pair_feature_names"])
        pair = features[0, 1, 0]
        self.assertEqual(features.shape, (2, 2, 9, 20))
        self.assertEqual(pair[names.index("approach_robot_intersection_over_source")], 1.0)
        self.assertEqual(pair[names.index("approach_robot_first_contact_progress")], 0.25)
        self.assertEqual(
            pair[names.index("approach_robot_longest_contiguous_contact_fraction")],
            0.25,
        )
        self.assertAlmostEqual(
            pair[names.index("grasp_robot_minimum_clearance_norm")],
            0.4,
        )
        self.assertEqual(
            pair[names.index("extraction_robot_first_contact_progress")],
            0.75,
        )
        self.assertEqual(
            pair[names.index("extraction_carried_target_first_contact_progress")],
            1.0,
        )
        infeasible_pair = features[1, 0, 0]
        for name in names:
            expected = 1.0 if name.endswith("_minimum_clearance_norm") else 0.0
            self.assertEqual(infeasible_pair[names.index(name)], expected)
        self.assertFalse(features[1, 1].any())
        self.assertFalse(result["safety"]["queries_dynamic_scene_objects"])

    def test_runtime_candidate_kinematic_mask_uses_ordered_ik_candidates(self) -> None:
        class FakeBullet:
            def __init__(self, env):
                self.env = env

            def resetJointState(self, *_args, **_kwargs):
                return None

            def getLinkState(self, *_args, **_kwargs):
                return (self.env.last_ik_position.tolist(), None)

        class FakeEnv:
            arm_joint_indices = tuple(range(6))
            robot_id = 1
            tool_tip_id = 2
            client_id = 3
            init_ori = [0.0, 0.0, 0.0, 1.0]

            def __init__(self):
                self.last_ik_position = np.zeros(3, dtype=np.float64)
                self._p = FakeBullet(self)

            def get_ik_joints(self, position, _orientation, link=None):
                self.last_ik_position = np.asarray(position, dtype=np.float64)
                if float(self.last_ik_position[0]) > 0.06:
                    return [float("nan")] * 6
                return [0.0] * 6

        result = build_runtime_candidate_kinematic_mask(
            FakeEnv(),
            target_world_aabbs=[[[0.0, 0.85, 0.91], [0.10, 0.95, 1.11]]],
            node_ids=["cnabu-node-0"],
            initial_arm_config=[0.0] * 6,
        )

        mask = np.asarray(result["kinematic_mask"], dtype=bool)
        self.assertEqual(mask.shape, (1, 9))
        self.assertTrue(mask[0, :6].all())
        self.assertFalse(mask[0, 6:].any())
        self.assertEqual(result["node_ids"], ["cnabu-node-0"])
        self.assertFalse(result["safety"]["uses_gt_or_simulator_instance_ids"])

    def test_composed_cnabu_runtime_adapter_preserves_node_order(self) -> None:
        class FakeHeightmapGeneration:
            bounds = np.asarray([[-0.5, 0.5], [0.0, 2.0], [0.0, 2.0]])

            def map_point_to_world_point(self, point):
                return np.asarray(point, dtype=np.float64) * 0.01

        class FakeBullet:
            def __init__(self, env):
                self.env = env

            def resetJointState(self, *_args, **_kwargs):
                return None

            def getLinkState(self, *_args, **_kwargs):
                return (self.env.last_ik_position.tolist(), None)

        class FakeEnv:
            arm_joint_indices = tuple(range(6))
            robot_id = 1
            tool_tip_id = 2
            client_id = 3
            init_ori = [0.0, 0.0, 0.0, 1.0]

            def __init__(self):
                self.last_ik_position = np.zeros(3, dtype=np.float64)
                self._p = FakeBullet(self)

            def get_ik_joints(self, position, _orientation, link=None):
                self.last_ik_position = np.asarray(position, dtype=np.float64)
                return [0.0] * 6

        result = build_cnabu_runtime_candidate_kinematic_mask(
            FakeEnv(),
            FakeHeightmapGeneration(),
            [
                np.asarray([[91, 85, 0], [110, 94, 9]], dtype=np.int16),
                np.asarray([[91, 85, 20], [110, 94, 29]], dtype=np.int16),
            ],
            crop_rows=(0, 100),
            node_ids=["learned-a", "learned-b"],
            initial_arm_config=[0.0] * 6,
        )

        self.assertEqual(result["node_ids"], ["learned-a", "learned-b"])
        self.assertEqual(np.asarray(result["kinematic_mask"]).shape, (2, 9))
        self.assertTrue(np.asarray(result["kinematic_mask"], dtype=bool).all())
        self.assertIn("cnabu_sparse_support_world_calibration", result["source"])

    def test_runtime_action_mask_combines_ik_with_known_fixed_collisions(self) -> None:
        class FakeBullet:
            GEOM_BOX = 1

            def __init__(self, env):
                self.env = env
                self.joints = np.zeros(6, dtype=np.float64)
                self.proxy_position = np.zeros(3, dtype=np.float64)
                self.removed_bodies = []

            def resetJointState(self, _body, joint, value, **_kwargs):
                self.joints[int(joint)] = float(value)

            def getLinkState(self, *_args, **_kwargs):
                return (self.joints[:3].tolist(), None)

            def getNumJoints(self, *_args, **_kwargs):
                return 6

            def getAABB(self, body, linkIndex=-1, **_kwargs):
                if int(body) != self.env.robot_id:
                    raise AssertionError("planner geometry may query only robot links")
                center = self.joints[:3] + np.asarray(
                    [0.001 * int(linkIndex), 0.0, 0.0], dtype=np.float64
                )
                return (center - 0.01).tolist(), (center + 0.01).tolist()

            def createCollisionShape(self, *_args, **_kwargs):
                return 77

            def createMultiBody(self, **_kwargs):
                self.proxy_position = np.asarray(_kwargs["basePosition"], dtype=np.float64)
                return 99

            def resetBasePositionAndOrientation(self, body, position, _orientation, **_kwargs):
                if int(body) == 99:
                    self.proxy_position = np.asarray(position, dtype=np.float64)

            def performCollisionDetection(self, **_kwargs):
                return None

            def getClosestPoints(self, *, bodyA, bodyB, **_kwargs):
                # Low candidate heights penetrate the known shelf with the
                # robot. Higher candidates remain fixed-environment-free.
                if int(bodyA) == self.env.robot_id and int(bodyB) == 50 and self.joints[2] < 0.99:
                    point = [0.0] * 9
                    point[3] = 0
                    point[8] = -0.003
                    return [tuple(point)]
                return []

            def removeBody(self, body, **_kwargs):
                self.removed_bodies.append(int(body))

            def removeCollisionShape(self, _shape, **_kwargs):
                raise AssertionError(
                    "PyBullet collision shapes used by a body must not be removed explicitly"
                )

        class FakeEnv:
            arm_joint_indices = tuple(range(6))
            robot_id = 1
            tool_tip_id = 2
            grasp_link_id = 2
            client_id = 3
            init_ori = [0.0, 0.0, 0.0, 1.0]

            def __init__(self):
                self._p = FakeBullet(self)

            def get_ik_joints(self, position, _orientation, link=None):
                position = np.asarray(position, dtype=np.float64)
                return [float(position[0]), float(position[1]), float(position[2]), 0.0, 0.0, 0.0]

        env = FakeEnv()
        result = build_runtime_candidate_action_mask(
            env,
            target_world_aabbs=[[[0.0, 0.85, 0.91], [0.10, 0.95, 1.11]]],
            node_ids=["learned-node"],
            initial_arm_config=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            fixed_body_ids={"shelf": 50},
            include_planner_swept_geometry=True,
        )

        kinematic = np.asarray(result["kinematic_mask"], dtype=bool)
        fixed_free = np.asarray(
            result["fixed_environment_collision_free_mask"], dtype=bool
        )
        eligible = np.asarray(result["action_eligible_mask"], dtype=bool)
        self.assertEqual(kinematic.shape, (1, 9))
        self.assertTrue(kinematic.all())
        self.assertTrue(np.array_equal(fixed_free, eligible))
        self.assertFalse(fixed_free[0, [0, 3, 6]].any())
        self.assertTrue(fixed_free[0, [1, 2, 4, 5, 7, 8]].all())
        self.assertEqual(env._p.removed_bodies, [99] * 9)
        self.assertFalse(result["safety"]["uses_gt_or_simulator_instance_ids"])
        self.assertFalse(result["safety"]["queries_dynamic_scene_objects"])
        self.assertTrue(result["safety"]["queries_known_fixed_environment_bodies"])
        geometry = result["targets"][0]["candidates"][1][
            "planner_swept_geometry"
        ]
        self.assertEqual(geometry["progress_bin_count"], 4)
        self.assertEqual(
            len(geometry["robot_link_aabbs_by_stage"]["approach"]),
            4,
        )
        self.assertTrue(
            all(geometry["robot_link_aabbs_by_stage"]["grasp"])
        )
        self.assertTrue(
            all(geometry["carried_target_aabbs_by_stage"]["extraction"])
        )

    def test_clean_extraction_uses_monitored_blocker_not_unrelated_settling(self) -> None:
        blocked = summarize_monitored_displacement(
            object_displacements_m={"10": 0.04, "20": 1.0},
            monitored_instance_ids=[10],
            maximum_displacement_m=0.01,
        )
        tolerated = summarize_monitored_displacement(
            object_displacements_m={"10": 0.001, "20": 1.0},
            monitored_instance_ids=[10],
            maximum_displacement_m=0.01,
        )

        self.assertFalse(blocked["monitored_objects_stable"])
        self.assertTrue(tolerated["monitored_objects_stable"])
        self.assertEqual(blocked["maximum_monitored_displacement_m"], 0.04)

    def test_extraction_progress_is_relative_to_candidate_plan(self) -> None:
        summary = summarize_extraction_progress(
            actual_displacement=[0.0, -0.0705, 0.0102],
            planned_carried_positions=[
                [0.0, 0.792, 1.025],
                [0.0, 0.707, 1.035],
            ],
            minimum_progress_fraction=0.8,
        )

        self.assertTrue(summary["target_extracted"])
        self.assertGreater(summary["progress_fraction"], 0.8)
        self.assertAlmostEqual(summary["planned_displacement_m"], np.hypot(0.085, 0.01))

    def test_penetration_threshold_can_separate_supported_and_tolerated_contacts(self) -> None:
        trials = [
            {
                "metadata": {
                    "contact_outcome": "hard_blockage_supported",
                    "minimum_blocker_signed_distance_m": -0.03,
                }
            },
            {
                "metadata": {
                    "contact_outcome": "contact_tolerated",
                    "minimum_blocker_signed_distance_m": -0.004,
                }
            },
        ]

        selection = select_hard_penetration_threshold(trials)

        self.assertEqual(selection["evaluable_trial_count"], 2)
        self.assertEqual(selection["best"]["f1"], 1.0)
        self.assertGreater(selection["best"]["threshold_m"], 0.004)
    def test_relation_threshold_uses_causally_evaluable_single_blockers(self) -> None:
        trials = [
            {
                "removed_instance_ids": [10],
                "metadata": {
                    "contact_outcome": "hard_blockage_supported",
                    "pair_scores": {"10": 0.8},
                },
            },
            {
                "removed_instance_ids": [20],
                "metadata": {
                    "contact_outcome": "contact_tolerated",
                    "pair_scores": {"20": 0.2},
                },
            },
            {
                "removed_instance_ids": [10, 20],
                "metadata": {
                    "contact_outcome": "hard_blockage_supported",
                    "pair_scores": {"10": 0.8, "20": 0.7},
                },
            },
        ]

        selection = select_relation_score_threshold(trials)

        self.assertEqual(selection["eligible_single_blocker_trial_count"], 2)
        self.assertEqual(selection["best"]["f1"], 1.0)
        self.assertGreater(selection["best"]["threshold"], 0.2)
    def test_signed_distance_summary_separates_contact_from_hard_penetration(self) -> None:
        tolerable = summarize_signed_distances([-0.0005, 0.0], hard_penetration_m=0.002)
        hard = summarize_signed_distances([-0.003], hard_penetration_m=0.002)

        self.assertTrue(tolerable["has_contact"])
        self.assertFalse(tolerable["has_hard_penetration"])
        self.assertTrue(hard["has_hard_penetration"])
        self.assertEqual(hard["minimum_signed_distance_m"], -0.003)

    def test_counterfactual_summary_separates_outcomes_and_fixed_contact(self) -> None:
        def record(intact: bool, intervention: bool, fixed: bool) -> dict:
            return {
                "trials": [
                    {
                        "intact_success": intact,
                        "intervention_success": intervention,
                        "metadata": {
                            "stratum": "single_all_geometry_positive",
                            "intervention_execution": {
                                "robot_fixed_contacts_by_stage": {
                                    "approach": [],
                                    "extraction": ["shelf"] if fixed else [],
                                }
                            },
                        },
                    }
                ],
                "interventions": [{"success_delta": float(intervention) - float(intact)}],
            }

        summary = aggregate_counterfactual_records(
            [record(False, True, False), record(True, True, False), record(False, False, True)]
        )

        self.assertEqual(
            summary["paired_outcome_counts"],
            {"failure_to_failure": 1, "failure_to_success": 1, "success_to_success": 1},
        )
        self.assertEqual(summary["intervention_fixed_environment_contact_count"], 1)

    def test_counterfactual_selector_uses_distinct_scenes_and_strata(self) -> None:
        def scene(sample_id: str, blockers: list[int], geometry_edge: int) -> dict:
            return {
                "sample_id": sample_id,
                "node_order_instance_ids": [10, 20, 30],
                "geometry_pseudo_gt_v0": {
                    "adjacency_matrix": [[0, geometry_edge, 0], [0, 0, 0], [0, geometry_edge, 0]]
                },
                "targets": [
                    {
                        "trajectories": [
                            {
                                "trajectory_id": "20/grasp",
                                "grasp_id": "grasp",
                                "target_instance_id": 20,
                                "eligible_for_scoring": True,
                                "blocked_by": blockers,
                                "blocked_by_stage": {"approach": blockers},
                            }
                        ]
                    }
                ],
            }

        scenes = [scene("a", [10], 0), scene("b", [10], 1), scene("c", [10, 30], 0)]

        selected = select_stratified_counterfactuals(scenes, limit=3)

        self.assertEqual(len({item[0]["sample_id"] for item in selected}), 3)
        self.assertEqual(
            {item[1]["stratum"] for item in selected},
            {
                "single_contains_action_only",
                "single_all_geometry_positive",
                "multiple_contains_action_only",
            },
        )

    def test_counterfactual_descriptor_distinguishes_action_only_single_blocker(self) -> None:
        scene = {
            "sample_id": "8/example",
            "node_order_instance_ids": [10, 20],
            "geometry_pseudo_gt_v0": {"adjacency_matrix": [[0, 0], [0, 0]]},
        }
        trajectory = {
            "trajectory_id": "20/front_z0.35",
            "grasp_id": "front_z0.35",
            "target_instance_id": 20,
            "eligible_for_scoring": True,
            "blocked_by": [10],
            "blocked_by_stage": {"approach": [10]},
        }

        descriptor = counterfactual_candidate_descriptor(scene, trajectory)

        self.assertEqual(descriptor["stratum"], "single_contains_action_only")
        self.assertEqual(descriptor["removed_instance_ids"], [10])

    def test_scene_selector_round_robins_across_groups(self) -> None:
        records = [
            {"sample_id": "0/a", "sample_dir": "/data/0/a/pre_action"},
            {"sample_id": "0/b", "sample_dir": "/data/0/b/pre_action"},
            {"sample_id": "1/c", "sample_dir": "/data/1/c/pre_action"},
            {"sample_id": "2/d", "sample_dir": "/data/2/d/pre_action"},
        ]

        selected = select_scene_disjoint_round_robin(records, limit=4)

        self.assertEqual(
            [str(item) for item in selected],
            [
                "/data/0/a/pre_action",
                "/data/1/c/pre_action",
                "/data/2/d/pre_action",
                "/data/0/b/pre_action",
            ],
        )

    def test_front_extraction_waypoints_move_toward_low_y_opening_and_lift(self) -> None:
        config = OracleActionFamilyConfig(opening_y=0.68)
        waypoints = front_extraction_waypoints(
            world_aabb=[[0.0, 0.85, 0.91], [0.10, 0.95, 1.11]],
            height_fraction=0.5,
            config=config,
        )

        self.assertLess(waypoints["pregrasp"][1], waypoints["grasp"][1])
        self.assertLess(waypoints["extraction"][1], waypoints["pregrasp"][1])
        self.assertGreater(waypoints["lift"][2], waypoints["grasp"][2])
        self.assertEqual(waypoints["grasp"][0], 0.05)

    def test_front_extraction_waypoints_span_lateral_grasp_grid(self) -> None:
        config = OracleActionFamilyConfig()
        left = front_extraction_waypoints(
            world_aabb=[[0.0, 0.85, 0.91], [0.10, 0.95, 1.11]],
            lateral_fraction=0.35,
            height_fraction=0.55,
            config=config,
        )
        right = front_extraction_waypoints(
            world_aabb=[[0.0, 0.85, 0.91], [0.10, 0.95, 1.11]],
            lateral_fraction=0.65,
            height_fraction=0.55,
            config=config,
        )

        self.assertAlmostEqual(left["grasp"][0], 0.035)
        self.assertAlmostEqual(right["grasp"][0], 0.065)
        self.assertGreater(config.lift_distance_m, 0.0)
        self.assertLess(config.lift_distance_m, 0.04)

    def test_interpolate_joint_configs_keeps_endpoints_without_duplicate_join(self) -> None:
        first = interpolate_joint_configs(np.zeros(2), np.ones(2), 3)
        second = interpolate_joint_configs(np.ones(2), np.full(2, 2.0), 3, include_start=False)
        joined = first + second

        self.assertEqual(len(joined), 5)
        self.assertTrue(np.array_equal(joined[0], np.zeros(2)))
        self.assertTrue(np.array_equal(joined[-1], np.full(2, 2.0)))
        self.assertEqual(sum(np.array_equal(item, np.ones(2)) for item in joined), 1)

    def test_gt_records_keep_only_object_classes_and_exact_instance_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gt_hms.npz"
            instances = np.full((2, 4, 5), -1, dtype=np.int32)
            instances[0, 0:2, 0:2] = 10
            instances[1, 2:4, 1:4] = 20
            instances[1, :, 4] = 3
            semantics = np.full((4, 5), 14, dtype=np.int32)
            semantics[0:2, 0:2] = 4
            semantics[2:4, 1:4] = 7
            np.savez(path, instance_maps=instances, semantic_2d=semantics)

            records = extract_gt_object_records(path)

        self.assertEqual([item["instance_id"] for item in records], [10, 20])
        self.assertEqual([item["semantic_class_id"] for item in records], [4, 7])
        self.assertEqual(records[0]["bbox_yx_minmax"], [0, 0, 1, 1])
        self.assertEqual(records[1]["pixel_count"], 6)

    def test_geometry_baseline_and_comparison_keep_undefined_oracle_pairs_separate(self) -> None:
        objects = [
            {"instance_id": 10, "bbox_yx_minmax": [0, 0, 2, 2], "centroid_yx": [1.0, 1.0]},
            {"instance_id": 20, "bbox_yx_minmax": [3, 1, 5, 3], "centroid_yx": [4.0, 2.0]},
            {"instance_id": 30, "bbox_yx_minmax": [6, 4, 8, 6], "centroid_yx": [7.0, 5.0]},
        ]
        geometry = build_geometry_pseudo_gt_adjacency(objects)
        self.assertEqual(geometry, [[0, 1, 0], [0, 0, 0], [0, 0, 0]])

        comparison = compare_relation_targets(
            node_order=[10, 20, 30],
            geometry_adjacency=geometry,
            action_adjacency=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            action_score_valid_mask=[
                [False, True, False],
                [True, False, False],
                [True, True, False],
            ],
        )

        self.assertEqual(comparison["comparable_pair_count"], 4)
        self.assertEqual(comparison["undefined_action_pair_count"], 2)
        self.assertEqual(comparison["geometry_only_pairs"], [[10, 20]])


if __name__ == "__main__":
    unittest.main()
