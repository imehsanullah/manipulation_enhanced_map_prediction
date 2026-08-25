from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from shelf_gym.utils.action_conditioned_relation_oracle import (
    evaluate_live_target_access_feasibility,
)


class _Physics:
    def __init__(self):
        self.poses = {
            10: ([0.0, 0.8, 1.0], [0.0, 0.0, 0.0, 1.0]),
            20: ([0.1, 0.9, 1.0], [0.0, 0.0, 0.0, 1.0]),
            99: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        }
        self.velocities = {
            body: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) for body in self.poses
        }
        self.joints = {0: [0.1, 0.01], 1: [0.2, 0.02], 2: [0.3, 0.03]}

    def getBasePositionAndOrientation(self, body, physicsClientId):
        return copy.deepcopy(self.poses[body])

    def resetBasePositionAndOrientation(
        self, body, position, orientation, physicsClientId
    ):
        self.poses[body] = (list(position), list(orientation))

    def getBaseVelocity(self, body, physicsClientId):
        return copy.deepcopy(self.velocities[body])

    def resetBaseVelocity(
        self, body, *, linearVelocity, angularVelocity, physicsClientId
    ):
        self.velocities[body] = (list(linearVelocity), list(angularVelocity))

    def getAABB(self, body, physicsClientId):
        x, y, z = self.poses[body][0]
        return ([x - 0.02, y - 0.02, z - 0.02], [x + 0.02, y + 0.02, z + 0.02])

    def getJointState(self, robot, joint, physicsClientId):
        return tuple(self.joints[joint])

    def resetJointState(
        self,
        robot,
        joint,
        targetValue,
        targetVelocity=0.0,
        physicsClientId=None,
    ):
        self.joints[joint] = [float(targetValue), float(targetVelocity)]

    def performCollisionDetection(self, physicsClientId):
        return None


class _Env:
    def __init__(self):
        self._p = _Physics()
        self.client_id = 1
        self.robot_id = 99
        self.current_obj_ids = [10, 20]
        self.arm_joint_indices = [0, 1]
        self.arm_and_gripper_joint_indices = [0, 1, 2]
        self.initial_parameters = [0.0]
        self.planeID = 30
        self.UR5Stand_id = 31
        self.shelf_id = 32
        self.wall_id = 33
        self.rack_ids = [34]

    def reset_robot(self, parameters):
        self._p.joints[0][0] = 9.0

    def move_gripper(self, width):
        self._p.joints[2][0] = 8.0

    def get_current_joint_config(self):
        return [self._p.joints[0][0], self._p.joints[1][0]]


class LiveAccessEvaluatorTest(unittest.TestCase):
    def test_live_evaluator_uses_hidden_target_only_and_restores_world_state(
        self,
    ) -> None:
        env = _Env()
        before_poses = copy.deepcopy(env._p.poses)
        before_velocities = copy.deepcopy(env._p.velocities)
        before_joints = copy.deepcopy(env._p.joints)
        candidate_index = 0

        def fake_candidate(*args, **kwargs):
            nonlocal candidate_index
            candidate_index += 1
            # Deliberately perturb state; the public evaluator must restore it.
            env._p.poses[20] = ([5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 1.0])
            env._p.velocities[20] = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
            clean = candidate_index == 4
            return (
                {
                    "trajectory_id": f"t/{candidate_index}",
                    "grasp_id": f"g/{candidate_index}",
                    "kinematically_feasible": True,
                    "eligible_for_scoring": True,
                    "fixed_environment_collision": False,
                    "blocked_by": [] if clean else [10],
                },
                {},
            )

        with patch(
            "shelf_gym.utils.action_conditioned_relation_oracle._build_candidate",
            side_effect=fake_candidate,
        ):
            result = evaluate_live_target_access_feasibility(env, target_instance_id=20)
        self.assertIs(result["access_feasible"], True)
        self.assertEqual(result["clean_candidate_count"], 1)
        self.assertEqual(result["candidate_count"], 9)
        self.assertIs(result["read_only"], True)
        self.assertIs(result["evaluation_only_simulator_target_id"], True)
        self.assertEqual(env._p.poses, before_poses)
        self.assertEqual(env._p.velocities, before_velocities)
        self.assertEqual(env._p.joints, before_joints)

    def test_live_evaluator_rejects_target_outside_current_scene(self) -> None:
        with self.assertRaisesRegex(ValueError, "current live scene"):
            evaluate_live_target_access_feasibility(_Env(), target_instance_id=999)


if __name__ == "__main__":
    unittest.main()
