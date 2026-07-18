import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shelf_gym.utils.cnabu_mem_experiment_control import (
    apply_initial_state_snapshot,
    array_sha256,
    build_initial_state_snapshot,
    candidate_set_fingerprint,
    capture_runtime_physics_state,
    initial_observation_hashes,
    load_initial_state_snapshot,
    physics_state_sha256,
    write_initial_state_snapshot,
)


class _FakeObjectRegistry:
    def __init__(self, classes):
        self._classes = classes

    def get_id_to_class_dict(self):
        return dict(self._classes)


class _FakeClient:
    JOINT_FIXED = 4

    def __init__(self):
        self.bodies = {
            10: {
                "position": [0.1, 0.2, 0.3],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear": [0.0, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
            },
            11: {
                "position": [0.4, 0.5, 0.6],
                "orientation": [0.0, 0.0, 0.1, 0.99498743710662],
                "linear": [0.01, 0.0, 0.0],
                "angular": [0.0, 0.02, 0.0],
            },
            99: {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear": [0.0, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
            },
        }
        self.joints = {0: [0.25, -0.1], 1: [0.0, 0.0]}
        self.detected = False

    def getBasePositionAndOrientation(self, body_id):
        body = self.bodies[body_id]
        return tuple(body["position"]), tuple(body["orientation"])

    def getBaseVelocity(self, body_id):
        body = self.bodies[body_id]
        return tuple(body["linear"]), tuple(body["angular"])

    def resetBasePositionAndOrientation(self, body_id, position, orientation):
        self.bodies[body_id]["position"] = list(position)
        self.bodies[body_id]["orientation"] = list(orientation)

    def resetBaseVelocity(self, body_id, linearVelocity, angularVelocity):
        self.bodies[body_id]["linear"] = list(linearVelocity)
        self.bodies[body_id]["angular"] = list(angularVelocity)

    def getNumJoints(self, _body_id):
        return 2

    def getJointInfo(self, _body_id, joint_index):
        values = [None] * 3
        values[2] = 0 if joint_index == 0 else self.JOINT_FIXED
        return values

    def getJointState(self, _body_id, joint_index):
        return self.joints[joint_index][0], self.joints[joint_index][1], (), 0.0

    def resetJointState(
        self, _body_id, joint_index, targetValue, targetVelocity
    ):
        self.joints[joint_index] = [targetValue, targetVelocity]

    def performCollisionDetection(self):
        self.detected = True


class _FakeEnvironment:
    def __init__(self):
        self._p = _FakeClient()
        self.current_obj_ids = [10, 11]
        self.robot_id = 99
        self.obj = _FakeObjectRegistry({10: 2, 11: 7})


class CnabuMemExperimentControlTest(unittest.TestCase):
    def test_snapshot_round_trip_restores_exact_state(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            scene = tmp_path / "scene.pkl"
            scene.write_bytes(b"frozen-scene")
            environment = _FakeEnvironment()
            snapshot = build_initial_state_snapshot(
                environment, scene_path=scene, seed=3
            )
            output = tmp_path / "state.json"
            write_initial_state_snapshot(snapshot, output)
            loaded = load_initial_state_snapshot(output)

            environment._p.bodies[10]["position"] = [9.0, 9.0, 9.0]
            environment._p.bodies[11]["linear"] = [1.0, 1.0, 1.0]
            environment._p.joints[0] = [2.0, 3.0]
            result = apply_initial_state_snapshot(
                environment, loaded, scene_path=scene
            )

            self.assertEqual(result["state_sha256"], snapshot["state_sha256"])
            self.assertTrue(environment._p.detected)
            self.assertEqual(
                physics_state_sha256(capture_runtime_physics_state(environment)),
                snapshot["state_sha256"],
            )

    def test_snapshot_rejects_scene_and_state_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            scene = tmp_path / "scene.pkl"
            scene.write_bytes(b"scene-a")
            environment = _FakeEnvironment()
            snapshot = build_initial_state_snapshot(
                environment, scene_path=scene, seed=0
            )
            output = tmp_path / "state.json"
            write_initial_state_snapshot(snapshot, output)
            scene.write_bytes(b"scene-b")
            with self.assertRaisesRegex(ValueError, "different scene"):
                apply_initial_state_snapshot(
                    environment,
                    load_initial_state_snapshot(output),
                    scene_path=scene,
                )

            payload = json.loads(output.read_text())
            payload["state"]["objects"][0]["body"]["position"][0] = 4.0
            output.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "state hash"):
                load_initial_state_snapshot(output)

    def test_initial_hashes_are_shape_dtype_and_value_sensitive(self):
        first = np.arange(12, dtype=np.int16).reshape(3, 4)
        same = first.copy()
        changed = first.copy()
        changed[0, 0] = 5
        self.assertEqual(array_sha256(first), array_sha256(same))
        self.assertNotEqual(
            array_sha256(first)["sha256"], array_sha256(changed)["sha256"]
        )
        self.assertNotEqual(
            array_sha256(first)["sha256"],
            array_sha256(first.astype(np.int32))["sha256"],
        )

        hashes_a = initial_observation_hashes(
            {"height_maps": first}, {"semantic_gt": first}
        )
        hashes_b = initial_observation_hashes(
            {"height_maps": same}, {"semantic_gt": same}
        )
        hashes_c = initial_observation_hashes(
            {"height_maps": changed}, {"semantic_gt": same}
        )
        self.assertEqual(hashes_a, hashes_b)
        self.assertNotEqual(
            hashes_a["combined_sha256"], hashes_c["combined_sha256"]
        )

    def test_physics_hash_ignores_only_sub_quantization_float_jitter(self):
        baseline = {"orientation": [0.1, 0.2, 0.3, 0.92736184955]}
        sub_roundoff = {
            "orientation": [0.1 + 2.0e-16, 0.2, 0.3, 0.92736184955]
        }
        material_change = {
            "orientation": [0.1 + 2.0e-9, 0.2, 0.3, 0.92736184955]
        }
        self.assertEqual(
            physics_state_sha256(baseline),
            physics_state_sha256(sub_roundoff),
        )
        self.assertNotEqual(
            physics_state_sha256(baseline),
            physics_state_sha256(material_change),
        )

    def test_candidate_fingerprint_tracks_count_and_content(self):
        result = {
            "paths": [np.asarray([[1.0, 2.0], [3.0, 4.0]])],
            "motion_parametrization": np.asarray([[1, 2, 3, 4, 5, 6]]),
            "path_annotations": [["free", "pushing"]],
        }
        first = candidate_set_fingerprint(result)
        second = candidate_set_fingerprint(result)
        changed = candidate_set_fingerprint(
            {
                **result,
                "paths": [np.asarray([[1.0, 2.0], [3.0, 4.1]])],
            }
        )
        self.assertEqual(first, second)
        self.assertEqual(first["path_count"], 1)
        self.assertNotEqual(first["combined_sha256"], changed["combined_sha256"])

    def test_array_hash_uses_explicit_device_copy_protocol(self):
        class DeviceArray:
            def get(self):
                return np.asarray([1, 2, 3], dtype=np.int16)

            def __array__(self):
                raise AssertionError("implicit device conversion is forbidden")

        self.assertEqual(
            array_sha256(DeviceArray()),
            array_sha256(np.asarray([1, 2, 3], dtype=np.int16)),
        )


if __name__ == "__main__":
    unittest.main()
