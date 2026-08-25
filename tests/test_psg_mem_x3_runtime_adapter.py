from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from shelf_gym.scripts.psg_mem_x3_runtime_adapter import (
    RUNTIME_SPEC_SCHEMA,
    validate_runtime_spec,
)


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class RuntimeAdapterSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        source = self.root / "src" / "scene_graph_mem"
        source.mkdir(parents=True)
        files = {}
        for name in (
            "scene",
            "initial_state_snapshot",
            "splitter_checkpoint",
            "relation_checkpoint",
            "relation_config",
            "scene_graph_python",
            "scene_graph_sidecar",
        ):
            path = (
                self.root / "tools" / f"{name}.artifact"
                if name == "scene_graph_sidecar"
                else self.root / f"{name}.artifact"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}\n".encode("utf-8"))
            files[name] = _identity(path)
        self.spec = {
            "episode_id": "runtime-spec-test",
            "arm_id": "d",
            "planner_query": {"class_id": 2, "coarse_region": "back"},
            "integration": {
                "schema": RUNTIME_SPEC_SCHEMA,
                **files,
                "scene_graph_mem_src": str((self.root / "src").resolve()),
                "seed": 0,
                "action_budget": 5,
                "max_sampled_pushes": 12,
                "render": False,
                "relation_device": "cuda:0",
                "raw_shape_hw": [140, 200],
                "crop_rows": [10, 130],
                "graph_config": {},
                "tracker_config": {},
                "extractor_config": {"write_snapshots": False},
                "sidecar_startup_timeout_seconds": 180.0,
                "sidecar_request_timeout_seconds": 120.0,
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_hashed_default_off_spec_contract_validates(self) -> None:
        result = validate_runtime_spec(self.spec)
        self.assertEqual(result["episode_id"], "runtime-spec-test")
        self.assertEqual(result["arm_id"], "d")
        self.assertIs(result["persistent"], True)
        self.assertEqual(
            result["files"]["scene"], Path(self.spec["integration"]["scene"]["path"])
        )

        reasoning = self.root / "reasoning_checkpoint.pth"
        reasoning.write_bytes(b"x4 reasoning fixture\n")
        learned = copy.deepcopy(self.spec)
        learned["integration"]["reasoning_checkpoint"] = _identity(reasoning)
        learned_result = validate_runtime_spec(learned)
        self.assertEqual(
            learned_result["files"]["reasoning_checkpoint"], reasoning.resolve()
        )

    def test_hash_mismatch_and_snapshot_writes_are_rejected(self) -> None:
        mismatch = copy.deepcopy(self.spec)
        mismatch["integration"]["scene"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_runtime_spec(mismatch)

        writing = copy.deepcopy(self.spec)
        writing["integration"]["extractor_config"]["write_snapshots"] = True
        with self.assertRaisesRegex(ValueError, "may not write"):
            validate_runtime_spec(writing)

    def test_privileged_fields_are_rejected_recursively(self) -> None:
        privileged_query = copy.deepcopy(self.spec)
        privileged_query["planner_query"]["nested"] = {"evaluation_object_id": 901}
        with self.assertRaisesRegex(ValueError, "privileged runtime field"):
            validate_runtime_spec(privileged_query)

        privileged_config = copy.deepcopy(self.spec)
        privileged_config["integration"]["graph_config"] = {"target_mask": [[1]]}
        with self.assertRaisesRegex(ValueError, "privileged runtime field"):
            validate_runtime_spec(privileged_config)

    def test_source_root_must_belong_to_the_pinned_sidecar_repository(self) -> None:
        divergent = copy.deepcopy(self.spec)
        other_source = self.root / "other" / "src" / "scene_graph_mem"
        other_source.mkdir(parents=True)
        divergent["integration"]["scene_graph_mem_src"] = str(other_source.parent)
        with self.assertRaisesRegex(ValueError, "hash-pinned sidecar repository"):
            validate_runtime_spec(divergent)


if __name__ == "__main__":
    unittest.main()
