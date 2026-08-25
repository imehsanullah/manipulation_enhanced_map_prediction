from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    PROJECT_ROOT / "shelf_gym" / "scripts" / ("generate_psg_mem_scene_bank.py")
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "test_psg_mem_bank_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PsgMemSceneBankGeneratorTest(unittest.TestCase):
    def _manifest(self, module, output_root: Path):
        state = module._git_state(PROJECT_ROOT)
        groups = module.build_scene_groups()
        return {
            "schema": module.SCHEMA,
            "bank_id": module.BANK_ID,
            "run_type": "x1_fresh_tuning_scene_generation",
            "host": socket.gethostname(),
            "group_count": module.GROUP_COUNT,
            "scene_groups": groups,
            "scene_groups_sha256": hashlib.sha256(
                module._canonical_bytes(groups)
            ).hexdigest(),
            "generator_config": {
                "use_ycb": True,
                "use_occupancy_for_placing": True,
                "max_obj_num": 25,
                "max_occupancy_threshold": 0.4,
                "hard_only": False,
                "planner_executed": False,
                "mapping_model_loaded": False,
                "graph_model_loaded": False,
            },
            "output_root": str(output_root.resolve()),
            "artifacts": {
                "generator": {
                    "path": str(GENERATOR_PATH.resolve()),
                    "sha256": hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest(),
                }
            },
            "source_repository": {
                "path": str(PROJECT_ROOT),
                "worktree_state_sha256": state["worktree_state_sha256"],
            },
            "artifact_policy": {
                "scene_arrangements": True,
                "physics_snapshots": True,
                "compact_records": True,
                "raw_maps": False,
                "datasets_or_hdf5": False,
                "checkpoints_or_models": False,
                "planner_or_oracle": False,
            },
        }

    def test_frozen_groups_are_balanced_unique_and_not_opened(self) -> None:
        module = _module()
        groups = module.build_scene_groups()
        self.assertEqual(len(groups), 21)
        self.assertEqual(len({row["generator_seed"] for row in groups}), 21)
        self.assertEqual(
            {row["occupancy_target"] for row in groups}, {0.35, 0.375, 0.4}
        )
        self.assertEqual(
            {
                value: sum(row["occupancy_target"] == value for row in groups)
                for value in (0.35, 0.375, 0.4)
            },
            {0.35: 7, 0.375: 7, 0.4: 7},
        )
        self.assertFalse(
            {row["generator_seed"] for row in groups}
            & set(range(2026071800, 2026071818))
        )

    def test_manifest_validation_is_hash_and_policy_strict(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(module, Path(temporary) / "bank")
            module.validate_manifest(manifest)
            mutated = json.loads(json.dumps(manifest))
            mutated["scene_groups"][0]["generator_seed"] += 1
            with self.assertRaisesRegex(ValueError, "frozen X1 scope"):
                module.validate_manifest(mutated)
            mutated = json.loads(json.dumps(manifest))
            mutated["artifact_policy"]["raw_maps"] = True
            with self.assertRaisesRegex(ValueError, "artifact policy"):
                module.validate_manifest(mutated)

    def test_new_file_helpers_refuse_overwrite(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            module._write_new_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                module._write_new_json(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
