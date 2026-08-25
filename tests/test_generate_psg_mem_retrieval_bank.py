from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    PROJECT_ROOT / "shelf_gym" / "scripts" / "generate_psg_mem_retrieval_bank.py"
)
CATALOG_BUILDER_PATH = (
    PROJECT_ROOT / "shelf_gym" / "scripts" / "prepare_psg_mem_x3_episode.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "test_psg_mem_retrieval_bank_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PsgMemRetrievalBankGeneratorTest(unittest.TestCase):
    def _manifest(self, module, output_root: Path) -> dict:
        groups = [
            {
                "scene_group_id": f"PSG_M7_CAND_{index:03d}",
                "scene_role": "m7_candidate",
                "generator_seed": 2026072000 + index,
                "occupancy_stratum": name,
                "occupancy_target": occupancy,
                "alignment": 0,
                "replicate": index // 3,
            }
            for index, (name, occupancy) in enumerate(
                [
                    ("low_official_range", 0.35),
                    ("medium_official_range", 0.375),
                    ("high_official_range", 0.4),
                ]
                * 10
            )
        ]
        state = module._git_state(PROJECT_ROOT)
        return {
            "schema": module.MANIFEST_SCHEMA,
            "bank_id": "psg_mem_m7_candidate_fixture_v1",
            "run_type": "m7_retrieval_candidate_bank_generation",
            "host": socket.gethostname(),
            "group_count": len(groups),
            "scene_groups": groups,
            "scene_groups_sha256": hashlib.sha256(
                module._canonical_bytes(groups)
            ).hexdigest(),
            "generator_config": dict(module.GENERATOR_CONFIG),
            "output_root": str(output_root.resolve()),
            "artifacts": {
                "generator": module._identity(GENERATOR_PATH),
                "target_catalog_builder": module._identity(CATALOG_BUILDER_PATH),
            },
            "source_repository": {
                "path": str(PROJECT_ROOT),
                "worktree_state_sha256": state["worktree_state_sha256"],
            },
            "artifact_policy": dict(module.ARTIFACT_POLICY),
            "authorization": {
                "authorized_by": "Ehsan",
                "authorization": "all work needed through completion of the given goal",
                "machine": socket.gethostname(),
                "devices": ["cuda:0", "egl:gpu0"],
                "script": str(GENERATOR_PATH),
                "dataset_path": None,
                "output_path": str(output_root.resolve()),
                "run_type": "comparison",
                "model_outputs_allowed": False,
            },
        }

    def test_manifest_is_general_but_fail_closed(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(module, Path(temporary) / "bank")
            module.validate_manifest(manifest)
            mutated = json.loads(json.dumps(manifest))
            mutated["scene_groups"][0]["generator_seed"] = 2026071900
            mutated["scene_groups_sha256"] = hashlib.sha256(
                module._canonical_bytes(mutated["scene_groups"])
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "opened bank"):
                module.validate_manifest(mutated)
            mutated = json.loads(json.dumps(manifest))
            mutated["artifact_policy"]["raw_maps"] = True
            with self.assertRaisesRegex(ValueError, "artifact policy"):
                module.validate_manifest(mutated)

    def test_catalog_payload_uses_raw_thirds_and_is_evaluator_only(self) -> None:
        module = _module()
        instances = np.full((1, 140, 200), -1, dtype=np.int64)
        semantics = np.full((140, 200), 14, dtype=np.int64)
        instances[0, 5:10, 10:15] = 11
        semantics[5:10, 10:15] = 3
        instances[0, 110:120, 50:60] = 22
        semantics[110:120, 50:60] = 7
        payload = module.build_target_catalog_payload(
            scene_group_id="fixture",
            instance_maps=instances,
            semantic_2d=semantics,
        )
        self.assertEqual(payload["schema"], module.TARGET_CATALOG_SCHEMA)
        self.assertFalse(payload["runtime_visible"])
        self.assertFalse(payload["planner_may_read"])
        self.assertTrue(payload["contains_simulator_instance_ids"])
        self.assertEqual(
            [row["coarse_region"] for row in payload["targets"]],
            ["front", "back"],
        )
        self.assertNotIn("instance_maps", payload)
        self.assertNotIn("semantic_2d", payload)

    def test_new_json_refuses_overwrite(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            module._write_new_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                module._write_new_json(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
