from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scene_graph_mem.runtime.cnabu_scene_graph import encode_binary_mask_rle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    PROJECT_ROOT / "shelf_gym" / "scripts" / "psg_mem_x2_evaluator_adapter.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "test_x2_evaluator_adapter", ADAPTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Objects:
    @staticmethod
    def get_id_to_class_dict():
        return {101: 1, 202: 2}


class _Mem:
    def __init__(self):
        self.current_obj_ids = [101, 202]
        self.obj = _Objects()
        self.positions = [[0.0, 0.6, 0.1], [0.1, 0.9, 0.1]]

    def get_gt_height_map(self, *, no_tqdm):
        assert no_tqdm is True
        instances = np.full((4, 4), -1, dtype=np.int64)
        instances[:2, :2] = 101
        instances[2:, 2:] = 202
        semantics = np.full((4, 4), 14, dtype=np.int64)
        semantics[:2, :2] = 1
        semantics[2:, 2:] = 2
        return {"instance_maps": instances, "semantic_gt": semantics}

    def physics_state(self):
        return {
            "objects": [
                {
                    "object_index": index,
                    "class_id": index + 1,
                    "body": {"position": list(position)},
                }
                for index, position in enumerate(self.positions)
            ]
        }


def _graph(*, episode_id: str, step: int, persistent: bool):
    first = np.zeros((4, 4), dtype=bool)
    first[:2, :2] = True
    second = np.zeros((4, 4), dtype=bool)
    second[2:, 2:] = True
    identifiers = (1, 2) if persistent else ("local-a", "local-b")
    nodes = []
    for identifier, class_id, mask in zip(identifiers, (1, 2), (first, second)):
        node = {
            "node_id": identifier,
            "node_type": "object",
            "class_id": class_id,
            "footprint_mask": encode_binary_mask_rle(mask),
            "source_payload": {"score": 0.9},
        }
        if persistent:
            node["tracking_state"] = "born" if step == 0 else "observed"
        nodes.append(node)
    edges = []
    if persistent and step > 0:
        edges = [
            {
                "edge_type": "same_object_as",
                "source": 1,
                "target": 1,
                "source_step": step - 1,
                "target_step": step,
                "displacement_xy": [2.0, 0.0],
            },
            {
                "edge_type": "same_object_as",
                "source": 2,
                "target": 2,
                "source_step": step - 1,
                "target_step": step,
                "displacement_xy": [0.0, 0.0],
            },
        ]
    return {
        "episode_id": episode_id,
        "step": step,
        "nodes": nodes,
        "edges": edges,
        "metadata": {"persistent_memory": persistent},
    }


def _state_hash(value):
    return json.dumps(value, sort_keys=True)


def _mapping_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_mapping_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_mapping_keys(child))
        return result
    return set()


def test_x2_tracker_evaluator_preserves_identity_and_detects_push_motion() -> None:
    module = _module()
    mem = _Mem()
    state = SimpleNamespace(action_count=0)
    handle = SimpleNamespace(mem=mem, bridge=SimpleNamespace(state=state))
    with (
        patch.object(module, "get_live_episode", return_value=handle),
        patch.object(
            module,
            "capture_runtime_physics_state",
            side_effect=lambda _mem: copy.deepcopy(_mem.physics_state()),
        ),
        patch.object(module, "physics_state_sha256", side_effect=_state_hash),
    ):
        adapter = module.build_evaluator_adapter(
            {
                "episode_id": "tracker-e",
                "arm_id": "tracker",
                "evaluation": module.DEFAULT_CONFIG,
            }
        )
        first = adapter["evaluate_graph"](
            graph=_graph(episode_id="tracker-e", step=0, persistent=True),
            step=0,
            previous_action_kind="episode_start",
        )
        mem.positions[0][0] += 0.02
        state.action_count = 1
        second = adapter["evaluate_graph"](
            graph=_graph(episode_id="tracker-e", step=1, persistent=True),
            step=1,
            previous_action_kind="push",
        )

    assert first["matched_object_count"] == 2
    assert second["physics_state_exactly_restored"] is True
    moved = next(
        row
        for row in second["identity_rows"]
        if row["gt_object_id"] == "physical_object_000"
    )
    assert moved["primary_track_id"] == 1
    assert moved["track_ids"] == [1]
    assert moved["memory_track_ids"] == [1]
    assert moved["gt_moved"] is True
    assert moved["displacement_detected"] is True
    assert moved["after_push"] is True
    forbidden = {
        "body_id",
        "evaluation_object_id",
        "gt_instance_id",
        "instance_id",
        "simulator_instance_id",
    }
    assert forbidden.isdisjoint(_mapping_keys(second))
    assert {
        row["gt_object_id"] for row in second["identity_rows"]
    } == {"physical_object_000", "physical_object_001"}


def test_x2_rebuild_evaluator_namespaces_steps_and_measures_memory_duplicates() -> None:
    module = _module()
    mem = _Mem()
    state = SimpleNamespace(action_count=0)
    handle = SimpleNamespace(mem=mem, bridge=SimpleNamespace(state=state))
    with (
        patch.object(module, "get_live_episode", return_value=handle),
        patch.object(
            module,
            "capture_runtime_physics_state",
            side_effect=lambda _mem: copy.deepcopy(_mem.physics_state()),
        ),
        patch.object(module, "physics_state_sha256", side_effect=_state_hash),
    ):
        adapter = module.build_evaluator_adapter(
            {
                "episode_id": "rebuild-e",
                "arm_id": "rebuild",
                "evaluation": module.DEFAULT_CONFIG,
            }
        )
        first = adapter["evaluate_graph"](
            graph=_graph(episode_id="rebuild-e", step=0, persistent=False),
            step=0,
            previous_action_kind="episode_start",
        )
        state.action_count = 1
        second = adapter["evaluate_graph"](
            graph=_graph(episode_id="rebuild-e", step=1, persistent=False),
            step=1,
            previous_action_kind="observe",
        )

    first_row = first["identity_rows"][0]
    second_row = second["identity_rows"][0]
    assert first_row["primary_track_id"].startswith("rebuild:0:")
    assert second_row["primary_track_id"].startswith("rebuild:1:")
    assert len(second_row["track_ids"]) == 1
    assert len(second_row["memory_track_ids"]) == 2
    assert second["memory_track_count"] == 4


def test_x2_evaluator_rejects_changed_contract_and_arm_semantics() -> None:
    module = _module()
    changed = dict(module.DEFAULT_CONFIG)
    changed["world_motion_threshold_m"] = 0.02
    with patch.object(module, "get_live_episode"):
        try:
            module.build_evaluator_adapter(
                {
                    "episode_id": "e",
                    "arm_id": "tracker",
                    "evaluation": changed,
                }
            )
        except ValueError as error:
            assert "frozen contract" in str(error)
        else:
            raise AssertionError("changed X2 evaluator contract was accepted")
