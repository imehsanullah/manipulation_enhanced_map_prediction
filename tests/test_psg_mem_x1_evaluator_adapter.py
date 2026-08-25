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
    PROJECT_ROOT / "shelf_gym" / "scripts" / "psg_mem_x1_evaluator_adapter.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "test_x1_evaluator_adapter", ADAPTER_PATH
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
        self.physics_token = "unchanged"

    def get_gt_height_map(self, *, no_tqdm):
        assert no_tqdm is True
        instances = np.full((4, 4), -1, dtype=np.int64)
        instances[:2, :2] = 101
        instances[2:, 2:] = 202
        semantics = np.full((4, 4), 14, dtype=np.int64)
        semantics[:2, :2] = 1
        semantics[2:, 2:] = 2
        return {"instance_maps": instances, "semantic_gt": semantics}


def _graph():
    first = np.zeros((4, 4), dtype=bool)
    first[:2, :2] = True
    second = np.zeros((4, 4), dtype=bool)
    second[2:, 2:] = True
    return {
        "episode_id": "x1-test",
        "step": 0,
        "nodes": [
            {
                "node_id": "pred-a",
                "node_type": "object",
                "class_id": 1,
                "footprint_mask": encode_binary_mask_rle(first),
                "source_payload": {"score": 0.9},
            },
            {
                "node_id": "pred-b",
                "node_type": "object",
                "class_id": 2,
                "footprint_mask": encode_binary_mask_rle(second),
                "source_payload": {"score": 0.8},
            },
        ],
        "edges": [
            {
                "edge_type": "blocks_access_to",
                "source": "pred-a",
                "target": "pred-b",
                "score": 0.95,
            },
            {
                "edge_type": "blocks_access_to",
                "source": "pred-b",
                "target": "pred-a",
                "score": 0.10,
            },
        ],
    }


def test_x1_evaluator_matches_nodes_scores_pairs_and_hides_simulator_ids() -> None:
    module = _module()
    mem = _Mem()
    handle = SimpleNamespace(
        mem=mem,
        bridge=SimpleNamespace(state=SimpleNamespace(action_count=0)),
    )

    def oracle(_mem, *, target_instance_id, include_evaluation_private_blockers):
        assert _mem is mem
        assert include_evaluation_private_blockers is True
        counts = {"101": 8} if target_instance_id == 202 else {}
        return {
            "environment_state_restored": True,
            "runtime_seconds": 0.01,
            "_evaluation_private": {
                "eligible_candidate_count": 9,
                "blocker_candidate_counts": counts,
            },
        }

    with (
        patch.object(module, "get_live_episode", return_value=handle),
        patch.object(
            module,
            "capture_runtime_physics_state",
            side_effect=lambda value: {"token": value.physics_token},
        ),
        patch.object(
            module,
            "physics_state_sha256",
            side_effect=lambda value: json.dumps(value, sort_keys=True),
        ),
        patch.object(
            module,
            "evaluate_live_target_access_feasibility",
            side_effect=oracle,
        ) as mocked_oracle,
    ):
        adapter = module.build_evaluator_adapter(
            {"episode_id": "x1-test", "evaluation": module.DEFAULT_CONFIG}
        )
        result = adapter["evaluate_graph"](graph=copy.deepcopy(_graph()), step=0)

    assert result["node_metrics"]["thresholds"]["0.50"]["f1"] == 1.0
    assert result["matched_object_count"] == 2
    assert result["matched_pair_count"] == 2
    by_pair = {
        (row["source_node_id"], row["target_node_id"]): row
        for row in result["edge_rows"]
    }
    assert by_pair[("pred-a", "pred-b")]["label"] is True
    assert by_pair[("pred-a", "pred-b")]["prediction"] is True
    assert by_pair[("pred-b", "pred-a")]["label"] is False
    assert mocked_oracle.call_count == 2
    assert result["physics_state_exactly_restored"] is True
    encoded = json.dumps(result, sort_keys=True)
    assert "101" not in encoded
    assert "202" not in encoded
    assert "body_id" not in encoded
    assert "instance_id" not in encoded


def test_x1_evaluator_rejects_changed_contract_and_missing_directed_pair() -> None:
    module = _module()
    changed = dict(module.DEFAULT_CONFIG)
    changed["node_iou_threshold"] = 0.25
    with patch.object(module, "get_live_episode"):
        try:
            module.build_evaluator_adapter(
                {"episode_id": "x1-test", "evaluation": changed}
            )
        except ValueError as error:
            assert "frozen contract" in str(error)
        else:
            raise AssertionError("changed X1 contract was accepted")

    graph = _graph()
    graph["edges"].pop()
    mem = _Mem()
    handle = SimpleNamespace(
        mem=mem,
        bridge=SimpleNamespace(state=SimpleNamespace(action_count=0)),
    )
    with (
        patch.object(module, "get_live_episode", return_value=handle),
        patch.object(
            module,
            "capture_runtime_physics_state",
            return_value={"token": "unchanged"},
        ),
        patch.object(module, "physics_state_sha256", return_value="same"),
        patch.object(
            module,
            "evaluate_live_target_access_feasibility",
            return_value={
                "environment_state_restored": True,
                "runtime_seconds": 0.01,
                "_evaluation_private": {
                    "eligible_candidate_count": 9,
                    "blocker_candidate_counts": {},
                },
            },
        ),
    ):
        adapter = module.build_evaluator_adapter(
            {"episode_id": "x1-test", "evaluation": module.DEFAULT_CONFIG}
        )
        try:
            adapter["evaluate_graph"](graph=graph, step=0)
        except ValueError as error:
            assert "every matched directed pair" in str(error)
        else:
            raise AssertionError("missing X1 directed pair was accepted")


def test_x1_evaluator_explicitly_transfers_device_arrays_to_host() -> None:
    module = _module()

    class DeviceArray:
        def __array__(self):
            raise TypeError("implicit conversion forbidden")

        def get(self):
            return np.asarray([[1, 2], [3, 4]], dtype=np.int64)

    assert module._host_array(DeviceArray()).tolist() == [[1, 2], [3, 4]]
