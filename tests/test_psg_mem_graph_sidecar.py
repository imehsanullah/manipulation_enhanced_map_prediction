from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shelf_gym.utils.psg_mem_graph_sidecar import PsgMemGraphSidecarClient


_FAKE_SIDECAR = r"""
import json
import sys
from multiprocessing import resource_tracker, shared_memory

SCHEMA = "psg_mem_graph_sidecar_v1"

def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)

emit({
    "schema": SCHEMA,
    "message_type": "ready",
    "ok": True,
    "extractor_config_hash": "a" * 64,
    "checkpoint_hashes": {},
})
for line in sys.stdin:
    message = json.loads(line)
    request_id = message["request_id"]
    if message["command"] == "shutdown":
        emit({
            "schema": SCHEMA,
            "message_type": "shutdown_response",
            "request_id": request_id,
            "ok": True,
        })
        break
    handles = []
    arrays = {}
    for name, descriptor in message["arrays"].items():
        handle = shared_memory.SharedMemory(name=descriptor["name"])
        handles.append(handle)
        arrays[name] = np.ndarray(
            descriptor["shape"], dtype=np.dtype(descriptor["dtype"]), buffer=handle.buf
        )
    emit({
        "schema": SCHEMA,
        "message_type": "candidate_context_request",
        "request_id": request_id,
        "runtime_graph": {"nodes": [{"id": 9, "class_id": 2}]},
        "crop_rows": message["extractor_inputs"]["crop_rows"],
        "raw_shape_hw": message["extractor_inputs"]["raw_shape_hw"],
        "image_id": "fixture:0",
    })
    context = json.loads(sys.stdin.readline())["context"]
    emit({
        "schema": SCHEMA,
        "message_type": "extract_response",
        "request_id": request_id,
        "ok": True,
        "graph": {
            "schema": "fixture_graph",
            "occupancy_sum": float(arrays["occupancy_mean"].sum()),
            "context": context,
        },
    })
    for handle in handles:
        handle.close()
        resource_tracker.unregister(handle._name, "shared_memory")
""".lstrip()

_FAKE_DIRECT_SIDECAR = r"""
import json
import sys

SCHEMA = "psg_mem_graph_sidecar_v1"

def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)

emit({
    "schema": SCHEMA,
    "message_type": "ready",
    "ok": True,
    "extractor_config_hash": "a" * 64,
    "checkpoint_hashes": {},
})
for line in sys.stdin:
    message = json.loads(line)
    request_id = message["request_id"]
    if message["command"] == "shutdown":
        emit({
            "schema": SCHEMA,
            "message_type": "shutdown_response",
            "request_id": request_id,
            "ok": True,
        })
        break
    emit({
        "schema": SCHEMA,
        "message_type": "extract_response",
        "request_id": request_id,
        "ok": True,
        "graph": {
            "schema": "frontier_only_fixture",
            "object_count": 0,
            "metadata": {
                "num_object_nodes": 0,
                "relation_source": "not_applicable_fewer_than_two_object_nodes",
                "extractor": {
                    "relation_mode": "not_applicable_fewer_than_two_object_nodes"
                },
            },
        },
    })
""".lstrip()


class GraphSidecarClientTest(unittest.TestCase):
    def test_shared_memory_and_interleaved_context_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "fake_sidecar.py"
            script.write_text("import numpy as np\n" + _FAKE_SIDECAR, encoding="utf-8")
            artifacts = []
            for name in ("splitter", "relation", "relation_config", "reasoning"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                artifacts.append(path)
            client = PsgMemGraphSidecarClient(
                python=sys.executable,
                sidecar_script=script,
                splitter_checkpoint=artifacts[0],
                relation_checkpoint=artifacts[1],
                relation_config=artifacts[2],
                reasoning_checkpoint=artifacts[3],
                device="cpu",
                seed=0,
                graph_config={},
                extractor_config={},
                candidate_context_identity={"schema": "fixture"},
                startup_timeout_seconds=5.0,
                request_timeout_seconds=5.0,
            )
            self.assertEqual(
                client.command[-2:],
                ["--reasoning-checkpoint", str(artifacts[3].resolve())],
            )
            contexts = []
            occupancy_variance = np.full((2, 3, 4), 0.12, dtype=np.float32)
            semantic_vacuity = np.full((3, 4), 0.11, dtype=np.float32)
            try:
                graph = client.extract(
                    episode_id="fixture",
                    step=0,
                    occupancy_mean=np.ones((2, 3, 4), dtype=np.float32),
                    semantic_mean=np.ones((5, 3, 4), dtype=np.float32),
                    occupancy_variance=occupancy_variance,
                    semantic_vacuity=semantic_vacuity,
                    raw_shape_hw=(140, 200),
                    crop_rows=(10, 130),
                    target_query={"class_id": 2, "coarse_region": "back"},
                    selected_view_indices=[],
                    component_config={"occupancy_threshold": 0.65},
                    metadata={"uses_gt": False},
                    candidate_context_provider=lambda **kwargs: contexts.append(kwargs)
                    or {
                        "candidate_action_mask": {
                            "uses_gt": False,
                            "eligible": np.asarray([True, False]),
                            "node_ids": np.asarray([4, 9], dtype=np.int64),
                            "score": np.float32(0.25),
                        },
                        "metadata": {
                            "uses_gt": False,
                            "source": np.str_("fixture"),
                        },
                    },
                )
            finally:
                client.close()
            self.assertEqual(graph["occupancy_sum"], 24.0)
            self.assertEqual(contexts[0]["runtime_graph"]["nodes"][0]["id"], 9)
            self.assertIs(contexts[0]["occupancy_epistemic"], occupancy_variance)
            self.assertIs(contexts[0]["semantic_vacuity"], semantic_vacuity)
            self.assertEqual(
                contexts[0]["component_config"],
                {"occupancy_threshold": 0.65},
            )
            self.assertIs(graph["context"]["metadata"]["uses_gt"], False)
            self.assertEqual(
                graph["context"]["candidate_action_mask"]["eligible"],
                [True, False],
            )
            self.assertEqual(
                graph["context"]["candidate_action_mask"]["node_ids"],
                [4, 9],
            )
            self.assertEqual(graph["context"]["candidate_action_mask"]["score"], 0.25)
            self.assertEqual(graph["context"]["metadata"]["source"], "fixture")
            self.assertIsNotNone(client._process.returncode)

    def test_direct_extract_response_skips_candidate_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "fake_direct_sidecar.py"
            script.write_text(_FAKE_DIRECT_SIDECAR, encoding="utf-8")
            artifacts = []
            for name in ("splitter", "relation", "relation_config"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                artifacts.append(path)
            client = PsgMemGraphSidecarClient(
                python=sys.executable,
                sidecar_script=script,
                splitter_checkpoint=artifacts[0],
                relation_checkpoint=artifacts[1],
                relation_config=artifacts[2],
                device="cpu",
                seed=0,
                graph_config={},
                extractor_config={},
                candidate_context_identity={"schema": "fixture"},
                startup_timeout_seconds=5.0,
                request_timeout_seconds=5.0,
            )
            context_calls = []
            try:
                graph = client.extract(
                    episode_id="fixture",
                    step=0,
                    occupancy_mean=np.ones((2, 3, 4), dtype=np.float32),
                    semantic_mean=np.ones((5, 3, 4), dtype=np.float32),
                    occupancy_variance=np.zeros((2, 3, 4), dtype=np.float32),
                    semantic_vacuity=np.zeros((3, 4), dtype=np.float32),
                    raw_shape_hw=(140, 200),
                    crop_rows=(10, 130),
                    target_query={"class_id": 2, "coarse_region": "back"},
                    selected_view_indices=[],
                    metadata={"uses_gt": False},
                    candidate_context_provider=lambda **kwargs: context_calls.append(
                        kwargs
                    ),
                )
            finally:
                client.close()
            self.assertEqual(graph["object_count"], 0)
            self.assertEqual(context_calls, [])


if __name__ == "__main__":
    unittest.main()
