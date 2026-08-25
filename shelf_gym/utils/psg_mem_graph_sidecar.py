"""Client for the separate-environment learned PSG-MEM graph sidecar."""

from __future__ import annotations

import copy
import json
import selectors
import subprocess
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np


PROTOCOL_SCHEMA = "psg_mem_graph_sidecar_v1"
_ARRAY_FIELDS = (
    "occupancy_mean",
    "semantic_mean",
    "occupancy_variance",
    "semantic_vacuity",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "?biufUS":
            raise TypeError(
                f"unsupported NumPy array dtype for sidecar JSON: {value.dtype}"
            )
        return value.tolist()
    if isinstance(value, np.generic):
        scalar = value.item()
        if scalar is None or isinstance(scalar, (bool, int, float, str)):
            return scalar
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not sidecar JSON serializable"
    )


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _process_memory(pid: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "psg_mem_process_memory_v1",
        "pid": int(pid),
        "available": False,
        "rss_bytes": None,
        "high_water_bytes": None,
    }
    try:
        values = {}
        for line in Path(f"/proc/{int(pid)}/status").read_text(
            encoding="utf-8"
        ).splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                fields = remainder.strip().split()
                if len(fields) == 2 and fields[1] == "kB":
                    values[name] = int(fields[0]) * 1024
        if {"VmRSS", "VmHWM"} <= set(values):
            result.update(
                {
                    "available": True,
                    "rss_bytes": int(values["VmRSS"]),
                    "high_water_bytes": int(values["VmHWM"]),
                }
            )
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return result


class PsgMemGraphSidecarClient:
    """Own one persistent graph-inference child and strict request stream."""

    def __init__(
        self,
        *,
        python: str | Path,
        sidecar_script: str | Path,
        splitter_checkpoint: str | Path,
        relation_checkpoint: str | Path,
        relation_config: str | Path,
        reasoning_checkpoint: str | Path | None = None,
        device: str,
        seed: int,
        graph_config: Mapping[str, Any],
        extractor_config: Mapping[str, Any],
        candidate_context_identity: Mapping[str, Any],
        startup_timeout_seconds: float = 180.0,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        if self.startup_timeout_seconds <= 0.0 or self.request_timeout_seconds <= 0.0:
            raise ValueError("sidecar timeouts must be positive")
        paths = {
            "python": Path(python).expanduser().resolve(),
            "sidecar_script": Path(sidecar_script).expanduser().resolve(),
            "splitter_checkpoint": Path(splitter_checkpoint).expanduser().resolve(),
            "relation_checkpoint": Path(relation_checkpoint).expanduser().resolve(),
            "relation_config": Path(relation_config).expanduser().resolve(),
        }
        if reasoning_checkpoint is not None:
            paths["reasoning_checkpoint"] = (
                Path(reasoning_checkpoint).expanduser().resolve()
            )
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "sidecar paths are missing: "
                + ", ".join(f"{name}={paths[name]}" for name in missing)
            )
        if device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("sidecar device must be cpu/cuda/cuda:0")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sidecar seed must be a non-negative integer")
        self.command = [
            str(paths["python"]),
            str(paths["sidecar_script"]),
            "--splitter-checkpoint",
            str(paths["splitter_checkpoint"]),
            "--relation-checkpoint",
            str(paths["relation_checkpoint"]),
            "--relation-config",
            str(paths["relation_config"]),
            "--device",
            str(device),
            "--seed",
            str(int(seed)),
            "--graph-config-json",
            _compact_json(dict(graph_config)),
            "--extractor-config-json",
            _compact_json(dict(extractor_config)),
            "--candidate-context-identity-json",
            _compact_json(dict(candidate_context_identity)),
        ]
        if "reasoning_checkpoint" in paths:
            self.command.extend(
                ["--reasoning-checkpoint", str(paths["reasoning_checkpoint"])]
            )
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.terminate()
            raise RuntimeError("graph sidecar pipes were not created")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._lock = threading.RLock()
        self._request_index = 0
        self._closed = False
        try:
            ready = self._read_message(self.startup_timeout_seconds)
            if (
                ready.get("schema") != PROTOCOL_SCHEMA
                or ready.get("message_type") != "ready"
                or ready.get("ok") is not True
            ):
                raise RuntimeError(f"graph sidecar did not become ready: {ready}")
        except BaseException:
            self._closed = True
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
            self._stdin.close()
            self._stdout.close()
            raise
        self.ready_provenance = copy.deepcopy(ready)

    def _write_message(self, payload: Mapping[str, Any]) -> None:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"graph sidecar exited with code {self._process.returncode}"
            )
        self._stdin.write(_compact_json(dict(payload)) + "\n")
        self._stdin.flush()

    def _read_message(self, timeout_seconds: float) -> Dict[str, Any]:
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._stdout, selectors.EVENT_READ)
            if not selector.select(timeout=float(timeout_seconds)):
                raise TimeoutError("timed out waiting for graph sidecar")
            line = self._stdout.readline()
        finally:
            selector.close()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"graph sidecar closed its output (exit={code})")
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError("graph sidecar response must be a mapping")
        return dict(payload)

    def _next_request_id(self) -> str:
        self._request_index += 1
        return f"request-{self._request_index:08d}"

    @staticmethod
    def _shared_arrays(arrays: Mapping[str, Any]):
        stack = []
        descriptors = {}
        try:
            for name in _ARRAY_FIELDS:
                array = np.ascontiguousarray(np.asarray(arrays[name]))
                if array.dtype.hasobject or array.nbytes < 1:
                    raise ValueError(
                        "sidecar belief arrays must be non-empty numeric arrays"
                    )
                handle = shared_memory.SharedMemory(create=True, size=array.nbytes)
                stack.append(handle)
                np.ndarray(array.shape, dtype=array.dtype, buffer=handle.buf)[...] = (
                    array
                )
                descriptors[name] = {
                    "name": handle.name,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "nbytes": int(array.nbytes),
                }
            return stack, descriptors
        except BaseException:
            for handle in reversed(stack):
                handle.close()
                handle.unlink()
            raise

    def extract(
        self,
        *,
        episode_id: str,
        step: int,
        occupancy_mean: Any,
        semantic_mean: Any,
        occupancy_variance: Any,
        semantic_vacuity: Any,
        raw_shape_hw: Sequence[int],
        crop_rows: Sequence[int],
        target_query: Mapping[str, Any],
        selected_view_indices: Sequence[int],
        component_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any],
        candidate_context_provider: Callable[..., Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not callable(candidate_context_provider):
            raise TypeError("candidate_context_provider must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("graph sidecar client is closed")
            total_started = time.perf_counter()
            request_id = self._next_request_id()
            shared_started = time.perf_counter()
            handles, descriptors = self._shared_arrays(
                {
                    "occupancy_mean": occupancy_mean,
                    "semantic_mean": semantic_mean,
                    "occupancy_variance": occupancy_variance,
                    "semantic_vacuity": semantic_vacuity,
                }
            )
            shared_seconds = time.perf_counter() - shared_started
            try:
                request_started = time.perf_counter()
                self._write_message(
                    {
                        "schema": PROTOCOL_SCHEMA,
                        "command": "extract",
                        "request_id": request_id,
                        "arrays": descriptors,
                        "extractor_inputs": {
                            "episode_id": str(episode_id),
                            "step": int(step),
                            "raw_shape_hw": [int(value) for value in raw_shape_hw],
                            "crop_rows": [int(value) for value in crop_rows],
                            "target_query": copy.deepcopy(dict(target_query)),
                            "selected_view_indices": [
                                int(value) for value in selected_view_indices
                            ],
                            "component_config": copy.deepcopy(
                                dict(component_config or {})
                            ),
                            "metadata": copy.deepcopy(dict(metadata)),
                        },
                    }
                )
                event = self._read_message(self.request_timeout_seconds)
                first_response_seconds = time.perf_counter() - request_started
                direct_response = False
                context_seconds = 0.0
                if (
                    event.get("schema") == PROTOCOL_SCHEMA
                    and event.get("message_type") == "extract_response"
                    and event.get("request_id") == request_id
                ):
                    response = event
                    direct_response = True
                else:
                    if (
                        event.get("schema") != PROTOCOL_SCHEMA
                        or event.get("message_type") != "candidate_context_request"
                        or event.get("request_id") != request_id
                    ):
                        self._raise_unexpected(event, request_id)
                    context_started = time.perf_counter()
                    context = candidate_context_provider(
                        runtime_graph=copy.deepcopy(dict(event["runtime_graph"])),
                        occupancy_mean=occupancy_mean,
                        semantic_mean=semantic_mean,
                        occupancy_epistemic=occupancy_variance,
                        semantic_vacuity=semantic_vacuity,
                        component_config=copy.deepcopy(
                            dict(component_config or {})
                        ),
                        crop_rows=list(event["crop_rows"]),
                        raw_shape_hw=list(event["raw_shape_hw"]),
                        image_id=str(event["image_id"]),
                    )
                    if not isinstance(context, Mapping):
                        raise TypeError(
                            "candidate context provider must return a mapping"
                        )
                    context_seconds = time.perf_counter() - context_started
                    self._write_message(
                        {
                            "schema": PROTOCOL_SCHEMA,
                            "message_type": "candidate_context_response",
                            "request_id": request_id,
                            "context": copy.deepcopy(dict(context)),
                        }
                    )
                    response = self._read_message(self.request_timeout_seconds)
                roundtrip_seconds = time.perf_counter() - request_started
                if (
                    response.get("schema") != PROTOCOL_SCHEMA
                    or response.get("message_type") != "extract_response"
                    or response.get("request_id") != request_id
                    or response.get("ok") is not True
                ):
                    self._raise_unexpected(response, request_id)
                graph = response.get("graph")
                if not isinstance(graph, Mapping):
                    raise TypeError("graph sidecar response requires a graph mapping")
                if direct_response:
                    graph_metadata = graph.get("metadata")
                    extractor_metadata = (
                        graph_metadata.get("extractor")
                        if isinstance(graph_metadata, Mapping)
                        else None
                    )
                    object_node_count = (
                        graph_metadata.get("num_object_nodes")
                        if isinstance(graph_metadata, Mapping)
                        else None
                    )
                    if (
                        not isinstance(graph_metadata, Mapping)
                        or isinstance(object_node_count, bool)
                        or not isinstance(object_node_count, int)
                        or not 0 <= object_node_count < 2
                        or graph_metadata.get("relation_source")
                        != "not_applicable_fewer_than_two_object_nodes"
                        or not isinstance(extractor_metadata, Mapping)
                        or extractor_metadata.get("relation_mode")
                        != "not_applicable_fewer_than_two_object_nodes"
                    ):
                        raise RuntimeError(
                            "sidecar may skip candidate context only for a "
                            "declared graph with fewer than two objects"
                        )
                result = copy.deepcopy(dict(graph))
                graph_metadata = result.setdefault("metadata", {})
                graph_metadata["sidecar_transport_timing_seconds"] = {
                    "shared_memory_publish": float(shared_seconds),
                    "request_to_first_response": float(first_response_seconds),
                    "candidate_context_parent": float(context_seconds),
                    "request_roundtrip": float(roundtrip_seconds),
                    "total_sidecar_client": float(
                        time.perf_counter() - total_started
                    ),
                }
                resources = copy.deepcopy(
                    dict(graph_metadata.get("runtime_resources") or {})
                )
                resources["sidecar_process_client_observed"] = _process_memory(
                    self._process.pid
                )
                graph_metadata["runtime_resources"] = resources
                return result
            finally:
                for handle in reversed(handles):
                    handle.close()
                    try:
                        handle.unlink()
                    except FileNotFoundError:
                        pass

    @staticmethod
    def _raise_unexpected(message: Mapping[str, Any], request_id: str) -> None:
        if message.get("message_type") == "error_response":
            raise RuntimeError(
                "graph sidecar request {} failed with {}: {}".format(
                    request_id,
                    message.get("exception_type"),
                    message.get("message"),
                )
            )
        raise RuntimeError(
            f"unexpected graph sidecar response for {request_id}: {message}"
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._process.poll() is None:
                    request_id = self._next_request_id()
                    self._write_message(
                        {
                            "schema": PROTOCOL_SCHEMA,
                            "command": "shutdown",
                            "request_id": request_id,
                        }
                    )
                    response = self._read_message(
                        min(self.request_timeout_seconds, 10.0)
                    )
                    if (
                        response.get("message_type") != "shutdown_response"
                        or response.get("request_id") != request_id
                    ):
                        raise RuntimeError("graph sidecar did not acknowledge shutdown")
                    self._process.wait(timeout=10.0)
            except Exception:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=5.0)
            finally:
                self._stdin.close()
                self._stdout.close()


__all__ = ["PROTOCOL_SCHEMA", "PsgMemGraphSidecarClient"]
