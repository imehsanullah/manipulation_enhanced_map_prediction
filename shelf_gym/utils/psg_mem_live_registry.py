"""Process-local handoff between separately loaded X3 runtime/evaluator adapters."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LiveEpisodeHandle:
    episode_id: str
    mem: Any
    bridge: Any
    scene_path: str
    latest_graph: Dict[str, Any] | None = None
    evaluation_cache: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    instance_map_cache: Dict[int, Any] = field(default_factory=dict)
    closed: bool = False


_LOCK = threading.RLock()
_EPISODES: Dict[str, LiveEpisodeHandle] = {}


def register_live_episode(handle: LiveEpisodeHandle) -> None:
    if not isinstance(handle, LiveEpisodeHandle):
        raise TypeError("handle must be LiveEpisodeHandle")
    if not isinstance(handle.episode_id, str) or not handle.episode_id:
        raise ValueError("live episode ID must be non-empty")
    with _LOCK:
        if handle.episode_id in _EPISODES:
            raise RuntimeError(
                f"live episode is already registered: {handle.episode_id}"
            )
        _EPISODES[handle.episode_id] = handle


def get_live_episode(episode_id: str) -> LiveEpisodeHandle:
    with _LOCK:
        handle = _EPISODES.get(str(episode_id))
        if handle is None or handle.closed:
            raise RuntimeError(f"live episode is unavailable: {episode_id}")
        return handle


def close_live_episode(episode_id: str) -> None:
    with _LOCK:
        handle = _EPISODES.pop(str(episode_id), None)
        if handle is not None:
            handle.closed = True
            handle.latest_graph = None
            handle.evaluation_cache.clear()
            handle.instance_map_cache.clear()


__all__ = [
    "LiveEpisodeHandle",
    "close_live_episode",
    "get_live_episode",
    "register_live_episode",
]
