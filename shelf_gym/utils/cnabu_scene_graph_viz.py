"""OpenCV visualization helpers for runtime CNABU scene graphs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from shelf_gym.utils.cnabu_scene_graph import (
    DEFAULT_YCB_CLASS_NAMES,
    decode_binary_mask_rle,
)


DEFAULT_CLASS_PALETTE_BGR: Tuple[Tuple[int, int, int], ...] = (
    (52, 128, 235),
    (65, 186, 85),
    (220, 120, 72),
    (190, 82, 210),
    (72, 184, 205),
    (80, 92, 220),
    (205, 165, 65),
    (116, 150, 76),
    (205, 96, 125),
    (98, 120, 180),
    (82, 178, 158),
    (156, 102, 210),
    (210, 150, 110),
    (120, 176, 232),
    (98, 104, 112),
)


def build_cnabu_map_context(
    *,
    occupancy_distribution: Any = None,
    occupancy_mean: Any = None,
    occupancy_alpha: Any = None,
    occupancy_beta: Any = None,
    semantic_concentration: Any = None,
    semantic_mean: Any = None,
    raw_shape_hw: Optional[Sequence[int]] = None,
    crop_rows: Optional[Sequence[int]] = None,
    class_palette_bgr: Sequence[Sequence[int]] = DEFAULT_CLASS_PALETTE_BGR,
) -> Dict[str, Any]:
    """Build raw-map-sized semantic/occupancy arrays for graph rendering.

    The returned dictionary is intentionally image-oriented and does not alter
    the graph payload. It supports the same runtime interleaved occupancy
    distribution shape used by the CNABU pipeline.
    """

    occupancy_projection = _derive_occupancy_projection(
        occupancy_distribution=occupancy_distribution,
        occupancy_mean=occupancy_mean,
        occupancy_alpha=occupancy_alpha,
        occupancy_beta=occupancy_beta,
    )
    semantic_labels, semantic_confidence = _derive_semantic_map(
        semantic_concentration=semantic_concentration,
        semantic_mean=semantic_mean,
    )

    if occupancy_projection is None and semantic_labels is None:
        raise ValueError("at least one occupancy or semantic CNABU array is required")

    crop_height, crop_width = _first_spatial_shape(occupancy_projection, semantic_labels)
    raw_height, raw_width = _shape_hw_from_value(raw_shape_hw, fallback_hw=(crop_height, crop_width))
    crop_start, crop_stop = _crop_rows_from_value(
        crop_rows,
        crop_height=crop_height,
        raw_height=raw_height,
    )
    if raw_width != crop_width:
        raise ValueError(f"raw width {raw_width} must match CNABU crop width {crop_width}")

    raw_occupancy = _pad_crop_2d(occupancy_projection, raw_shape_hw=(raw_height, raw_width), crop_rows=(crop_start, crop_stop))
    raw_semantic = _pad_crop_2d(semantic_labels, raw_shape_hw=(raw_height, raw_width), crop_rows=(crop_start, crop_stop))
    raw_confidence = _pad_crop_2d(
        semantic_confidence,
        raw_shape_hw=(raw_height, raw_width),
        crop_rows=(crop_start, crop_stop),
    )
    background = render_cnabu_context_background(
        occupancy_projection=raw_occupancy,
        semantic_labels=raw_semantic,
        semantic_confidence=raw_confidence,
        class_palette_bgr=class_palette_bgr,
    )

    return {
        "background_bgr": background,
        "occupancy_projection": raw_occupancy,
        "semantic_labels": raw_semantic,
        "semantic_confidence": raw_confidence,
        "raw_shape_hw": [int(raw_height), int(raw_width)],
        "crop_rows": [int(crop_start), int(crop_stop)],
        "class_palette_bgr": [list(map(int, color)) for color in class_palette_bgr],
    }


def render_cnabu_context_background(
    *,
    occupancy_projection: Optional[Any] = None,
    semantic_labels: Optional[Any] = None,
    semantic_confidence: Optional[Any] = None,
    class_palette_bgr: Sequence[Sequence[int]] = DEFAULT_CLASS_PALETTE_BGR,
) -> np.ndarray:
    """Render a semantic argmax plus occupancy-projection background."""

    if occupancy_projection is None and semantic_labels is None:
        raise ValueError("occupancy_projection or semantic_labels is required")

    if semantic_labels is not None:
        labels = np.asarray(semantic_labels)
        if labels.ndim != 2:
            raise ValueError(f"semantic_labels must be 2D, got {labels.shape}")
        height, width = labels.shape
        palette = np.asarray(class_palette_bgr, dtype=np.uint8)
        clipped = np.clip(labels.astype(np.int32), 0, len(palette) - 1)
        semantic_rgb = palette[clipped].astype(np.float32)
    else:
        occupancy_shape = np.asarray(occupancy_projection).shape
        if len(occupancy_shape) != 2:
            raise ValueError(f"occupancy_projection must be 2D, got {occupancy_shape}")
        height, width = occupancy_shape
        semantic_rgb = np.full((height, width, 3), 118, dtype=np.float32)

    if occupancy_projection is None:
        occupancy = np.ones((height, width), dtype=np.float32) * 0.65
    else:
        occupancy = _normalise01(np.asarray(occupancy_projection, dtype=np.float32))
        if occupancy.shape != (height, width):
            raise ValueError(
                f"occupancy_projection shape {occupancy.shape} must match semantic shape {(height, width)}"
            )

    if semantic_confidence is None:
        confidence = np.ones((height, width), dtype=np.float32)
    else:
        confidence = np.clip(np.asarray(semantic_confidence, dtype=np.float32), 0.0, 1.0)
        if confidence.shape != (height, width):
            raise ValueError(
                f"semantic_confidence shape {confidence.shape} must match semantic shape {(height, width)}"
            )

    neutral = np.full_like(semantic_rgb, 228.0)
    semantic_weight = (0.28 + 0.52 * confidence)[..., None]
    occupied_weight = (0.18 + 0.82 * occupancy)[..., None]
    blended = neutral * (1.0 - semantic_weight) + semantic_rgb * semantic_weight
    blended = blended * (0.55 + 0.45 * occupied_weight)

    grid = _subtle_grid((height, width))
    blended = np.clip(blended * 0.94 + grid[..., None] * 0.06, 0, 255)
    return blended.astype(np.uint8)


def render_cnabu_scene_graph_view(
    graph: Mapping[str, Any],
    *,
    context: Optional[Mapping[str, Any]] = None,
    update_index: Optional[int] = None,
    width: int = 960,
    height: int = 760,
    max_edges: int = 32,
    max_labels: int = 18,
    show_context_background: bool = True,
    plain_background_bgr: Sequence[int] = (120, 154, 188),
    class_names: Sequence[str] = DEFAULT_YCB_CLASS_NAMES,
    class_palette_bgr: Sequence[Sequence[int]] = DEFAULT_CLASS_PALETTE_BGR,
    rotate_map_180: bool = False,
) -> np.ndarray:
    """Render a readable top-down CNABU scene-graph view as a BGR image."""

    if width < 420 or height < 360:
        raise ValueError("render size must be at least 420x360")

    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    metadata = graph.get("metadata", {})
    raw_height, raw_width = _graph_shape_hw(graph, context)
    palette = np.asarray(class_palette_bgr, dtype=np.uint8)

    canvas = np.full((int(height), int(width), 3), 246, dtype=np.uint8)
    header_h = 62
    bottom_h = 44
    margin = 24
    map_left, map_top, map_w, map_h, scale = _fit_rect(
        raw_width,
        raw_height,
        width - 2 * margin,
        height - header_h - bottom_h - margin,
        margin,
        header_h,
    )

    if bool(show_context_background):
        background = _context_background(context, raw_shape_hw=(raw_height, raw_width))
        map_layer = cv2.resize(background, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
    else:
        background_color = _plain_background_color(plain_background_bgr)
        map_layer = np.empty((map_h, map_w, 3), dtype=np.uint8)
        map_layer[:, :] = background_color

    for node in nodes:
        _overlay_node_mask(
            map_layer,
            node,
            raw_shape_hw=(raw_height, raw_width),
            palette=palette,
            alpha=0.34,
        )
    if bool(rotate_map_180):
        map_layer = cv2.rotate(map_layer, cv2.ROTATE_180)

    canvas[map_top:map_top + map_h, map_left:map_left + map_w] = map_layer
    _draw_edges(
        canvas,
        nodes=nodes,
        edges=edges,
        raw_shape_hw=(raw_height, raw_width),
        map_left=map_left,
        map_top=map_top,
        scale=scale,
        max_edges=max_edges,
        rotate_map_180=bool(rotate_map_180),
    )
    _draw_nodes(
        canvas,
        nodes=nodes,
        raw_shape_hw=(raw_height, raw_width),
        map_left=map_left,
        map_top=map_top,
        map_w=map_w,
        map_h=map_h,
        scale=scale,
        palette=palette,
        class_names=class_names,
        max_labels=max_labels,
        rotate_map_180=bool(rotate_map_180),
    )
    _draw_access_axis(
        canvas,
        map_left=map_left,
        map_top=map_top,
        map_h=map_h,
        rotate_map_180=bool(rotate_map_180),
    )
    _draw_header(
        canvas,
        graph,
    )
    _draw_footer(
        canvas,
        map_left=map_left,
        map_top=map_top,
        map_w=map_w,
        map_h=map_h,
        graph=graph,
        metadata=metadata,
        update_index=update_index,
        visible_edges=min(max_edges, len(edges)),
    )
    return canvas


def render_cnabu_belief_map_view(
    *,
    context: Mapping[str, Any],
    update_index: Optional[int] = None,
    width: int = 640,
    height: int = 520,
    title: str = "Live CNABU/MEM belief map",
    rotate_map_180: bool = False,
) -> np.ndarray:
    """Render the CNABU belief map without graph node/edge overlays."""

    if width < 420 or height < 360:
        raise ValueError("render size must be at least 420x360")

    raw_height, raw_width = _context_shape_hw(context)
    canvas = np.full((int(height), int(width), 3), 246, dtype=np.uint8)
    header_h = 62
    bottom_h = 44
    margin = 24
    map_left, map_top, map_w, map_h, _ = _fit_rect(
        raw_width,
        raw_height,
        width - 2 * margin,
        height - header_h - bottom_h - margin,
        margin,
        header_h,
    )

    background = _context_background(context, raw_shape_hw=(raw_height, raw_width))
    map_layer = cv2.resize(background, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
    if bool(rotate_map_180):
        map_layer = cv2.rotate(map_layer, cv2.ROTATE_180)
    canvas[map_top:map_top + map_h, map_left:map_left + map_w] = map_layer
    _draw_access_axis(
        canvas,
        map_left=map_left,
        map_top=map_top,
        map_h=map_h,
        rotate_map_180=bool(rotate_map_180),
    )
    _draw_belief_header(canvas, title=title)
    _draw_belief_footer(
        canvas,
        map_left=map_left,
        map_top=map_top,
        map_w=map_w,
        map_h=map_h,
        update_index=update_index,
    )
    return canvas


def compose_runtime_demo_panel(
    scene_rgb: Any,
    graph_bgr: Any,
    *,
    title: str = "PyBullet shelf view + live CNABU scene graph",
    width: int = 1600,
    height: int = 760,
) -> np.ndarray:
    """Compose a camera RGB view beside the graph BGR view."""

    scene = _to_uint8_image(_to_numpy(scene_rgb, "scene_rgb"))
    if scene.ndim != 3 or scene.shape[2] < 3:
        raise ValueError(f"scene_rgb must be HxWx3, got {scene.shape}")
    scene_bgr = cv2.cvtColor(scene[:, :, :3], cv2.COLOR_RGB2BGR)
    graph = _to_uint8_image(_to_numpy(graph_bgr, "graph_bgr"))
    if graph.ndim != 3 or graph.shape[2] != 3:
        raise ValueError(f"graph_bgr must be HxWx3, got {graph.shape}")

    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    header_h = 50
    gap = 18
    margin = 18
    left_w = int(width * 0.36)
    right_w = width - left_w - gap - 2 * margin
    panel_h = height - header_h - 2 * margin

    cv2.putText(canvas, title, (margin, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (42, 48, 54), 2, cv2.LINE_AA)
    _paste_fit(canvas, scene_bgr, (margin, header_h), (left_w, panel_h), label="PyBullet shelf")
    _paste_fit(
        canvas,
        graph,
        (margin + left_w + gap, header_h),
        (right_w, panel_h),
        label="CNABU/MEM graph",
    )
    return canvas


def _derive_occupancy_projection(
    *,
    occupancy_distribution: Any,
    occupancy_mean: Any,
    occupancy_alpha: Any,
    occupancy_beta: Any,
) -> Optional[np.ndarray]:
    if occupancy_distribution is not None and (occupancy_alpha is None or occupancy_beta is None):
        distribution = _squeeze_to_ndim(_to_numpy(occupancy_distribution, "occupancy_distribution"), 3)
        if distribution.shape[0] % 2 != 0:
            raise ValueError("occupancy_distribution first axis must interleave beta/alpha channels")
        occupancy_beta = distribution[0::2]
        occupancy_alpha = distribution[1::2]

    mean = None
    if occupancy_mean is not None:
        mean = _squeeze_to_ndim(_to_numpy(occupancy_mean, "occupancy_mean"), 3)
    elif occupancy_alpha is not None and occupancy_beta is not None:
        alpha = _squeeze_to_ndim(_to_numpy(occupancy_alpha, "occupancy_alpha"), 3)
        beta = _squeeze_to_ndim(_to_numpy(occupancy_beta, "occupancy_beta"), 3)
        mean = alpha / np.maximum(alpha + beta, 1e-8)

    if mean is None:
        return None
    mean = np.asarray(mean, dtype=np.float32)
    if mean.ndim == 2:
        return np.clip(mean, 0.0, 1.0)
    if mean.ndim != 3:
        raise ValueError(f"occupancy mean must be 2D or 3D after squeeze, got {mean.shape}")
    return np.clip(mean.max(axis=0), 0.0, 1.0)


def _derive_semantic_map(
    *,
    semantic_concentration: Any,
    semantic_mean: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mean = None
    if semantic_mean is not None:
        mean = _squeeze_to_ndim(_to_numpy(semantic_mean, "semantic_mean"), 3)
    elif semantic_concentration is not None:
        concentration = _squeeze_to_ndim(_to_numpy(semantic_concentration, "semantic_concentration"), 3)
        mean = concentration / np.maximum(concentration.sum(axis=0, keepdims=True), 1e-8)

    if mean is None:
        return None, None
    if mean.ndim != 3:
        raise ValueError(f"semantic mean must be CxHxW after squeeze, got {mean.shape}")
    labels = mean.argmax(axis=0).astype(np.int32)
    confidence = np.clip(mean.max(axis=0), 0.0, 1.0).astype(np.float32)
    return labels, confidence


def _context_background(context: Optional[Mapping[str, Any]], *, raw_shape_hw: Tuple[int, int]) -> np.ndarray:
    if context is not None and context.get("background_bgr") is not None:
        background = _to_uint8_image(_to_numpy(context["background_bgr"], "background_bgr"))
        if background.shape[:2] != raw_shape_hw:
            background = cv2.resize(background, (raw_shape_hw[1], raw_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        return background[:, :, :3]
    return render_cnabu_context_background(
        occupancy_projection=np.zeros(raw_shape_hw, dtype=np.float32),
        semantic_labels=np.full(raw_shape_hw, len(DEFAULT_CLASS_PALETTE_BGR) - 1, dtype=np.int32),
    )


def _context_shape_hw(context: Mapping[str, Any]) -> Tuple[int, int]:
    if context.get("raw_shape_hw") is not None:
        return _shape_hw_from_value(context["raw_shape_hw"], fallback_hw=(1, 1))
    if context.get("background_bgr") is not None:
        background = _to_uint8_image(_to_numpy(context["background_bgr"], "background_bgr"))
        return int(background.shape[0]), int(background.shape[1])
    raise ValueError("belief-map context must include raw_shape_hw or background_bgr")


def _overlay_node_mask(
    image: np.ndarray,
    node: Mapping[str, Any],
    *,
    raw_shape_hw: Tuple[int, int],
    palette: np.ndarray,
    alpha: float,
) -> None:
    encoded_mask = node.get("mask")
    if not encoded_mask:
        return
    try:
        mask = decode_binary_mask_rle(encoded_mask)
    except (KeyError, TypeError, ValueError):
        return
    if mask.shape != raw_shape_hw:
        mask = cv2.resize(mask.astype(np.uint8), (raw_shape_hw[1], raw_shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    mask_large = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if not bool(mask_large.any()):
        return
    color = _class_color(node, palette).astype(np.float32)
    image[mask_large] = np.clip(image[mask_large].astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255)


def _draw_edges(
    canvas: np.ndarray,
    *,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    raw_shape_hw: Tuple[int, int],
    map_left: int,
    map_top: int,
    scale: float,
    max_edges: int,
    rotate_map_180: bool,
) -> None:
    node_by_id = {int(node["id"]): node for node in nodes if "id" in node}
    ranked_edges = sorted(
        edges,
        key=lambda edge: (
            float(edge.get("score", 0.0)),
            float(edge.get("lateral_overlap_pixels", 0.0)),
            -float(edge.get("access_coordinate_gap", 0.0)),
        ),
        reverse=True,
    )[: max(0, int(max_edges))]
    overlay = canvas.copy()
    for edge in ranked_edges:
        source = node_by_id.get(int(edge.get("source", -1)))
        target = node_by_id.get(int(edge.get("target", -1)))
        if source is None or target is None:
            continue
        start = _node_point(
            source,
            raw_shape_hw=raw_shape_hw,
            map_left=map_left,
            map_top=map_top,
            scale=scale,
            rotate_map_180=rotate_map_180,
        )
        end = _node_point(
            target,
            raw_shape_hw=raw_shape_hw,
            map_left=map_left,
            map_top=map_top,
            scale=scale,
            rotate_map_180=rotate_map_180,
        )
        start, end = _shorten_segment(start, end, offset=11)
        score = float(edge.get("score", 0.0))
        thickness = 3 + int(score >= 0.45) + int(score >= 0.70)
        cv2.arrowedLine(
            overlay,
            start,
            end,
            (44, 178, 94),
            thickness,
            cv2.LINE_AA,
            tipLength=0.22,
        )
    cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0.0, dst=canvas)


def _draw_nodes(
    canvas: np.ndarray,
    *,
    nodes: Sequence[Mapping[str, Any]],
    raw_shape_hw: Tuple[int, int],
    map_left: int,
    map_top: int,
    map_w: int,
    map_h: int,
    scale: float,
    palette: np.ndarray,
    class_names: Sequence[str],
    max_labels: int,
    rotate_map_180: bool,
) -> None:
    for node in nodes:
        color = tuple(int(v) for v in _class_color(node, palette).tolist())
        x1, y1, x2, y2 = _node_bbox(
            node,
            raw_shape_hw=raw_shape_hw,
            rotate_map_180=rotate_map_180,
        )
        p1 = (int(round(map_left + x1 * scale)), int(round(map_top + y1 * scale)))
        p2 = (int(round(map_left + x2 * scale)), int(round(map_top + y2 * scale)))
        cv2.rectangle(canvas, p1, p2, color, 2, cv2.LINE_AA)
        center = _node_point(
            node,
            raw_shape_hw=raw_shape_hw,
            map_left=map_left,
            map_top=map_top,
            scale=scale,
            rotate_map_180=rotate_map_180,
        )
        cv2.circle(canvas, center, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 6, color, -1, cv2.LINE_AA)
        node_id = str(int(node.get("id", 0)))
        text_size = cv2.getTextSize(node_id, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]
        cv2.putText(
            canvas,
            node_id,
            (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    label_nodes = sorted(
        nodes,
        key=lambda node: (int(node.get("area_pixels", 0)), float(node.get("score", 0.0))),
        reverse=True,
    )[: max(0, int(max_labels))]
    placed: List[Tuple[int, int, int, int]] = []
    bounds = (map_left, map_top, map_left + map_w, map_top + map_h)
    for node in label_nodes:
        _draw_node_label(
            canvas,
            node,
            placed=placed,
            bounds=bounds,
            raw_shape_hw=raw_shape_hw,
            map_left=map_left,
            map_top=map_top,
            scale=scale,
            palette=palette,
            class_names=class_names,
            rotate_map_180=rotate_map_180,
        )


def _draw_node_label(
    canvas: np.ndarray,
    node: Mapping[str, Any],
    *,
    placed: List[Tuple[int, int, int, int]],
    bounds: Tuple[int, int, int, int],
    raw_shape_hw: Tuple[int, int],
    map_left: int,
    map_top: int,
    scale: float,
    palette: np.ndarray,
    class_names: Sequence[str],
    rotate_map_180: bool,
) -> None:
    center = _node_point(
        node,
        raw_shape_hw=raw_shape_hw,
        map_left=map_left,
        map_top=map_top,
        scale=scale,
        rotate_map_180=rotate_map_180,
    )
    label = f"{int(node.get('id', 0))} {_short_class_name(node, class_names)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    text_w, text_h = cv2.getTextSize(label, font, font_scale, thickness)[0]
    pad_x, pad_y = 5, 4
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y
    candidates = (
        (10, -box_h - 6),
        (10, 12),
        (-box_w - 10, -box_h - 6),
        (-box_w - 10, 12),
        (-box_w // 2, -box_h - 16),
        (-box_w // 2, 22),
    )

    chosen = None
    for dx, dy in candidates:
        rect = (center[0] + dx, center[1] + dy, center[0] + dx + box_w, center[1] + dy + box_h)
        if not _rect_inside(rect, bounds):
            continue
        if any(_rects_overlap(rect, used, margin=3) for used in placed):
            continue
        chosen = rect
        break
    if chosen is None:
        dx, dy = candidates[0]
        chosen = _clip_rect((center[0] + dx, center[1] + dy, center[0] + dx + box_w, center[1] + dy + box_h), bounds)

    color = _class_color(node, palette)
    fill = tuple(int(v) for v in (color.astype(np.float32) * 0.55).clip(0, 255).tolist())
    x1, y1, x2, y2 = chosen
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x1 + pad_x, y2 - pad_y - 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    placed.append(chosen)


def _draw_header(
    canvas: np.ndarray,
    graph: Mapping[str, Any],
) -> None:
    _, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width, 58), (236, 240, 242), -1)
    title = _scene_graph_title(graph)
    title_scale = _fit_text_scale(title, max_width=width - 44, base_scale=0.70, min_scale=0.42, thickness=2)
    cv2.putText(canvas, title, (22, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (36, 42, 48), 2, cv2.LINE_AA)


def _draw_footer(
    canvas: np.ndarray,
    *,
    map_left: int,
    map_top: int,
    map_w: int,
    map_h: int,
    graph: Mapping[str, Any],
    metadata: Mapping[str, Any],
    update_index: Optional[int],
    visible_edges: int,
) -> None:
    _ = metadata
    y1 = map_top + map_h + 22
    y2 = min(canvas.shape[0] - 10, y1 + 20)
    num_nodes = int(graph.get("metadata", {}).get("num_nodes", len(graph.get("nodes", []))))
    num_edges = int(graph.get("metadata", {}).get("num_edges", len(graph.get("edges", []))))
    opening_side = graph.get("thresholds", {}).get("edge_rule", {}).get("opening_side", "low")
    update_text = "update ?" if update_index is None else f"update {int(update_index)}"
    mode = str(graph.get("metadata", {}).get("runtime_mode") or graph.get("metadata", {}).get("node_source") or "?")
    timing = graph.get("metadata", {}).get("runtime_timing_seconds", {})
    total_seconds = timing.get("total_graph_generation") if isinstance(timing, Mapping) else None
    timing_text = "" if total_seconds is None else f" | graph={float(total_seconds):.3f}s"
    counts_text = (
        f"{update_text} | mode={mode} | {num_nodes} nodes | {num_edges} edges | "
        f"opening_side={opening_side}{timing_text}"
    )
    edge_text = f"showing top {visible_edges}/{num_edges} directed blocks_access_to arrows"
    counts_scale = _fit_text_scale(counts_text, max_width=map_w, base_scale=0.43, min_scale=0.30, thickness=1)
    edge_scale = _fit_text_scale(edge_text, max_width=map_w, base_scale=0.43, min_scale=0.30, thickness=1)
    cv2.putText(canvas, counts_text, (map_left, y1), cv2.FONT_HERSHEY_SIMPLEX, counts_scale, (70, 78, 86), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        edge_text,
        (map_left, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        edge_scale,
        (70, 78, 86),
        1,
        cv2.LINE_AA,
    )


def _draw_belief_header(canvas: np.ndarray, *, title: str) -> None:
    _, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width, 58), (236, 240, 242), -1)
    title_scale = _fit_text_scale(title, max_width=width - 44, base_scale=0.70, min_scale=0.42, thickness=2)
    cv2.putText(canvas, title, (22, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (36, 42, 48), 2, cv2.LINE_AA)


def _draw_belief_footer(
    canvas: np.ndarray,
    *,
    map_left: int,
    map_top: int,
    map_w: int,
    map_h: int,
    update_index: Optional[int],
) -> None:
    y1 = map_top + map_h + 22
    y2 = min(canvas.shape[0] - 10, y1 + 20)
    update_text = "update ?" if update_index is None else f"update {int(update_index)}"
    counts_text = f"{update_text} | belief map only"
    detail_text = "2D occupancy projection + semantic argmax/confidence"
    counts_scale = _fit_text_scale(counts_text, max_width=map_w, base_scale=0.43, min_scale=0.30, thickness=1)
    detail_scale = _fit_text_scale(detail_text, max_width=map_w, base_scale=0.43, min_scale=0.30, thickness=1)
    cv2.putText(canvas, counts_text, (map_left, y1), cv2.FONT_HERSHEY_SIMPLEX, counts_scale, (70, 78, 86), 1, cv2.LINE_AA)
    cv2.putText(canvas, detail_text, (map_left, y2), cv2.FONT_HERSHEY_SIMPLEX, detail_scale, (70, 78, 86), 1, cv2.LINE_AA)


def _draw_access_axis(
    canvas: np.ndarray,
    *,
    map_left: int,
    map_top: int,
    map_h: int,
    rotate_map_180: bool = False,
) -> None:
    x = max(8, map_left - 16)
    y1 = map_top + 16
    y2 = map_top + map_h - 16
    start, end = ((x, y2), (x, y1)) if bool(rotate_map_180) else ((x, y1), (x, y2))
    cv2.arrowedLine(canvas, start, end, (72, 82, 92), 2, cv2.LINE_AA, tipLength=0.04)


def _scene_graph_title(graph: Mapping[str, Any]) -> str:
    metadata = graph.get("metadata", {})
    source_bits = [
        metadata.get("sample_id"),
        metadata.get("source"),
        metadata.get("source_path"),
    ]
    caller_metadata = metadata.get("caller_metadata")
    if isinstance(caller_metadata, Mapping):
        source_bits.extend(
            [
                caller_metadata.get("sample_id"),
                caller_metadata.get("source"),
            ]
        )
    source_text = " ".join(str(bit).lower() for bit in source_bits if bit is not None)
    if "live" in source_text or "pybullet" in source_text:
        return "Live CNABU/MEM scene graph"
    if "saved_cnabu" in source_text or "cnabu_hms" in source_text or ".npz" in source_text:
        return "Offline CNABU/MEM scene graph"
    return "CNABU/MEM scene graph"


def _paste_fit(
    canvas: np.ndarray,
    image: np.ndarray,
    origin: Tuple[int, int],
    size: Tuple[int, int],
    *,
    label: str,
) -> None:
    x, y = origin
    width, height = size
    fitted, offset_x, offset_y = _letterbox(image, width, height)
    canvas[y:y + height, x:x + width] = fitted
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (82, 90, 96), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + width, y + 28), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    _ = (offset_x, offset_y)


def _letterbox(image: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, int, int]:
    image_h, image_w = image.shape[:2]
    scale = min(width / max(image_w, 1), height / max(image_h, 1))
    resized_w = max(1, int(round(image_w * scale)))
    resized_h = max(1, int(round(image_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 230, dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized[:, :, :3]
    return canvas, x, y


def _fit_rect(
    raw_width: int,
    raw_height: int,
    available_w: int,
    available_h: int,
    offset_x: int,
    offset_y: int,
) -> Tuple[int, int, int, int, float]:
    scale = min(available_w / max(raw_width, 1), available_h / max(raw_height, 1))
    map_w = max(1, int(round(raw_width * scale)))
    map_h = max(1, int(round(raw_height * scale)))
    left = offset_x + (available_w - map_w) // 2
    top = offset_y + (available_h - map_h) // 2
    return left, top, map_w, map_h, float(scale)


def _plain_background_color(color_bgr: Sequence[int]) -> np.ndarray:
    color = np.asarray(color_bgr, dtype=np.int32).reshape(-1)
    if color.size != 3:
        raise ValueError(f"plain_background_bgr must contain exactly 3 values, got {color_bgr!r}")
    return np.clip(color, 0, 255).astype(np.uint8)


def _graph_shape_hw(graph: Mapping[str, Any], context: Optional[Mapping[str, Any]]) -> Tuple[int, int]:
    if context is not None and context.get("raw_shape_hw") is not None:
        return _shape_hw_from_value(context["raw_shape_hw"], fallback_hw=(1, 1))
    metadata = graph.get("metadata", {})
    if metadata.get("raw_shape_hw") is not None:
        return _shape_hw_from_value(metadata["raw_shape_hw"], fallback_hw=(1, 1))
    for node in graph.get("nodes", []):
        mask = node.get("mask")
        if mask and mask.get("size"):
            return _shape_hw_from_value(mask["size"], fallback_hw=(1, 1))
    return 140, 200


def _node_bbox(
    node: Mapping[str, Any],
    *,
    raw_shape_hw: Tuple[int, int],
    rotate_map_180: bool = False,
) -> Tuple[float, float, float, float]:
    bbox = node.get("bbox_xyxy_abs")
    if bbox is not None:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    else:
        y, x = _node_centroid_yx(node)
        x1, y1, x2, y2 = (
            max(0.0, x - 2.0),
            max(0.0, y - 2.0),
            min(float(raw_shape_hw[1]), x + 2.0),
            min(float(raw_shape_hw[0]), y + 2.0),
        )
    if not bool(rotate_map_180):
        return x1, y1, x2, y2
    raw_height, raw_width = [float(value) for value in raw_shape_hw]
    return raw_width - x2, raw_height - y2, raw_width - x1, raw_height - y1


def _node_point(
    node: Mapping[str, Any],
    *,
    raw_shape_hw: Tuple[int, int],
    map_left: int,
    map_top: int,
    scale: float,
    rotate_map_180: bool = False,
) -> Tuple[int, int]:
    y, x = _node_centroid_yx(node)
    if bool(rotate_map_180):
        raw_height, raw_width = [float(value) for value in raw_shape_hw]
        x = raw_width - 1.0 - x
        y = raw_height - 1.0 - y
    return int(round(map_left + x * scale)), int(round(map_top + y * scale))


def _node_centroid_yx(node: Mapping[str, Any]) -> Tuple[float, float]:
    if node.get("centroid_yx") is not None:
        y, x = node["centroid_yx"]
        return float(y), float(x)
    if node.get("centroid_xy") is not None:
        x, y = node["centroid_xy"]
        return float(y), float(x)
    x1, y1, x2, y2 = [float(value) for value in node["bbox_xyxy_abs"]]
    return (y1 + y2) / 2.0, (x1 + x2) / 2.0


def _class_color(node: Mapping[str, Any], palette: np.ndarray) -> np.ndarray:
    class_id = int(node.get("class_id", len(palette) - 1))
    class_id = max(0, min(class_id, len(palette) - 1))
    return palette[class_id]


def _short_class_name(node: Mapping[str, Any], class_names: Sequence[str]) -> str:
    class_name = node.get("class_name")
    if not class_name:
        class_id = int(node.get("class_id", -1))
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class{class_id}"
    class_name = str(class_name).replace("Ycb", "")
    replacements = {
        "TomatoSoupCan": "Tomato",
        "PottedMeatCan": "Potted",
        "MasterChefCan": "Chef",
        "GelatinBox": "Gelatin",
        "CrackerBox": "Cracker",
        "ChipsCan": "Chips",
        "BleachCleanser": "Bleach",
        "MustardBottle": "Mustard",
    }
    class_name = replacements.get(class_name, class_name)
    return class_name[:14]


def _shorten_segment(start: Tuple[int, int], end: Tuple[int, int], *, offset: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    vector = np.asarray([end[0] - start[0], end[1] - start[1]], dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return start, end
    unit = vector / length
    start_arr = np.asarray(start, dtype=np.float32) + unit * float(offset)
    end_arr = np.asarray(end, dtype=np.float32) - unit * float(offset)
    return (int(round(start_arr[0])), int(round(start_arr[1]))), (int(round(end_arr[0])), int(round(end_arr[1])))


def _rect_inside(rect: Tuple[int, int, int, int], bounds: Tuple[int, int, int, int]) -> bool:
    return rect[0] >= bounds[0] and rect[1] >= bounds[1] and rect[2] <= bounds[2] and rect[3] <= bounds[3]


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], *, margin: int = 0) -> bool:
    return not (
        a[2] + margin <= b[0]
        or b[2] + margin <= a[0]
        or a[3] + margin <= b[1]
        or b[3] + margin <= a[1]
    )


def _fit_text_scale(
    text: str,
    *,
    max_width: int,
    base_scale: float,
    min_scale: float,
    thickness: int,
) -> float:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, base_scale, thickness)[0][0]
    if width <= int(max_width):
        return float(base_scale)
    if width <= 0:
        return float(base_scale)
    return float(max(min_scale, base_scale * float(max_width) / float(width)))


def _clip_rect(rect: Tuple[int, int, int, int], bounds: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    x1 = min(max(rect[0], bounds[0]), max(bounds[0], bounds[2] - width))
    y1 = min(max(rect[1], bounds[1]), max(bounds[1], bounds[3] - height))
    return x1, y1, x1 + width, y1 + height


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    elif value.__class__.__module__.split(".")[0] == "cupy" and hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()
    array = np.asarray(array, dtype=np.float32)
    if array.size and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _squeeze_to_ndim(value: np.ndarray, min_ndim: int) -> np.ndarray:
    result = np.asarray(value)
    while result.ndim > min_ndim and result.shape[0] == 1:
        result = result[0]
    return result.astype(np.float32, copy=False)


def _first_spatial_shape(*arrays: Optional[np.ndarray]) -> Tuple[int, int]:
    for array in arrays:
        if array is not None:
            if array.ndim != 2:
                raise ValueError(f"context array must be 2D after projection, got {array.shape}")
            return int(array.shape[0]), int(array.shape[1])
    raise ValueError("missing context arrays")


def _shape_hw_from_value(value: Optional[Sequence[int]], *, fallback_hw: Sequence[int]) -> Tuple[int, int]:
    if value is None:
        return int(fallback_hw[0]), int(fallback_hw[1])
    array = np.asarray(value).reshape(-1)
    if array.size == 2:
        return int(array[0]), int(array[1])
    if array.size >= 3:
        return int(array[1]), int(array[2])
    raise ValueError(f"shape value must have at least two entries, got {value!r}")


def _crop_rows_from_value(
    value: Optional[Sequence[int]],
    *,
    crop_height: int,
    raw_height: int,
) -> Tuple[int, int]:
    if value is None:
        return 0, int(crop_height)
    rows = np.asarray(value).reshape(-1)
    if rows.size != 2:
        raise ValueError(f"crop_rows must have two entries, got {value!r}")
    start, stop = int(rows[0]), int(rows[1])
    if stop - start != int(crop_height):
        raise ValueError(f"crop_rows {[start, stop]} do not match crop height {crop_height}")
    if start < 0 or stop > int(raw_height):
        raise ValueError(f"crop_rows {[start, stop]} outside raw height {raw_height}")
    return start, stop


def _pad_crop_2d(
    array: Optional[np.ndarray],
    *,
    raw_shape_hw: Tuple[int, int],
    crop_rows: Tuple[int, int],
) -> Optional[np.ndarray]:
    if array is None:
        return None
    result = np.zeros(raw_shape_hw, dtype=array.dtype)
    result[crop_rows[0]:crop_rows[1], :] = array
    return result


def _normalise01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not bool(finite.any()):
        return np.zeros_like(array, dtype=np.float32)
    result = np.zeros_like(array, dtype=np.float32)
    values = array[finite]
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-8:
        result[finite] = np.clip(values, 0.0, 1.0)
    else:
        result[finite] = (values - min_value) / (max_value - min_value)
    return np.clip(result, 0.0, 1.0)


def _subtle_grid(shape_hw: Tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    grid = np.full((height, width), 215, dtype=np.float32)
    step = max(8, min(height, width) // 10)
    grid[::step, :] = 145
    grid[:, ::step] = 145
    return grid


__all__ = [
    "DEFAULT_CLASS_PALETTE_BGR",
    "build_cnabu_map_context",
    "compose_runtime_demo_panel",
    "render_cnabu_belief_map_view",
    "render_cnabu_context_background",
    "render_cnabu_scene_graph_view",
]
