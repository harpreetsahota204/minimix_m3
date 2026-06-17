"""Parse MiniMax-M3's raw output into FiftyOne label objects.

Unlike tag-grammar VLMs, M3 is prompted to emit **JSON**. `to_fiftyone(content,
annotation_format, ...)` is the single dispatcher. The JSON-extraction strategy
mirrors the vibe-check notebook: strip ``<think>`` blocks, strip ``` code
fences, regex-grab the first ``[...]`` / ``{...}`` span, and ``json.loads`` it.

Conventions:
    * Coordinates are NORMALIZED [0, 1] (``COORD_SCALE = 1.0``), consumed by
      FiftyOne directly.
    * Boxes: ``[x1, y1, x2, y2]`` (top-left / bottom-right) ->
      ``fo.Detection(bounding_box=[x1, y1, x2 - x1, y2 - y1])``.
    * Keypoints: ``[x, y]`` -> ``fo.Keypoint(points=[[x, y]])``.
    * Temporal: ``{start, end}`` seconds -> ``fo.TemporalDetection`` with a
      ``support=[start_frame, end_frame]`` derived from the video frame rate
      (plus raw ``t_start_seconds`` / ``t_end_seconds`` attributes).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeAlias

import fiftyone as fo

logger = logging.getLogger("minimax_m3")


# M3 returns NORMALIZED [0, 1] coordinates, so no rescaling is needed.
COORD_SCALE: float = 1.0


# Return shape of `to_fiftyone`.
FOLabel: TypeAlias = (
    fo.Detections
    | fo.Keypoints
    | fo.TemporalDetections
    | fo.Classification
    | fo.Classifications
    | str
    | None
)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)
_JSON_SPAN_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Strip thinking blocks + code fences and parse the first JSON span.

    Returns the parsed value (list / dict) or ``None`` if no parseable JSON is
    found. Tolerant of prose before/after the JSON, which M3 occasionally adds.
    """
    if not text:
        return None
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned)
    m = _JSON_SPAN_RE.search(cleaned)
    if m is None:
        logger.info("[minimax_m3] extract_json: no JSON span found")
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError as exc:
        logger.info("[minimax_m3] extract_json: parse error (%s)", exc)
        return None


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and trim, for free-text outputs."""
    return _THINK_BLOCK_RE.sub("", text or "").strip()


# ---------------------------------------------------------------------------
# Per-item coordinate helpers
# ---------------------------------------------------------------------------


def _coerce_box(raw: Any) -> tuple[float, float, float, float] | None:
    """Coerce a raw box value to normalized ``(x1, y1, x2, y2)``, or ``None``.

    Accepts a 4-length list/tuple. Swaps corners so ``x1<=x2`` / ``y1<=y2``.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) / COORD_SCALE for v in raw[:4])
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _box_field(item: dict[str, Any]) -> Any:
    """Return the box value from an item, tolerating box / bbox / bbox_2d keys."""
    for key in ("box", "bbox", "bbox_2d", "bounding_box"):
        if key in item:
            return item[key]
    return None


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def to_fiftyone(
    content: str,
    annotation_format: str,
    *,
    target: str | None = None,
    frame_rate: float | None = None,
) -> FOLabel:
    """Convert M3's raw response text to a FiftyOne label container.

    ``annotation_format`` values and their output types:
        * ``"box"``             -> `fo.Detections`
        * ``"point"``           -> `fo.Keypoints`
        * ``"temporal"``        -> `fo.TemporalDetections`
        * ``"caption"`` | ``"vqa"`` -> ``str``
        * ``"classify_single"`` -> `fo.Classification`
        * ``"classify_multi"``  -> `fo.Classifications`

    Args:
        content: Raw ``message.content`` string from the API.
        annotation_format: One of the values listed above.
        target: Class-label fallback for unlabeled items.
        frame_rate: Video frame rate, used for ``support`` mapping on temporal
            tasks; optional otherwise.
    """
    preview = content.replace("\n", " ")
    if len(preview) > 200:
        preview = preview[:200] + "..."
    logger.info(
        "[minimax_m3] to_fiftyone: format=%s len=%d preview=%r",
        annotation_format,
        len(content),
        preview,
    )

    if not content:
        logger.info("[minimax_m3] empty content; returning empty container for format=%s", annotation_format)
        return _empty_container_for(annotation_format)

    match annotation_format:
        case "box":
            return _parse_boxes(content, target=target)
        case "point":
            return _parse_points(content, target=target)
        case "temporal":
            return _parse_temporal(content, frame_rate=frame_rate)
        case "caption" | "vqa":
            cleaned = strip_thinking(content)
            logger.info("[minimax_m3] free-text result: %d chars", len(cleaned))
            return cleaned
        case "classify_single":
            return _parse_classifications(content, multi=False)
        case "classify_multi":
            return _parse_classifications(content, multi=True)
        case _:
            raise ValueError(f"unsupported annotation_format: {annotation_format!r}")


def _empty_container_for(annotation_format: str) -> FOLabel:
    """Return the empty FiftyOne container matching ``annotation_format``."""
    match annotation_format:
        case "box":
            return fo.Detections(detections=[])
        case "point":
            return fo.Keypoints(keypoints=[])
        case "temporal":
            return fo.TemporalDetections(detections=[])
        case "caption" | "vqa":
            return ""
        case "classify_single" | "classify_multi":
            return None
        case _:
            raise ValueError(f"unsupported annotation_format: {annotation_format!r}")


def _parse_boxes(content: str, *, target: str | None) -> fo.Detections:
    """Parse a JSON list of ``{"label", "box":[x1,y1,x2,y2]}`` into Detections."""
    parsed = extract_json(content)
    if not isinstance(parsed, list):
        logger.info("[minimax_m3] boxes: expected a JSON list, got %s", type(parsed).__name__)
        return fo.Detections(detections=[])

    detections: list[fo.Detection] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coords = _coerce_box(_box_field(item))
        if coords is None:
            logger.info("[minimax_m3] boxes: skipping item with missing/invalid box: %r", item)
            continue
        x1, y1, x2, y2 = coords
        label = str(item.get("label") or target or "object")
        detections.append(
            fo.Detection(label=label, bounding_box=[x1, y1, x2 - x1, y2 - y1])
        )

    logger.info("[minimax_m3] parsed %d box(es)", len(detections))
    return fo.Detections(detections=detections)


def _parse_points(content: str, *, target: str | None) -> fo.Keypoints:
    """Parse a JSON list of ``{"label", "point":[x,y]}`` into Keypoints.

    Falls back to box centers if the model returned boxes instead of points.
    """
    parsed = extract_json(content)
    if not isinstance(parsed, list):
        logger.info("[minimax_m3] points: expected a JSON list, got %s", type(parsed).__name__)
        return fo.Keypoints(keypoints=[])

    keypoints: list[fo.Keypoint] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or target or "point")
        point = item.get("point") or item.get("keypoint")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                x, y = float(point[0]) / COORD_SCALE, float(point[1]) / COORD_SCALE
            except (TypeError, ValueError):
                continue
            keypoints.append(fo.Keypoint(label=label, points=[(x, y)]))
            continue
        # Fallback: model returned a box; use its center.
        coords = _coerce_box(_box_field(item))
        if coords is not None:
            x1, y1, x2, y2 = coords
            keypoints.append(
                fo.Keypoint(label=label, points=[((x1 + x2) / 2, (y1 + y2) / 2)])
            )
        else:
            logger.info("[minimax_m3] points: skipping item with no point/box: %r", item)

    logger.info("[minimax_m3] parsed %d keypoint(s)", len(keypoints))
    return fo.Keypoints(keypoints=keypoints)


# FiftyOne `TemporalDetection` requires a non-null ``support=[first, last]``.
# When ``frame_rate`` is unknown we fall back to this placeholder; the raw
# seconds live on ``t_start_seconds`` / ``t_end_seconds`` attributes.
_PLACEHOLDER_SUPPORT: list[int] = [1, 1]


def _seconds_to_frame(seconds: float, frame_rate: float | None) -> int | None:
    """Convert seconds to a 1-indexed FiftyOne frame number, or ``None``."""
    if frame_rate is None or frame_rate <= 0:
        return None
    return max(1, int(round(seconds * frame_rate)))


def _parse_temporal(content: str, *, frame_rate: float | None) -> fo.TemporalDetections:
    """Parse a JSON list of ``{"label","start","end"}`` into TemporalDetections.

    Each detection carries raw ``t_start_seconds`` / ``t_end_seconds`` plus a
    ``support=[start_frame, end_frame]`` when ``frame_rate`` is known.
    """
    parsed = extract_json(content)
    if not isinstance(parsed, list):
        logger.info("[minimax_m3] temporal: expected a JSON list, got %s", type(parsed).__name__)
        return fo.TemporalDetections(detections=[])

    detections: list[fo.TemporalDetection] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            logger.info("[minimax_m3] temporal: skipping item without start/end: %r", item)
            continue
        try:
            t_start = float(item["start"])
            t_end = float(item["end"])
        except (TypeError, ValueError):
            logger.info("[minimax_m3] temporal: unparseable start/end: %r", item)
            continue
        label = str(item.get("label") or "event")
        detections.append(
            _build_temporal_detection(
                label=label, t_start=t_start, t_end=t_end, frame_rate=frame_rate
            )
        )

    logger.info("[minimax_m3] parsed %d temporal detection(s)", len(detections))
    return fo.TemporalDetections(detections=detections)


def _build_temporal_detection(
    *, label: str, t_start: float, t_end: float, frame_rate: float | None
) -> fo.TemporalDetection:
    """Build a `TemporalDetection` with raw seconds + frame-index ``support``."""
    f_start = _seconds_to_frame(t_start, frame_rate)
    f_end = _seconds_to_frame(t_end, frame_rate)

    if f_start is None or f_end is None:
        logger.warning(
            "[minimax_m3] frame_rate unavailable; TemporalDetection for %r uses "
            "placeholder support=%s (raw seconds kept on t_start_seconds / "
            "t_end_seconds).",
            label,
            _PLACEHOLDER_SUPPORT,
        )
        det = fo.TemporalDetection(label=label, support=list(_PLACEHOLDER_SUPPORT))
        det["support_is_placeholder"] = True
    else:
        if f_end < f_start:
            f_end = f_start
        det = fo.TemporalDetection(label=label, support=[f_start, f_end])

    det["t_start_seconds"] = t_start
    det["t_end_seconds"] = t_end
    return det


def _parse_classifications(
    content: str, *, multi: bool
) -> fo.Classification | fo.Classifications | None:
    """Parse classification JSON into FiftyOne classification label(s).

    Accepts ``{"classifications": [{"label","confidence"}]}``, a bare list, or
    a single ``{"label", ...}`` object. Returns `Classification` when
    ``multi=False`` (first item only), `Classifications` when ``multi=True``,
    or ``None`` if nothing parses.
    """
    parsed = extract_json(content)
    if parsed is None:
        logger.info("[minimax_m3] classifications: no JSON parsed")
        return None

    items: list[dict[str, Any]] = []
    match parsed:
        case dict() as d if isinstance(d.get("classifications"), list):
            items = [c for c in d["classifications"] if isinstance(c, dict)]
        case dict() as d if "label" in d:
            items = [d]
        case list() as lst:
            items = [c for c in lst if isinstance(c, dict)]
        case _:
            logger.info(
                "[minimax_m3] classifications JSON had unexpected shape: %s",
                type(parsed).__name__,
            )
            return None

    classifications: list[fo.Classification] = []
    for item in items:
        label = item.get("label")
        if label is None:
            continue
        kwargs: dict[str, Any] = {"label": str(label)}
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            kwargs["confidence"] = float(confidence)
        classifications.append(fo.Classification(**kwargs))

    if not classifications:
        logger.info("[minimax_m3] classifications JSON parsed but yielded no items")
        return None

    if not multi:
        if len(classifications) > 1:
            logger.info(
                "[minimax_m3] classify_single got %d items; keeping first only",
                len(classifications),
            )
        return classifications[0]
    return fo.Classifications(classifications=classifications)


# ---------------------------------------------------------------------------
# Per-frame writeback for video samples.
# ---------------------------------------------------------------------------


def write_per_frame_labels(
    sample: fo.Sample,
    label_obj: FOLabel,
    field: str,
    *,
    frame_rate: float | None,
) -> dict[str, int]:
    """Route a label container to ``sample.frames[i][field]`` or ``sample[field]``.

    Routing rules:

    * **Video + Detections/Keypoints**: items go on frames (the App renders
      spatial labels off frames). Items need both a ``t`` (seconds) attribute
      and a usable ``frame_rate``; otherwise they're dropped with a warning.
    * **Video + non-spatial** (Classification(s), str, TemporalDetections):
      these describe the whole clip, so they go on ``sample[field]``.
    * **Image samples**: always ``sample[field]``.

    Returns a summary dict with ``per_frame_count`` / ``sample_level`` /
    ``dropped`` / ``frames_written`` keys.
    """
    if label_obj is None:
        logger.info("[minimax_m3] write_per_frame_labels: label_obj is None; nothing to write")
        return _writeback_summary()

    is_video = getattr(sample, "media_type", None) == "video"

    # Non-spatial labels are always sample-level (image or video).
    if not isinstance(label_obj, (fo.Detections, fo.Keypoints)):
        sample[field] = label_obj
        sample.save()
        n_items = _count_sample_level_items(label_obj)
        logger.info(
            "[minimax_m3] wrote sample-level %s (%d item(s)) -> sample[%s]",
            type(label_obj).__name__,
            n_items,
            field,
        )
        return _writeback_summary(sample_level=n_items)

    items = (
        label_obj.detections
        if isinstance(label_obj, fo.Detections)
        else label_obj.keypoints
    )

    if not items:
        logger.info(
            "[minimax_m3] %s container is empty; nothing to write for sample[%s]",
            type(label_obj).__name__,
            field,
        )
        return _writeback_summary()

    # Image case: always sample-level.
    if not is_video:
        sample[field] = label_obj
        sample.save()
        logger.info(
            "[minimax_m3] image sample: wrote sample-level %s (%d item(s)) -> sample[%s]",
            type(label_obj).__name__,
            len(items),
            field,
        )
        return _writeback_summary(sample_level=len(items))

    # Video case: spatial labels MUST live on frames.
    if frame_rate is None or frame_rate <= 0:
        logger.error(
            "[minimax_m3] video sample %s has frame_rate=%r; %d %s label(s) cannot "
            "be placed on a frame and will be dropped. Run "
            "`sample.compute_metadata()` and re-run to keep these labels.",
            sample.filepath,
            frame_rate,
            len(items),
            type(label_obj).__name__,
        )
        return _writeback_summary(dropped=len(items))

    timed = [it for it in items if getattr(it, "t", None) is not None]
    n_untimed = len(items) - len(timed)
    if n_untimed:
        logger.warning(
            "[minimax_m3] video sample %s: %d/%d %s label(s) had no `t=` timestamp "
            "and will be dropped (only timestamped labels can be placed on a frame).",
            sample.filepath,
            n_untimed,
            len(items),
            type(label_obj).__name__,
        )

    if not timed:
        return _writeback_summary(dropped=len(items))

    written = _write_items_to_frames(sample, timed, field, frame_rate=frame_rate)
    written["dropped"] += n_untimed
    return written


def _writeback_summary(
    *,
    per_frame_count: int = 0,
    sample_level: int = 0,
    dropped: int = 0,
    frames_written: int = 0,
) -> dict[str, int]:
    return {
        "per_frame_count": per_frame_count,
        "sample_level": sample_level,
        "dropped": dropped,
        "frames_written": frames_written,
    }


def _count_sample_level_items(label_obj: Any) -> int:
    match label_obj:
        case fo.Classifications():
            return len(label_obj.classifications)
        case fo.TemporalDetections():
            return len(label_obj.detections)
        case _:
            return 1


def _write_items_to_frames(
    sample: fo.Sample,
    items: list[Any],
    field: str,
    *,
    frame_rate: float,
) -> dict[str, int]:
    """Group timed `items` by frame index and write to `sample.frames[i][field]`."""
    by_frame: dict[int, list[Any]] = {}
    for item in items:
        idx = _seconds_to_frame(item.t, frame_rate)
        assert idx is not None  # noqa: S101 -- frame_rate validated by caller
        by_frame.setdefault(idx, []).append(item)

    is_keypoint = isinstance(items[0], fo.Keypoint)
    for frame_idx, batch in sorted(by_frame.items()):
        sample.frames[frame_idx][field] = (
            fo.Keypoints(keypoints=batch)
            if is_keypoint
            else fo.Detections(detections=batch)
        )

    sample.save()
    logger.info(
        "[minimax_m3] wrote per-frame labels to sample[%s]: %d label(s) across %d frame(s)",
        field,
        len(items),
        len(by_frame),
    )
    return _writeback_summary(
        per_frame_count=len(items),
        frames_written=len(by_frame),
    )
