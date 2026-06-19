"""Parse MiniMax-M3's raw output into FiftyOne label objects.

Unlike tag-grammar VLMs, M3 is prompted to emit **JSON**. `to_fiftyone(content,
annotation_format, ...)` is the single dispatcher. JSON extraction strips
``<think>`` blocks and ``` code fences, then scans left-to-right for the first
*balanced* ``[...]`` / ``{...}`` span that ``json.loads`` accepts (tolerating
preamble/trailing prose and stray brackets).

Conventions:
    * Coordinates are NORMALIZED [0, 1] and consumed by FiftyOne directly (M3
      reliably emits this scale, so no rescaling is applied).
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

from ._shared import preview_text

logger = logging.getLogger("minimax_m3")


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

_OPEN_TO_CLOSE: dict[str, str] = {"[": "]", "{": "}"}
_CLOSE_TO_OPEN: dict[str, str] = {"]": "[", "}": "{"}


def _balanced_span(text: str, start: int) -> str | None:
    """Return the balanced JSON value (``[...]`` / ``{...}``) starting at ``start``.

    Walks forward tracking bracket depth while respecting string literals and
    escapes, so braces inside quoted strings don't end the span. Returns the
    substring, or ``None`` if the brackets never balance.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if not stack or stack[-1] != _CLOSE_TO_OPEN[ch]:
                return None
            stack.pop()
            if not stack:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any | None:
    """Strip thinking blocks + code fences and parse the first JSON value.

    Returns the parsed value (list / dict) or ``None`` if no parseable JSON is
    found. Robust to preamble/trailing prose and stray brackets in that prose:
    it scans for balanced ``[...]`` / ``{...}`` spans left-to-right and returns
    the first one that parses, rather than greedily matching first-open to
    last-close (which breaks when prose contains a stray ``[`` or ``{``).
    """
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", _THINK_BLOCK_RE.sub("", text))

    # Fast path: the whole (cleaned) payload is valid JSON.
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # Otherwise try each balanced span starting at an opening bracket.
    for i, ch in enumerate(cleaned):
        if ch not in _OPEN_TO_CLOSE:
            continue
        span = _balanced_span(cleaned, i)
        if span is None:
            continue
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue

    logger.info("[minimax_m3] extract_json: no parseable JSON span found")
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
        x1, y1, x2, y2 = (float(v) for v in raw[:4])
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


def _coerce_xy(raw: Any) -> tuple[float, float] | None:
    """Coerce a raw point value to normalized ``(x, y)``, or ``None``.

    Accepts ``[x, y]`` or singly-nested ``[[x, y]]`` (some replies wrap points).
    """
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _box_center(coords: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the center ``(x, y)`` of a normalized ``(x1, y1, x2, y2)`` box."""
    x1, y1, x2, y2 = coords
    return (x1 + x2) / 2, (y1 + y2) / 2


def _all_numbers(seq: Any) -> bool:
    """Whether ``seq`` is a non-empty sequence of plain numbers (not bools)."""
    if not isinstance(seq, (list, tuple)) or not seq:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq)


# Keys that mark a dict as a single grounding item (vs. a wrapper object).
_ITEM_MARKER_KEYS: tuple[str, ...] = (
    "box", "bbox", "bbox_2d", "bounding_box",
    "point", "keypoint", "points",
    "start", "end",
)

# Keys tried, in order, when reading an item's class label.
_LABEL_KEYS: tuple[str, ...] = ("label", "class", "category", "name", "type")


def _coerce_item_list(parsed: Any) -> list[Any]:
    """Normalize parsed JSON into a flat list of item candidates.

    Handles a bare list, a single item dict (``{"label": ..., "box": [...]}``),
    and a wrapper dict (``{"detections": [...]}``) — so output that isn't a
    plain list still parses.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if any(key in parsed for key in _ITEM_MARKER_KEYS):
            return [parsed]
        for value in parsed.values():
            if isinstance(value, list) and (not value or isinstance(value[0], (dict, list))):
                return value
        return [parsed]
    return []


def _item_label(
    item: dict[str, Any], fallback: str | None, *, extra_keys: tuple[str, ...] = ()
) -> str | None:
    """Read an item's label, tolerating alternate key names.

    Returns ``fallback`` when no label-like key is present (or ``None`` when
    ``fallback`` is ``None``, signalling the caller to skip the item).
    """
    for key in (*_LABEL_KEYS, *extra_keys):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


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

    Coordinates are consumed as the normalized ``[0, 1]`` values M3 emits; no
    rescaling is applied.
    """
    logger.info(
        "[minimax_m3] to_fiftyone: format=%s len=%d preview=%r",
        annotation_format,
        len(content),
        preview_text(content),
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
    """Parse boxes into Detections.

    Tolerates: wrapper dicts (``{"detections": [...]}``), single-item dicts,
    items missing a ``label`` key (falls back to ``target`` / ``"object"``),
    alternate label keys, and bare ``[x1, y1, x2, y2]`` arrays. Coordinates are
    used as the normalized ``[0, 1]`` values M3 emits.
    """
    fallback = target or "object"
    detections: list[fo.Detection] = []
    for item in _coerce_item_list(extract_json(content)):
        label = fallback
        if isinstance(item, dict):
            coords = _coerce_box(_box_field(item))
            label = _item_label(item, fallback) or fallback
        elif _all_numbers(item):
            coords = _coerce_box(item)
        else:
            coords = None
        if coords is None:
            logger.info("[minimax_m3] boxes: skipping item with missing/invalid box: %r", item)
            continue
        x1, y1, x2, y2 = coords
        detections.append(
            fo.Detection(label=label, bounding_box=[x1, y1, x2 - x1, y2 - y1])
        )

    logger.info("[minimax_m3] parsed %d box(es)", len(detections))
    return fo.Detections(detections=detections)


def _parse_points(content: str, *, target: str | None) -> fo.Keypoints:
    """Parse keypoints into Keypoints.

    Tolerates wrapper/single-item dicts, items missing a ``label`` key, bare
    ``[x, y]`` arrays, and falls back to box centers when the model returns a
    box (or a bare 4-number array) instead of a point. Coordinates are used as
    the normalized ``[0, 1]`` values M3 emits.
    """
    fallback = target or "point"
    keypoints: list[fo.Keypoint] = []
    for item in _coerce_item_list(extract_json(content)):
        label = fallback
        if isinstance(item, dict):
            label = _item_label(item, fallback) or fallback
            xy = _coerce_xy(item.get("point") or item.get("keypoint") or item.get("points"))
            if xy is None:
                coords = _coerce_box(_box_field(item))
                xy = _box_center(coords) if coords else None
        elif _all_numbers(item):
            # A bare 4-number array is a box (use its center); 2 numbers are a point.
            if len(item) >= 4:
                coords = _coerce_box(item)
                xy = _box_center(coords) if coords else None
            else:
                xy = _coerce_xy(item)
        else:
            xy = None
        if xy is None:
            logger.info("[minimax_m3] points: skipping item with no point/box: %r", item)
            continue
        keypoints.append(fo.Keypoint(label=label, points=[[xy[0], xy[1]]]))

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


_START_KEYS: tuple[str, ...] = ("start", "start_time", "from", "t_start", "begin")
_END_KEYS: tuple[str, ...] = ("end", "end_time", "to", "t_end", "stop")


def _get_first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        if item.get(key) is not None:
            return item[key]
    return None


def _parse_temporal(content: str, *, frame_rate: float | None) -> fo.TemporalDetections:
    """Parse temporal events into TemporalDetections.

    Tolerates wrapper/single-item dicts, items missing a ``label`` key (falls
    back to ``"event"``), and alternate start/end key names. Each detection
    carries raw ``t_start_seconds`` / ``t_end_seconds`` plus a
    ``support=[start_frame, end_frame]`` when ``frame_rate`` is known.
    """
    detections: list[fo.TemporalDetection] = []
    for item in _coerce_item_list(extract_json(content)):
        if not isinstance(item, dict):
            continue
        start_raw = _get_first(item, _START_KEYS)
        end_raw = _get_first(item, _END_KEYS)
        if start_raw is None or end_raw is None:
            logger.info("[minimax_m3] temporal: skipping item without start/end: %r", item)
            continue
        try:
            t_start = float(start_raw)
            t_end = float(end_raw)
        except (TypeError, ValueError):
            logger.info("[minimax_m3] temporal: unparseable start/end: %r", item)
            continue
        detections.append(
            _build_temporal_detection(
                label=_item_label(item, "event") or "event",
                t_start=t_start,
                t_end=t_end,
                frame_rate=frame_rate,
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

    _cls_label_keys = (*_LABEL_KEYS, "value", "answer", "prediction")
    items: list[dict[str, Any]] = []
    match parsed:
        case dict() as d if isinstance(d.get("classifications"), list):
            items = [c for c in d["classifications"] if isinstance(c, dict)]
        case dict() as d if any(k in d for k in _cls_label_keys):
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
        label = _item_label(item, None, extra_keys=("value", "answer", "prediction"))
        if label is None:
            continue
        kwargs: dict[str, Any] = {"label": label}
        confidence = item.get("confidence")
        if confidence is None:
            confidence = item.get("score")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
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
        n_items = count_label_items(label_obj)
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


def count_label_items(label: Any) -> int:
    """Return the number of items in a FiftyOne label container (1 for scalars)."""
    match label:
        case fo.Detections():
            return len(label.detections)
        case fo.Keypoints():
            return len(label.keypoints)
        case fo.Classifications():
            return len(label.classifications)
        case fo.TemporalDetections():
            return len(label.detections)
        case _:
            return 1


def label_items(label: Any) -> list[Any]:
    """Return the individual label objects inside ``label`` for per-object edits.

    Containers yield their members; a scalar ``Classification`` yields itself;
    non-label values (e.g. free-text captions) yield ``[]``.
    """
    match label:
        case fo.Detections():
            return label.detections
        case fo.Keypoints():
            return label.keypoints
        case fo.Classifications():
            return label.classifications
        case fo.TemporalDetections():
            return label.detections
        case fo.Classification():
            return [label]
        case _:
            return []


def attach_provenance(label: Any, provenance: dict[str, Any]) -> None:
    """Stamp ``provenance`` attributes onto each label object in ``label``.

    Stored per-object (not on the container) so the metadata survives merging
    multiple runs into one field: each item keeps the prompt / resolution / raw
    output of the run that produced it. A no-op for free-text labels.
    """
    for item in label_items(label):
        for key, value in provenance.items():
            item[key] = value


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
