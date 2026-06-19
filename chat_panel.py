"""MiniMax-M3 Chat Panel — ask questions about the current image or video.

Architecture
------------
A hybrid modal panel. Python lifecycle hooks push the current sample's
filepath and media type to React via ``ctx.panel.set_state()``. The React
component calls two panel methods:

    ``ask``             — validates params, starts an inference daemon thread
                          that streams tokens to a file, returns a run_id.
    ``get_stream_chunk``— reads new bytes from the stream file since the
                          caller's last cursor position; React polls every
                          250 ms to produce a live typing effect.

Converting a response to a FiftyOne label is handled by the ``save_minimax_label``
operator (see operators.py), which reuses ``save_stream_as_label`` here to parse
and write the label, then refreshes the App so the overlay appears in the open
modal. Only offered when the response looked like a recognised JSON shape.

Streaming runs in a daemon thread that writes token chunks to an append-only
file under ``~/.fiftyone/minimax_chat/`` and lifecycle state to a sibling JSON
file. This file-based IPC is required because FiftyOne reimports the plugin
module on every panel-method call.

Video handling: M3 has no native video input on the HF router, so video
samples are frame-sampled (see ``minimax_api.sample_video_frames``) and the
timestamped frame strip is embedded in the first user message.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import bson
from openai import OpenAI

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone import ViewField as F

from ._shared import get_api_key, has_api_key
from .minimax_api import (
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE_MAX_SIDE,
    DEFAULT_MODEL_ID,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOP_K,
    encode_image,
    frame_parts,
    resolution_label,
    sample_video_frames,
)
from .minimax_parser import attach_provenance, count_label_items, to_fiftyone
from .prompts import JSON_SHAPE_BY_FORMAT

logger = logging.getLogger("minimax_m3")

# Runtime files live OUTSIDE the plugin directory to avoid invalidating
# FiftyOne's plugin cache (writing inside the dir changes its mtime).
_STATUS_DIR = Path.home() / ".fiftyone" / "minimax_chat"

# JSON-shape instructions the panel surfaces and lets the user edit; the edited
# text is sent back as `hint_text`. Sourced verbatim from
# `prompts.JSON_SHAPE_BY_FORMAT` so the panel and the operator demand the exact
# same shape (single source of truth).
_HINT_TEMPLATES: dict[str, str] = dict(JSON_SHAPE_BY_FORMAT)

# Default output field names shown in the Convert UI, one per format.
_DEFAULT_FIELDS: dict[str, str] = {
    "box": "m3_detections",
    "point": "m3_keypoints",
    "temporal": "m3_events",
}


def _stream_path(run_id: str) -> Path:
    return _STATUS_DIR / f".stream_{run_id}.txt"


def _status_path(run_id: str) -> Path:
    return _STATUS_DIR / f".status_{run_id}.json"


def _ensure_dir() -> None:
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    _ensure_dir()
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _append_stream(run_id: str, text: str) -> None:
    _ensure_dir()
    with open(_stream_path(run_id), "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


def _meta_path(run_id: str) -> Path:
    return _STATUS_DIR / f".meta_{run_id}.json"


def _clear_run(run_id: str) -> None:
    for p in (_stream_path(run_id), _status_path(run_id), _meta_path(run_id)):
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Format detection (JSON shapes)
# ---------------------------------------------------------------------------


def _detect_format(content: str) -> str | None:
    """Detect a JSON grounding shape from a completed response.

    Priority: temporal (start+end) > box > point. Returns ``None`` for plain
    text, which hides the Convert button.
    """
    if re.search(r'"start"\s*:', content) and re.search(r'"end"\s*:', content):
        return "temporal"
    if re.search(r'"(?:box|bbox|bbox_2d|bounding_box)"\s*:', content):
        return "box"
    if re.search(r'"point"\s*:', content):
        return "point"
    return None


def _append_label(existing: Any, incoming: Any, field: str) -> Any:
    """Append compatible converted labels to an existing sample field value."""
    if existing is None:
        return incoming

    if isinstance(existing, fo.Detections) and isinstance(incoming, fo.Detections):
        return fo.Detections(detections=[*existing.detections, *incoming.detections])

    if isinstance(existing, fo.Keypoints) and isinstance(incoming, fo.Keypoints):
        return fo.Keypoints(keypoints=[*existing.keypoints, *incoming.keypoints])

    if isinstance(existing, fo.TemporalDetections) and isinstance(
        incoming, fo.TemporalDetections
    ):
        return fo.TemporalDetections(
            detections=[*existing.detections, *incoming.detections]
        )

    raise ValueError(
        f"field {field!r} already contains {type(existing).__name__}; "
        f"cannot append {type(incoming).__name__}"
    )


def _stringify_object_ids(value: Any) -> Any:
    """Recursively convert BSON ObjectIds to strings.

    The App's sample JSON (what the modal looker renders) uses plain hex-string
    ids, so a label serialized via ``to_dict()`` must have its ObjectIds
    stringified before it can be merged into the modal sample client-side.
    """
    if isinstance(value, bson.ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify_object_ids(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_object_ids(v) for v in value]
    return value


def save_stream_as_label(
    dataset: Any,
    run_id: str,
    sample_id: str,
    field: str,
    fmt: str,
    frame_rate: Any = None,
) -> dict:
    """Parse a completed stream as JSON and write it to a sample as a label.

    Each saved label object is stamped with ``minimax_prompt``,
    ``minimax_raw_output``, and (for images) ``minimax_resolution`` provenance
    attributes read from the run's meta file.

    On success returns a summary dict including ``field_is_new`` and
    ``label_json`` -- the saved field serialized to the App's sample-JSON shape
    so the caller can refresh the open modal in place. Returns ``{"error": ...}``
    on any failure.
    """
    if not run_id:
        return {"error": "No run_id provided."}
    if not sample_id:
        return {"error": "No sample_id provided."}
    if not field:
        return {"error": "Field name is required."}
    if not fmt:
        return {"error": "No detected format — nothing to convert."}

    stream_file = _stream_path(run_id)
    if not stream_file.exists():
        return {"error": (
            f"Stream file for run '{run_id}' no longer exists. "
            "Re-ask the question to generate a new stream."
        )}

    try:
        content = stream_file.read_text(encoding="utf-8")
        fr = float(frame_rate) if frame_rate else None

        field_is_new = field not in dataset.get_field_schema()
        sample = dataset[sample_id]

        label = to_fiftyone(content, fmt, frame_rate=fr)

        meta = _read_json(_meta_path(run_id)) or {}
        provenance: dict[str, Any] = {
            "minimax_prompt": meta.get("prompt", ""),
            "minimax_raw_output": content,
        }
        if getattr(sample, "media_type", "image") != "video" and "image_max_side" in meta:
            provenance["minimax_resolution"] = resolution_label(int(meta["image_max_side"]))
        attach_provenance(label, provenance)

        existing_label = None if field_is_new else sample.get_field(field)
        label = _append_label(existing_label, label, field)

        sample[field] = label
        sample.save()

        return {
            "saved": True,
            "label_type": type(label).__name__,
            "count": count_label_items(label),
            "field": field,
            "field_is_new": field_is_new,
            "label_json": _stringify_object_ids(label.to_dict()),
        }
    except Exception as exc:
        return {"error": f"Parse / save failed: {exc}"}


# ---------------------------------------------------------------------------
# Media part construction
# ---------------------------------------------------------------------------


def _build_media_parts(
    filepath: str, media_type: str, n_frames: int, image_max_side: int
) -> list[dict[str, Any]]:
    """Return the OpenAI content parts representing the media.

    For images, a single ``image_url`` part encoded at ``image_max_side``
    (longest side; ``<= 0`` sends native resolution). For videos, a timestamped
    frame strip (text marker + ``image_url`` per sampled frame).
    """
    if media_type == "video":
        frames, _fps, _total = sample_video_frames(filepath, n=n_frames)
        return frame_parts(frames)
    return [
        {"type": "image_url", "image_url": {"url": encode_image(filepath, max_side=image_max_side)}}
    ]


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------


def _run_stream_thread(
    api_key: str,
    messages: list[dict[str, Any]],
    thinking: str,
    run_id: str,
) -> None:
    """Stream an M3 chat completion and write tokens to the stream file."""
    _write_json(_status_path(run_id), {"status": "streaming", "start_time": time.time()})
    try:
        client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT_SECONDS)

        stream = client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=1.0,
            top_p=0.95,
            max_tokens=1500,
            extra_body={"top_k": DEFAULT_TOP_K, "thinking": {"type": thinking}},
        )

        t0 = time.time()
        prompt_tokens = completion_tokens = 0

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            delta = getattr(choice, "delta", None) if choice else None
            if delta and getattr(delta, "content", None):
                _append_stream(run_id, delta.content)
            if getattr(chunk, "usage", None):
                prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

        try:
            full_content = _stream_path(run_id).read_text(encoding="utf-8")
        except OSError:
            full_content = ""
        detected_format = _detect_format(full_content)

        _write_json(_status_path(run_id), {
            "status": "done",
            "latency_ms": int((time.time() - t0) * 1000),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "detected_format": detected_format,
            "default_field": _DEFAULT_FIELDS.get(detected_format, "") if detected_format else "",
        })

    except Exception:
        tb = traceback.format_exc()
        _write_json(_status_path(run_id), {"status": "error", "error": tb})
        logger.error("[minimax_m3] chat stream error:\n%s", tb)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class MiniMaxChatPanel(foo.Panel):
    """Modal panel for asking questions about the current image or video."""

    @property
    def config(self):
        return foo.PanelConfig(
            name="minimax_chat",
            label="Ask MiniMax-M3",
            surfaces="modal",
            icon="/assets/icon.svg",
            help_markdown=(
                "Ask free-form questions about the current image or video. "
                "Responses stream live from MiniMax-M3. Use the output hint to "
                "steer M3 toward boxes / keypoints / temporal JSON you can save "
                "as FiftyOne labels."
            ),
        )

    # ── Lifecycle hooks ──────────────────────────────────────────────────────

    def on_load(self, ctx):
        ctx.panel.set_state("api_key_missing", not has_api_key(ctx))
        ctx.panel.set_state("hint_templates", _HINT_TEMPLATES)
        self._sync_sample(ctx)

    def on_change_current_sample(self, ctx):
        self._sync_sample(ctx)

    def on_change_group_slice(self, ctx):
        self._sync_sample(ctx)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sync_sample(self, ctx) -> None:
        if not ctx.current_sample:
            return

        dataset = ctx.dataset
        sample = dataset[ctx.current_sample]
        gf = dataset.group_field

        resolved = sample
        filepath = sample.filepath
        sample_id = ctx.current_sample
        media_type = getattr(sample, "media_type", None) or "image"

        if gf and ctx.group_slice:
            group_elem = sample[gf]
            if group_elem and group_elem.name != ctx.group_slice:
                try:
                    slice_sample = (
                        dataset
                        .select_group_slices(ctx.group_slice)
                        .match(F(f"{gf}._id") == bson.ObjectId(group_elem.id))
                        .first()
                    )
                    if slice_sample is not None:
                        resolved = slice_sample
                        filepath = slice_sample.filepath
                        sample_id = slice_sample.id
                        media_type = getattr(slice_sample, "media_type", None) or "image"
                except Exception as exc:
                    logger.warning("[minimax_m3] group-slice lookup error: %s", exc)

        frame_rate: float | None = None
        meta = resolved.metadata
        if meta is not None:
            raw_fr = getattr(meta, "frame_rate", None)
            if raw_fr:
                frame_rate = float(raw_fr)

        ctx.panel.set_state("filepath", filepath)
        ctx.panel.set_state("sample_id", sample_id)
        ctx.panel.set_state("media_type", media_type)
        ctx.panel.set_state("frame_rate", frame_rate)

    # ── Panel methods (called from React via usePanelEvent) ──────────────────

    def ask(self, ctx) -> dict:
        """Start a streaming inference and return a run_id for React to poll.

        Parameters (via ctx.params): filepath, media_type, question, history,
        enable_thinking (bool -> adaptive/disabled), hint_format
        ("auto"|"box"|"point"|"temporal"), hint_text (the editable format
        instruction; overrides the default suffix), n_frames, image_max_side
        (image encode resolution; ``<= 0`` = native).
        """
        if not has_api_key(ctx):
            return {"error": "HF_TOKEN is not set."}

        filepath = ctx.params.get("filepath", "")
        media_type = ctx.params.get("media_type", "image")
        question = (ctx.params.get("question") or "").strip()
        history: list[dict] = ctx.params.get("history", [])
        enable_thinking = bool(ctx.params.get("enable_thinking", False))
        hint_format = ctx.params.get("hint_format", "auto")
        hint_text = (ctx.params.get("hint_text") or "").strip()
        n_frames = int(ctx.params.get("n_frames") or 8)
        image_max_side = int(ctx.params.get("image_max_side", DEFAULT_IMAGE_MAX_SIDE))

        if not question:
            return {"error": "Question cannot be empty."}
        if not filepath:
            return {"error": "No filepath provided."}

        # Steer M3's output shape by appending a format instruction to the new
        # question. Prefer the user-edited `hint_text`; fall back to the default
        # instruction for the selected format; "auto" appends nothing.
        question_to_send = question
        if hint_text:
            question_to_send = f"{question}\n\n{hint_text}"
        elif hint_format and hint_format != "auto" and hint_format in _HINT_TEMPLATES:
            question_to_send = f"{question} {_HINT_TEMPLATES[hint_format]}"

        media_parts = _build_media_parts(filepath, media_type, n_frames, image_max_side)

        # Full turn sequence is the prior history plus the new question. The
        # media is embedded once, in the first user message; every other turn is
        # plain text. The new question is itself a user turn, so the media always
        # lands somewhere even when history has no user turn.
        turns = [*history, {"role": "user", "content": question_to_send}]
        messages: list[dict[str, Any]] = []
        media_injected = False
        for turn in turns:
            if turn["role"] == "user" and not media_injected:
                messages.append({
                    "role": "user",
                    "content": [*media_parts, {"type": "text", "text": turn["content"]}],
                })
                media_injected = True
            else:
                messages.append({"role": turn["role"], "content": turn["content"]})

        thinking = "adaptive" if enable_thinking else "disabled"

        run_id = f"{ctx.current_sample or 'x'}_{int(time.time() * 1000)}"
        _clear_run(run_id)
        # Persist provenance so the (separate) convert call can stamp the saved
        # label with the prompt and image resolution this run actually used.
        _write_json(
            _meta_path(run_id),
            {"prompt": question_to_send, "image_max_side": image_max_side},
        )

        thread = threading.Thread(
            target=_run_stream_thread,
            kwargs=dict(
                api_key=get_api_key(ctx),
                messages=messages,
                thinking=thinking,
                run_id=run_id,
            ),
            daemon=True,
        )
        thread.start()
        return {"status": "started", "run_id": run_id}

    def get_stream_chunk(self, ctx) -> dict:
        run_id = ctx.params.get("run_id", "")
        cursor = int(ctx.params.get("cursor", 0))

        try:
            with open(_stream_path(run_id), "rb") as f:
                f.seek(cursor)
                new_bytes = f.read()
            new_cursor = cursor + len(new_bytes)
            new_text = new_bytes.decode("utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            new_text = ""
            new_cursor = cursor

        status = _read_json(_status_path(run_id)) or {}
        done = status.get("status") in ("done", "error")

        return {
            "text": new_text,
            "cursor": new_cursor,
            "done": done,
            "final_status": status if done else None,
        }

    def render(self, ctx):
        return types.Property(
            types.Object(),
            view=types.View(
                component="MiniMaxChatPanel",
                composite_view=True,
                ask=self.ask,
                get_stream_chunk=self.get_stream_chunk,
            ),
        )
