"""Single source of truth for MiniMax-M3 task definitions and prompt templates.

Defines the `Task` enum, the parser-format mapping, task-set frozensets, and
the default user-prompt templates. Unlike tag-grammar VLMs, M3 is steered
purely through the prompt: every grounded task asks for **ONLY JSON** in a
specific shape, and `minimax_parser` strips code fences and `json.loads` the
first JSON span.

Task -> media type -> output shape:

    Image-only tasks (single image_url call per image):
        DETECT    -> fo.Detections   (JSON [{"label","box":[x1,y1,x2,y2]}])
        KEYPOINTS -> fo.Keypoints    (JSON [{"label","point":[x,y]}])

    Video-only tasks (frame-sampled, one or more image_url parts):
        FRAME_DETECT -> per-frame fo.Detections via dense frame decomposition
                        (N image-mode API calls at a chosen stride).
        FIND_EVENT   -> fo.TemporalDetections, targeted moment search over the
                        sampled-frame strip. One API call per video.
        KEY_MOMENTS  -> fo.TemporalDetections, unconstrained event summary over
                        the sampled-frame strip. One API call per video.

    Shared tasks (image or video, one API call per sample):
        CAPTION_CONCISE  -> free text.
        CAPTION_DETAILED -> free text.
        CLASSIFY_SINGLE  -> fo.Classification w/ confidence (JSON).
        CLASSIFY_MULTI   -> fo.Classifications (JSON).
        VQA              -> free text (user-supplied question).

Coordinates: M3 emits NORMALIZED [0, 1] coordinates, so `minimax_parser` uses
a `COORD_SCALE` of 1.0.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Task(StrEnum):
    """Every task the plugin exposes.

    Members are plain strings (``Task.DETECT == "detect"``) so they round-trip
    cleanly through JSON ``ctx.params`` values while still giving us type-safe
    dispatch in ``match`` statements.
    """

    # Image-only grounding tasks (single image_url call per image)
    DETECT = "detect"
    KEYPOINTS = "keypoints"

    # Video-only tasks (frame-sampled)
    FRAME_DETECT = "frame_detect"
    FIND_EVENT = "find_event"
    KEY_MOMENTS = "key_moments"

    # Shared tasks (work on both image and video samples)
    CAPTION_CONCISE = "caption_concise"
    CAPTION_DETAILED = "caption_detailed"
    CLASSIFY_SINGLE = "classify_single"
    CLASSIFY_MULTI = "classify_multi"
    VQA = "vqa"


# ---------------------------------------------------------------------------
# Parser-format mapping -> dispatch key for minimax_parser.to_fiftyone().
# ---------------------------------------------------------------------------

TASK_TO_PARSER_FORMAT: Final[dict[Task, str]] = {
    Task.DETECT: "box",
    Task.KEYPOINTS: "point",
    Task.FRAME_DETECT: "box",
    Task.FIND_EVENT: "temporal",
    Task.KEY_MOMENTS: "temporal",
    Task.CAPTION_CONCISE: "caption",
    Task.CAPTION_DETAILED: "caption",
    Task.CLASSIFY_SINGLE: "classify_single",
    Task.CLASSIFY_MULTI: "classify_multi",
    Task.VQA: "vqa",
}


# Default M3 thinking mode per task. Grounded / structured tasks run with
# thinking off for clean JSON; temporal reasoning benefits from "adaptive".
# Valid values: "disabled", "adaptive", "enabled".
TASK_DEFAULT_THINKING: Final[dict[Task, str]] = {
    Task.DETECT: "disabled",
    Task.KEYPOINTS: "disabled",
    Task.FRAME_DETECT: "disabled",
    Task.FIND_EVENT: "adaptive",
    Task.KEY_MOMENTS: "adaptive",
    Task.CAPTION_CONCISE: "disabled",
    Task.CAPTION_DETAILED: "disabled",
    Task.CLASSIFY_SINGLE: "disabled",
    Task.CLASSIFY_MULTI: "disabled",
    Task.VQA: "adaptive",
}


# ---------------------------------------------------------------------------
# Task-set constants. Used by the operator (to build per-media-type form lists)
# and the model (to dispatch between image / video / dense predict paths).
# ---------------------------------------------------------------------------

# Single-shot image grounding; go through _predict_image (one image_url call).
TASKS_IMAGE_GROUNDING: Final[frozenset[Task]] = frozenset(
    {Task.DETECT, Task.KEYPOINTS}
)

# Video-only; not shown on image datasets.
TASKS_VIDEO_ONLY: Final[frozenset[Task]] = frozenset(
    {Task.FRAME_DETECT, Task.FIND_EVENT, Task.KEY_MOMENTS}
)

# Available on both image and video datasets.
TASKS_SHARED: Final[frozenset[Task]] = frozenset(
    {
        Task.CAPTION_CONCISE,
        Task.CAPTION_DETAILED,
        Task.CLASSIFY_SINGLE,
        Task.CLASSIFY_MULTI,
        Task.VQA,
    }
)

# Dense video path: decompose into per-frame image-mode requests.
TASKS_DENSE_FRAME_MODE: Final[frozenset[Task]] = frozenset({Task.FRAME_DETECT})

# Tasks that produce temporal output over the whole sampled-frame strip.
TASKS_TEMPORAL: Final[frozenset[Task]] = frozenset(
    {Task.FIND_EVENT, Task.KEY_MOMENTS}
)


# ---------------------------------------------------------------------------
# Default user-prompt templates. Every grounded task demands "ONLY JSON".
# ---------------------------------------------------------------------------

# JSON-shape suffixes appended to grounding prompts. M3 returns NORMALIZED
# [0, 1] coordinates, which the parser consumes directly (COORD_SCALE = 1.0).
_BOX_JSON_SHAPE: Final[str] = (
    'Return ONLY JSON (no prose): a list like '
    '[{"label": "dog", "box": [x1, y1, x2, y2]}] '
    "where box is NORMALIZED in [0, 1], (x1,y1)=top-left, (x2,y2)=bottom-right."
)
_POINT_JSON_SHAPE: Final[str] = (
    'Return ONLY JSON (no prose): a list like '
    '[{"label": str, "point": [x, y]}] with NORMALIZED [0, 1] coordinates.'
)
_TEMPORAL_JSON_SHAPE: Final[str] = (
    'Return ONLY JSON (no prose): a list like '
    '[{"label": str, "start": <seconds>, "end": <seconds>}].'
)
_CLASSIFY_JSON_SHAPE: Final[str] = (
    'Return ONLY JSON (no prose) with this shape: '
    '{"classifications": [{"label": str, "confidence": <0..1 float>}]}.'
)


def _detect_template(target: str | None, media_type: str) -> str:
    subject = f"every {target}" if target else "the main objects"
    return f"Detect {subject} in this {media_type}. {_BOX_JSON_SHAPE}"


def _keypoints_template(target: str | None, media_type: str) -> str:
    subject = f"each {target}" if target else "notable points (object centers / distinctive parts)"
    return f"Point to {subject} in this {media_type}. {_POINT_JSON_SHAPE}"


_FIND_EVENT_TEMPLATE: Final[str] = (
    "These frames are sampled from one video; each is labeled with its "
    "timestamp in seconds. Identify when the following event occurs: {target}. "
    + _TEMPORAL_JSON_SHAPE
)

_KEY_MOMENTS_TEMPLATE: Final[str] = (
    "These frames are sampled from one video; each is labeled with its "
    "timestamp in seconds. Identify distinct events / activities and WHEN they "
    "occur. " + _TEMPORAL_JSON_SHAPE
)

_CAPTION_CONCISE_TEMPLATE: Final[str] = (
    "Provide a concise, human-friendly caption for this {media_type} "
    "(one sentence). Respond with the caption text only."
)
_CAPTION_DETAILED_TEMPLATE: Final[str] = (
    "Provide a detailed caption describing key objects, relationships, and "
    "context in this {media_type}, including any visible text or signage. "
    "Respond with the caption text only."
)

_CLASSIFY_SINGLE_TEMPLATE: Final[str] = (
    "Classify the primary {target} shown in this {media_type}. "
    + _CLASSIFY_JSON_SHAPE
)
_CLASSIFY_MULTI_TEMPLATE: Final[str] = (
    "List every relevant category that describes this {media_type} -- "
    "{target}. " + _CLASSIFY_JSON_SHAPE
)


def default_user_prompt(
    task: Task,
    target: str | None = None,
    *,
    media_type: str = "image",
) -> str:
    """Return the default user prompt for ``task``, formatted with ``target``.

    Args:
        task: The task being run.
        target: Semantics vary by task:
            - DETECT / KEYPOINTS / FRAME_DETECT: optional class restriction;
              falls back to "main objects" / "notable points" when empty.
            - FIND_EVENT / VQA: required; raises ``ValueError`` when empty.
            - CLASSIFY_SINGLE / CLASSIFY_MULTI: optional aspect hint
              (e.g. ``"scene type"``); falls back to a sensible default.
            - CAPTION_* / KEY_MOMENTS: ignored.
        media_type: ``"image"`` or ``"video"``. Substituted into templates so
            prompts read naturally for both.
    """
    match task:
        case Task.DETECT | Task.FRAME_DETECT:
            return _detect_template(target, media_type)

        case Task.KEYPOINTS:
            return _keypoints_template(target, media_type)

        case Task.FIND_EVENT:
            if not target:
                raise ValueError(
                    "task=find_event requires a non-empty target "
                    "(event description, e.g. 'a pedestrian crosses the street')"
                )
            return _FIND_EVENT_TEMPLATE.format(target=target)

        case Task.KEY_MOMENTS:
            return _KEY_MOMENTS_TEMPLATE

        case Task.CAPTION_CONCISE:
            return _CAPTION_CONCISE_TEMPLATE.format(media_type=media_type)

        case Task.CAPTION_DETAILED:
            return _CAPTION_DETAILED_TEMPLATE.format(media_type=media_type)

        case Task.CLASSIFY_SINGLE:
            aspect = target or "category"
            return _CLASSIFY_SINGLE_TEMPLATE.format(target=aspect, media_type=media_type)

        case Task.CLASSIFY_MULTI:
            aspects = target or (
                "scene type, objects present, and any other salient attributes"
            )
            return _CLASSIFY_MULTI_TEMPLATE.format(target=aspects, media_type=media_type)

        case Task.VQA:
            if not target:
                raise ValueError(
                    "task=vqa requires a non-empty target (the question to ask)"
                )
            return target

        case _:
            raise ValueError(f"unsupported task: {task!r}")
