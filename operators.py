"""FiftyOne operator for the MiniMax-M3 plugin.

One operator: `RunMiniMax`. Accessible from the operator browser (backtick) or
the "MiniMax-M3" grid-action button. The form is *conditional* -- the mode you
pick at the top drives which fields below appear and what `execute` does.

The available modes depend on ``dataset.media_type``:

    Image datasets:
        Semantic Search  -- score each sample yes/no against a free-text query.
        Bootstrap Labels -- single-shot grounding (detect / keypoints) plus
                            caption, classify, and VQA.

    Video datasets:
        Event Search     -- find targeted moments over a sampled-frame strip;
                            writes fo.TemporalDetections and switches the App
                            into a clips view of matches.
        Semantic Search  -- same as above (scores the sampled-frame strip).
        Bootstrap Labels -- per-frame detection (FRAME_DETECT), unconstrained
                            temporal events (KEY_MOMENTS), plus caption,
                            classify, VQA.

Mixed-media datasets are not supported and surface an error in the form.
Failures bubble up untouched.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone import ViewField as F
from fiftyone.operators import types

from ._shared import (
    PLUGIN_VERSION,
    get_api_key,
    has_api_key,
    make_run_key,
    notify,
    task_supports_per_frame,
)
from .chat_panel import save_stream_as_label
from .minimax_model import DEFAULT_MODEL_NAME, DEFAULT_N_FRAMES, MiniMaxModel
from .minimax_parser import write_per_frame_labels
from .prompts import Task, default_user_prompt

logger = logging.getLogger("minimax_m3")


# ---------------------------------------------------------------------------
# Mode + task taxonomy. Strings are user-facing AND ctx.params values.
# ---------------------------------------------------------------------------

MODE_EVENT_SEARCH: str = "event_search"
MODE_SEMANTIC_SEARCH: str = "semantic_search"
MODE_BOOTSTRAP: str = "bootstrap"

MODE_LABELS: dict[str, str] = {
    MODE_EVENT_SEARCH: "Event Search",
    MODE_SEMANTIC_SEARCH: "Semantic Search",
    MODE_BOOTSTRAP: "Bootstrap Labels",
}

MODE_CHOICE_DESCRIPTIONS: dict[str, str] = {
    MODE_EVENT_SEARCH: "Find moments matching a free-text event description. (Video only)",
    MODE_SEMANTIC_SEARCH: "Score each sample yes/no against a free-text query.",
    MODE_BOOTSTRAP: "Bootstrap annotation labels across the target view.",
}

MODE_LONG_DESCRIPTIONS: dict[str, str] = {
    MODE_EVENT_SEARCH: (
        "**Event Search**  \n"
        "Samples frames from each video, labels them with timestamps, and asks M3 when the event "
        "you describe occurs. For every match it writes a `fo.TemporalDetection` "
        "(with `support=[start_frame, end_frame]` plus raw `t_start_seconds` / `t_end_seconds`).  \n\n"
        "After the run, the view switches to a clips view with one row per detected event."
    ),
    MODE_SEMANTIC_SEARCH: (
        "**Semantic Search**  \n"
        "Scores each sample as a yes/no answer with a confidence score, written as "
        "`fo.Classification(label, confidence)` on the sample.  \n"
        "After the run, the view filters to samples where `label == \"yes\"` and `confidence >= threshold`."
    ),
    MODE_BOOTSTRAP: (
        "**Bootstrap Labels**  \n"
        "Pick a task below to write detections, keypoints, captions, classifications, or VQA "
        "answers across the target view. Available tasks and output placement (per-frame vs. "
        "sample-level) depend on the dataset's media type."
    ),
}


# Bootstrap task lists, split by media type. Display order = priority order.
_BOOTSTRAP_VIDEO_TASKS: list[Task] = [
    Task.FRAME_DETECT,   # per-frame detection (one API call per sampled frame)
    Task.KEY_MOMENTS,    # unconstrained temporal event extraction
    Task.CLASSIFY_SINGLE,
    Task.CLASSIFY_MULTI,
    Task.VQA,
    Task.CAPTION_CONCISE,
    Task.CAPTION_DETAILED,
]

_BOOTSTRAP_IMAGE_TASKS: list[Task] = [
    Task.DETECT,
    Task.KEYPOINTS,
    Task.CLASSIFY_SINGLE,
    Task.CLASSIFY_MULTI,
    Task.VQA,
    Task.CAPTION_CONCISE,
    Task.CAPTION_DETAILED,
]


_BOOTSTRAP_TASK_SHORT: dict[Task, str] = {
    Task.DETECT: "Detect and locate objects with bounding boxes. One API call per image.",
    Task.KEYPOINTS: "Point to objects by class. One API call per image.",
    Task.FRAME_DETECT: "Per-frame object detection on sampled frames. N API calls per video.",
    Task.KEY_MOMENTS: "Identify all key events in each video. One API call per video.",
    Task.CLASSIFY_SINGLE: "One-aspect classification with confidence.",
    Task.CLASSIFY_MULTI: "Multi-aspect classification with confidences.",
    Task.VQA: "Free-form Q&A; plain-text answer per sample.",
    Task.CAPTION_CONCISE: "One-sentence caption per sample.",
    Task.CAPTION_DETAILED: "Paragraph caption (picks up signage / fine detail).",
}


_BOOTSTRAP_TASK_LONG: dict[Task, str] = {
    Task.DETECT: (
        "**Detect**  \n"
        "Single-shot box grounding on images. Sends one `image_url` request per sample and writes "
        "`fo.Detections` at the sample level. M3 returns normalized `[x1,y1,x2,y2]` boxes as JSON."
    ),
    Task.KEYPOINTS: (
        "**Keypoints**  \n"
        "Single-shot point grounding on images. Sends one `image_url` request per sample and writes "
        "`fo.Keypoints` at the sample level. When the model returns a box instead of a point, the "
        "parser substitutes the box center."
    ),
    Task.FRAME_DETECT: (
        "**Frame Detection**  \n"
        "Samples N frames from each video and detects objects on each one. Writes `fo.Detections` to "
        "`sample.frames[i][field]`. **No cross-frame instance IDs** -- run a downstream tracker for "
        "ID continuity. Cost scales with the frame count you sample.  \n\n"
        "Requires `metadata.frame_rate` (run `dataset.compute_metadata()` once) so per-frame boxes "
        "map to the correct frame index."
    ),
    Task.KEY_MOMENTS: (
        "**Key Moments**  \n"
        "Sends the sampled-frame strip in one request and asks M3 to freely identify every noteworthy "
        "event, writing `fo.TemporalDetections` at the sample level. No target required.  \n\n"
        "For targeted event search, use the **Event Search** mode instead."
    ),
    Task.CLASSIFY_SINGLE: (
        "**Classify (single label)**  \n"
        "Assign one classification label per sample with a confidence score. Produces "
        "`fo.Classification(label, confidence)` at the sample level."
    ),
    Task.CLASSIFY_MULTI: (
        "**Classify (multi-label)**  \n"
        "Assign multiple classification labels per sample, each with a confidence. Produces "
        "`fo.Classifications` at the sample level."
    ),
    Task.VQA: (
        "**Visual Q&A**  \n"
        "Ask a free-text question about each sample. Produces a plain-text answer written to the "
        "output field as a string."
    ),
    Task.CAPTION_CONCISE: (
        "**Caption (concise)**  \n"
        "One-sentence caption per sample. Pairs well with FiftyOne's brain text-similarity methods."
    ),
    Task.CAPTION_DETAILED: (
        "**Caption (detailed)**  \n"
        "Paragraph-length description per sample, picking up visible text (signage, plates) and "
        "fine-grained context."
    ),
}


# Per-task spec for the conditional ``target`` input field.
_TARGET_FIELD_SPEC: dict[Task, dict[str, Any]] = {
    Task.DETECT: {
        "label": "Classes",
        "description": "One or more object classes to detect (e.g. 'car', 'person'). Leave empty to detect the main objects.",
        "required": False,
    },
    Task.KEYPOINTS: {
        "label": "Classes",
        "description": "One or more object classes to point at (e.g. 'pedestrian'). Leave empty for notable points.",
        "required": False,
    },
    Task.FRAME_DETECT: {
        "label": "Classes",
        "description": "One or more object classes to detect per frame. Leave empty to detect the main objects.",
        "required": False,
    },
    Task.CLASSIFY_SINGLE: {
        "label": "Aspect to classify (optional)",
        "description": "What to classify (e.g. 'scene type', 'weather').",
        "required": False,
    },
    Task.CLASSIFY_MULTI: {
        "label": "Aspects to list (optional)",
        "description": "Aspects to enumerate (e.g. 'scene', 'vehicles', 'lighting').",
        "required": False,
    },
    Task.VQA: {
        "label": "Question",
        "description": "Free-text question to ask about each sample.",
        "required": True,
    },
}


_BOOTSTRAP_DEFAULT_FIELDS: dict[Task, str] = {
    Task.DETECT: "m3_detections",
    Task.KEYPOINTS: "m3_keypoints",
    Task.FRAME_DETECT: "m3_detections",
    Task.KEY_MOMENTS: "m3_key_moments",
    Task.CLASSIFY_SINGLE: "m3_class",
    Task.CLASSIFY_MULTI: "m3_class",
    Task.VQA: "m3_answer",
    Task.CAPTION_CONCISE: "m3_caption",
    Task.CAPTION_DETAILED: "m3_caption",
}


# Thinking-mode dropdown. "auto" defers to each task's recommended default.
_THINKING_CHOICES: list[tuple[str, str]] = [
    ("auto", "Auto (recommended per task)"),
    ("disabled", "Disabled (fast; best for structured output)"),
    ("adaptive", "Adaptive (reasoning when helpful)"),
    ("enabled", "Enabled (always reason)"),
]

_SEMANTIC_SEARCH_TEMPLATE: str = (
    'Does this sample match this description: "{query}"? '
    'Return ONLY JSON: {{"label": "yes" or "no", "confidence": <0..1 float>}}.'
)
_SEMANTIC_DEFAULT_THRESHOLD: float = 0.7

_PROGRESS_TICK_INTERVAL_S: float = 0.5
_SANITIZE_FIELD_RE = re.compile(r"[^a-zA-Z0-9_]+")


_METADATA_FIX_INSTRUCTIONS_MD: str = """\
**To fix this:**

- Press `Cancel` to close this operator.
- Press the backtick key (**`**) to open the operator browser.
- Search for and run the built-in **`compute_metadata`** operator on this dataset.
- Re-open this operator.

Metadata is cached on each sample after the first call, so this is a one-time cost per dataset.
""".strip()


# ---------------------------------------------------------------------------
# The operator.
# ---------------------------------------------------------------------------


class RunMiniMax(foo.Operator):
    """Run a MiniMax-M3 task on an image or video dataset, with a mode-driven form."""

    version: str = PLUGIN_VERSION

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="run_minimax",
            label="MiniMax-M3: run task",
            description=(
                "Run a MiniMax-M3 task across a dataset. Pick a mode for event "
                "search, semantic search, or label bootstrapping; the form "
                "updates based on your selection."
            ),
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            icon="/assets/icon.svg",
        )

    def resolve_placement(self, ctx: Any) -> Any:
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="MiniMax-M3",
                icon="/assets/icon.svg",
                prompt=True,
            ),
        )

    # ----------------------------------------------------------------- inputs

    def resolve_input(self, ctx: Any) -> Any:
        inputs = types.Object()

        # 0. API-key gate.
        if not has_api_key(ctx):
            inputs.message(
                "no_key",
                label=(
                    "HF_TOKEN is not set. Export a Hugging Face token with "
                    "Inference Providers access before launching FiftyOne "
                    "(see the plugin README)."
                ),
            )
            return types.Property(inputs)

        # 1. Media-type gate.
        media_type = _resolve_media_type(ctx)
        if media_type == "mixed":
            inputs.view(
                "mixed_media_error",
                types.Error(
                    label="Mixed-media datasets are not supported",
                    description=(
                        "This dataset contains both images and videos. "
                        "Filter the view to a single media type before running MiniMax-M3."
                    ),
                    space=12,
                ),
            )
            return types.Property(inputs, view=types.View(label="MiniMax-M3"))

        # 2. Metadata pre-flight (only per-frame video output needs frame_rate).
        mode = ctx.params.get(
            "mode", MODE_EVENT_SEARCH if media_type == "video" else MODE_SEMANTIC_SEARCH
        )
        if _mode_requires_frame_rate(ctx, mode):
            view = ctx.target_view()
            n_missing = _count_missing_metadata(view)
            if n_missing > 0:
                _render_missing_metadata_error(inputs, n_missing, len(view))
                return types.Property(inputs, view=types.View(label="MiniMax-M3"))

        # 3. Target view selection.
        inputs.view_target(ctx)

        # 4. Mode selector.
        available_modes = (
            [MODE_EVENT_SEARCH, MODE_SEMANTIC_SEARCH, MODE_BOOTSTRAP]
            if media_type == "video"
            else [MODE_SEMANTIC_SEARCH, MODE_BOOTSTRAP]
        )
        default_mode = available_modes[0]
        if mode not in available_modes:
            mode = default_mode

        mode_choices = types.DropdownView()
        for mode_value in available_modes:
            mode_choices.add_choice(
                mode_value,
                label=MODE_LABELS[mode_value],
                description=MODE_CHOICE_DESCRIPTIONS[mode_value],
            )
        inputs.enum(
            "mode",
            mode_choices.values(),
            default=default_mode,
            required=True,
            label="Mode",
            view=mode_choices,
        )
        inputs.str(
            "_mode_description",
            view=types.MarkdownView(read_only=True),
            default=MODE_LONG_DESCRIPTIONS.get(mode, ""),
        )

        # 5. Mode-specific form fields.
        match mode:
            case "event_search":
                _render_event_search_inputs(inputs, ctx)
            case "semantic_search":
                _render_semantic_search_inputs(inputs, ctx, media_type=media_type)
            case "bootstrap":
                _render_bootstrap_inputs(inputs, ctx, media_type=media_type)
            case _:
                raise ValueError(f"unexpected mode: {mode!r}")

        # 6. Thinking-mode dropdown (M3 reasoning control).
        thinking_choices = types.DropdownView()
        for value, label in _THINKING_CHOICES:
            thinking_choices.add_choice(value, label=label)
        inputs.enum(
            "thinking_mode",
            thinking_choices.values(),
            default="auto",
            required=True,
            label="Thinking mode (advanced)",
            description=(
                "M3's reasoning control. 'Auto' uses the recommended mode per task "
                "(off for grounding/structured output, adaptive for temporal/VQA)."
            ),
            view=thinking_choices,
        )

        # 7. Execution-mode checkbox.
        _render_execution_mode(inputs, ctx)

        return types.Property(inputs, view=types.View(label="MiniMax-M3"))

    def resolve_delegation(self, ctx: Any) -> bool:
        return bool(ctx.params.get("delegate", False))

    # ------------------------------------------------------------- execution

    async def execute(self, ctx: Any) -> AsyncIterator[dict[str, Any]]:
        mode = ctx.params["mode"]
        with foo.ProgressHandler(ctx, logger=logger), _capture_logs() as logbuf:
            match mode:
                case "event_search":
                    async for msg in _execute_event_search(ctx, self.version, logbuf):
                        yield msg
                case "semantic_search":
                    async for msg in _execute_semantic_search(ctx, self.version, logbuf):
                        yield msg
                case "bootstrap":
                    async for msg in _execute_bootstrap(ctx, self.version, logbuf):
                        yield msg
                case _:
                    raise ValueError(f"unexpected mode in execute: {mode!r}")

    def resolve_output(self, ctx: Any) -> Any:
        outputs = types.Object()
        outputs.str("mode", label="Mode")
        outputs.str("summary", label="Summary", view=types.MarkdownView())
        outputs.str("run_key", label="Custom run key")
        outputs.float("elapsed_seconds", label="Elapsed (s)")
        return types.Property(outputs, view=types.View(label="MiniMax-M3 run results"))


# ---------------------------------------------------------------------------
# Live-log -> progress-modal plumbing.
# ---------------------------------------------------------------------------


class _ProgressLogBuffer(logging.Handler):
    """Bounded logging handler that buffers formatted records for later draining."""

    def __init__(self, capacity: int = 256, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.records: collections.deque[str] = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            msg = record.getMessage()
        self.records.append(msg)


@contextlib.contextmanager
def _capture_logs() -> Iterator[_ProgressLogBuffer]:
    buffer = _ProgressLogBuffer()
    minimax_logger = logging.getLogger("minimax_m3")
    buffer.setLevel(minimax_logger.getEffectiveLevel())
    minimax_logger.addHandler(buffer)
    try:
        yield buffer
    finally:
        minimax_logger.removeHandler(buffer)


def _drain_log_buffer(ctx: Any, buffer: _ProgressLogBuffer) -> Iterator[Any]:
    while buffer.records:
        msg = buffer.records.popleft()
        yield ctx.ops.set_progress(label=msg)


async def _predict_with_progress(
    ctx: Any,
    view: Any,
    *,
    model: MiniMaxModel,
    mode_label: str,
    logbuf: _ProgressLogBuffer,
    started_at: float,
    on_sample: Callable[[fo.Sample, Any, int], str],
) -> AsyncIterator[Any]:
    total = len(view)
    for i, sample in enumerate(view):
        predict_task = asyncio.create_task(asyncio.to_thread(_predict, model, sample))
        async for tick in _tick_until_done(
            predict_task,
            ctx,
            mode_label=mode_label,
            done_so_far=i,
            total=total,
            started_at=started_at,
            model=model,
            logbuf=logbuf,
        ):
            yield tick

        label = predict_task.result()
        status = on_sample(sample, label, i)

        for msg in _drain_log_buffer(ctx, logbuf):
            yield msg
        yield _yield_progress(
            ctx,
            mode_label=mode_label,
            status=status,
            done=i + 1,
            total=total,
            model=model,
            started_at=started_at,
        )


# ---------------------------------------------------------------------------
# Per-mode form renderers.
# ---------------------------------------------------------------------------


def _render_frame_sampling(inputs: Any) -> None:
    """``Frames to sample`` control, shown for any frame-sampled video task."""
    inputs.int(
        "n_frames",
        label="Frames to sample",
        description=(
            "Evenly sampled frames sent to M3 per video. More frames = better "
            "coverage but larger, slower, costlier requests. Default 8."
        ),
        default=DEFAULT_N_FRAMES,
        min=1,
    )


def _render_event_search_inputs(inputs: Any, ctx: Any) -> None:
    inputs.str(
        "event_query",
        label="Event description",
        description=(
            "Natural-language description of the event to find "
            "(e.g. 'a pedestrian crosses in front of a moving vehicle')."
        ),
        required=True,
    )

    query = ctx.params.get("event_query") or ""
    preview_target = query or "<event description>"
    preview = default_user_prompt(Task.FIND_EVENT, preview_target)
    inputs.str(
        "_prompt_preview",
        view=types.MarkdownView(read_only=True),
        default=f"**Prompt:** `{preview}`",
    )

    _render_frame_sampling(inputs)

    default_field = _derive_field_name("event", query)
    inputs.str(
        "event_field",
        label="Output field",
        description="Where to write the TemporalDetections (per video).",
        default=default_field,
        required=True,
    )


def _render_semantic_search_inputs(inputs: Any, ctx: Any, *, media_type: str) -> None:
    inputs.str(
        "semantic_query",
        label="Search query",
        description=(
            "Natural-language description of the samples you want "
            "(e.g. 'images with motorcycles', 'aerial coastline shots')."
        ),
        required=True,
    )

    query = ctx.params.get("semantic_query") or ""
    preview_query = query or "<search query>"
    preview = _SEMANTIC_SEARCH_TEMPLATE.format(query=preview_query)
    inputs.str(
        "_prompt_preview",
        view=types.MarkdownView(read_only=True),
        default=f"**Prompt:** `{preview}`",
    )

    if media_type == "video":
        _render_frame_sampling(inputs)

    default_field = _derive_field_name("match", query)
    inputs.str(
        "semantic_field",
        label="Output field",
        description="Where to write the per-sample Classification.",
        default=default_field,
        required=True,
    )

    inputs.float(
        "semantic_threshold",
        label="Confidence threshold",
        description=(
            "After scoring, only samples with `label == \"yes\"` AND "
            "`confidence >= threshold` will be kept in the filtered view."
        ),
        default=_SEMANTIC_DEFAULT_THRESHOLD,
    )


def _render_bootstrap_inputs(inputs: Any, ctx: Any, *, media_type: str) -> None:
    task_list = _BOOTSTRAP_VIDEO_TASKS if media_type == "video" else _BOOTSTRAP_IMAGE_TASKS
    default_task = task_list[0].value

    task_choices = types.DropdownView()
    for task in task_list:
        task_choices.add_choice(
            task.value,
            label=_humanize_task(task),
            description=_BOOTSTRAP_TASK_SHORT[task],
        )

    inputs.enum(
        "bootstrap_task",
        task_choices.values(),
        default=default_task,
        required=True,
        label="Task",
        view=task_choices,
    )

    selected = ctx.params.get("bootstrap_task", default_task)
    try:
        task = Task(selected)
    except ValueError:
        task = task_list[0]

    inputs.str(
        "_bootstrap_task_description",
        view=types.MarkdownView(read_only=True),
        default=_BOOTSTRAP_TASK_LONG.get(task, ""),
    )

    spec = _TARGET_FIELD_SPEC.get(task)
    prompt_source = "classes"

    if spec is not None:
        source_view = types.RadioView()
        source_view.add_choice("classes", label=f"Enter {spec['label'].lower()}")
        source_view.add_choice("field", label="Read from sample field")
        inputs.enum(
            "bootstrap_prompt_source",
            source_view.values(),
            default="classes",
            required=True,
            label="Target source",
            view=source_view,
        )
        prompt_source = ctx.params.get("bootstrap_prompt_source", "classes")

        if prompt_source == "classes":
            class_view = types.ListView()
            inputs.list(
                "bootstrap_target",
                types.String(),
                label=spec["label"],
                required=spec.get("required", False),
                description=spec.get("description", ""),
                view=class_view,
            )
        else:
            inputs.str(
                "bootstrap_prompt_prefix",
                label="Prompt prefix (optional)",
                description=(
                    "Text prepended to the field value to form the full prompt "
                    "(e.g. for a VQA question: ``What is the ``). Leave blank to "
                    "use the field value as the entire prompt."
                ),
                required=False,
            )
            str_field_names = sorted(
                name for name, fld in ctx.dataset.get_field_schema().items()
                if isinstance(fld, fo.StringField) and not name.startswith("_")
                and name not in ("id", "filepath")
            )
            if str_field_names:
                field_choices = types.DropdownView()
                for fn in str_field_names:
                    field_choices.add_choice(fn, label=fn)
                inputs.enum(
                    "bootstrap_prompt_field",
                    field_choices.values(),
                    default=str_field_names[0],
                    required=True,
                    label="Prompt field",
                    description=(
                        "String field whose value is used as (or appended to the "
                        "prefix to form) the prompt for each sample."
                    ),
                    view=field_choices,
                )
            else:
                inputs.str(
                    "_no_string_fields_notice",
                    view=types.MarkdownView(read_only=True),
                    default=(
                        "> **No string fields found.** Add a string field to your "
                        "dataset first (e.g. `dataset.add_sample_field('my_prompt', "
                        "fo.StringField())`)."
                    ),
                )

    # Live prompt preview.
    if prompt_source == "field":
        _prompt_prefix = (ctx.params.get("bootstrap_prompt_prefix") or "").strip()
        _prompt_field = ctx.params.get("bootstrap_prompt_field") or ""
        preview = f"{_prompt_prefix}<{_prompt_field}>" if _prompt_prefix else f"<{_prompt_field}>"
        inputs.str(
            "_prompt_preview",
            view=types.MarkdownView(read_only=True),
            default=f"**Prompt (per sample):** `{preview}`",
        )
    else:
        _raw_target = ctx.params.get("bootstrap_target") or []
        if isinstance(_raw_target, list):
            target_str = ", ".join(t.strip() for t in _raw_target if t and t.strip())
        else:
            target_str = str(_raw_target).strip()
        placeholder = f"<{spec['label'].lower()}>" if spec else None
        preview_target = target_str or placeholder
        try:
            preview = default_user_prompt(task, preview_target, media_type=media_type)
            inputs.str(
                "_prompt_preview",
                view=types.MarkdownView(read_only=True),
                default=f"**Prompt:** `{preview}`",
            )
        except ValueError:
            pass

    # Frame-sampling controls for any video task (all video tasks frame-sample).
    if media_type == "video":
        _render_frame_sampling(inputs)
        if task_supports_per_frame(task):
            _render_dense_cost_preview(inputs, ctx)

    inputs.str(
        "bootstrap_field",
        label="Output field",
        description="Where to write the labels.",
        default=_BOOTSTRAP_DEFAULT_FIELDS[task],
        required=True,
    )


def _render_dense_cost_preview(inputs: Any, ctx: Any) -> None:
    n_frames = max(1, int(ctx.params.get("n_frames") or DEFAULT_N_FRAMES))
    view = ctx.target_view()
    n_videos = len(view)
    total = n_frames * n_videos
    preview = (
        f"**Cost preview**: ~{n_frames} API call(s) per video across "
        f"**{n_videos}** sample(s) -> **~{total} total call(s)**."
    )
    inputs.str(
        "_dense_cost_preview",
        view=types.MarkdownView(read_only=True, space=12),
        default=preview,
    )


def _humanize_task(task: Task) -> str:
    return task.value.replace("_", " ").capitalize()


def _render_missing_metadata_error(inputs: Any, n_missing: int, n_total: int) -> None:
    inputs.view(
        "metadata_missing_error",
        types.Error(
            label="Compute video metadata before running MiniMax-M3",
            description=(
                f"{n_missing} of {n_total} samples in the target view are "
                f"missing the `metadata` field, which is needed to map "
                f"per-frame model output back to frames."
            ),
            space=12,
        ),
    )
    inputs.str(
        "_metadata_fix_instructions",
        view=types.MarkdownView(read_only=True, space=12),
        default=_METADATA_FIX_INSTRUCTIONS_MD,
    )


def _render_execution_mode(inputs: Any, ctx: Any) -> None:
    delegate = bool(ctx.params.get("delegate", False))
    description = (
        "Uncheck this box to run immediately in the foreground."
        if delegate
        else "Check this box to delegate this run to a background queue."
    )
    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )
    if delegate:
        inputs.view(
            "delegate_notice",
            types.Notice(
                label=(
                    "Delegated execution requires a FiftyOne delegated-operation "
                    "service running in this environment."
                )
            ),
        )


# ---------------------------------------------------------------------------
# Per-mode execute() bodies.
# ---------------------------------------------------------------------------


@dataclass
class _RunOutcome:
    """The mode-specific result of a run, produced after the predict loop.

    Carries everything the shared runner needs to finish: the custom-run
    ``summary`` payload, the Markdown ``summary_md`` for the output panel, the
    App toast (``notify_message`` / ``notify_variant``), the final progress
    ``status`` line, and any ``view_ops`` (``ctx.ops.*`` results) to yield.
    """

    summary: dict[str, Any]
    summary_md: str
    notify_message: str
    final_status: str
    notify_variant: str = "success"
    view_ops: list[Any] = dataclass_field(default_factory=list)


def _usage_summary(model: MiniMaxModel, elapsed_s: float) -> dict[str, Any]:
    """The token / call / elapsed fields shared by every mode's run summary."""
    usage = model.usage_totals
    return {
        "elapsed_seconds": round(elapsed_s, 2),
        "api_calls": usage["calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }


async def _run_mode(
    ctx: Any,
    version: str,
    logbuf: _ProgressLogBuffer,
    *,
    view: Any,
    mode_value: str,
    mode_label: str,
    operation: str,
    start_notify: str,
    start_status: str,
    model: MiniMaxModel,
    on_sample: Callable[[fo.Sample, Any, int], str],
    finalize: Callable[[float], _RunOutcome],
) -> AsyncIterator[dict[str, Any]]:
    """Drive the shared per-mode run lifecycle.

    Handles the boilerplate every mode shares -- start toast / progress, log
    draining, usage reset, the predict-with-progress loop, custom-run
    registration, the success toast, view ops, and the final progress + result.
    The mode supplies its ``on_sample`` writeback and a ``finalize`` callback
    that computes the post-run ``_RunOutcome``.
    """
    total = len(view)
    notify(ctx, start_notify)
    yield _yield_progress(ctx, mode_label=mode_label, status=start_status, done=0, total=total)

    for msg in _drain_log_buffer(ctx, logbuf):
        yield msg
    model.reset_usage_totals()

    started_at = time.perf_counter()
    async for msg in _predict_with_progress(
        ctx, view, model=model, mode_label=mode_label, logbuf=logbuf,
        started_at=started_at, on_sample=on_sample,
    ):
        yield msg

    elapsed_s = time.perf_counter() - started_at
    outcome = finalize(elapsed_s)

    run_key = _register_custom_run(
        ctx, version=version, operation=operation, summary=outcome.summary
    )

    notify(ctx, outcome.notify_message, variant=outcome.notify_variant)
    for op in outcome.view_ops:
        yield op
    yield _yield_progress(
        ctx, mode_label=mode_label, status=outcome.final_status,
        done=total, total=total, model=model, started_at=started_at,
    )

    yield {
        "mode": mode_value,
        "summary": outcome.summary_md,
        "run_key": run_key,
        "elapsed_seconds": round(elapsed_s, 2),
    }


async def _execute_event_search(
    ctx: Any, version: str, logbuf: _ProgressLogBuffer
) -> AsyncIterator[dict[str, Any]]:
    query = (ctx.params.get("event_query") or "").strip()
    if not query:
        raise ValueError("Event Search requires a non-empty event description.")
    field = (ctx.params.get("event_field") or _derive_field_name("event", query)).strip()
    if not field:
        raise ValueError("Event Search requires a non-empty output field name.")

    view = ctx.target_view()
    total = len(view)
    model = _build_model(ctx, task=Task.FIND_EVENT, target=query)
    counters = {"clips": 0, "matched": 0}

    def on_sample(sample: fo.Sample, label: Any, _i: int) -> str:
        sample[field] = label
        sample.save()
        if label is not None and getattr(label, "detections", None):
            counters["clips"] += len(label.detections)
            counters["matched"] += 1
        return f"{counters['matched']} match(es), {counters['clips']} clip(s)"

    def finalize(elapsed_s: float) -> _RunOutcome:
        clips_view = ctx.dataset.to_clips(field)
        n_matched_clips = len(clips_view)
        n_matched_samples = len(
            ctx.dataset.match(F(field) != None).match(  # noqa: E711
                F(f"{field}.detections").length() > 0
            )
        )
        usage = model.usage_totals
        summary_md = (
            f"**Event Search** completed.  \n"
            f"- Query: `{query}`  \n"
            f"- Matched: **{n_matched_samples}** / {total} videos "
            f"(**{n_matched_clips}** event clip(s))  \n"
            f"- Output field: `{field}`  \n"
            f"- Elapsed: **{_format_elapsed(elapsed_s)}** across **{usage['calls']}** API call(s)  \n"
            f"- Tokens: **{usage['prompt_tokens']:,}** in / **{usage['completion_tokens']:,}** out  \n"
            f"- View auto-switched to a clips view, one row per detected event."
        )
        return _RunOutcome(
            summary={
                "mode": MODE_EVENT_SEARCH,
                "query": query,
                "field": field,
                "n_total": total,
                "n_matched_samples": n_matched_samples,
                "n_matched_clips": n_matched_clips,
                "n_temporal_detections": counters["clips"],
                **_usage_summary(model, elapsed_s),
            },
            summary_md=summary_md,
            notify_message=(
                f"Event Search done: {n_matched_clips} clip(s) across "
                f"{n_matched_samples}/{total} video(s)."
            ),
            final_status=(
                f"Complete -- {n_matched_clips} clip(s) across "
                f"{n_matched_samples}/{total} video(s)."
            ),
            view_ops=[ctx.ops.set_view(view=clips_view), ctx.ops.reload_samples()],
        )

    async for msg in _run_mode(
        ctx, version, logbuf,
        view=view, mode_value=MODE_EVENT_SEARCH, mode_label="Event Search",
        operation="event_search",
        start_notify=f"Searching {total} video(s) for event '{query}'...",
        start_status=f"Starting on {total} video(s)...",
        model=model, on_sample=on_sample, finalize=finalize,
    ):
        yield msg


async def _execute_semantic_search(
    ctx: Any, version: str, logbuf: _ProgressLogBuffer
) -> AsyncIterator[dict[str, Any]]:
    query = (ctx.params.get("semantic_query") or "").strip()
    if not query:
        raise ValueError("Semantic Search requires a non-empty query.")
    field = (ctx.params.get("semantic_field") or _derive_field_name("match", query)).strip()
    if not field:
        raise ValueError("Semantic Search requires a non-empty output field name.")
    threshold = float(ctx.params.get("semantic_threshold", _SEMANTIC_DEFAULT_THRESHOLD))

    view = ctx.target_view()
    total = len(view)
    model = _build_model(
        ctx,
        task=Task.CLASSIFY_SINGLE,
        target=None,
        prompt=_SEMANTIC_SEARCH_TEMPLATE.format(query=query),
    )
    counters = {"yes": 0}

    def on_sample(sample: fo.Sample, label: Any, _i: int) -> str:
        sample[field] = label
        sample.save()
        if label is not None and getattr(label, "label", None) == "yes":
            counters["yes"] += 1
        return f"{counters['yes']} 'yes' so far"

    def finalize(elapsed_s: float) -> _RunOutcome:
        n_yes = counters["yes"]
        matched_view = ctx.dataset.match(
            (F(f"{field}.label") == "yes") & (F(f"{field}.confidence") >= threshold)
        )
        n_matched = len(matched_view)
        usage = model.usage_totals
        summary_md = (
            f"**Semantic Search** completed.  \n"
            f"- Query: `{query}`  \n"
            f"- Threshold: `{threshold:.2f}`  \n"
            f"- 'yes' answers: **{n_yes}** / {total}  \n"
            f"- Above threshold: **{n_matched}** samples  \n"
            f"- Output field: `{field}`  \n"
            f"- Elapsed: **{_format_elapsed(elapsed_s)}** across **{usage['calls']}** API call(s)  \n"
            f"- Tokens: **{usage['prompt_tokens']:,}** in / **{usage['completion_tokens']:,}** out  \n"
            f"- View auto-filtered to above-threshold samples."
        )
        return _RunOutcome(
            summary={
                "mode": MODE_SEMANTIC_SEARCH,
                "query": query,
                "field": field,
                "threshold": threshold,
                "n_total": total,
                "n_yes": n_yes,
                "n_matched_samples": n_matched,
                **_usage_summary(model, elapsed_s),
            },
            summary_md=summary_md,
            notify_message=f"Semantic Search done: {n_matched}/{total} matched at >= {threshold:.2f}.",
            final_status=f"Complete -- {n_matched}/{total} matched at >= {threshold:.2f}",
            view_ops=[ctx.ops.set_view(view=matched_view), ctx.ops.reload_samples()],
        )

    async for msg in _run_mode(
        ctx, version, logbuf,
        view=view, mode_value=MODE_SEMANTIC_SEARCH, mode_label="Semantic Search",
        operation="semantic_search",
        start_notify=f"Scoring {total} sample(s) for '{query}'...",
        start_status=f"Starting on {total} sample(s)...",
        model=model, on_sample=on_sample, finalize=finalize,
    ):
        yield msg


async def _execute_bootstrap(
    ctx: Any, version: str, logbuf: _ProgressLogBuffer
) -> AsyncIterator[dict[str, Any]]:
    task = Task(ctx.params["bootstrap_task"])
    _raw_target = ctx.params.get("bootstrap_target") or []
    if isinstance(_raw_target, list):
        target: str | None = ", ".join(t.strip() for t in _raw_target if t and t.strip()) or None
    else:
        target = str(_raw_target).strip() or None
    field = (ctx.params.get("bootstrap_field") or _BOOTSTRAP_DEFAULT_FIELDS[task]).strip()
    if not field:
        raise ValueError("Bootstrap Labels requires a non-empty output field name.")

    view = ctx.target_view()
    total = len(view)
    per_frame = task_supports_per_frame(task)
    mode_label = f"Bootstrap {task.value}"
    model = _build_model(ctx, task=task, target=target)
    stats = {"frame_labels": 0, "sample_labels": 0, "frames_touched": 0, "dropped": 0}

    def on_sample(sample: fo.Sample, label: Any, _i: int) -> str:
        frame_rate = _frame_rate(sample) if per_frame else None
        summary = write_per_frame_labels(sample, label, field, frame_rate=frame_rate)
        stats["frame_labels"] += summary["per_frame_count"]
        stats["sample_labels"] += summary["sample_level"]
        stats["frames_touched"] += summary["frames_written"]
        stats["dropped"] += summary["dropped"]
        n_labels = stats["frame_labels"] + stats["sample_labels"]
        drop_tail = f"; {stats['dropped']} dropped" if stats["dropped"] else ""
        return f"{n_labels} label(s) written{drop_tail}"

    def finalize(elapsed_s: float) -> _RunOutcome:
        n_frame_labels = stats["frame_labels"]
        n_sample_labels = stats["sample_labels"]
        n_frames_touched = stats["frames_touched"]
        n_dropped = stats["dropped"]
        n_labels = n_frame_labels + n_sample_labels
        usage = model.usage_totals

        summary_md_lines = [
            f"**Bootstrap Labels** ({task.value}) completed.  ",
            f"- Target: `{target or '(none)'}`  ",
            f"- Output field: `{field}`  ",
            f"- Processed: {total} sample(s)  ",
            f"- Frame-level labels: **{n_frame_labels}** across {n_frames_touched} frame(s)  ",
            f"- Sample-level labels: **{n_sample_labels}**  ",
            f"- Elapsed: **{_format_elapsed(elapsed_s)}** across **{usage['calls']}** API call(s)  ",
            f"- Tokens: **{usage['prompt_tokens']:,}** in / **{usage['completion_tokens']:,}** out  ",
        ]
        if n_dropped:
            summary_md_lines.append(f"- Dropped (no `t=` or missing `frame_rate`): **{n_dropped}**  ")

        success_msg = f"Bootstrap {task.value} done: {n_labels} label(s) written to '{field}'"
        if n_dropped:
            success_msg += f" ({n_dropped} dropped -- see logs)"
        final_drop_tail = f"; {n_dropped} dropped" if n_dropped else ""

        return _RunOutcome(
            summary={
                "mode": MODE_BOOTSTRAP,
                "task": task.value,
                "prompt_source": ctx.params.get("bootstrap_prompt_source", "classes"),
                "target": target,
                "prompt_field": ctx.params.get("bootstrap_prompt_field") or None,
                "prompt_prefix": (ctx.params.get("bootstrap_prompt_prefix") or "").strip() or None,
                "field": field,
                "n_total": total,
                "n_labels": n_labels,
                "n_frame_labels": n_frame_labels,
                "n_sample_labels": n_sample_labels,
                "n_frames_touched": n_frames_touched,
                "n_dropped": n_dropped,
                **_usage_summary(model, elapsed_s),
            },
            summary_md="\n".join(summary_md_lines),
            notify_message=success_msg + ".",
            notify_variant="success" if n_dropped == 0 else "warning",
            final_status=f"Complete -- {n_labels} label(s) written{final_drop_tail}",
            view_ops=[ctx.ops.reload_dataset()],
        )

    async for msg in _run_mode(
        ctx, version, logbuf,
        view=view, mode_value=MODE_BOOTSTRAP, mode_label=mode_label,
        operation=f"bootstrap_{task.value}",
        start_notify=f"Running {task.value} on {total} sample(s)...",
        start_status=f"Starting on {total} sample(s)...",
        model=model, on_sample=on_sample, finalize=finalize,
    ):
        yield msg


# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------


def _resolve_media_type(ctx: Any) -> str:
    """Resolve the effective media type, collapsing uniform groups."""
    raw_media_type = ctx.dataset.media_type
    if raw_media_type == "group":
        slice_types = set(ctx.dataset.group_media_types.values())
        if len(slice_types) == 1:
            return slice_types.pop()
        return "mixed"
    return raw_media_type


def _build_model(
    ctx: Any,
    *,
    task: Task,
    target: str | None,
    prompt: str | None = None,
) -> MiniMaxModel:
    """Construct a `MiniMaxModel` from the operator context."""
    thinking_param = ctx.params.get("thinking_mode", "auto")
    n_frames_param = ctx.params.get("n_frames")
    cfg: dict[str, Any] = {
        "model": DEFAULT_MODEL_NAME,
        "task": task,
        "media_type": _resolve_media_type(ctx),
        "target": target,
        "prompt": prompt,
        "prompt_prefix": (ctx.params.get("bootstrap_prompt_prefix") or "").strip() or None,
        "prompt_field": ctx.params.get("bootstrap_prompt_field") or None,
        "thinking": None if thinking_param == "auto" else thinking_param,
        "api_key": get_api_key(ctx),
    }
    if n_frames_param:
        cfg["n_frames"] = int(n_frames_param)
    return MiniMaxModel(config=cfg)


def _predict(model: MiniMaxModel, sample: fo.Sample) -> Any:
    try:
        return model.predict(sample.filepath, sample=sample)
    except Exception as exc:
        exc.add_note(f"While processing sample {sample.id!s} ({sample.filepath})")
        raise


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60}m"


def _format_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


async def _tick_until_done(
    predict_task: asyncio.Task[Any],
    ctx: Any,
    *,
    mode_label: str,
    done_so_far: int,
    total: int,
    started_at: float,
    model: MiniMaxModel,
    logbuf: _ProgressLogBuffer,
) -> AsyncIterator[Any]:
    pending_status = "Waiting for API response..."
    while not predict_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(predict_task), timeout=_PROGRESS_TICK_INTERVAL_S)
        except asyncio.TimeoutError:
            for msg in _drain_log_buffer(ctx, logbuf):
                yield msg
            yield _yield_progress(
                ctx, mode_label=mode_label, status=pending_status,
                done=done_so_far, total=total, model=model, started_at=started_at,
            )


def _yield_progress(
    ctx: Any,
    *,
    mode_label: str,
    status: str,
    done: int,
    total: int,
    model: MiniMaxModel | None = None,
    started_at: float | None = None,
) -> Any:
    pct = int(round(done / total * 100)) if total else 0
    spinner_caption = f"{mode_label}  -  {done}/{total} ({pct}%)"

    schema = types.Object()
    schema.int("progress", view=types.ProgressView(variant="circular", label=spinner_caption))
    schema.str("status", label="Status", view=types.LabelValueView())
    schema.str("elapsed", label="Elapsed", view=types.LabelValueView())
    schema.str("tokens_in", label="Total Tokens In", view=types.LabelValueView())
    schema.str("tokens_out", label="Total Tokens Out", view=types.LabelValueView())

    if model is not None and started_at is not None:
        elapsed_str = _format_elapsed(time.perf_counter() - started_at)
        usage = model.usage_totals
        tokens_in_str = _format_tokens(usage["prompt_tokens"])
        tokens_out_str = _format_tokens(usage["completion_tokens"])
    else:
        elapsed_str = tokens_in_str = tokens_out_str = "--"

    results: dict[str, Any] = {
        "status": status,
        "elapsed": elapsed_str,
        "tokens_in": tokens_in_str,
        "tokens_out": tokens_out_str,
    }
    if total and done >= total:
        results["progress"] = 1

    return ctx.trigger(
        "show_output",
        {"outputs": types.Property(schema).to_json(), "results": results},
    )


def _missing_metadata_view(view: Any) -> Any:
    return view.exists("metadata", False)


def _count_missing_metadata(view: Any) -> int:
    return len(_missing_metadata_view(view))


def _mode_requires_frame_rate(ctx: Any, mode: str) -> bool:
    """Whether the active mode/task needs ``metadata.frame_rate``.

    Only FRAME_DETECT (per-frame writeback) hard-requires it. Temporal tasks
    fall back to the video's own fps, and image tasks never need it.
    """
    if mode != "bootstrap":
        return False
    task_str = ctx.params.get("bootstrap_task")
    if not task_str:
        return False
    try:
        return task_supports_per_frame(Task(task_str))
    except ValueError:
        return False


def _frame_rate(sample: fo.Sample) -> float | None:
    metadata = sample.metadata
    if metadata is None:
        return None
    fr = getattr(metadata, "frame_rate", None)
    return float(fr) if fr else None


def _register_custom_run(
    ctx: Any, *, version: str, operation: str, summary: dict[str, Any]
) -> str:
    run_key = make_run_key(operation)
    run_config = ctx.dataset.init_run(
        operator="run_minimax",
        version=version,
        params=dict(ctx.params),
        dataset_name=ctx.dataset.name,
    )
    ctx.dataset.register_run(run_key, run_config)
    run_results = ctx.dataset.init_run_results(run_key)
    run_results.summary = summary
    ctx.dataset.save_run_results(run_key, run_results, overwrite=True)
    logger.info("[minimax_m3] registered custom run %s with summary keys=%s", run_key, list(summary.keys()))
    return run_key


def _derive_field_name(prefix: str, query: str) -> str:
    slug = _SANITIZE_FIELD_RE.sub("_", query.strip().lower()).strip("_")
    if not slug:
        return prefix
    combined = f"{prefix}_{slug}"
    return combined[:40].rstrip("_") or prefix


class SaveMiniMaxLabel(foo.Operator):
    """Write a chat-panel response to a sample as a FiftyOne label.

    Invoked from the chat panel's "Convert to FiftyOne" button via
    ``useOperatorExecutor``. Runs in the foreground (never delegated) so
    ``ctx.ops`` triggers and ``ctx.log`` (browser console) work. The open modal
    is refreshed in place on the client (handleConvert -> useRefreshSample using
    the returned ``label_json``), which shows the new overlay without closing
    the modal so prev/next navigation survives. A brand-new field also reloads
    the dataset here so the App registers it in the schema/sidebar.
    """

    version: str = PLUGIN_VERSION

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="save_minimax_label",
            label="MiniMax-M3: save label",
            description=(
                "Save a generated label to a sample so its overlay appears in "
                "the open modal."
            ),
            unlisted=True,
            allow_immediate_execution=True,
            allow_delegated_execution=False,
        )

    def execute(self, ctx: Any) -> dict[str, Any]:
        run_id = ctx.params.get("run_id", "")
        sample_id = ctx.params.get("sample_id", "") or ctx.current_sample
        field = (ctx.params.get("field_name") or "").strip()
        fmt = ctx.params.get("detected_format", "")
        frame_rate = ctx.params.get("frame_rate")

        ctx.log(
            f"[save_minimax_label] params run_id={run_id} sample_id={sample_id} "
            f"field={field!r} fmt={fmt} frame_rate={frame_rate}"
        )

        result = save_stream_as_label(
            ctx.dataset, run_id, sample_id, field, fmt, frame_rate
        )
        if result.get("error"):
            ctx.log(f"[save_minimax_label] write failed: {result['error']}")
            return result

        ctx.log(
            f"[save_minimax_label] wrote field={result['field']} "
            f"type={result['label_type']} count={result['count']} "
            f"field_is_new={result['field_is_new']}"
        )

        # A new field changes the dataset schema; reload so the App's sidebar
        # registers it. The modal sample itself is refreshed client-side, which
        # keeps the modal open (navigation preserved). A new path only renders
        # in the modal after a remount, so prompt the user to reopen once.
        if result["field_is_new"]:
            ctx.ops.reload_dataset()
            ctx.log("[save_minimax_label] reload_dataset (new field)")
            notify(
                ctx,
                f"Added new field '{result['field']}'. Close and reopen this "
                "sample once to display it; further conversions show instantly.",
                variant="info",
            )

        return result
