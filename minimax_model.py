"""`MiniMaxModel` -- the FiftyOne `Model` wrapper for MiniMax-M3.

Owns configuration, per-sample request construction, the call to
`MiniMaxClient.chat_completion(...)`, and the handoff to
`minimax_parser.to_fiftyone(...)`.

Three dispatch paths inside `predict()`:

    * Dense frame-mode (FRAME_DETECT) -- evenly sample N frames from the video
      and send each as its own ``image_url`` request. Returns a single
      `fo.Detections` container with per-item ``t`` (seconds) attributes that
      `write_per_frame_labels` routes to ``sample.frames[i]``.
    * Image mode (DETECT, KEYPOINTS, and shared tasks on image datasets) --
      single ``image_url`` request per sample, returns a sample-level label.
    * Video frame-strip mode (FIND_EVENT, KEY_MOMENTS, and shared tasks on
      video datasets) -- sample N frames, send them all in one message as a
      timestamped strip, returns a sample-level label.

M3 has no native video input on the HF router, so all video handling goes
through frame sampling (see `minimax_api.sample_video_frames`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import fiftyone as fo
from fiftyone import Model

from .minimax_api import (
    DEFAULT_MODEL_ID,
    MiniMaxClient,
    build_frame_strip_content,
    encode_image,
    sample_video_frames,
)
from .minimax_parser import FOLabel, to_fiftyone
from .prompts import (
    TASK_DEFAULT_THINKING,
    TASK_TO_PARSER_FORMAT,
    TASKS_DENSE_FRAME_MODE,
    TASKS_IMAGE_GROUNDING,
    TASKS_TEMPORAL,
    Task,
    default_user_prompt,
)

logger = logging.getLogger("minimax_m3")


# Friendlier alias surfaced to FiftyOne; the wire model id lives in minimax_api.
DEFAULT_MODEL_NAME: str = DEFAULT_MODEL_ID

# Token ceilings. Temporal reasoning over a frame strip wants more headroom.
DEFAULT_MAX_TOKENS: int = 1024
TEMPORAL_MAX_TOKENS: int = 1500

# Frames evenly sampled from each video for frame-strip and dense tasks.
# Kept small to respect provider payload limits (mirrors the notebook).
DEFAULT_N_FRAMES: int = 8


@dataclass(slots=True, kw_only=True, frozen=True)
class MiniMaxConfig:
    """Immutable configuration for one `MiniMaxModel` instance.

    Attributes:
        model: M3 model id on the HF router (e.g. ``"MiniMaxAI/MiniMax-M3"``,
            optionally provider-pinned like ``"MiniMaxAI/MiniMax-M3:novita"``).
        task: Drives prompt selection and predict dispatch.
        media_type: ``"image"`` or ``"video"``. Controls FiftyOne's
            ``Model.media_type`` and the predict dispatch for shared tasks.
        target: User-supplied target -- class string, event description,
            classify aspect, or VQA question, depending on ``task``.
        prompt: Override for the auto-generated user prompt. Takes precedence
            over ``prompt_prefix`` / ``prompt_field``.
        prompt_prefix: Text prepended to the per-sample field value when
            ``prompt_field`` is set.
        prompt_field: Name of a sample-level string field used as (or appended
            to ``prompt_prefix`` to form) the per-sample prompt.
        thinking: M3 reasoning mode -- ``"disabled"`` / ``"adaptive"`` /
            ``"enabled"``. ``None`` -> use the task's default.
        max_tokens: Output ceiling per call. ``None`` -> task-appropriate
            default.
        n_frames: Frames to sample per video for frame-strip and dense tasks.
        api_key: Explicit HF token. The operator passes ``ctx.secrets["HF_TOKEN"]``;
            the zoo loader / scripts leave it ``None`` to fall back to
            ``os.environ["HF_TOKEN"]``.
    """

    model: str = DEFAULT_MODEL_NAME
    task: Task = Task.DETECT
    media_type: str = "image"
    target: str | None = None
    prompt: str | None = None
    prompt_prefix: str | None = None
    prompt_field: str | None = None
    thinking: str | None = None
    max_tokens: int | None = None
    n_frames: int = DEFAULT_N_FRAMES
    api_key: str | None = None


class MiniMaxModel(Model):
    """FiftyOne `Model` for the MiniMax-M3 API (via the HF Inference router)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg_dict = config or {}
        task_value = cfg_dict.get("task", Task.DETECT)
        task = task_value if isinstance(task_value, Task) else Task(task_value)

        self._config = MiniMaxConfig(
            model=str(cfg_dict.get("model", DEFAULT_MODEL_NAME)),
            task=task,
            media_type=str(cfg_dict.get("media_type", "image")),
            target=cfg_dict.get("target"),
            prompt=cfg_dict.get("prompt"),
            prompt_prefix=cfg_dict.get("prompt_prefix") or None,
            prompt_field=cfg_dict.get("prompt_field") or None,
            thinking=cfg_dict.get("thinking") or None,
            max_tokens=cfg_dict.get("max_tokens"),
            n_frames=int(cfg_dict.get("n_frames", DEFAULT_N_FRAMES)),
            api_key=cfg_dict.get("api_key"),
        )

        if self._config.api_key:
            self._client = MiniMaxClient(api_key=self._config.api_key)
            key_source = "config (ctx.secrets)"
        else:
            self._client = MiniMaxClient.from_env()
            key_source = "os.environ[HF_TOKEN]"

        logger.info(
            "[minimax_m3] MiniMaxModel ready: model=%s task=%s target=%r "
            "thinking=%s n_frames=%d api_key_source=%s",
            self._config.model,
            self._config.task.value,
            self._config.target,
            self._thinking,
            self._config.n_frames,
            key_source,
        )

    # -- FiftyOne Model interface ------------------------------------------------

    @property
    def media_type(self) -> str:
        return self._config.media_type

    # -- Public access for callers ----------------------------------------------

    @property
    def config(self) -> MiniMaxConfig:
        return self._config

    @property
    def parser_format(self) -> str:
        return TASK_TO_PARSER_FORMAT[self._config.task]

    @property
    def _thinking(self) -> str:
        return self._config.thinking or TASK_DEFAULT_THINKING.get(self._config.task, "disabled")

    @property
    def _max_tokens(self) -> int:
        if self._config.max_tokens:
            return self._config.max_tokens
        if self._config.task in TASKS_TEMPORAL:
            return TEMPORAL_MAX_TOKENS
        return DEFAULT_MAX_TOKENS

    @property
    def usage_totals(self) -> dict[str, int]:
        return self._client.usage_totals

    def reset_usage_totals(self) -> None:
        self._client.reset_usage_totals()

    def predict(self, filepath: str, sample: fo.Sample | None = None) -> FOLabel:
        """Run inference on one media file (image or video).

        Dispatch order:
            1. TASKS_DENSE_FRAME_MODE (FRAME_DETECT) -> ``_predict_dense``.
            2. TASKS_IMAGE_GROUNDING or media_type == "image" -> ``_predict_image``.
            3. Everything else (video shared / temporal) -> ``_predict_video_frames``.
        """
        if self._config.task in TASKS_DENSE_FRAME_MODE:
            return self._predict_dense(filepath, sample)
        if self._config.task in TASKS_IMAGE_GROUNDING or self._config.media_type == "image":
            return self._predict_image(filepath, sample)
        return self._predict_video_frames(filepath, sample)

    # -- Image-mode predict (one ``image_url`` request) --------------------------

    def _predict_image(self, filepath: str, sample: fo.Sample | None = None) -> FOLabel:
        prompt = self._resolve_prompt(sample)
        content = _image_content(prompt, encode_image(filepath))
        logger.info(
            "[minimax_m3] predict (image-mode): task=%s target=%r path=%s prompt=%r",
            self._config.task.value,
            self._config.target,
            filepath,
            _preview(prompt),
        )
        response = self._chat(content)
        text = response.choices[0].message.content or ""
        label = to_fiftyone(text, self.parser_format, target=self._config.target)
        logger.info("[minimax_m3] predict produced %s", _typename(label))
        return label

    # -- Video frame-strip predict (one request with N timestamped frames) -------

    def _predict_video_frames(self, filepath: str, sample: fo.Sample | None) -> FOLabel:
        frames, fps, _total = sample_video_frames(filepath, n=self._config.n_frames)
        prompt = self._resolve_prompt(sample)
        content = build_frame_strip_content(prompt, frames)
        logger.info(
            "[minimax_m3] predict (video frame-strip): task=%s target=%r path=%s "
            "n_frames=%d prompt=%r",
            self._config.task.value,
            self._config.target,
            filepath,
            len(frames),
            _preview(prompt),
        )
        response = self._chat(content)
        text = response.choices[0].message.content or ""
        label = to_fiftyone(
            text,
            self.parser_format,
            target=self._config.target,
            frame_rate=self._extract_frame_rate(sample) or fps,
        )
        logger.info("[minimax_m3] predict produced %s", _typename(label))
        return label

    # -- Dense frame-mode predict (N per-frame ``image_url`` requests) -----------

    def _predict_dense(self, filepath: str, sample: fo.Sample | None) -> FOLabel:
        """Sample frames and run per-frame image-mode detection.

        Used only for FRAME_DETECT. Each sampled frame is sent as its own
        ``image_url`` request; results are accumulated into one `fo.Detections`
        container with ``t`` (seconds) attributes so `write_per_frame_labels`
        can route each detection to the correct ``sample.frames[i]``.
        """
        frames, _fps, _total = sample_video_frames(filepath, n=self._config.n_frames)
        prompt = self._resolve_prompt(sample)
        parser_format = self.parser_format

        logger.info(
            "[minimax_m3] predict (dense): task=%s target=%r path=%s -> %d frame call(s); prompt=%r",
            self._config.task.value,
            self._config.target,
            filepath,
            len(frames),
            _preview(prompt),
        )

        accumulated: list[Any] = []
        for f in frames:
            response = self._chat(_image_content(prompt, f["url"]))
            text = response.choices[0].message.content or ""
            label = to_fiftyone(text, parser_format, target=self._config.target)
            if isinstance(label, fo.Detections):
                for det in label.detections:
                    det["t"] = f["t"]
                    accumulated.append(det)

        container = fo.Detections(detections=accumulated)
        logger.info(
            "[minimax_m3] dense predict produced %s with %d item(s) across %d frame(s)",
            type(container).__name__,
            len(accumulated),
            len(frames),
        )
        return container

    # -- Internal helpers --------------------------------------------------------

    def _chat(self, content: list[dict[str, Any]]) -> Any:
        return self._client.chat_completion(
            model=self._config.model,
            messages=[{"role": "user", "content": content}],
            thinking=self._thinking,
            max_tokens=self._max_tokens,
        )

    def _resolve_prompt(self, sample: fo.Sample | None = None) -> str:
        """Return the user prompt for this request.

        Resolution order: explicit ``config.prompt`` > per-sample
        ``config.prompt_field`` (with optional ``prompt_prefix``) > default
        template from ``prompts.default_user_prompt``.
        """
        if self._config.prompt:
            return self._config.prompt

        if self._config.prompt_field and sample is not None:
            field_value = sample.get_field(self._config.prompt_field)
            if field_value:
                prefix = self._config.prompt_prefix or ""
                return f"{prefix}{field_value}"
            logger.warning(
                "[minimax_m3] sample %s has no value for prompt_field=%r; "
                "falling back to default prompt",
                sample.id,
                self._config.prompt_field,
            )

        return default_user_prompt(
            self._config.task,
            self._config.target,
            media_type=self._config.media_type,
        )

    @staticmethod
    def _extract_frame_rate(sample: fo.Sample | None) -> float | None:
        meta = sample.metadata if sample is not None else None
        fr = getattr(meta, "frame_rate", None) if meta is not None else None
        return float(fr) if fr else None


def _image_content(prompt: str, image_url: str) -> list[dict[str, Any]]:
    """Build a single-image user-message content list (prompt + one image)."""
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


def _preview(text: str, limit: int = 200) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _typename(label: Any) -> str:
    return type(label).__name__ if label is not None else "None"
