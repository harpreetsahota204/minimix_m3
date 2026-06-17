"""MiniMax-M3 API client wrapper.

Thin OpenAI-compatible client targeted at the Hugging Face Inference router
(``router.huggingface.co/v1``), exactly as the vibe-check notebook and
``explore_minimax_m3.py`` use it:

* `MiniMaxClient.chat_completion(...)` -- one shot of `/chat/completions` with
  M3's recommended sampling (``temperature=1.0, top_p=0.95, top_k=40``) and the
  ``thinking`` object passed via ``extra_body``. Retries on 429.
* `encode_image(...)` / `to_image_data_uri(...)` -- base64-encode local images
  into the ``data:image/jpeg;base64,...`` URI the API requires.
* `sample_video_frames(...)` -- evenly sample N frames from a local video with
  OpenCV, returning each frame's timestamp (seconds), 1-indexed frame number,
  and a base64 data URI.

Every notable action emits a ``[minimax_m3]`` log line.

NOTE on ``thinking``: M3 expects an *object* (``{"type": "disabled"}`` /
``{"type": "adaptive"}`` / ``{"type": "enabled"}``), NOT a bare string. Passing
a bare string is a common mistake that the API rejects.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Final, Self

import cv2
import numpy as np
from openai import OpenAI, RateLimitError
from PIL import Image

logger = logging.getLogger("minimax_m3")


DEFAULT_BASE_URL: Final[str] = "https://router.huggingface.co/v1"

# M3 is served on the HF router under its full repo id. A provider can be pinned
# by appending ``:novita`` etc., e.g. ``MiniMaxAI/MiniMax-M3:novita``.
DEFAULT_MODEL_ID: Final[str] = "MiniMaxAI/MiniMax-M3"

# Long timeout: multi-frame video payloads are large and reasoning modes can
# take tens of seconds.
DEFAULT_TIMEOUT_SECONDS: Final[int] = 300

# Three attempts total (1 initial + 2 retries) with `Retry-After` preferred,
# exponential backoff as fallback.
MAX_RETRY_ATTEMPTS: Final[int] = 3

# M3-recommended sampling (from the model card / notebook).
DEFAULT_TEMPERATURE: Final[float] = 1.0
DEFAULT_TOP_P: Final[float] = 0.95
DEFAULT_TOP_K: Final[int] = 40

# Default frame-encoding ceilings (mirror the notebook).
DEFAULT_IMAGE_MAX_SIDE: Final[int] = 1280
DEFAULT_FRAME_MAX_SIDE: Final[int] = 1024


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale ``img`` so its longest side is ``max_side`` px (never upscales)."""
    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def _pil_to_jpeg_data_uri(img: Image.Image, *, quality: int = 90) -> str:
    """Encode a PIL image as a ``data:image/jpeg;base64,...`` URI."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_image(filepath: str | Path, *, max_side: int = DEFAULT_IMAGE_MAX_SIDE) -> str:
    """Read a local image, optionally downscale, and return a JPEG data URI.

    Downscaling to ``max_side`` (longest side) keeps payloads small; M3 returns
    NORMALIZED coordinates so the resize doesn't affect downstream parsing.
    """
    img = _downscale(Image.open(filepath).convert("RGB"), max_side)
    return _pil_to_jpeg_data_uri(img)


def to_image_data_uri(data: bytes, *, mime: str = "image/jpeg") -> str:
    """Encode raw image bytes as a ``data:<mime>;base64,...`` URI.

    Used by the chat panel (which holds raw bytes) and any caller that already
    has encoded image bytes.
    """
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def sample_video_frames(
    filepath: str | Path,
    *,
    n: int = 8,
    max_side: int = DEFAULT_FRAME_MAX_SIDE,
) -> tuple[list[dict[str, Any]], float, int]:
    """Evenly sample ``n`` frames from a video with OpenCV.

    Returns ``(frames, fps, total_frames)`` where each ``frames`` entry is::

        {"t": <seconds>, "frame_no": <1-indexed int>, "url": <jpeg data URI>}

    The 1-indexed ``frame_no`` matches FiftyOne's frame numbering so callers
    can write directly to ``sample.frames[frame_no]``. Mirrors the notebook's
    ``sample_frames`` helper.
    """
    cap = cv2.VideoCapture(str(filepath))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: list[dict[str, Any]] = []
        indices = sorted(set(np.linspace(0, total - 1, n).astype(int).tolist()))
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                logger.warning("[minimax_m3] could not read frame %d from %s; skipping", i, filepath)
                continue
            img = _downscale(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), max_side)
            frames.append(
                {
                    "t": float(i) / fps,
                    "frame_no": int(i) + 1,
                    "url": _pil_to_jpeg_data_uri(img, quality=85),
                }
            )
    finally:
        cap.release()
    logger.info(
        "[minimax_m3] sampled %d/%d frame(s) from %s (fps=%.2f)",
        len(frames),
        total,
        Path(filepath).name,
        fps,
    )
    return frames, fps, total


def build_frame_strip_content(prompt: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a single user message's content from a prompt + timestamped frames.

    Each frame is preceded by a ``[Frame at t=...s]`` text marker so the model
    can reason about *when* something happens. Mirrors the notebook's
    ``build_frame_content``.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for f in frames:
        content.append({"type": "text", "text": f"[Frame at t={f['t']:.2f}s]"})
        content.append({"type": "image_url", "image_url": {"url": f["url"]}})
    return content


def _truncate(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MiniMaxClient:
    """OpenAI client targeted at the Hugging Face Inference router for M3.

    Handles base URL / auth / timeout, M3 sampling defaults, 429 retry with
    backoff, and request / response logging. Higher-level concerns (prompt
    construction, parsing, FiftyOne writeback) live in
    `minimax_model.MiniMaxModel`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        # Running totals across every successful chat_completion call. Reset
        # between operator runs via `reset_usage_totals()`.
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_calls: int = 0
        logger.info(
            "[minimax_m3] MiniMaxClient ready: base_url=%s timeout=%ds",
            base_url,
            timeout,
        )

    @property
    def usage_totals(self) -> dict[str, int]:
        """Running token / call totals."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "calls": self._total_calls,
        }

    def reset_usage_totals(self) -> None:
        """Zero the token / call totals. Idempotent."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

    @classmethod
    def from_env(cls, env_var: str = "HF_TOKEN", **kwargs: Any) -> Self:
        """Build a client from ``os.environ[env_var]``.

        Raises:
            RuntimeError: If the env var is unset or empty.
        """
        value = os.environ.get(env_var, "")
        if not value:
            raise RuntimeError(
                f"environment variable {env_var!r} is not set or is empty; "
                f"export a Hugging Face token with Inference Providers access "
                f"before launching FiftyOne or the script that instantiates "
                f"the model"
            )
        return cls(api_key=value, **kwargs)

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: str = "disabled",
        max_tokens: int = 1024,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> Any:
        """Call ``/chat/completions`` once, retrying on 429.

        Args:
            model: Model id (e.g. ``"MiniMaxAI/MiniMax-M3"``).
            messages: OpenAI-style message list (caller builds it).
            thinking: M3 reasoning mode -- ``"disabled"``, ``"adaptive"``, or
                ``"enabled"``. Sent as ``extra_body.thinking = {"type": ...}``.
            max_tokens: Output token ceiling.
            temperature / top_p / top_k: M3-recommended sampling. ``top_k`` is
                passed via ``extra_body`` (not a standard OpenAI field).

        Returns:
            The raw OpenAI ``ChatCompletion`` response.

        Raises:
            openai.APIError: For any non-retried error.
        """
        extra_body: dict[str, Any] = {
            "top_k": top_k,
            "thinking": {"type": thinking},
        }
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }

        text_preview = _last_user_text_preview(messages)
        logger.info(
            "[minimax_m3] POST /chat/completions model=%s thinking=%s max_tokens=%d user_text=%r",
            model,
            thinking,
            max_tokens,
            text_preview,
        )

        # The retry loop is the ONLY try/except in the call path. Every
        # iteration either returns or raises.
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._log_response_summary(response, attempt)
                self._accumulate_usage(response)
                return response
            except RateLimitError as exc:
                if attempt == MAX_RETRY_ATTEMPTS:
                    logger.error(
                        "[minimax_m3] rate-limited after %d attempts; giving up", attempt
                    )
                    raise
                wait = self._compute_backoff(exc, attempt)
                logger.warning(
                    "[minimax_m3] 429 on attempt %d/%d; sleeping %.1fs before retry",
                    attempt,
                    MAX_RETRY_ATTEMPTS,
                    wait,
                )
                time.sleep(wait)

    def _accumulate_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self._total_calls += 1

    @staticmethod
    def _compute_backoff(exc: RateLimitError, attempt: int) -> float:
        """Prefer the server's ``Retry-After`` hint, else exponential backoff."""
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "headers"):
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return float(2**attempt)

    @staticmethod
    def _log_response_summary(response: Any, attempt: int) -> None:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        content_preview = _truncate((choice.message.content or "").replace("\n", " "))
        if usage is not None:
            logger.info(
                "[minimax_m3] response id=%s attempt=%d finish_reason=%s "
                "tokens={prompt=%d completion=%d total=%d} content=%r",
                getattr(response, "id", "?"),
                attempt,
                choice.finish_reason,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                content_preview,
            )
        else:
            logger.info(
                "[minimax_m3] response id=%s attempt=%d finish_reason=%s "
                "tokens=unavailable content=%r",
                getattr(response, "id", "?"),
                attempt,
                choice.finish_reason,
                content_preview,
            )


def _last_user_text_preview(messages: list[dict[str, Any]]) -> str:
    """Return a truncated text preview from the last user message, or ``""``."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    content = user_msgs[-1].get("content", "")
    if isinstance(content, str):
        return _truncate(content)
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            return _truncate(part.get("text", ""))
    return ""
