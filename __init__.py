"""FiftyOne plugin entry point for the MiniMax-M3 integration.

This module is intentionally thin: it registers a single operator and a chat
panel, and exposes the two hooks the FiftyOne zoo uses when this directory is
treated as a remote zoo model source.

Module layout:

    minimax_api.py     -- HF-router OpenAI client + retry/backoff + data URI /
                          frame-sampling helpers.
    minimax_parser.py  -- raw JSON model output -> FiftyOne label types.
    minimax_model.py   -- `MiniMaxModel` (image / video-frame / dense dispatch).
    prompts.py         -- Task enum, task-set constants, "ONLY JSON" prompt
                          templates.
    _shared.py         -- helpers shared by the operator and zoo registration.
    operators.py       -- `RunMiniMax`: conditional-input form operator for
                          Event Search / Semantic Search / Bootstrap Labels.
    chat_panel.py      -- `MiniMaxChatPanel`: streaming modal panel for asking
                          free-form questions about a sample.

Dual distribution:
    * As a plugin: drop this directory into FiftyOne's plugins dir; the
      plugin loader calls `register(plugin)`.
    * As a zoo model source: register the repo with
      `foz.register_zoo_model_source(...)` then `foz.load_zoo_model(
      "minimax/minimax-m3", task=..., ...)`. The zoo loader calls
      `download_model` (a no-op marker file -- nothing to actually download for
      a remote API model) then `load_model`, which returns a configured
      `MiniMaxModel`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .chat_panel import MiniMaxChatPanel
from .operators import RunMiniMax

logger = logging.getLogger("minimax_m3")


def register(plugin) -> None:
    """Register the RunMiniMax operator and MiniMaxChatPanel."""
    plugin.register(RunMiniMax)
    plugin.register(MiniMaxChatPanel)


def download_model(model_name: str, model_path: str) -> None:
    """Marker-file stand-in for a real downloader.

    M3 is a remote API, so there's nothing to actually download. The
    convention is to touch an empty file at `model_path` so FiftyOne's
    existence check passes. The real API call happens later inside
    `MiniMaxModel.predict(...)`.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    logger.info("[minimax_m3] download_model: %s -> marker file at %s", model_name, path)


def load_model(model_name: str, model_path: str, **kwargs: Any):
    """Construct a `MiniMaxModel` for the FiftyOne zoo loader.

    `model_name` and `model_path` are ignored -- configuration comes from
    `**kwargs`, which is forwarded to `MiniMaxModel(config=...)`.
    """
    # Lazy import so listing plugins doesn't pull in the OpenAI client.
    from .minimax_model import MiniMaxModel

    logger.info(
        "[minimax_m3] load_model: name=%s path=%s kwargs=%s",
        model_name,
        model_path,
        kwargs,
    )
    return MiniMaxModel(config=kwargs)
