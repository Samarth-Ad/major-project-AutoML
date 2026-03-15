"""
utils/logger.py
---------------
Centralized logging utility for the Agentic Pipeline Builder System.

Provides:
- A factory function `get_logger()` that returns a named logger.
- Console handler with colour-coded output (via ANSI codes — no extra deps).
- File handler that writes plain-text logs to `logs/pipeline.log`.
- A `PipelineLogger` helper that wraps the standard logger and exposes
  convenience methods used throughout the system (step_start, step_end,
  agent_event, code_written, error).

Usage
-----
    from utils.logger import get_logger, PipelineLogger

    log = get_logger(__name__)          # standard logging.Logger
    plog = PipelineLogger(__name__)     # richer helper
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ANSI colour helpers (works on most Unix terminals and Windows 10+)
# ---------------------------------------------------------------------------

class _Colours:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    # log-level colours
    DEBUG   = "\033[36m"   # cyan
    INFO    = "\033[32m"   # green
    WARNING = "\033[33m"   # yellow
    ERROR   = "\033[31m"   # red
    CRITICAL= "\033[35m"   # magenta
    # semantic colours
    STEP    = "\033[34m"   # blue  — pipeline step events
    CODE    = "\033[90m"   # dark-grey — code writer events
    AGENT   = "\033[96m"   # bright-cyan — agent lifecycle


_LEVEL_COLOURS = {
    "DEBUG":    _Colours.DEBUG,
    "INFO":     _Colours.INFO,
    "WARNING":  _Colours.WARNING,
    "ERROR":    _Colours.ERROR,
    "CRITICAL": _Colours.CRITICAL,
}


class _ColourFormatter(logging.Formatter):
    """
    Custom log formatter that injects ANSI colours into console output.
    Falls back to plain text if the stream is not a TTY (e.g. redirected).
    Also encodes emoji/Unicode safely for Windows consoles (cp1252).
    """

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    DATE_FMT = "%H:%M:%S"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        formatted = super().format(record)
        if self._use_colour:
            colour = _LEVEL_COLOURS.get(record.levelname, "")
            formatted = f"{colour}{formatted}{_Colours.RESET}"
        # Safe-encode for Windows PowerShell/cmd which may use cp1252
        # Replace any character that cannot be encoded with its closest ASCII
        try:
            formatted.encode(sys.stdout.encoding or "utf-8")
        except (UnicodeEncodeError, LookupError):
            formatted = formatted.encode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ).decode(sys.stdout.encoding or "utf-8", errors="replace")
        return formatted


class _PlainFormatter(logging.Formatter):
    """Plain formatter used for the rotating file handler."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


# ---------------------------------------------------------------------------
# Internal registry — avoids duplicate handlers when get_logger is called
# multiple times for the same name (common in Python logging).
# ---------------------------------------------------------------------------

_CONFIGURED_LOGGERS: set[str] = set()
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "pipeline.log"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Return a configured :class:`logging.Logger` instance.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` of the calling module.
    level:
        Minimum log level captured.  Defaults to ``DEBUG`` so that all
        messages are recorded.
    log_to_file:
        When *True* (default), a plain-text file handler is also attached.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # prevent double-output via root logger

    # --- Console handler ------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    use_colour = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    ch.setFormatter(_ColourFormatter(use_colour=use_colour))
    logger.addHandler(ch)

    # --- File handler ---------------------------------------------------
    if log_to_file:
        _ensure_log_dir()
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_PlainFormatter())
        logger.addHandler(fh)

    _CONFIGURED_LOGGERS.add(name)
    return logger


# ---------------------------------------------------------------------------
# PipelineLogger — higher-level helper used by agents & orchestrator
# ---------------------------------------------------------------------------

class PipelineLogger:
    """
    Thin wrapper around :func:`get_logger` providing semantic convenience
    methods aligned with pipeline lifecycle events.

    Parameters
    ----------
    name:
        Logger namespace (use ``__name__``).
    """

    def __init__(self, name: str) -> None:
        self._log = get_logger(name)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def step_start(self, step_name: str, task_id: str) -> None:
        """Log that a pipeline step is beginning execution."""
        self._log.info(
            f"{_Colours.STEP}▶  STEP START{_Colours.RESET} "
            f"[{task_id}] {step_name}"
        )

    def step_end(
        self,
        step_name: str,
        task_id: str,
        status: str,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """Log that a pipeline step has finished."""
        timing = f"  ({elapsed_ms:.1f} ms)" if elapsed_ms is not None else ""
        icon = "✔" if status == "success" else "✘"
        level = logging.INFO if status == "success" else logging.ERROR
        self._log.log(
            level,
            f"{_Colours.STEP}{icon}  STEP END  {_Colours.RESET} "
            f"[{task_id}] {step_name} → {status.upper()}{timing}",
        )

    def agent_event(self, agent_name: str, message: str) -> None:
        """General agent lifecycle event."""
        self._log.debug(
            f"{_Colours.AGENT}⚙  AGENT     {_Colours.RESET} "
            f"[{agent_name}] {message}"
        )

    def code_written(self, step_name: str, lines: int) -> None:
        """Log that CodeWriterAgent has appended code to the script."""
        self._log.info(
            f"{_Colours.CODE}✎  CODE      {_Colours.RESET} "
            f"Appended {lines} line(s) for step '{step_name}'"
        )

    def pipeline_start(self, steps: list[str]) -> None:
        """Log the start of the full pipeline."""
        self._log.info(
            f"{_Colours.BOLD}{'─' * 60}{_Colours.RESET}\n"
            f"  🚀  Pipeline starting — {len(steps)} step(s): "
            f"{' → '.join(steps)}\n"
            f"{_Colours.BOLD}{'─' * 60}{_Colours.RESET}"
        )

    def pipeline_end(self, success: bool, elapsed_s: float) -> None:
        """Log the completion of the full pipeline."""
        status_str = "COMPLETED ✔" if success else "FAILED ✘"
        self._log.info(
            f"{_Colours.BOLD}{'─' * 60}{_Colours.RESET}\n"
            f"  🏁  Pipeline {status_str}  "
            f"(total: {elapsed_s:.2f}s)\n"
            f"{_Colours.BOLD}{'─' * 60}{_Colours.RESET}"
        )

    def retry(self, step_name: str, attempt: int, max_attempts: int) -> None:
        """Log a retry attempt for a failed step."""
        self._log.warning(
            f"🔄  RETRY  [{step_name}]  attempt {attempt}/{max_attempts}"
        )

    def error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log an error, optionally with exception details."""
        if exc:
            self._log.error(f"💥  {message}", exc_info=exc)
        else:
            self._log.error(f"💥  {message}")

    def info(self, message: str) -> None:
        """Passthrough info log."""
        self._log.info(message)

    def debug(self, message: str) -> None:
        """Passthrough debug log."""
        self._log.debug(message)

    def warning(self, message: str) -> None:
        """Passthrough warning log."""
        self._log.warning(message)


# ---------------------------------------------------------------------------
# Module-level convenience logger (used by this module itself)
# ---------------------------------------------------------------------------

_module_log = get_logger(__name__)
_module_log.debug("Logger utility initialised — log file: %s", _LOG_FILE)