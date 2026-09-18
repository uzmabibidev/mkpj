"""Braille spinner shown while a background task runs.

Parity: the same ten frames, the same 0.08s cadence, cursor hidden while
spinning, line cleared afterwards, and a plain ``msg ... done/failed`` fallback
whenever colour is unavailable or stderr is not a terminal.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from . import console
from .console import STYLE

__all__ = ["SPIN_FRAMES", "run_with_spinner", "run_command_with_spinner"]

SPIN_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_INTERVAL = 0.08


def _animated() -> bool:
    try:
        return STYLE.enabled and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def run_with_spinner(message: str, work: Callable[[], int]) -> int:
    result: dict = {"rc": 1, "error": None}

    def runner() -> None:
        try:
            result["rc"] = int(work())
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            result["error"] = exc
            result["rc"] = 1

    thread = threading.Thread(target=runner, daemon=True)

    if not _animated():
        sys.stderr.write(f"{message} ")
        sys.stderr.flush()
        thread.start()
        thread.join()
        if result["error"] is not None:
            sys.stderr.write("failed\n")
            raise result["error"]
        sys.stderr.write("done\n" if result["rc"] == 0 else "failed\n")
        sys.stderr.flush()
        return result["rc"]

    console.hide_cursor()
    thread.start()
    index = 0
    try:
        while thread.is_alive():
            frame = SPIN_FRAMES[index % len(SPIN_FRAMES)]
            sys.stderr.write(f"\r\033[K{STYLE.cyan}{frame}{STYLE.reset} {message}")
            sys.stderr.flush()
            time.sleep(_INTERVAL)
            index += 1
        thread.join()
    finally:
        console.restore_terminal()

    if result["error"] is not None:
        raise result["error"]
    return result["rc"]


def run_command_with_spinner(
    message: str,
    argv: Sequence[str],
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> int:

    def work() -> int:
        handle = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
        try:
            process = subprocess.Popen(
                list(argv),
                stdout=handle if log_path else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
                cwd=str(cwd) if cwd else None,
            )
            console.set_current_child(process.pid)
            try:
                return process.wait()
            finally:
                console.set_current_child(None)
        finally:
            if log_path and handle is not subprocess.DEVNULL:
                handle.close()

    return run_with_spinner(message, work)
