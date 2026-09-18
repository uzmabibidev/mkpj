"""pip runner with live, coloured download progress.

Watches ``pip install --progress-bar raw`` and turns its ``Collecting`` /
``Downloading`` / ``Progress N of M`` chatter into a three-line block per
artifact::

      numpy
      ██████████████████████░░░░░░   72.4%
      7.5/10.4 MB   2.5 MB/s   ETA 00:01

Each block is finished in green before the next one starts, both numbers in a
size pair always share one unit, wheels pip already has are reported as cached
instead of pretending to download them, and the first artifact over 250 MB
prints a single warning.

Everything except :func:`run_pip` is pure, which makes the parsing and the
rendering testable without touching the network.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from . import console
from .console import STYLE

__all__ = [
    "LARGE_DOWNLOAD_THRESHOLD",
    "UNIT_MAP",
    "clean_package_name",
    "format_bytes_pair",
    "format_rate",
    "format_eta",
    "parse_download",
    "parse_progress",
    "parse_collecting",
    "parse_cached",
    "ProgressRenderer",
    "run_pip",
]

LARGE_DOWNLOAD_THRESHOLD = 250 * 1024 * 1024  # 250 MB
DEFAULT_FLAG_FILE = Path(
    os.environ.get(
        "MKPJ_LARGE_DOWNLOAD_FLAG",
        Path(tempfile.gettempdir()) / "mkpj_large_download_warned",
    )
)

UNIT_MAP = {
    "B": 1,
    "KB": 1024,
    "KiB": 1024,
    "MB": 1024**2,
    "MiB": 1024**2,
    "GB": 1024**3,
    "GiB": 1024**3,
}

_COLLECTING_RE = re.compile(r"Collecting\s+([A-Za-z0-9_.-]+)")
_DOWNLOAD_RE = re.compile(r"Downloading\s+(.+?)\s+\(([\d.]+)\s*(B|KB|MB|GB|KiB|MiB|GiB)\)")
_PROGRESS_RE = re.compile(r"Progress\s+(\d+)\s+of\s+(\d+)")
_CACHED_RE = re.compile(r"Using cached\s+(\S+)")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def clean_package_name(raw_name: str) -> str:
    name = raw_name.split("/")[-1]
    name = re.sub(r"\-(py2|py3|py2\.py3).*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.(whl|tar\.gz|zip|gz|bz2)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-\d+.*$", "", name)
    return name.replace("_", "-")


def format_bytes_pair(downloaded: int, total: int) -> str:
    # Both halves share one unit; "519.2 KB/519.2 KB MB" is what that avoids.
    if total < 1024:
        return f"{downloaded:.0f}/{total:.0f} B"
    if total < 1024**2:
        unit, scale = "KB", 1024
    elif total < 1024**3:
        unit, scale = "MB", 1024**2
    else:
        unit, scale = "GB", 1024**3
    return f"{downloaded / scale:.1f}/{total / scale:.1f} {unit}"


def format_rate(bytes_per_sec: float) -> str:
    if bytes_per_sec <= 0:
        return "-- kB/s"
    if bytes_per_sec >= 1024**2:
        return f"{bytes_per_sec / 1024**2:.1f} MB/s"
    return f"{bytes_per_sec / 1024:.1f} kB/s"


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def parse_collecting(line: str) -> str | None:
    match = _COLLECTING_RE.search(line)
    return match.group(1) if match else None


def parse_download(line: str):
    match = _DOWNLOAD_RE.search(line)
    if not match:
        return None
    raw_filename = match.group(1)
    if raw_filename.endswith(".metadata"):
        return None
    total = int(float(match.group(2)) * UNIT_MAP.get(match.group(3), 1))
    return clean_package_name(raw_filename), total


def parse_cached(line: str) -> str | None:
    # pip reports the metadata fetch as cached too; counting both would name
    # every package twice.
    match = _CACHED_RE.search(line)
    if not match or match.group(1).endswith(".metadata"):
        return None
    return clean_package_name(match.group(1))


def parse_progress(line: str):
    match = _PROGRESS_RE.search(line)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class ProgressRenderer:

    BAR_WIDTH = 28
    FILL = "█"
    EMPTY = "░"
    _REDRAW_INTERVAL = 0.05

    def __init__(self, stream=None, flag_file: Path = DEFAULT_FLAG_FILE) -> None:
        self.stream = stream or sys.stderr
        self.flag_file = Path(flag_file)
        self.current_package: str | None = None
        self.current_artifact: str | None = None
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.started_at = 0.0
        self.warned = self.flag_file.exists()
        self._block_open = False
        self._last_draw = 0.0

    # -- colours ----------------------------------------------------------
    @property
    def animated(self) -> bool:
        """In-place redraws need a terminal; logs get one line per artifact."""
        if not STYLE.enabled:
            return False
        try:
            return self.stream.isatty()
        except (AttributeError, ValueError):
            return False

    @property
    def _pink(self) -> str:
        return "\033[38;5;212m" if STYLE.enabled else ""

    @property
    def _green(self) -> str:
        return "\033[38;5;40m" if STYLE.enabled else ""

    @property
    def _gray(self) -> str:
        return "\033[38;5;240m" if STYLE.enabled else ""

    @property
    def _dim(self) -> str:
        return "\033[2m" if STYLE.enabled else ""

    @property
    def _reset(self) -> str:
        return "\033[0m" if STYLE.enabled else ""

    # -- rendering --------------------------------------------------------
    def percentage(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(self.downloaded_bytes / self.total_bytes * 100, 100.0)

    def bar_line(self, done: bool = False) -> str:
        percentage = 100.0 if done else self.percentage()
        filled = int(round(self.BAR_WIDTH * percentage / 100))
        color = self._green if done else self._pink
        bar = (
            color
            + (self.FILL * filled)
            + self._reset
            + self._gray
            + (self.EMPTY * (self.BAR_WIDTH - filled))
            + self._reset
        )
        return f"  {bar}  {percentage:5.1f}%"

    def stats_line(self, done: bool = False) -> str:
        downloaded = self.total_bytes if done else self.downloaded_bytes
        size = format_bytes_pair(downloaded, self.total_bytes)
        elapsed = max(time.time() - self.started_at, 0.001)
        rate = downloaded / elapsed
        if done:
            tail = format_eta(elapsed)
        else:
            remaining = (self.total_bytes - downloaded) / rate if rate > 0 else 0
            tail = f"ETA {format_eta(remaining)}"
        return f"  {size}   {format_rate(rate)}   {tail}"

    def draw(self, done: bool = False) -> None:
        if not self.total_bytes:
            return
        now = time.time()
        if not done and self._block_open and now - self._last_draw < self._REDRAW_INTERVAL:
            return
        self._last_draw = now

        if not self.animated:
            # No terminal to rewrite: one summary line when the artifact lands.
            if done:
                self.stream.write(f"{self.bar_line(True)}\n{self.stats_line(True)}\n")
                self.stream.flush()
            return

        if self._block_open:
            self.stream.write("\033[2A")
        self.stream.write(f"\r\033[K{self.bar_line(done)}\n\r\033[K{self.stats_line(done)}\n")
        self.stream.flush()
        self._block_open = True

    def finish_current(self, ok: bool = True) -> None:
        if not (self.current_artifact and self.total_bytes):
            return
        if ok:
            self.downloaded_bytes = self.total_bytes
            self.draw(done=True)
        self._block_open = False
        self.current_artifact = None
        self.total_bytes = 0
        self.downloaded_bytes = 0

    def _start(self, artifact: str, total: int) -> None:
        self.finish_current()
        self.current_artifact = artifact
        self.total_bytes = total
        self.downloaded_bytes = 0
        self.started_at = time.time()
        self._last_draw = 0.0

        display_name = artifact or self.current_package or "dependency"
        self.stream.write(f"\n  {display_name}\n")
        self.stream.flush()

        if total > LARGE_DOWNLOAD_THRESHOLD and not self.warned:
            self.stream.write("Large download detected — this may take a while.\n")
            self.stream.flush()
            self.warned = True
            try:
                self.flag_file.touch()
            except OSError:
                pass

    def feed(self, line: str) -> None:
        clean_line = line.strip()

        collected = parse_collecting(clean_line)
        if collected:
            self.current_package = collected

        download = parse_download(clean_line)
        if download:
            self._start(*download)
            return

        cached = parse_cached(clean_line)
        if cached:
            self.finish_current()
            self.stream.write(f"  {cached} {self._dim}(cached){self._reset}\n")
            self.stream.flush()
            return

        progress = parse_progress(clean_line)
        if progress and self.total_bytes:
            downloaded, prog_total = progress
            self.downloaded_bytes = downloaded
            if prog_total > 0:
                self.total_bytes = prog_total
            self.draw()

    def close(self, return_code: int) -> None:
        self.finish_current(ok=return_code == 0)


# ---------------------------------------------------------------------------
# Process driver
# ---------------------------------------------------------------------------
def run_pip(python_exe: str, log_path: Path, args: Sequence[str]) -> int:
    argv: list[str] = [python_exe, "-m", "pip", *args]
    renderer = ProgressRenderer()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        console.set_current_child(process.pid)
        # A plain blocking read works on every platform. select() cannot be
        # used here: on Windows it only accepts sockets, never pipes.
        try:
            for line in process.stdout:
                log.write(line)
                renderer.feed(line)
            return_code = process.wait()
        except KeyboardInterrupt:  # pragma: no cover - interactive path
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
            console.restore_terminal()
            raise
        finally:
            console.set_current_child(None)

    renderer.close(return_code)
    return return_code
