"""Non-scrolling ↑/↓ wheel menu and the prompt helpers around it.

Drawn entirely on stderr, so stdout stays clean for piping. Each redraw moves
the cursor up ``len(options) + 1`` lines and clears down, which is what makes
the wheel look like it stays put. Arrows and vim's k/j move, Enter confirms,
the selection wraps, and the cursor is hidden while the menu is live.

Without a TTY (CI, pipes, ``mkpj < /dev/null``) it falls back to a numbered
prompt rather than blocking on raw reads that can never arrive.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager

from . import console
from .console import STYLE

try:  # POSIX
    import select
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - Windows
    _HAS_TERMIOS = False

try:  # Windows
    import msvcrt

    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

__all__ = [
    "raw_mode",
    "KEY_UP",
    "KEY_DOWN",
    "KEY_ENTER",
    "KEY_OTHER",
    "decode_key",
    "next_index",
    "render_menu",
    "select_menu",
    "flush_stdin",
    "ask",
    "confirm",
]

_IN_RAW_MODE = False

KEY_UP = "up"
KEY_DOWN = "down"
KEY_ENTER = "enter"
KEY_OTHER = "other"


# ---------------------------------------------------------------------------
# Pure logic (unit-tested without a terminal)
# ---------------------------------------------------------------------------
def decode_key(first: str, rest: str = "") -> str:
    if first == "\x1b":
        if rest.startswith("[A") or rest.startswith("OA"):
            return KEY_UP
        if rest.startswith("[B") or rest.startswith("OB"):
            return KEY_DOWN
        return KEY_OTHER
    if first in ("", "\r", "\n"):
        return KEY_ENTER
    if first == "k":
        return KEY_UP
    if first == "j":
        return KEY_DOWN
    return KEY_OTHER


def next_index(current: int, action: str, count: int) -> int:
    if count <= 0:
        return 0
    if action == KEY_UP:
        return count - 1 if current - 1 < 0 else current - 1
    if action == KEY_DOWN:
        return 0 if current + 1 >= count else current + 1
    return current


def render_menu(prompt: str, options: Sequence[str], selected: int) -> str:
    # CRLF, not LF: a bare newline staircases wherever the driver is not
    # translating (raw mode, some multiplexers, some remote sessions).
    lines = [f"{STYLE.cyan}{STYLE.bold}{prompt}{STYLE.reset}"]
    for index, option in enumerate(options):
        if index == selected:
            lines.append(f"  {STYLE.blue}{STYLE.white}{STYLE.bold} ❯ {option} {STYLE.reset}")
        else:
            lines.append(f"    {STYLE.dim}  {option}{STYLE.reset}")
    return "\r\n".join(lines) + "\r\n"


# ---------------------------------------------------------------------------
# Terminal I/O
# ---------------------------------------------------------------------------
def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def flush_stdin() -> None:
    if not (_HAS_TERMIOS and _stdin_is_tty()):
        return
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (termios.error, OSError, ValueError):
        pass


@contextmanager
def raw_mode():
    # cbreak, not raw: raw also clears OPOST, so "\n" stops becoming "\r\n"
    # and the multi-line wheel walks diagonally down the screen.
    # TCSANOW, not TCSAFLUSH: flushing drops keys typed while a frame was
    # drawing, so holding down an arrow would lose most of them.
    global _IN_RAW_MODE
    if _IN_RAW_MODE or not (_HAS_TERMIOS and _stdin_is_tty()):
        yield
        return
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
    except (termios.error, OSError, ValueError):
        yield
        return
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        _IN_RAW_MODE = True
        yield
    finally:
        _IN_RAW_MODE = False
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, OSError, ValueError):
            pass


def _read_key_posix() -> str:
    fd = sys.stdin.fileno()
    with raw_mode():
        first = os.read(fd, 1).decode("utf-8", "replace")
        if first == "\x03":
            raise KeyboardInterrupt
        if first == "\x04":  # Ctrl-D behaves like EOF -> confirm
            return KEY_ENTER
        rest = ""
        if first == "\x1b":
            ready, _, _ = select.select([fd], [], [], 0.01)
            if ready:
                rest = os.read(fd, 2).decode("utf-8", "replace")
        return decode_key(first, rest)


def _read_key_windows() -> str:  # pragma: no cover - Windows only
    char = msvcrt.getwch()
    if char == "\x03":
        raise KeyboardInterrupt
    if char in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        if code == "H":
            return KEY_UP
        if code == "P":
            return KEY_DOWN
        return KEY_OTHER
    return decode_key(char)


def read_key() -> str:
    if _HAS_TERMIOS and _stdin_is_tty():
        return _read_key_posix()
    if _HAS_MSVCRT:  # pragma: no cover - Windows only
        return _read_key_windows()
    return KEY_ENTER


def _fallback_select(prompt: str, options: Sequence[str], default: int) -> int:
    console.plain("")
    sys.stderr.write(f"{STYLE.cyan}{STYLE.bold}{prompt}{STYLE.reset}\n")
    for index, option in enumerate(options, start=1):
        sys.stderr.write(f"  {index}) {option}\n")
    sys.stderr.flush()
    try:
        raw = input(f"Select [1-{len(options)}] (default {default + 1}): ").strip()
    except EOFError:
        return default
    if not raw:
        return default
    try:
        choice = int(raw)
    except ValueError:
        return default
    if 1 <= choice <= len(options):
        return choice - 1
    return default


def select_menu(prompt: str, options: Sequence[str], default: int = 0) -> int:
    options = list(options)
    if not options:
        raise ValueError("select_menu requires at least one option")

    if not (_stdin_is_tty() and STYLE.enabled):
        return _fallback_select(prompt, options, default)

    selected = max(0, min(default, len(options) - 1))
    first_draw = True

    console.hide_cursor()
    flush_stdin()
    try:
        # One raw-mode window for the whole wheel: keys that arrive while a
        # frame is being drawn stay queued instead of being flushed away.
        with raw_mode():
            while True:
                if not first_draw:
                    sys.stderr.write(f"\033[{len(options) + 1}A\033[J")
                first_draw = False
                sys.stderr.write(render_menu(prompt, options, selected))
                sys.stderr.flush()

                action = read_key()
                if action == KEY_ENTER:
                    break
                selected = next_index(selected, action, len(options))
    finally:
        console.restore_terminal()
    return selected


# ---------------------------------------------------------------------------
# Line prompts
# ---------------------------------------------------------------------------
def ask(prompt: str, default: str = "") -> str:
    flush_stdin()
    try:
        sys.stderr.write(prompt)
        sys.stderr.flush()
        raw = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        console.plain("")
        return default
    if raw == "":  # EOF
        console.plain("")
        return default
    value = raw.strip()
    return value if value else default


def confirm(prompt: str, default: bool = False) -> bool:
    answer = ask(prompt).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def ask_list(message: str, default: str = "", assume_yes: bool = False) -> str:
    if assume_yes:
        return default
    return ask(f"{message}: ", default)


def choose(prompt: str, options: Sequence[str], values: Sequence[str], default: int = 0) -> str:
    index = select_menu(prompt, options, default)
    return values[index]


def maybe_index(value: int | None, count: int) -> int:
    if value is None:
        return 0
    return max(0, min(value, count - 1))


def option_labels(*labels: str) -> list[str]:
    return list(labels)
