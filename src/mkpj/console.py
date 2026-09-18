"""Terminal output: palette, semantic log helpers, cursor and signal handling.

Every message goes to stderr, so stdout stays usable for piping.
"""

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass

__all__ = [
    "configure_output_encoding",
    "Style",
    "STYLE",
    "refresh_style",
    "info",
    "success",
    "ok",
    "warn",
    "error",
    "err",
    "step",
    "muted",
    "plain",
    "print_banner",
    "restore_terminal",
    "hide_cursor",
    "show_cursor",
    "install_signal_handlers",
    "set_current_child",
    "MkpjExit",
]


class MkpjExit(SystemExit):
    """Carries mkpj's exit code up to main()."""


@dataclass(frozen=True)
class Style:

    enabled: bool = True
    reset: str = "\033[0m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"
    blue: str = "\033[38;5;39m"
    cyan: str = "\033[38;5;45m"
    green: str = "\033[38;5;40m"
    yellow: str = "\033[38;5;220m"
    red: str = "\033[38;5;196m"
    white: str = "\033[38;5;255m"
    gray: str = "\033[38;5;245m"

    @classmethod
    def detect(cls, stream=None) -> Style:
        stream = stream or sys.stderr
        if os.environ.get("NO_COLOR"):
            return cls.disabled()
        if os.environ.get("TERM") == "dumb":
            return cls.disabled()
        try:
            if not stream.isatty():
                return cls.disabled()
        except (AttributeError, ValueError):
            return cls.disabled()
        return cls()

    @classmethod
    def disabled(cls) -> Style:
        return cls(
            enabled=False,
            reset="",
            bold="",
            dim="",
            blue="",
            cyan="",
            green="",
            yellow="",
            red="",
            white="",
            gray="",
        )


class _StyleProxy:

    __slots__ = ("_style",)

    def __init__(self, style: Style) -> None:
        object.__setattr__(self, "_style", style)

    def _replace(self, style: Style) -> None:
        object.__setattr__(self, "_style", style)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_style"), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<StyleProxy {object.__getattribute__(self, '_style')!r}>"


STYLE = _StyleProxy(Style.detect())


def refresh_style(stream=None) -> Style:
    style = Style.detect(stream)
    STYLE._replace(style)
    return style


# Arrows, ticks and box-drawing appear in the help text, the menus and the
# progress bars.
_UI_CHARACTERS = "↑↓◆✓✗❯█░━"


def configure_output_encoding() -> None:
    # A redirected stdout on Windows falls back to cp1252, which has no
    # arrows: "mkpj --help > notes.txt" died on them. errors="replace" is the
    # backstop, so no terminal can turn output into a traceback.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # a test double, or a stream we don't own
            continue
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            _UI_CHARACTERS.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            settings = {"encoding": "utf-8", "errors": "replace"}
        else:
            settings = {"errors": "replace"}
        try:
            reconfigure(**settings)
        except (ValueError, OSError):
            pass


def _emit(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def info(message: str) -> None:
    _emit(f"{STYLE.blue}◆{STYLE.reset} {message}\n")


def success(message: str) -> None:
    _emit(f"{STYLE.green}✓{STYLE.reset} {message}\n")


def warn(message: str) -> None:
    _emit(f"{STYLE.yellow}!{STYLE.reset} {message}\n")


def error(message: str) -> None:
    _emit(f"{STYLE.red}✗{STYLE.reset} {message}\n")


def step(message: str) -> None:
    _emit(f"{STYLE.cyan}→{STYLE.reset} {message}\n")


def muted(message: str) -> None:
    _emit(f"{STYLE.gray}{message}{STYLE.reset}\n")


def plain(message: str = "") -> None:
    _emit(f"{message}\n")


ok = success
err = error


def print_banner() -> None:
    _emit(f"\n{STYLE.blue}{STYLE.bold}mkpj{STYLE.reset}\n")
    _emit(f"{STYLE.cyan}Python Project Generator{STYLE.reset}\n")
    _emit(f"{STYLE.blue}{'─' * 40}{STYLE.reset}\n\n")


# ---------------------------------------------------------------------------
# Cursor / terminal state
# ---------------------------------------------------------------------------
def hide_cursor() -> None:
    if STYLE.enabled:
        _emit("\033[?25l")


def show_cursor() -> None:
    if STYLE.enabled:
        _emit("\033[?25h")


def restore_terminal() -> None:
    show_cursor()
    if not STYLE.enabled:
        return
    try:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    except (ValueError, OSError):
        pass


# ---------------------------------------------------------------------------
# Signals — no stray cursor or orphan child may be left behind, and
# neither does this one.
# ---------------------------------------------------------------------------
_current_child: int | None = None
_exiting = False


def set_current_child(pid: int | None) -> None:
    global _current_child
    _current_child = pid


def _kill_current_child() -> None:
    if _current_child is None:
        return
    try:
        os.kill(_current_child, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _terminate(code: int, message: str | None = None, extra: str | None = None) -> None:
    global _exiting
    if _exiting:
        raise MkpjExit(code)
    _exiting = True
    restore_terminal()
    _emit("\n")
    if message:
        warn(message)
    if extra:
        warn(extra)
    _kill_current_child()
    raise MkpjExit(code)


def _on_interrupt(_signum, _frame) -> None:
    _terminate(130, "Interrupted by user.", "mkpj stopped safely.")


def _on_terminate(_signum, _frame) -> None:
    _terminate(143, "mkpj was terminated.")


def _on_hup(_signum, _frame) -> None:
    _terminate(129, "Terminal/session was closed.")


def _on_suspend(_signum, _frame) -> None:
    restore_terminal()
    signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTSTP)
    # Execution resumes here on SIGCONT.
    signal.signal(signal.SIGTSTP, _on_suspend)


def _on_continue(_signum, _frame) -> None:
    hide_cursor()


def install_signal_handlers() -> None:
    handlers = {
        "SIGINT": _on_interrupt,
        "SIGTERM": _on_terminate,
        "SIGHUP": _on_hup,
        "SIGTSTP": _on_suspend,
        "SIGCONT": _on_continue,
    }
    for name, handler in handlers.items():
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError, RuntimeError):
            pass
