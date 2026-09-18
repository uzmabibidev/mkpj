"""End-to-end tests of the interactive terminal UI, driven through a real pty.

A terminal UI is easy to break silently. Here mkpj is spawned
on a pseudo-terminal and driven with the exact byte sequences a terminal sends
for ↑/↓, so navigation, wrap-around, buffered keypresses, Ctrl-C handling and
cursor restoration are all covered automatically.
"""

from __future__ import annotations

import os
import re
import select
import sys
import time
from pathlib import Path

import pytest
from conftest import SRC_DIR, read

pty = pytest.importorskip("pty")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not hasattr(pty, "fork"), reason="pty.fork is POSIX-only"),
]

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

DOWN = b"\x1b[B"
UP = b"\x1b[A"
ENTER = b"\r"
CTRL_C = b"\x03"


class PtySession:
    """Minimal expect-style driver for a process attached to a pty."""

    def __init__(self, argv, cwd: Path, env: dict) -> None:
        self.buffer = ""
        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # pragma: no cover - child process
            os.chdir(str(cwd))
            os.environ.update(env)
            os.execve(argv[0], argv, os.environ)
        self._exit_code = None

    def _drain(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            self.buffer += chunk.decode("utf-8", "replace")

    def expect(self, needle: str, timeout: float = 90.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in self.plain:
                return self.plain
            self._drain(0.2)
        raise AssertionError(f"timed out waiting for {needle!r}\n--- seen ---\n{self.plain}")

    def send(self, data: bytes, settle: float = 0.4) -> None:
        os.write(self.fd, data)
        self._drain(settle)

    @property
    def plain(self) -> str:
        return ANSI.sub("", self.buffer)

    @property
    def selected_rows(self):
        return [line.strip() for line in self.plain.splitlines() if "❯" in line]

    def wait(self, timeout: float = 120.0) -> int:
        if self._exit_code is not None:
            return self._exit_code
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._drain(0.2)
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self._drain(0.2)
                self._exit_code = os.waitstatus_to_exitcode(status)
                return self._exit_code
        raise AssertionError("mkpj did not exit in time")


@pytest.fixture
def session(base_command, tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    sessions = []

    def start(*args: str) -> PtySession:
        env = {
            "TERM": "xterm-256color",
            "PYTHONPATH": str(SRC_DIR),
            "MKPJ_INSTALL_LOG": str(tmp_path / "install.log"),
            "MKPJ_LARGE_DOWNLOAD_FLAG": str(tmp_path / "large.flag"),
        }
        env.pop("NO_COLOR", None)
        argv = [*base_command, *args]
        if argv[0] == sys.executable:
            argv[0] = sys.executable
        started = PtySession(argv, sandbox, env)
        sessions.append(started)
        return started

    start.sandbox = sandbox
    yield start

    for started in sessions:
        try:
            os.close(started.fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Menu navigation
# ---------------------------------------------------------------------------
def test_profile_menu_opens_on_the_first_row(session):
    app = session("-n", "nav", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    assert app.selected_rows[-1].endswith("❯ Minimal")
    app.send(CTRL_C)
    app.wait()


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ((DOWN,), "Learner"),
        ((DOWN, DOWN), "Developer"),
        ((UP,), "Automation"),
        ((b"j", b"j", b"j"), "Library"),
        ((b"k", b"k"), "Research"),
    ],
)
def test_arrow_and_vim_keys_move_the_wheel(session, keys, expected):
    app = session("-n", "nav", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    for key in keys:
        app.send(key)
    assert app.selected_rows[-1].endswith(f"❯ {expected}")
    app.send(CTRL_C)
    app.wait()


def test_selection_wraps_around_both_ends(session):
    app = session("-n", "nav", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    app.send(UP)  # Minimal (index 0) wraps to the last row
    assert app.selected_rows[-1].endswith("❯ Automation")
    app.send(DOWN)
    assert app.selected_rows[-1].endswith("❯ Minimal")
    app.send(CTRL_C)
    app.wait()


def test_keys_buffered_during_a_redraw_are_not_dropped(session):
    """Regression guard: raw mode must not flush the pending input queue.

    Holding ↓ delivers several escape sequences in one read; TCSAFLUSH would
    silently discard every one that arrived mid-frame.
    """
    app = session("-n", "nav", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    app.send(DOWN + DOWN + DOWN)  # Minimal -> Learner -> Developer -> Library
    assert app.selected_rows[-1].endswith("❯ Library")
    app.send(CTRL_C)
    app.wait()


def test_every_line_returns_to_the_left_margin(session):
    """Regression guard: the wheel must not walk diagonally down the screen.

    ``tty.setraw`` clears OPOST, which stops the terminal translating "\n"
    into "\r\n". Each drawn line then starts where the previous one ended,
    the redraw rewinds to the wrong column, and the menu turns into a
    staircase of half-erased frames. Text-based assertions cannot see this,
    so check the bytes: while the menu is live, every newline must be paired
    with a carriage return.
    """
    app = session("-n", "margin", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    app.send(DOWN)
    app.send(DOWN)
    app.send(UP)

    bare_newlines = [
        index
        for index, char in enumerate(app.buffer)
        if char == "\n" and (index == 0 or app.buffer[index - 1] != "\r")
    ]
    assert not bare_newlines, (
        f"{len(bare_newlines)} newline(s) without a carriage return while the "
        f"menu was open — the terminal is in raw mode when it should be cbreak"
    )

    app.send(CTRL_C)
    app.wait()


def test_menu_hides_the_cursor_while_it_is_open(session):
    app = session("-n", "nav", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    assert "\x1b[?25l" in app.buffer
    app.send(CTRL_C)
    app.wait()
    assert "\x1b[?25h" in app.buffer  # and gives it back on the way out


# ---------------------------------------------------------------------------
# Interrupts
# ---------------------------------------------------------------------------
def test_ctrl_c_exits_cleanly_without_leaving_a_half_built_project(session):
    app = session("-n", "aborted", "--no-install", "--no-git", "--no-vscode")
    app.expect("Choose a project profile")
    app.send(CTRL_C)

    assert app.wait() == 130
    assert "Interrupted by user." in app.plain
    assert "mkpj stopped safely." in app.plain
    assert not (session.sandbox / "aborted").exists()


# ---------------------------------------------------------------------------
# A full interactive run
# ---------------------------------------------------------------------------
def test_full_interactive_run_generates_the_selected_profile(session):
    app = session("-n", "chosen", "--no-install", "--no-git", "--no-vscode")

    app.expect("Choose a project profile")
    app.send(DOWN + DOWN + DOWN)  # Minimal -> Learner -> Developer -> Library
    assert app.selected_rows[-1].endswith("❯ Library")
    app.send(ENTER)

    app.expect("Customize tooling bundles?")
    app.send(b"n\r")

    app.expect("Additional packages", timeout=120)
    app.send(ENTER)

    # The editor menu always ends with "Skip"; one k wraps straight to it.
    app.expect("Where do you want to open", timeout=120)
    app.send(b"k")
    assert app.selected_rows[-1].endswith("❯ Skip — do not open an editor")
    app.send(ENTER)

    assert app.wait() == 0
    assert "chosen is ready." in app.plain

    root = session.sandbox / "chosen"
    pyproject = read(root / "pyproject.toml")
    assert '"mkdocs"' in pyproject  # library profile default
    assert (root / "src" / "chosen" / "py.typed").is_file()
    assert (root / ".venv").is_dir()


def test_typed_project_name_is_validated_before_anything_is_created(session):
    app = session("--no-install", "--no-git", "--no-vscode")
    app.expect("Project name:")
    app.send(b"bad/name\r")
    app.expect("must be a directory name")
    app.send(b"good_name\r")
    app.expect("Choose a project profile")
    app.send(CTRL_C)
    app.wait()
    assert not (session.sandbox / "good_name").exists()
