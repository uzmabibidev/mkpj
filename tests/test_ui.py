"""Interactive UI parity: the wheel menu, the spinner and the pip progress bar.

A terminal UI is easy to break silently, so each piece of it is
split into pure logic plus a thin IO layer, so the behaviour people actually
notice — wrap-around selection, the ❯ marker, frame order, bar colours, "did
you mean" ordering — is pinned by tests.
"""

from __future__ import annotations

import io
import re
import time

import pytest

from mkpj import menu, pipdriver, spinner
from mkpj.console import Style
from mkpj.suggest import fuzzy_suggest
from mkpj.textutil import dedupe, package_name_for, split_list, strip_item, trim


# ---------------------------------------------------------------------------
# Key decoding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("first", "rest", "expected"),
    [
        ("\x1b", "[A", menu.KEY_UP),
        ("\x1b", "[B", menu.KEY_DOWN),
        ("\x1b", "OA", menu.KEY_UP),      # application cursor mode
        ("\x1b", "OB", menu.KEY_DOWN),
        ("\x1b", "", menu.KEY_OTHER),     # bare Escape does nothing
        ("k", "", menu.KEY_UP),           # vim-style navigation
        ("j", "", menu.KEY_DOWN),
        ("\r", "", menu.KEY_ENTER),
        ("\n", "", menu.KEY_ENTER),
        ("", "", menu.KEY_ENTER),         # some terminals deliver Enter as an empty read
        ("q", "", menu.KEY_OTHER),
    ],
)
def test_decode_key(first, rest, expected):
    assert menu.decode_key(first, rest) == expected


@pytest.mark.parametrize(
    ("current", "action", "expected"),
    [
        (0, menu.KEY_UP, 3),      # wraps to the bottom
        (3, menu.KEY_DOWN, 0),    # wraps to the top
        (1, menu.KEY_UP, 0),
        (1, menu.KEY_DOWN, 2),
        (2, menu.KEY_OTHER, 2),   # unknown keys never move the wheel
    ],
)
def test_menu_selection_wraps_around(current, action, expected):
    assert menu.next_index(current, action, 4) == expected


def test_menu_render_marks_only_the_selected_row(monkeypatch):
    monkeypatch.setattr(menu, "STYLE", Style.disabled())
    rendered = menu.render_menu("Choose a project profile", ["Minimal", "Developer"], selected=1)
    lines = rendered.splitlines()
    assert lines[0] == "Choose a project profile"
    assert lines[1] == "      Minimal"
    assert lines[2] == "   ❯ Developer "
    assert rendered.count("❯") == 1


def test_menu_lines_end_with_crlf(monkeypatch):
    """A bare LF staircases wherever the driver is not translating newlines."""
    monkeypatch.setattr(menu, "STYLE", Style.disabled())
    rendered = menu.render_menu("Pick", ["One", "Two"], selected=0)

    assert rendered.endswith("\r\n")
    assert rendered.count("\r\n") == 3  # prompt + two options
    assert "\n" not in rendered.replace("\r\n", "")


def test_menu_redraw_sequence_keeps_the_wheel_in_place(monkeypatch, capsys):
    """Second and later draws rewind exactly len(options) + 1 lines."""
    options = ["One", "Two", "Three"]
    keys = iter([menu.KEY_DOWN, menu.KEY_DOWN, menu.KEY_UP, menu.KEY_ENTER])

    monkeypatch.setattr(menu, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(menu, "STYLE", Style.disabled())
    monkeypatch.setattr(menu, "read_key", lambda: next(keys))
    monkeypatch.setattr(menu.console, "hide_cursor", lambda: None)
    monkeypatch.setattr(menu.console, "restore_terminal", lambda: None)
    monkeypatch.setattr(menu, "flush_stdin", lambda: None)
    # STYLE.enabled is consulted before drawing; force the interactive path.
    monkeypatch.setattr(menu, "STYLE", Style(enabled=True, reset="", bold="", dim="", blue="",
                                             cyan="", green="", yellow="", red="", white="",
                                             gray=""))

    selected = menu.select_menu("Pick one", options)
    captured = capsys.readouterr().err

    assert selected == 1
    assert captured.count(f"\033[{len(options) + 1}A\033[J") == 3


def test_select_menu_falls_back_to_a_numbered_prompt_without_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(menu, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "3")
    assert menu.select_menu("Pick", ["a", "b", "c"]) == 2
    assert "3) c" in capsys.readouterr().err


def test_fallback_prompt_uses_the_default_on_eof(monkeypatch):
    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(menu, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr("builtins.input", raise_eof)
    assert menu.select_menu("Pick", ["a", "b", "c"], default=1) == 1


class _FakeStdin:
    def fileno(self):
        return 0

    def isatty(self):
        return True


def test_raw_mode_uses_cbreak_and_keeps_queued_keystrokes(monkeypatch):
    """Two things this mode must get right.

    ``cbreak`` not ``raw``: raw clears OPOST, so "\n" stops being translated
    to "\r\n" and the multi-line wheel walks diagonally across the screen.
    ``TCSANOW`` not ``TCSAFLUSH``: flushing discards keys typed while a frame
    was being drawn.
    """
    termios = pytest.importorskip("termios")
    tty = pytest.importorskip("tty")
    recorded = {}

    monkeypatch.setattr(menu, "_HAS_TERMIOS", True)
    monkeypatch.setattr(menu.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: "saved")
    monkeypatch.setattr(
        termios, "tcsetattr", lambda fd, when, attrs: recorded.__setitem__("restore", when)
    )
    monkeypatch.setattr(
        tty, "setcbreak", lambda fd, when=None: recorded.__setitem__("enter", when)
    )
    monkeypatch.setattr(
        tty, "setraw", lambda fd, when=None: recorded.__setitem__("raw_was_used", when)
    )

    with menu.raw_mode():
        assert menu._IN_RAW_MODE is True
        with menu.raw_mode():  # nesting must not re-enter
            pass

    assert recorded["enter"] == termios.TCSANOW
    assert recorded["restore"] == termios.TCSADRAIN
    assert "raw_was_used" not in recorded
    assert menu._IN_RAW_MODE is False


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------
def test_spinner_uses_the_ten_frame_braille_wheel():
    assert spinner.SPIN_FRAMES == ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def test_spinner_returns_the_workers_exit_code(monkeypatch):
    monkeypatch.setattr(spinner, "_animated", lambda: False)
    assert spinner.run_with_spinner("working", lambda: 0) == 0
    assert spinner.run_with_spinner("working", lambda: 7) == 7


def test_spinner_reports_done_or_failed_without_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(spinner, "_animated", lambda: False)
    spinner.run_with_spinner("Setting up .venv", lambda: 0)
    spinner.run_with_spinner("Upgrading pip", lambda: 1)
    err = capsys.readouterr().err
    assert "Setting up .venv done" in err
    assert "Upgrading pip failed" in err


def test_spinner_animates_and_clears_the_line(monkeypatch, capsys):
    monkeypatch.setattr(spinner, "_animated", lambda: True)
    monkeypatch.setattr(spinner, "_INTERVAL", 0.001)
    monkeypatch.setattr(spinner.console, "STYLE", Style(enabled=True))
    monkeypatch.setattr(spinner.console, "hide_cursor", lambda: None)

    def slow_work() -> int:
        time.sleep(0.02)
        return 0

    assert spinner.run_with_spinner("Installing", slow_work) == 0
    err = capsys.readouterr().err
    assert "Installing" in err
    assert any(frame in err for frame in spinner.SPIN_FRAMES)
    assert "\r\033[K" in err  # the spinner line is cleared when it finishes


def test_spinner_propagates_worker_exceptions(monkeypatch):
    monkeypatch.setattr(spinner, "_animated", lambda: False)

    def boom() -> int:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        spinner.run_with_spinner("Installing", boom)


# ---------------------------------------------------------------------------
# pip progress parsing and rendering
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


class _TtyStream(io.StringIO):
    """A stream that claims to be a terminal, so in-place redraws happen."""

    def isatty(self) -> bool:
        return True
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.whl", "numpy"),
        ("https://files.pythonhosted.org/x/scikit_learn-1.5.1-cp312.whl", "scikit-learn"),
        ("pytest-8.2.0-py3-none-any.whl", "pytest"),
        ("torch-2.3.1+cpu-cp312-cp312-linux_x86_64.whl", "torch"),
    ],
)
def test_artifact_names_are_normalised(filename, expected):
    assert pipdriver.clean_package_name(filename) == expected


def test_download_line_is_parsed_with_its_unit():
    assert pipdriver.parse_download("Downloading numpy-2.1.0.whl (16.3 MB)") == (
        "numpy",
        int(16.3 * 1024**2),
    )


def test_metadata_only_downloads_are_ignored():
    assert pipdriver.parse_download("Downloading numpy-2.1.0.whl.metadata (2.0 kB)") is None


def test_progress_and_collecting_lines_are_parsed():
    assert pipdriver.parse_progress("Progress 1024 of 4096") == (1024, 4096)
    assert pipdriver.parse_collecting("Collecting pytest-cov") == "pytest-cov"
    assert pipdriver.parse_collecting("Installing collected packages") is None


@pytest.mark.parametrize(
    ("downloaded", "total", "expected"),
    [
        (200, 900, "200/900 B"),
        (512, 1024, "0.5/1.0 KB"),
        (5 * 1024**2, 10 * 1024**2, "5.0/10.0 MB"),
        (1024**3, 3 * 1024**3, "1.0/3.0 GB"),
    ],
)
def test_size_pairs_use_the_unit_of_the_total(downloaded, total, expected):
    assert pipdriver.format_bytes_pair(downloaded, total) == expected


def test_size_pairs_never_mix_units():
    """Regression guard against output like ``519.2 KB/519.2 KB MB``."""
    rendered = pipdriver.format_bytes_pair(531661, 531661)
    assert rendered == "519.2/519.2 KB"
    assert rendered.count("KB") == 1
    assert "MB" not in rendered


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(0, "-- kB/s"), (2048, "2.0 kB/s"), (5 * 1024**2, "5.0 MB/s")],
)
def test_transfer_rate_formatting(rate, expected):
    assert pipdriver.format_rate(rate) == expected


@pytest.mark.parametrize(("seconds", "expected"), [(0, "00:00"), (75, "01:15"), (3725, "1:02:05")])
def test_eta_formatting(seconds, expected):
    assert pipdriver.format_eta(seconds) == expected


def _renderer(tmp_path, stream=None):
    return pipdriver.ProgressRenderer(
        stream=stream or io.StringIO(), flag_file=tmp_path / "flag"
    )


def test_progress_bar_fills_proportionally(tmp_path):
    renderer = _renderer(tmp_path)
    renderer.current_artifact = "numpy"
    renderer.total_bytes = 1000
    renderer.downloaded_bytes = 500
    renderer.started_at = time.time() - 1

    line = renderer.bar_line()
    half = renderer.BAR_WIDTH // 2
    assert line.count(renderer.FILL) == half
    assert line.count(renderer.EMPTY) == renderer.BAR_WIDTH - half
    assert "50.0%" in line

    done = renderer.bar_line(done=True)
    assert done.count(renderer.FILL) == renderer.BAR_WIDTH
    assert "100.0%" in done


def test_stats_line_shows_size_speed_and_eta(tmp_path):
    renderer = _renderer(tmp_path)
    renderer.current_artifact = "numpy"
    renderer.total_bytes = 10 * 1024**2
    renderer.downloaded_bytes = 5 * 1024**2
    renderer.started_at = time.time() - 2

    line = renderer.stats_line()
    assert "5.0/10.0 MB" in line
    assert "MB/s" in line
    assert "ETA " in line

    # A finished artifact reports how long it took instead of an estimate.
    assert "ETA" not in renderer.stats_line(done=True)


def test_download_block_is_name_then_bar_then_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(pipdriver, "STYLE", Style(enabled=True))
    stream = _TtyStream()
    renderer = _renderer(tmp_path, stream)
    renderer.feed("Collecting numpy")
    renderer.feed("Downloading numpy-2.1.0-cp312.whl (10.4 MB)")
    renderer.started_at = time.time() - 3
    renderer.feed("Progress 7864320 of 10905190")

    lines = [line for line in _strip_ansi(stream.getvalue()).split("\n") if line.strip()]
    assert lines[0].strip() == "numpy"
    assert renderer.FILL in lines[1] and "%" in lines[1]
    assert "/10.4 MB" in lines[2] and "ETA" in lines[2]


def test_cached_metadata_does_not_name_a_package_twice(tmp_path):
    """pip reports the metadata fetch as cached too; only the wheel counts."""
    stream = io.StringIO()
    renderer = _renderer(tmp_path, stream)
    renderer.feed("Using cached ruff-0.5.0-py3-none-any.whl.metadata (25 kB)")
    renderer.feed("Using cached ruff-0.5.0-py3-none-any.whl (11.0 MB)")

    assert _strip_ansi(stream.getvalue()).count("ruff (cached)") == 1


def test_cached_wheels_are_not_reported_as_downloads(tmp_path):
    stream = io.StringIO()
    renderer = _renderer(tmp_path, stream)
    renderer.feed("Using cached pandas-2.2.0-py3-none-any.whl (12.0 MB)")

    output = _strip_ansi(stream.getvalue())
    assert "pandas (cached)" in output
    assert renderer.FILL not in output
    assert renderer.total_bytes == 0


def test_renderer_warns_once_about_large_downloads(tmp_path):
    stream = io.StringIO()
    flag = tmp_path / "flag"
    renderer = pipdriver.ProgressRenderer(stream=stream, flag_file=flag)

    renderer.feed("Downloading torch-2.3.1-cp312.whl (780.0 MB)")
    renderer.feed("Downloading nvidia_cudnn-9.1.0-cp312.whl (700.0 MB)")

    assert stream.getvalue().count("Large download detected") == 1
    assert flag.exists()


def test_finished_bars_are_closed_before_the_next_one_starts(tmp_path):
    stream = io.StringIO()
    renderer = pipdriver.ProgressRenderer(stream=stream, flag_file=tmp_path / "flag")

    renderer.feed("Downloading numpy-2.1.0.whl (10.0 MB)")
    renderer.feed("Progress 5242880 of 10485760")
    renderer.feed("Downloading pandas-2.2.0.whl (12.0 MB)")

    output = _strip_ansi(stream.getvalue())
    assert "  numpy\n" in output
    assert "  pandas\n" in output
    # The abandoned numpy bar is completed before pandas takes over.
    assert output.index("10.0/10.0 MB") < output.index("  pandas")


# ---------------------------------------------------------------------------
# Fuzzy "did you mean …"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("typo", "expected"),
    [
        ("nunpy", "numpy"),
        ("pandsa", "pandas"),
        ("pytst", "pytest"),
        ("fastpi", "fastapi"),
        ("scikitlearn", "scikit-learn"),
    ],
)
def test_typos_suggest_the_intended_package(typo, expected):
    from mkpj.installer import KNOWN_PACKAGES

    assert expected in fuzzy_suggest(typo, KNOWN_PACKAGES)


def test_suggestions_are_capped_at_three():
    from mkpj.installer import KNOWN_PACKAGES

    assert len(fuzzy_suggest("py", KNOWN_PACKAGES)) <= 3


def test_nonsense_names_suggest_nothing():
    from mkpj.installer import KNOWN_PACKAGES

    assert fuzzy_suggest("zzzzzzzzzz", KNOWN_PACKAGES) == []


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a, b c", ["a", "b", "c"]),
        ("numpy\npandas\n", ["numpy", "pandas"]),
        ("  spaced  ,  out ", ["spaced", "out"]),
        (",,,", []),
        ("", []),
    ],
)
def test_list_input_accepts_commas_spaces_and_newlines(raw, expected):
    assert split_list(raw) == expected


def test_dedupe_is_case_insensitive_and_order_preserving():
    assert dedupe(["Ruff", "pytest", "ruff", "PyTest", "mypy"]) == ["Ruff", "pytest", "mypy"]


def test_strip_item_removes_every_casing():
    assert strip_item(["Torch", "numpy", "torch"], "torch") == ["numpy"]


@pytest.mark.parametrize(
    ("project", "package"),
    [("my-app", "my_app"), ("My-App", "my_app"), ("data_tool", "data_tool")],
)
def test_project_names_map_to_importable_packages(project, package):
    assert package_name_for(project) == package


def test_trim_strips_surrounding_whitespace():
    assert trim("  myapp \n") == "myapp"


def test_run_pip_reads_real_pip_output(tmp_path):
    # Runs a real (offline) pip command so every CI platform exercises the
    # pipe-reading loop, not just the parsing helpers.
    import sys

    log = tmp_path / "pip.log"
    assert pipdriver.run_pip(sys.executable, log, ["--version"]) == 0
    assert "pip" in log.read_text(encoding="utf-8")
