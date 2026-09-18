"""CLI contract: flag parsing, help output and exit codes.

These run in-process (no venv, no network), so they stay fast enough to run on
every save.
"""

from __future__ import annotations

import pytest

from mkpj.cli import HELP_TEXT, Options, parse_args
from mkpj.console import MkpjExit


def test_positional_name_is_accepted():
    assert parse_args(["myapp"]).project_name == "myapp"


def test_named_flags_override_defaults():
    opts = parse_args(
        [
            "-n", "myapp",
            "--profile", "ML",
            "--framework", "PyTorch",
            "--pytorch", "CPU",
            "-p", "fastapi,pydantic",
            "-e", "charliermarsh.ruff",
            "--yes", "--no-git", "--no-vscode", "--no-install",
            "--python", "python3.12",
        ]
    )
    assert opts == Options(
        project_name="myapp",
        profile="ml",
        framework="pytorch",
        packages="fastapi,pydantic",
        extensions="charliermarsh.ruff",
        example=None,
        pytorch="cpu",
        assume_yes=True,
        do_git=False,
        do_vscode=False,
        do_install=False,
        python_bin="python3.12",
    )


def test_type_is_an_alias_for_profile():
    assert parse_args(["--type", "web"]).profile == "web"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(["--example"], True), (["--no-example"], False), ([], None)],
)
def test_example_flag_is_tri_state(argv, expected):
    """Unset means 'ask'; that is why it is not a plain boolean."""
    assert parse_args(argv).example is expected


@pytest.mark.parametrize("flag", ["--profile", "--framework", "--pytorch", "-n", "-p", "-e",
                                  "--python"])
def test_missing_option_value_exits_with_2(flag, capsys):
    with pytest.raises(MkpjExit) as excinfo:
        parse_args([flag])
    assert excinfo.value.code == 2
    assert "requires a value" in capsys.readouterr().err


def test_unknown_option_exits_with_1(capsys):
    with pytest.raises(MkpjExit) as excinfo:
        parse_args(["--wat"])
    assert excinfo.value.code == 1
    assert "Unknown option: --wat" in capsys.readouterr().err


def test_help_exits_zero_and_prints_usage(capsys):
    with pytest.raises(MkpjExit) as excinfo:
        parse_args(["--help"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == HELP_TEXT


@pytest.mark.parametrize(
    "flag", ["--example", "--no-example", "--no-install", "--pytorch <mode>", "--framework",
             "--profile, --type", "--no-git", "--no-vscode"],
)
def test_help_documents_every_flag(flag):
    assert flag in HELP_TEXT


def test_help_is_written_to_stdout_by_the_real_process(runner):
    result = runner.run("--help")
    assert result.returncode == 0
    assert "--pytorch <mode>" in result.stdout
    assert result.stderr == ""


def test_help_survives_a_stream_that_cannot_encode_arrows(runner):
    """Regression guard for Windows.

    A redirected stdout there defaults to cp1252, which has no arrows, so
    printing the help died with UnicodeEncodeError. PYTHONIOENCODING
    reproduces the same stream on any platform.
    """
    result = runner.run("--help", env_overrides={"PYTHONIOENCODING": "cp1252"})

    assert result.returncode == 0, str(result)
    assert "mkpj" in result.stdout
    assert "--pytorch <mode>" in result.stdout


def test_project_name_may_not_be_a_path(runner):
    result = runner.run("-n", "nested/app", "--profile", "minimal", "--yes", "--no-install")
    assert result.returncode == 2
    assert "not a path" in result.output
