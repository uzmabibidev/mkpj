"""Install batching and the interactive retry flow, driven by a fake pip.

No network, no venv: ``run_pip`` and the menu are replaced, so the decision
logic people hit when an install fails is testable end to end.
"""

from __future__ import annotations

import pytest

from mkpj import installer as installer_module
from mkpj import menu
from mkpj.installer import EXTENSION, PACKAGE, Installer

RETRY_AS_IS, ENTER_ALTERNATIVES, SKIP_ONE, SKIP_ALL = 0, 1, 2, 3


class FakePip:
    """Records pip invocations and fails for a configurable set of names."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def __call__(self, python_exe, log_path, args):
        self.calls.append(list(args))
        names = [a for a in args if not a.startswith("-") and a != "install"]
        return 1 if any(name in self.failing for name in names) else 0


@pytest.fixture
def pip(monkeypatch):
    fake = FakePip()
    monkeypatch.setattr(installer_module, "run_pip", fake)
    return fake


@pytest.fixture
def installer(pip, tmp_path):
    return Installer(python_exe="python", log_path=tmp_path / "install.log")


def fake_menu(monkeypatch, choices):
    answers = iter(choices)
    monkeypatch.setattr(menu, "select_menu", lambda *a, **k: next(answers))


def fake_input(monkeypatch, replies):
    answers = iter(replies)
    monkeypatch.setattr(menu, "ask", lambda *a, **k: next(answers))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def test_multiple_packages_are_resolved_in_one_call(installer, pip, capsys):
    installer.install_batch(PACKAGE, ["ruff", "pytest", "pytest-cov", "mypy"])

    assert len(pip.calls) == 1
    assert pip.calls[0][-4:] == ["ruff", "pytest", "pytest-cov", "mypy"]
    assert "Resolving 4 packages together" in capsys.readouterr().err
    assert installer.installed_packages == ["ruff", "pytest", "pytest-cov", "mypy"]


def test_a_single_package_skips_the_batch_message(installer, pip, capsys):
    installer.install_batch(PACKAGE, ["ruff"])
    assert "Resolving" not in capsys.readouterr().err
    assert pip.calls == [["install", "--progress-bar", "raw", "ruff"]]


def test_a_failing_batch_falls_back_to_one_package_at_a_time(monkeypatch, tmp_path, capsys):
    pip = FakePip(failing={"broken"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    installer = Installer(python_exe="python", log_path=tmp_path / "log", interactive=False)

    installer.install_batch(PACKAGE, ["ruff", "broken", "pytest"])

    assert len(pip.calls) == 4  # one batched attempt + three individual ones
    assert installer.installed_packages == ["ruff", "pytest"]
    assert installer.failed_packages == ["broken"]
    assert "retrying packages individually" in capsys.readouterr().err


def test_progress_bar_flag_is_always_passed(installer, pip):
    installer.install_batch(PACKAGE, ["ruff", "pytest"])
    assert "--progress-bar" in pip.calls[0]
    assert "raw" in pip.calls[0]


# ---------------------------------------------------------------------------
# Retry flow
# ---------------------------------------------------------------------------
def test_failure_shows_a_did_you_mean_suggestion(monkeypatch, tmp_path, capsys):
    pip = FakePip(failing={"nunpy"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    fake_menu(monkeypatch, [SKIP_ONE])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["nunpy"])

    err = capsys.readouterr().err
    assert "Did you mean:" in err
    assert "numpy" in err


def test_retry_as_is_reinstalls_the_same_name(monkeypatch, tmp_path):
    pip = FakePip(failing={"flaky"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    fake_menu(monkeypatch, [RETRY_AS_IS])
    fake_input(monkeypatch, ["n"])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["flaky"])

    assert [call[-1] for call in pip.calls] == ["flaky", "flaky"]
    assert installer.failed_packages == ["flaky"]


def test_alternatives_accept_a_comma_separated_list(monkeypatch, tmp_path):
    pip = FakePip(failing={"nunpy"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    fake_menu(monkeypatch, [ENTER_ALTERNATIVES])
    fake_input(monkeypatch, ["numpy, scipy"])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["nunpy"])

    assert installer.installed_packages == ["numpy", "scipy"]
    assert installer.failed_packages == []


def test_empty_alternatives_keep_the_original_failure(monkeypatch, tmp_path):
    pip = FakePip(failing={"nunpy"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    fake_menu(monkeypatch, [ENTER_ALTERNATIVES])
    fake_input(monkeypatch, ["", "n"])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["nunpy"])
    assert installer.failed_packages == ["nunpy"]


def test_skip_one_drops_only_that_package(monkeypatch, tmp_path):
    pip = FakePip(failing={"bad1", "bad2"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    fake_menu(monkeypatch, [SKIP_ONE, SKIP_ONE])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["bad1", "bad2"])
    assert installer.failed_packages == []


def test_skip_all_stops_asking_about_the_rest(monkeypatch, tmp_path, capsys):
    pip = FakePip(failing={"bad1", "bad2", "bad3"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    fake_menu(monkeypatch, [SKIP_ALL])  # a second prompt would raise StopIteration
    fake_input(monkeypatch, ["n"])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["bad1", "bad2", "bad3"])

    assert installer.failed_packages == ["bad1", "bad2", "bad3"]
    assert "Giving up on: bad1 bad2 bad3" in capsys.readouterr().err


def test_retry_prompt_counts_remaining_failures(monkeypatch, tmp_path):
    seen = []
    pip = FakePip(failing={"bad1", "bad2"})
    monkeypatch.setattr(installer_module, "run_pip", pip)
    monkeypatch.setattr(
        menu, "select_menu", lambda prompt, options, **k: seen.append(options[SKIP_ALL]) or SKIP_ALL
    )
    fake_input(monkeypatch, ["n"])
    installer = Installer(python_exe="python", log_path=tmp_path / "log")

    installer.install_batch(PACKAGE, ["bad1", "bad2"])
    assert seen == ["Skip all remaining (2 left)"]


def test_non_interactive_runs_never_prompt(monkeypatch, tmp_path):
    pip = FakePip(failing={"bad"})
    monkeypatch.setattr(installer_module, "run_pip", pip)

    def explode(*_args, **_kwargs):
        raise AssertionError("the menu must not open with --yes")

    monkeypatch.setattr(menu, "select_menu", explode)
    installer = Installer(python_exe="python", log_path=tmp_path / "log", interactive=False)

    installer.install_batch(PACKAGE, ["bad"])
    assert installer.failed_packages == ["bad"]


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
def test_cpu_mode_uses_the_dedicated_index(installer, pip):
    assert installer.install_pytorch("cpu") is True
    assert pip.calls[0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert installer.installed_packages == ["torch"]


def test_cuda_mode_uses_the_default_index(installer, pip, capsys):
    installer.install_pytorch("cuda")
    assert "--index-url" not in pip.calls[0]
    assert "several GB" in capsys.readouterr().err


def test_skip_mode_installs_nothing(installer, pip, capsys):
    assert installer.install_pytorch("skip") is True
    assert pip.calls == []
    assert "Skipping PyTorch installation." in capsys.readouterr().err


# ---------------------------------------------------------------------------
# VS Code extensions
# ---------------------------------------------------------------------------
def test_extensions_are_installed_one_at_a_time(monkeypatch, installer):
    calls = []
    monkeypatch.setattr(installer, "_install_extension", lambda name: calls.append(name) or 0)

    installer.install_batch(EXTENSION, ["ms-python.python", "charliermarsh.ruff"])

    assert calls == ["ms-python.python", "charliermarsh.ruff"]
    assert installer.installed_extensions == calls


def test_missing_code_cli_marks_extensions_as_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(installer_module.shutil, "which", lambda _cmd: None)
    installer = Installer(python_exe="python", log_path=tmp_path / "log", interactive=False)

    installer.install_batch(EXTENSION, ["ms-python.python"])
    assert installer.failed_extensions == ["ms-python.python"]
