"""The PyTorch path: its own pip call, and the dependency it leaves behind.

PyTorch cannot go through the normal batch because the CPU/CUDA wheel index
has to be passed as its own flag. These tests drive the real CLI with pip
stubbed out, so the wiring is covered without downloading several GB. The
actual download is exercised by the opt-in ``--run-heavy`` suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import PYTORCH_SKIP_REASON, PYTORCH_SUPPORTED, read

from mkpj import cli, installer

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not PYTORCH_SUPPORTED, reason=PYTORCH_SKIP_REASON),
]


@pytest.fixture
def pip(monkeypatch, tmp_path):
    """Record pip invocations instead of running them; keep the venv real."""
    calls: list[list[str]] = []

    def fake_run_pip(python_exe, log_path, args):
        calls.append(list(args))
        return 0

    real_spinner = cli.run_command_with_spinner

    def spinner(message, argv, **kwargs):
        if "pip" in argv:  # the pip self-upgrade; venv creation still runs
            return 0
        return real_spinner(message, argv, **kwargs)

    monkeypatch.setattr(installer, "run_pip", fake_run_pip)
    monkeypatch.setattr(cli, "run_command_with_spinner", spinner)
    monkeypatch.setattr(cli, "INSTALL_LOG", tmp_path / "install.log")
    monkeypatch.setattr(cli, "LARGE_DOWNLOAD_FLAG", tmp_path / "large.flag")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    return SimpleNamespace(calls=calls, sandbox=sandbox)


def generate(name: str, *args: str) -> int:
    return cli.run_mkpj(["-n", name, "--yes", "--no-git", "--no-vscode", *args])


DL_CPU = ("--profile", "dl", "--framework", "pytorch", "--pytorch", "cpu", "--no-example")


def test_torch_is_installed_from_the_cpu_index_on_its_own(pip):
    assert generate("dl_cpu", *DL_CPU) == 0

    torch_calls = [call for call in pip.calls if "torch" in call]
    assert len(torch_calls) == 1
    assert torch_calls[0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]

    # …and is kept out of the batched calls, which have no index of their own.
    batched = [call for call in pip.calls if call is not torch_calls[0]]
    assert batched and all("torch" not in call for call in batched)


def test_installed_torch_still_appears_in_the_dependencies(pip):
    """Installing torch separately must not drop it from the dependencies.

    torch cannot go through the normal batch, because the CPU/CUDA wheel index
    has to be passed as its own flag. It is therefore removed from the list
    handed to pip — but not from the list written to pyproject.toml, which is
    what this pins.
    """
    assert generate("dl_deps", *DL_CPU) == 0

    pyproject = read(pip.sandbox / "dl_deps" / "pyproject.toml")
    assert pyproject.count('"torch"') == 1
    assert 'requires-python = ">=3.10"' in pyproject


def test_cuda_mode_installs_from_the_default_index(pip):
    assert generate("dl_cuda", "--profile", "dl", "--framework", "pytorch",
                    "--pytorch", "cuda", "--no-example") == 0

    torch_calls = [call for call in pip.calls if "torch" in call]
    assert len(torch_calls) == 1
    assert "--index-url" not in torch_calls[0]


def test_skip_mode_installs_no_torch_and_declares_none(pip):
    assert generate("dl_skip", "--profile", "dl", "--framework", "pytorch",
                    "--pytorch", "skip", "--no-example") == 0

    assert all("torch" not in call for call in pip.calls)

    pyproject = read(pip.sandbox / "dl_skip" / "pyproject.toml")
    assert '"torch"' not in pyproject
    assert 'requires-python = ">=3.9"' in pyproject


def test_ml_profile_with_pytorch_keeps_the_rest_of_the_stack(pip):
    assert generate("ml_torch", "--profile", "ml", "--framework", "pytorch",
                    "--pytorch", "cpu", "--no-example") == 0

    pyproject = read(pip.sandbox / "ml_torch" / "pyproject.toml")
    for package in ('"torch"', '"numpy"', '"pandas"', '"matplotlib"'):
        assert package in pyproject
