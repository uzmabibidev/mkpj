"""Shared fixtures for the mkpj test suites.

The suites always exercise the mkpj in *this* repository, never a stale copy
that happens to be earlier on ``PATH``:

* by default the CLI runs as ``python -m mkpj`` with ``src/`` prepended to
  ``PYTHONPATH``;
* set ``MKPJ_BIN=/path/to/mkpj`` to test an installed console script instead
  (useful for verifying a ``pipx``/``uv tool install``).

Network-dependent suites are opt-in::

    pytest --run-network          # real pip installs
    pytest --run-network --run-extra    # + optional framework packages
    pytest --run-network --run-heavy    # + PyTorch (multi-GB download)

``MKPJ_TEST_NETWORK`` / ``MKPJ_TEST_EXTRA`` / ``MKPJ_TEST_HEAVY`` set to ``1``
work too, for CI entry points that prefer environment variables.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Unit tests import mkpj in-process; make the working tree importable even when
# the package has not been pip-installed into the current environment.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# The interpreter mkpj will actually use
#
# mkpj builds venvs with "python3" unless told otherwise, which is not
# necessarily the interpreter running the tests. PyTorch dropped 3.9, and mkpj
# refuses to install it below 3.10, so anything exercising a real PyTorch
# profile has to check the interpreter mkpj would pick.
# ---------------------------------------------------------------------------
def _default_venv_python_version() -> tuple:
    from mkpj.cli import resolve_interpreter

    executable = resolve_interpreter("python3") or sys.executable
    output = subprocess.check_output(
        [executable, "-c", "import sys; print(*sys.version_info[:2])"], text=True
    )
    major, minor = output.split()
    return int(major), int(minor)


DEFAULT_VENV_PYTHON = _default_venv_python_version()
PYTORCH_SUPPORTED = DEFAULT_VENV_PYTHON >= (3, 10)
PYTORCH_SKIP_REASON = (
    "mkpj refuses PyTorch below Python 3.10; the default interpreter is "
    f"Python {DEFAULT_VENV_PYTHON[0]}.{DEFAULT_VENV_PYTHON[1]}"
)

requires_pytorch_python = pytest.mark.skipif(not PYTORCH_SUPPORTED, reason=PYTORCH_SKIP_REASON)


# ---------------------------------------------------------------------------
# Options / markers
# ---------------------------------------------------------------------------
_OPT_TO_ENV = {
    "network": "MKPJ_TEST_NETWORK",
    "heavy": "MKPJ_TEST_HEAVY",
    "extra": "MKPJ_TEST_EXTRA",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", help="run tests that install packages")
    parser.addoption("--run-heavy", action="store_true", help="run the PyTorch download test")
    parser.addoption("--run-extra", action="store_true", help="run optional framework tests")


def _enabled(config: pytest.Config, name: str) -> bool:
    if config.getoption(f"--run-{name}"):
        return True
    return os.environ.get(_OPT_TO_ENV[name]) == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    reasons = {
        "network": "needs network access — pass --run-network (or MKPJ_TEST_NETWORK=1)",
        "heavy": "downloads several GB — pass --run-heavy (or MKPJ_TEST_HEAVY=1)",
        "extra": "optional framework packages — pass --run-extra (or MKPJ_TEST_EXTRA=1)",
    }
    enabled = {name: _enabled(config, name) for name in reasons}
    for item in items:
        for name, reason in reasons.items():
            if name in item.keywords and not enabled[name]:
                item.add_marker(pytest.mark.skip(reason=reason))
                break


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------
@dataclass
class Result:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def __str__(self) -> str:  # pragma: no cover - only used in failure output
        rendered = " ".join(shlex.quote(a) for a in self.args)
        return f"mkpj {rendered} -> {self.returncode}\n--- stderr ---\n{self.stderr}"


class MkpjRunner:
    """Invokes the CLI under test in a throwaway working directory."""

    def __init__(self, base_command: Sequence[str], workdir: Path, log_path: Path) -> None:
        self.base_command = list(base_command)
        self.workdir = workdir
        self.log_path = log_path

    def _env(self, overrides: dict | None = None) -> dict:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), existing]) if existing else str(SRC_DIR)
        env["MKPJ_INSTALL_LOG"] = str(self.log_path)
        env["MKPJ_LARGE_DOWNLOAD_FLAG"] = str(self.log_path.with_suffix(".flag"))
        env["NO_COLOR"] = "1"
        env.pop("VIRTUAL_ENV", None)
        if overrides:
            env.update(overrides)
        return env

    def run(
        self,
        *args: str,
        cwd: Path | None = None,
        timeout: int = 900,
        env_overrides: dict | None = None,
    ) -> Result:
        argv = [*self.base_command, *args]
        completed = subprocess.run(
            argv,
            cwd=str(cwd or self.workdir),
            env=self._env(env_overrides),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            # mkpj writes UTF-8; without this the harness decodes with the
            # system locale, which on Windows is cp1252 and cannot read the
            # ✓/◆/─ characters in mkpj's own output.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return Result(args, completed.returncode, completed.stdout, completed.stderr)

    def generate(self, name: str, *args: str, **kwargs) -> Path:
        """Generate a project and assert it succeeded; return its root path."""
        result = self.run("-n", name, *args, **kwargs)
        assert result.returncode == 0, str(result)
        root = (kwargs.get("cwd") or self.workdir) / name
        assert root.is_dir(), f"project directory missing: {root}"
        return root


@pytest.fixture(scope="session")
def base_command() -> list[str]:
    override = os.environ.get("MKPJ_BIN")
    if override:
        binary = Path(override)
        assert binary.exists(), f"MKPJ_BIN does not exist: {binary}"
        return [str(binary)]
    return [sys.executable, "-m", "mkpj"]


@pytest.fixture(scope="session")
def session_runner(base_command, tmp_path_factory) -> MkpjRunner:
    """Runner backed by one sandbox for the whole session (shared projects)."""
    sandbox = tmp_path_factory.mktemp("mkpj_session")
    return MkpjRunner(base_command, sandbox, sandbox / "install.log")


@pytest.fixture
def runner(base_command, tmp_path) -> MkpjRunner:
    """Runner with a fresh sandbox per test."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return MkpjRunner(base_command, sandbox, tmp_path / "install.log")


# ---------------------------------------------------------------------------
# Assertion helpers (importable from the test modules)
# ---------------------------------------------------------------------------
def read(path: Path) -> str:
    assert path.is_file(), f"file does not exist: {path}"
    return path.read_text(encoding="utf-8")


def assert_contains(path: Path, needle: str) -> None:
    content = read(path)
    assert needle in content, f"{needle!r} not found in {path}"


def assert_not_contains(path: Path, needle: str) -> None:
    content = read(path)
    assert needle not in content, f"{needle!r} unexpectedly found in {path}"


def venv_python(root: Path) -> Path:
    posix = root / ".venv" / "bin" / "python"
    return posix if posix.exists() else root / ".venv" / "Scripts" / "python.exe"


def venv_script(root: Path, name: str) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / f"{name}.exe"
    return root / ".venv" / "bin" / name


def run_in_venv(root: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(venv_python(root)), "-c", code], capture_output=True, text=True, timeout=300
    )
