"""Package-installation suite.

These tests install real packages and therefore need network access. They are
skipped unless you opt in::

    pytest --run-network
    pytest --run-network --run-extra   # optional ML frameworks
    pytest --run-network --run-heavy   # PyTorch, multi-GB download

Why the batching assertion matters: mkpj resolves each profile's packages in a
single combined ``pip install`` call. Installing one at a time lets pip
re-resolve against whatever the previous call already pinned, which can send
the resolver backtracking through dozens of releases — it was once observed
walking pytest all the way back from 8.4.0 to 6.1.0 to satisfy a constraint
introduced by an earlier, separately installed package.
"""

from __future__ import annotations

import pytest
from conftest import run_in_venv, venv_python, venv_script

pytestmark = [pytest.mark.network, pytest.mark.slow]

INSTALL_ARGS = ("--yes", "--no-git", "--no-vscode")

# profile -> (extra CLI args, modules that must import afterwards)
PROFILE_IMPORTS = {
    "learner": ((), ["IPython", "pytest", "ruff"]),
    "data": ((), ["jupyter", "matplotlib", "numpy", "pandas", "pytest", "ruff"]),
    "ml": (("--no-example",), ["jupyter", "matplotlib", "numpy", "pandas", "pytest", "ruff",
                               "sklearn"]),
    "automation": ((), ["dotenv", "pytest", "requests", "rich", "ruff"]),
    "library": ((), ["build", "mkdocs", "mypy", "pytest", "pytest_cov", "ruff"]),
}


def assert_imports(root, modules):
    code = "\n".join(f"import {module}" for module in modules)
    result = run_in_venv(root, code)
    assert result.returncode == 0, f"import failed:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Developer profile — also the batching regression guard
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def developer_install(session_runner):
    result = session_runner.run(
        "-n", "package_dev", "--profile", "developer", *INSTALL_ARGS, timeout=1800
    )
    assert result.returncode == 0, str(result)
    return result, session_runner.workdir / "package_dev"


def test_developer_profile_installs_its_dev_tools(developer_install):
    _, root = developer_install
    assert (root / ".venv").is_dir()
    assert venv_python(root).is_file()
    assert venv_script(root, "ruff").exists()
    assert venv_script(root, "pytest").exists()
    assert_imports(root, ["mypy", "pytest", "pytest_cov", "ruff"])


def test_dev_packages_are_resolved_in_a_single_pip_call(developer_install):
    result, _ = developer_install
    assert "Resolving 4 packages together" in result.output, str(result)


# ---------------------------------------------------------------------------
# Remaining profiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("profile", "extra_args", "modules"),
    [(name, args, modules) for name, (args, modules) in PROFILE_IMPORTS.items()],
    ids=list(PROFILE_IMPORTS),
)
def test_profile_installs_its_packages(session_runner, profile, extra_args, modules):
    root = session_runner.generate(
        f"package_{profile}", "--profile", profile, *extra_args, *INSTALL_ARGS, timeout=1800
    )
    assert_imports(root, modules)


def test_fastapi_web_profile_installs_its_framework_packages(session_runner):
    root = session_runner.generate(
        "package_web", "--profile", "web", "--framework", "fastapi", *INSTALL_ARGS, timeout=1800
    )
    assert_imports(root, ["fastapi", "mypy", "pytest", "pytest_cov", "ruff", "uvicorn"])


# ---------------------------------------------------------------------------
# Opt-in extras
# ---------------------------------------------------------------------------
@pytest.mark.extra
def test_xgboost_ml_framework_installs(session_runner):
    root = session_runner.generate(
        "package_xgb", "--profile", "ml", "--framework", "xgboost", "--no-example",
        *INSTALL_ARGS, timeout=1800,
    )
    assert_imports(root, ["xgboost"])


@pytest.mark.heavy
def test_dl_profile_installs_pytorch(session_runner):
    """PyTorch can pull several GB of CUDA/NVIDIA wheels and Triton.

    Watch progress with ``tail -f`` on the log path printed by the runner.
    """
    root = session_runner.generate(
        "package_dl", "--profile", "dl", "--framework", "pytorch", "--pytorch", "cpu",
        "--no-example", *INSTALL_ARGS, timeout=5400,
    )
    result = run_in_venv(root, "import torch; print(torch.__version__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.heavy
def test_pytorch_cpu_wheels_come_from_the_cpu_index(session_runner):
    """CPU mode must pass the dedicated index, not plain ``pip install torch``."""
    log = session_runner.log_path
    assert log.exists()
    assert "download.pytorch.org/whl/cpu" in log.read_text(encoding="utf-8", errors="ignore")
