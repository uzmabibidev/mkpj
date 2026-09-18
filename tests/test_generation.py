"""Generation / integration suite.

Everything here runs with ``--no-install``: it validates generated structure,
configuration, CLI flags, framework selection and optional examples without
downloading a single package. Real installation lives in ``test_packages.py``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from conftest import (
    PYTORCH_SKIP_REASON,
    PYTORCH_SUPPORTED,
    assert_contains,
    assert_not_contains,
    read,
    venv_python,
)

pytestmark = pytest.mark.slow

ALL_PROFILES = [
    "minimal", "learner", "developer", "library", "cli",
    "web", "data", "ml", "dl", "research", "automation",
]

OFFLINE = ("--yes", "--no-install", "--no-vscode", "--no-git")


# ---------------------------------------------------------------------------
# Shared projects — generated once per session because each one builds a venv
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def minimal_project(session_runner):
    return session_runner.generate("mini_app", "--profile", "minimal", *OFFLINE)


@pytest.fixture(scope="session")
def developer_project(session_runner):
    return session_runner.generate("dev_app", "--profile", "developer", *OFFLINE)


@pytest.fixture(scope="session")
def ml_clean_project(session_runner):
    return session_runner.generate("ml_clean", "--profile", "ml", "--no-example", *OFFLINE)


@pytest.fixture(scope="session")
def ml_example_project(session_runner):
    return session_runner.generate(
        "ml_example", "--profile", "ml", "--example", "--yes", "--no-install", "--no-git"
    )


@pytest.fixture(scope="session")
def dl_cpu_project(session_runner):
    if not PYTORCH_SUPPORTED:
        pytest.skip(PYTORCH_SKIP_REASON)
    return session_runner.generate(
        "dl_cpu",
        "--profile", "dl", "--framework", "pytorch", "--pytorch", "cpu", "--example",
        *OFFLINE,
    )


# ---------------------------------------------------------------------------
# Base structure
# ---------------------------------------------------------------------------
def test_minimal_profile_generates_the_base_project(minimal_project):
    root = minimal_project
    assert (root / "src" / "mini_app").is_dir()
    assert (root / ".venv").is_dir()
    for name in ("pyproject.toml", "README.md", ".gitignore"):
        assert (root / name).is_file()
    assert (root / "src" / "mini_app" / "__init__.py").is_file()
    assert (root / "src" / "mini_app" / "main.py").is_file()


def test_minimal_profile_has_no_tooling_configuration(minimal_project):
    pyproject = minimal_project / "pyproject.toml"
    assert_contains(pyproject, 'requires-python = ">=3.9"')
    assert_not_contains(pyproject, "[tool.ruff]")
    assert_not_contains(pyproject, "[tool.pytest.ini_options]")


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_every_supported_profile_generates(session_runner, profile):
    # dl defaults to PyTorch, which mkpj refuses below Python 3.10.
    if profile == "dl" and not PYTORCH_SUPPORTED:
        pytest.skip(PYTORCH_SKIP_REASON)
    root = session_runner.generate(f"matrix_{profile}", "--profile", profile, *OFFLINE)
    assert (root / ".venv").is_dir()
    assert (root / "pyproject.toml").is_file()
    assert (root / "README.md").is_file()
    assert (root / ".gitignore").is_file()
    assert (root / "src" / f"matrix_{profile}").is_dir()


# ---------------------------------------------------------------------------
# Developer profile
# ---------------------------------------------------------------------------
def test_developer_profile_creates_source_and_test_layout(developer_project):
    assert (developer_project / "src" / "dev_app").is_dir()
    assert (developer_project / "tests").is_dir()
    assert (developer_project / "pyproject.toml").is_file()


@pytest.mark.parametrize("package", ['"ruff"', '"pytest"', '"pytest-cov"', '"mypy"'])
def test_developer_dev_dependencies_live_in_the_pep_735_group(developer_project, package):
    pyproject = developer_project / "pyproject.toml"
    assert_contains(pyproject, "[dependency-groups]")
    assert_contains(pyproject, "dev = [")
    assert_contains(pyproject, package)


@pytest.mark.parametrize("section", ["[tool.ruff]", "[tool.pytest.ini_options]", "[tool.mypy]"])
def test_developer_profile_configures_quality_tooling(developer_project, section):
    assert_contains(developer_project / "pyproject.toml", section)


# ---------------------------------------------------------------------------
# ML / DL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "directory",
    ["data/raw", "data/processed", "models", "notebooks", "configs", "src/ml_clean"],
)
def test_ml_profile_creates_the_clean_structure(ml_clean_project, directory):
    assert (ml_clean_project / directory).is_dir()


def test_ml_profile_declares_its_runtime_stack(ml_clean_project):
    pyproject = ml_clean_project / "pyproject.toml"
    for package in ('"numpy"', '"pandas"', '"matplotlib"', '"scikit-learn"'):
        assert_contains(pyproject, package)
    assert_not_contains(pyproject, '"torch"')
    assert_not_contains(pyproject, '"tensorflow"')


def test_ml_clean_structure_has_no_example(ml_clean_project):
    assert not (ml_clean_project / "examples" / "baseline.py").exists()


def test_ml_example_workflow_is_runnable_scikit_learn_code(ml_example_project):
    example = ml_example_project / "examples" / "baseline.py"
    content = read(example)
    for needle in (
        "example",
        "RandomForestClassifier",
        "train_test_split",
        "random_state",
        "production",
    ):
        assert needle in content


def test_ml_example_run_still_writes_vscode_config(ml_example_project):
    assert (ml_example_project / ".vscode" / "settings.json").is_file()
    assert (ml_example_project / ".vscode" / "extensions.json").is_file()
    assert_contains(ml_example_project / "pyproject.toml", '"scikit-learn"')


@pytest.mark.parametrize("directory", ["data", "models", "notebooks", "src/dl_cpu"])
def test_dl_pytorch_cpu_creates_the_expected_layout(dl_cpu_project, directory):
    assert (dl_cpu_project / directory).is_dir()


def test_dl_pytorch_cpu_generates_the_example_workflow(dl_cpu_project):
    content = read(dl_cpu_project / "examples" / "baseline.py")
    assert "torch" in content
    assert "production" in content


def test_dl_pytorch_cpu_pins_the_pytorch_python_floor(dl_cpu_project):
    pyproject = dl_cpu_project / "pyproject.toml"
    assert_contains(pyproject, '"torch"')
    assert_contains(pyproject, '"matplotlib"')
    assert_contains(pyproject, 'requires-python = ">=3.10"')


def test_torch_is_declared_exactly_once(dl_cpu_project):
    """Regression guard: the framework package must not be added twice."""
    assert read(dl_cpu_project / "pyproject.toml").count('"torch"') == 1


def test_pytorch_skip_mode_omits_the_dependency(session_runner):
    root = session_runner.generate(
        "dl_skip",
        "--profile", "dl", "--framework", "pytorch", "--pytorch", "skip", "--no-example",
        *OFFLINE,
    )
    pyproject = root / "pyproject.toml"
    assert_contains(pyproject, 'requires-python = ">=3.9"')
    assert_not_contains(pyproject, '"torch"')
    assert not (root / "examples" / "baseline.py").exists()


def test_ml_pytorch_skip_keeps_the_rest_of_the_stack(session_runner):
    root = session_runner.generate(
        "ml_skip",
        "--profile", "ml", "--framework", "pytorch", "--pytorch", "skip", "--no-example",
        *OFFLINE,
    )
    pyproject = root / "pyproject.toml"
    assert_not_contains(pyproject, '"torch"')
    for package in ('"numpy"', '"pandas"', '"matplotlib"'):
        assert_contains(pyproject, package)


# ---------------------------------------------------------------------------
# Environment, git and VS Code integration
# ---------------------------------------------------------------------------
def test_no_install_creates_the_environment_without_integration_setup(session_runner):
    root = session_runner.generate(
        "no_install_app", "--profile", "developer", "--yes", "--no-install", "--no-git",
        "--no-vscode",
    )
    # The exact contents of a fresh venv vary by Python distribution, so only
    # the interpreter and the absence of integration setup are checked.
    assert (root / ".venv").is_dir()
    assert venv_python(root).is_file()
    assert (root / "pyproject.toml").is_file()
    assert not (root / ".vscode").exists()
    assert not (root / ".git").exists()


def test_git_and_vscode_can_be_suppressed(session_runner):
    root = session_runner.generate("isolated_app", "--profile", "minimal", *OFFLINE)
    assert not (root / ".git").exists()
    assert not (root / ".vscode").exists()
    assert (root / "pyproject.toml").is_file()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_git_and_vscode_are_generated_by_default(session_runner):
    root = session_runner.generate(
        "integrated_app", "--profile", "minimal", "--yes", "--no-install"
    )
    assert (root / ".git").is_dir()
    assert (root / ".vscode" / "settings.json").is_file()
    assert (root / ".vscode" / "extensions.json").is_file()
    assert_contains(root / ".vscode" / "settings.json", "python.defaultInterpreterPath")


MODEL_ARTIFACTS = ["model.pkl", "model.pt", "model.pth", "model.keras", "model.onnx"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_model_artifacts_are_actually_ignored_by_git(session_runner):
    root = session_runner.generate("artifact_app", "--profile", "ml", "--no-example", *OFFLINE)

    gitignore = read(root / ".gitignore")
    for pattern in ("models/*.pkl", "models/*.pt", "models/*.pth",
                    "models/*.keras", "models/*.onnx"):
        assert pattern in gitignore

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for artifact in MODEL_ARTIFACTS:
        (root / "models" / artifact).touch()

    for artifact in MODEL_ARTIFACTS:
        result = subprocess.run(
            ["git", "check-ignore", "-q", f"models/{artifact}"], cwd=root
        )
        assert result.returncode == 0, f"{artifact} is not an active git-ignore rule"


# ---------------------------------------------------------------------------
# pyproject correctness
# ---------------------------------------------------------------------------
def test_pyproject_python_version_placeholders_are_expanded(session_runner):
    # The version digits are interpolated, not left as literal placeholders.
    root = session_runner.generate("version_check", "--profile", "developer", *OFFLINE)
    pyproject = root / "pyproject.toml"

    assert_not_contains(pyproject, "${PY_MAJOR")
    assert_not_contains(pyproject, "${PY_MINOR")

    major, minor = subprocess.check_output(
        [str(venv_python(root)), "-c",
         "import sys; print(sys.version_info.major, sys.version_info.minor)"],
        text=True,
    ).split()

    assert_contains(pyproject, f'target-version = "py{major}{minor}"')
    assert_contains(pyproject, f'python_version = "{major}.{minor}"')


def test_generated_pyproject_is_valid_toml(developer_project):
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads(read(developer_project / "pyproject.toml"))
    assert data["project"]["name"] == "dev_app"
    assert "ruff" in data["dependency-groups"]["dev"]


# ---------------------------------------------------------------------------
# Frameworks and CLI contract
# ---------------------------------------------------------------------------
def test_cli_framework_selection_adds_the_selected_framework(session_runner):
    root = session_runner.generate(
        "web_fastapi", "--profile", "web", "--framework", "fastapi", *OFFLINE
    )
    pyproject = root / "pyproject.toml"
    assert_contains(pyproject, '"fastapi"')
    assert_contains(pyproject, '"uvicorn"')
    assert_not_contains(pyproject, '"flask"')
    assert_not_contains(pyproject, '"django"')


def test_cli_profile_declares_a_console_script(session_runner):
    root = session_runner.generate("tool_app", "--profile", "cli", "--framework", "typer", *OFFLINE)
    assert_contains(root / "pyproject.toml", "[project.scripts]")
    assert_contains(root / "pyproject.toml", 'tool_app = "tool_app.cli:main"')
    assert_contains(root / "src" / "tool_app" / "cli.py", "typer")


def test_library_profile_ships_a_py_typed_marker(session_runner):
    root = session_runner.generate("lib_app", "--profile", "library", *OFFLINE)
    assert (root / "src" / "lib_app" / "py.typed").is_file()


def test_extra_extensions_are_merged_into_the_recommendations(session_runner):
    root = session_runner.generate(
        "ext_app", "--profile", "minimal", "-e", "eamodio.gitlens, usernamehw.errorlens",
        "--yes", "--no-install", "--no-git",
    )
    recommendations = read(root / ".vscode" / "extensions.json")
    assert "eamodio.gitlens" in recommendations
    assert "usernamehw.errorlens" in recommendations


def test_invalid_pytorch_mode_is_rejected(runner):
    result = runner.run(
        "-n", "invalid_torch", "--profile", "dl", "--framework", "pytorch",
        "--pytorch", "potato", *OFFLINE,
    )
    assert result.returncode == 2, str(result)
    assert not (runner.workdir / "invalid_torch").exists()


def test_unsupported_framework_for_profile_is_rejected(runner):
    result = runner.run("-n", "bad_fw", "--profile", "web", "--framework", "typer", *OFFLINE)
    assert result.returncode == 2, str(result)
    assert not (runner.workdir / "bad_fw").exists()


def test_existing_directory_is_never_overwritten(runner):
    (runner.workdir / "taken").mkdir()
    result = runner.run("-n", "taken", "--profile", "minimal", *OFFLINE)
    assert result.returncode != 0
    assert "already exists" in result.output
    assert not (runner.workdir / "taken" / "pyproject.toml").exists()
