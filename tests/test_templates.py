"""Profile resolution and file rendering, tested without touching the disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from mkpj.generator import (
    ProjectSpec,
    render_pyproject,
    render_vscode_extensions,
    render_vscode_settings,
)
from mkpj.profiles import (
    PROFILE_CHOICES,
    PROFILE_LABELS,
    apply_tooling_bundles,
    build_profile,
    default_framework,
    framework_packages,
    is_dev_package,
    split_dev_and_runtime,
    validate_framework,
)


def make_spec(**overrides) -> ProjectSpec:
    state = build_profile(overrides.pop("profile", "developer"), "demo")
    apply_tooling_bundles(state)
    dev, runtime = split_dev_and_runtime(state.packages)
    spec = ProjectSpec(
        project_name="demo",
        pkg_name="demo",
        profile=state.name,
        files_mode=state.files_mode,
        framework="",
        tooling=state.tooling,
        root=Path("/tmp/demo"),
        dev_packages=dev,
        runtime_packages=runtime,
        extensions=state.extensions,
        py_major=3,
        py_minor=12,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def test_profile_choices_and_labels_stay_in_sync():
    assert len(PROFILE_CHOICES) == len(PROFILE_LABELS)


@pytest.mark.parametrize("profile", PROFILE_CHOICES)
def test_every_profile_resolves(profile):
    state = build_profile(profile, "demo")
    assert state.summary
    assert "src/demo" in state.dirs
    assert "ms-python.python" in state.extensions


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        build_profile("nope", "demo")


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("cli", "argparse"), ("web", "fastapi"), ("ml", "scikit-learn"), ("dl", "pytorch")],
)
def test_framework_defaults_match_the_documented_ones(profile, expected):
    assert default_framework(profile) == expected


@pytest.mark.parametrize(
    ("profile", "framework", "valid"),
    [
        ("web", "fastapi", True),
        ("web", "typer", False),
        ("dl", "jax", True),
        ("dl", "xgboost", False),
        ("ml", "pytorch", True),
        ("developer", "", True),
    ],
)
def test_framework_validation(profile, framework, valid):
    assert validate_framework(profile, framework) is valid


def test_pytorch_skip_contributes_no_package():
    assert framework_packages("dl", "pytorch", "skip") == []
    assert framework_packages("dl", "pytorch", "cpu") == ["torch"]


def test_fastapi_brings_its_server():
    assert framework_packages("web", "fastapi", "") == ["fastapi", "uvicorn"]


def test_polars_replaces_pandas_when_selected():
    state = build_profile("ml", "demo")
    state.tooling.data_frame = "polars"
    apply_tooling_bundles(state)
    assert "polars" in state.packages
    assert "pandas" not in state.packages


def test_pandas_and_polars_can_coexist():
    state = build_profile("ml", "demo")
    state.tooling.data_frame = "pandas_polars"
    apply_tooling_bundles(state)
    assert {"pandas", "polars"} <= set(state.packages)


@pytest.mark.parametrize("package", ["ruff", "pytest", "mypy", "build", "mkdocs"])
def test_tooling_packages_are_dev_dependencies(package):
    assert is_dev_package(package)


@pytest.mark.parametrize("package", ["numpy", "fastapi", "torch", "requests"])
def test_runtime_packages_are_not_dev_dependencies(package):
    assert not is_dev_package(package)


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------
def test_pyproject_splits_dev_and_runtime_dependencies():
    content = render_pyproject(make_spec(profile="ml"))
    runtime_block = content.split("[dependency-groups]")[0]
    dev_block = content.split("[dependency-groups]")[1]
    assert '"numpy"' in runtime_block
    assert '"ruff"' in dev_block
    assert '"ruff"' not in runtime_block


def test_pyproject_targets_the_interpreter_of_the_venv():
    content = render_pyproject(make_spec(py_major=3, py_minor=11))
    assert 'target-version = "py311"' in content
    assert 'python_version = "3.11"' in content


def test_minimal_profile_emits_no_tool_sections():
    content = render_pyproject(make_spec(profile="minimal"))
    assert "[tool." not in content


def test_black_selection_writes_a_black_section():
    spec = make_spec()
    spec.tooling.quality = "black_pylint"
    assert "[tool.black]" in render_pyproject(spec)
    assert "[tool.ruff]" not in render_pyproject(spec)


def test_empty_dependency_lists_still_produce_valid_toml():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads(render_pyproject(make_spec(profile="minimal")))
    assert data["project"]["dependencies"] == []
    assert data["dependency-groups"]["dev"] == []


# ---------------------------------------------------------------------------
# VS Code
# ---------------------------------------------------------------------------
def test_vscode_settings_are_valid_json_and_point_at_the_venv():
    import json

    from mkpj.generator import venv_python_path

    settings = json.loads(render_vscode_settings(make_spec()))
    assert settings["python.defaultInterpreterPath"] == (
        "${workspaceFolder}/" + venv_python_path()
    )
    assert settings["python.testing.pytestEnabled"] is True
    assert settings["[python]"]["editor.defaultFormatter"] == "charliermarsh.ruff"
    assert settings["editor.codeActionsOnSave"]["source.fixAll"] == "explicit"


def test_learner_profile_does_not_format_on_save():
    import json

    settings = json.loads(render_vscode_settings(make_spec(profile="learner")))
    assert settings["editor.formatOnSave"] is False


def test_settings_without_a_formatter_omit_the_python_block():
    import json

    spec = make_spec(profile="minimal")
    settings = json.loads(render_vscode_settings(spec))
    assert "[python]" not in settings
    assert "source.fixAll" not in settings["editor.codeActionsOnSave"]


def test_extension_recommendations_are_valid_json():
    import json

    data = json.loads(render_vscode_extensions(["ms-python.python", "charliermarsh.ruff"]))
    assert data["recommendations"] == ["ms-python.python", "charliermarsh.ruff"]


def test_empty_recommendations_are_valid_json():
    import json

    assert json.loads(render_vscode_extensions([]))["recommendations"] == []


# ---------------------------------------------------------------------------
# Platform-specific paths
#
# The Windows branches are reachable on any OS by overriding generator's own
# IS_WINDOWS flag, so the layout they produce is checked here rather than only
# on a Windows CI runner. Patching os.name would do it too — and would also
# repoint pathlib, which then refuses to build a WindowsPath on POSIX and takes
# the whole test session down with it.
# ---------------------------------------------------------------------------
def test_venv_interpreter_path_follows_the_platform(monkeypatch):
    from mkpj import generator

    monkeypatch.setattr(generator, "IS_WINDOWS", False)
    assert generator.venv_python_path() == ".venv/bin/python"

    monkeypatch.setattr(generator, "IS_WINDOWS", True)
    assert generator.venv_python_path() == ".venv/Scripts/python.exe"


def test_vscode_settings_point_at_the_windows_interpreter(monkeypatch):
    import json

    from mkpj import generator

    monkeypatch.setattr(generator, "IS_WINDOWS", True)
    settings = json.loads(generator.render_vscode_settings(make_spec()))
    assert settings["python.defaultInterpreterPath"] == (
        "${workspaceFolder}/.venv/Scripts/python.exe"
    )


def test_generated_readme_uses_the_platform_interpreter(monkeypatch):
    from mkpj import generator

    spec = make_spec()
    monkeypatch.setattr(generator, "IS_WINDOWS", True)
    assert ".venv/Scripts/python.exe -m pytest" in generator._readme(spec)

    monkeypatch.setattr(generator, "IS_WINDOWS", False)
    assert ".venv/bin/python -m pytest" in generator._readme(spec)
