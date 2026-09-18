"""Profile catalog and the tooling bundles each profile turns on.

A profile only supplies sensible defaults. Alternatives appear when the user
asks to customise; ``--yes`` keeps the documented defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .textutil import dedupe

__all__ = [
    "PROFILE_CHOICES",
    "PROFILE_LABELS",
    "Tooling",
    "ProfileState",
    "build_profile",
    "apply_tooling_bundles",
    "is_dev_package",
    "framework_options",
    "default_framework",
    "validate_framework",
    "framework_packages",
    "DEV_PACKAGE_NAMES",
]

PROFILE_CHOICES: tuple[str, ...] = (
    "minimal", "learner", "developer", "library", "cli", "web",
    "data", "ml", "dl", "research", "automation",
)

PROFILE_LABELS: tuple[str, ...] = (
    "Minimal", "Learner", "Developer", "Library", "CLI", "Web / API",
    "Data Science", "Machine Learning", "Deep Learning", "Research", "Automation",
)

DEV_PACKAGE_NAMES = {
    "ruff", "pytest", "pytest-cov", "pytest-mock", "mypy", "pyright", "black", "ipython",
    "build", "pre-commit", "tox", "coverage", "flake8", "pylint", "bandit", "mkdocs", "sphinx",
}


def is_dev_package(name: str) -> bool:
    return name.lower() in DEV_PACKAGE_NAMES


@dataclass
class Tooling:
    quality: str = "none"
    testing: str = "none"
    type_checker: str = "none"
    visualization: str = "none"
    data_frame: str = "none"
    docs: str = "none"
    security: str = "none"
    git_workflow: str = "none"


@dataclass
class ProfileState:

    name: str
    packages: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    files_mode: str = "standard"
    tooling: Tooling = field(default_factory=Tooling)
    summary: str = ""

    def add_package(self, candidate: str) -> None:
        if candidate.lower() not in {p.lower() for p in self.packages}:
            self.packages.append(candidate)

    def add_packages(self, *candidates: str) -> None:
        for candidate in candidates:
            self.add_package(candidate)

    def remove_package(self, name: str) -> None:
        self.packages = [p for p in self.packages if p.lower() != name.lower()]


_BASE_EXTENSIONS = ("ms-python.python", "ms-python.vscode-pylance")


def build_profile(name: str, pkg_name: str) -> ProfileState:
    src = f"src/{pkg_name}"
    state = ProfileState(name=name, extensions=list(_BASE_EXTENSIONS), dirs=[src])
    t = state.tooling

    if name == "minimal":
        state.files_mode = "minimal"
        state.summary = "No quality, testing, or type-checking tools"
    elif name == "learner":
        state.packages = ["ipython"]
        state.extensions += ["ms-python.black-formatter", "ms-python.pylint"]
        state.dirs = [src, "tests"]
        t.quality, t.testing = "ruff", "pytest"
        state.summary = "Ruff + pytest + IPython; type checking off"
    elif name == "developer":
        state.dirs = [src, "tests"]
        t.quality, t.testing, t.type_checker = "ruff", "pytest_cov", "mypy"
        state.summary = "Ruff + pytest + coverage + mypy"
    elif name == "library":
        state.packages = ["build"]
        state.dirs = [src, "tests"]
        state.files_mode = "library"
        t.quality, t.testing, t.type_checker, t.docs = "ruff", "pytest_cov", "mypy", "mkdocs"
        state.summary = "Ruff + pytest + coverage + mypy + build + MkDocs"
    elif name == "cli":
        state.dirs = [src, "tests"]
        state.files_mode = "cli"
        t.quality, t.testing, t.type_checker = "ruff", "pytest_cov", "mypy"
        state.summary = "Ruff + pytest + coverage + mypy"
    elif name == "web":
        state.dirs = [src, "tests"]
        state.files_mode = "web"
        t.quality, t.testing, t.type_checker = "ruff", "pytest_cov", "mypy"
        state.summary = "Ruff + pytest + coverage + mypy; choose a web framework"
    elif name == "data":
        state.packages = ["numpy", "pandas", "jupyter"]
        state.extensions += ["ms-toolsai.jupyter"]
        state.dirs = [
            "data/raw", "data/interim", "data/processed", "notebooks", "reports", src, "tests",
        ]
        state.files_mode = "data"
        t.quality, t.testing = "ruff", "pytest"
        t.visualization, t.data_frame = "matplotlib", "pandas"
        state.summary = "NumPy + pandas + Jupyter + Matplotlib + Ruff + pytest"
    elif name == "ml":
        state.packages = ["numpy", "pandas", "jupyter"]
        state.extensions += ["ms-toolsai.jupyter"]
        state.dirs = [
            "data/raw", "data/processed", "notebooks", "models", "configs", src, "tests",
        ]
        state.files_mode = "ml"
        t.quality, t.testing = "ruff", "pytest"
        t.visualization, t.data_frame = "matplotlib", "pandas"
        state.summary = "NumPy + pandas + scikit-learn + Jupyter + Matplotlib + Ruff + pytest"
    elif name == "dl":
        state.extensions += ["ms-toolsai.jupyter"]
        state.dirs = [
            "data/raw", "data/processed", "notebooks", "models", "configs", src, "tests",
        ]
        state.files_mode = "dl"
        t.quality, t.testing = "ruff", "pytest"
        t.visualization = "matplotlib"
        state.summary = "PyTorch + NumPy + Jupyter + Matplotlib + Ruff + pytest"
    elif name == "research":
        state.packages = ["numpy", "pandas", "jupyter"]
        state.extensions += ["ms-toolsai.jupyter"]
        state.dirs = [
            "data", "notebooks", "experiments", "results", "figures", "configs", src, "tests",
        ]
        state.files_mode = "research"
        t.quality, t.testing = "ruff", "pytest"
        t.visualization, t.data_frame = "matplotlib", "pandas"
        state.summary = "NumPy + pandas + Jupyter + Matplotlib + Ruff + pytest"
    elif name == "automation":
        state.packages = ["requests", "rich", "python-dotenv"]
        state.extensions += ["mikestead.dotenv"]
        state.dirs = [src, "tests"]
        state.files_mode = "automation"
        t.quality, t.testing = "ruff", "pytest"
        state.summary = "Requests + Rich + dotenv + Ruff + pytest"
    else:
        raise KeyError(name)

    return state


def apply_tooling_bundles(state: ProfileState) -> None:
    t = state.tooling
    packages: list[str] = []
    extensions: list[str] = []

    if t.quality == "ruff":
        packages.append("ruff")
        extensions.append("charliermarsh.ruff")
    elif t.quality == "black_pylint":
        packages += ["black", "pylint"]
        extensions += ["ms-python.black-formatter", "ms-python.pylint"]
    elif t.quality == "black_flake8":
        packages += ["black", "flake8"]
        extensions.append("ms-python.black-formatter")

    if t.testing == "pytest":
        packages.append("pytest")
    elif t.testing == "pytest_cov":
        packages += ["pytest", "pytest-cov"]

    if t.type_checker == "mypy":
        packages.append("mypy")
    elif t.type_checker == "pyright":
        packages.append("pyright")

    if t.visualization in ("matplotlib", "seaborn", "plotly"):
        packages.append(t.visualization)

    if t.data_frame == "polars":
        packages.append("polars")
        state.remove_package("pandas")
    elif t.data_frame == "pandas_polars":
        packages.append("polars")

    if t.docs in ("mkdocs", "sphinx"):
        packages.append(t.docs)
    if t.security == "bandit":
        packages.append("bandit")
    if t.git_workflow == "pre_commit":
        packages.append("pre-commit")

    state.packages += packages
    state.extensions += extensions


# ---------------------------------------------------------------------------
# Frameworks
# ---------------------------------------------------------------------------
_FRAMEWORK_MENUS: dict[str, tuple[str, Sequence[str], Sequence[str], str]] = {
    "cli": (
        "CLI framework",
        ("None / argparse", "Typer", "Click"),
        ("argparse", "typer", "click"),
        "argparse",
    ),
    "web": (
        "Web framework",
        ("FastAPI", "Flask", "Django", "None"),
        ("fastapi", "flask", "django", "none"),
        "fastapi",
    ),
    "ml": (
        "Machine-learning framework",
        ("scikit-learn", "XGBoost", "PyTorch", "TensorFlow", "None"),
        ("scikit-learn", "xgboost", "pytorch", "tensorflow", "none"),
        "scikit-learn",
    ),
    "dl": (
        "Deep-learning framework",
        ("PyTorch", "TensorFlow", "JAX"),
        ("pytorch", "tensorflow", "jax"),
        "pytorch",
    ),
}

_VALID_COMBINATIONS = {
    ("cli", "argparse"), ("cli", "typer"), ("cli", "click"),
    ("web", "fastapi"), ("web", "flask"), ("web", "django"), ("web", "none"),
    ("ml", "scikit-learn"), ("ml", "xgboost"), ("ml", "pytorch"),
    ("ml", "tensorflow"), ("ml", "none"),
    ("dl", "pytorch"), ("dl", "tensorflow"), ("dl", "jax"),
    ("minimal", ""), ("learner", ""), ("developer", ""), ("library", ""),
    ("data", ""), ("research", ""), ("automation", ""),
}


def framework_options(profile: str) -> tuple[str, Sequence[str], Sequence[str], str] | None:
    return _FRAMEWORK_MENUS.get(profile)


def default_framework(profile: str) -> str:
    entry = _FRAMEWORK_MENUS.get(profile)
    return entry[3] if entry else ""


def validate_framework(profile: str, framework: str) -> bool:
    return (profile, framework) in _VALID_COMBINATIONS


def framework_packages(profile: str, framework: str, pytorch_mode: str) -> list[str]:
    key = (profile, framework)
    if key == ("cli", "typer"):
        return ["typer"]
    if key == ("cli", "click"):
        return ["click"]
    if key == ("web", "fastapi"):
        return ["fastapi", "uvicorn"]
    if key == ("web", "flask"):
        return ["flask"]
    if key == ("web", "django"):
        return ["django"]
    if key == ("ml", "scikit-learn"):
        return ["scikit-learn"]
    if key == ("ml", "xgboost"):
        return ["xgboost"]
    if framework == "pytorch" and profile in ("ml", "dl"):
        return [] if pytorch_mode == "skip" else ["torch"]
    if framework == "tensorflow" and profile in ("ml", "dl"):
        return ["tensorflow"]
    if key == ("dl", "jax"):
        return ["jax"]
    return []


def split_dev_and_runtime(packages: Sequence[str]) -> tuple[list[str], list[str]]:
    dev = [p for p in packages if is_dev_package(p)]
    runtime = [p for p in packages if not is_dev_package(p)]
    return dedupe(dev), dedupe(runtime)
