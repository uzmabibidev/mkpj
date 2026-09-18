"""Package / extension installation, including the interactive retry flow.

The batching behaviour is load-bearing and deliberately preserved: a profile's
packages are resolved in a *single* pip call. Installing one at a time lets pip
re-resolve against whatever the previous call already pinned, which can send
the resolver backtracking through dozens of releases (it was once observed
walking pytest back from 8.4.0 to 6.1.0 to satisfy a constraint introduced by
an earlier, separately installed package).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import console, menu
from .console import STYLE
from .pipdriver import run_pip
from .suggest import format_suggestions, fuzzy_suggest
from .textutil import split_list, trim

__all__ = ["Installer", "KNOWN_PACKAGES", "KNOWN_EXTENSIONS", "PACKAGE", "EXTENSION"]

PACKAGE = "package"
EXTENSION = "extension"

KNOWN_PACKAGES = [
    "numpy", "pandas", "requests", "flask", "django", "fastapi", "black", "ruff", "mypy",
    "pytest", "pytest-cov", "pytest-mock", "ipython", "python-dotenv", "sqlalchemy", "pydantic",
    "httpx", "click", "typer", "rich", "loguru", "pillow", "beautifulsoup4", "scrapy", "celery",
    "redis", "pymongo", "psycopg2-binary", "uvicorn", "gunicorn", "jinja2", "pyyaml", "toml",
    "poetry", "pre-commit", "tox", "coverage", "flake8", "isort", "autopep8", "pylint", "bandit",
    "safety", "matplotlib", "scikit-learn", "scipy", "jupyter", "notebook", "tqdm", "attrs",
    "build", "torch", "tensorflow", "jax", "xgboost", "lightgbm", "polars", "seaborn", "plotly",
]

KNOWN_EXTENSIONS = [
    "ms-python.python", "ms-python.vscode-pylance", "ms-python.black-formatter",
    "ms-python.pylint", "charliermarsh.ruff", "njpwerner.autodocstring", "ms-toolsai.jupyter",
    "eamodio.gitlens", "streetsidesoftware.code-spell-checker", "gruntfuggly.todo-tree",
    "usernamehw.errorlens", "formulahendry.code-runner", "mikestead.dotenv",
    "redhat.vscode-yaml", "tamasfe.even-better-toml", "github.copilot",
]


@dataclass
class Installer:

    python_exe: str
    log_path: Path
    interactive: bool = True

    installed_packages: list[str] = field(default_factory=list)
    failed_packages: list[str] = field(default_factory=list)
    installed_extensions: list[str] = field(default_factory=list)
    failed_extensions: list[str] = field(default_factory=list)

    # -- low level --------------------------------------------------------
    def _record_success(self, kind: str, name: str) -> None:
        console.ok(f"Installed {kind}: {name}")
        target = self.installed_packages if kind == PACKAGE else self.installed_extensions
        target.append(name)

    def _record_failure(self, kind: str, name: str) -> None:
        console.err(
            f"Failed to install {kind}: {name}  {STYLE.dim}(see {self.log_path}){STYLE.reset}"
        )
        target = self.failed_packages if kind == PACKAGE else self.failed_extensions
        target.append(name)

    def install_one(self, kind: str, name: str) -> bool:
        if kind == PACKAGE:
            rc = run_pip(self.python_exe, self.log_path, ["install", "--progress-bar", "raw", name])
        else:
            rc = self._install_extension(name)

        if rc == 0:
            self._record_success(kind, name)
            return True
        self._record_failure(kind, name)
        return False

    def _install_extension(self, name: str) -> int:
        if shutil.which("code") is None:
            return 127
        with open(self.log_path, "a", encoding="utf-8") as log:
            return subprocess.call(
                ["code", "--install-extension", name], stdout=log, stderr=subprocess.STDOUT
            )

    def install_pytorch(self, mode: str) -> bool:
        """PyTorch needs its own call: the CPU/CUDA index is a separate flag."""
        if mode == "skip":
            console.warn("Skipping PyTorch installation.")
            return True

        if mode == "cpu":
            args = [
                "install", "--progress-bar", "raw", "torch",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ]
        else:
            console.warn("CUDA builds can download several GB of packages. This may take a while.")
            console.warn(f"Monitor progress with: tail -f {self.log_path}")
            args = ["install", "--progress-bar", "raw", "torch"]

        rc = run_pip(self.python_exe, self.log_path, args)
        if rc == 0:
            console.ok(f"Installed package: torch ({mode})")
            self.installed_packages.append("torch")
            return True

        console.err(
            f"Failed to install package: torch ({mode})  "
            f"{STYLE.dim}(see {self.log_path}){STYLE.reset}"
        )
        self.failed_packages.append("torch")
        return False

    # -- batch ------------------------------------------------------------
    def install_batch(self, kind: str, items: Sequence[str]) -> list[str]:
        items = list(items)
        known = KNOWN_PACKAGES if kind == PACKAGE else KNOWN_EXTENSIONS
        failed: list[str] = []

        if kind == PACKAGE and len(items) > 1:
            console.info(f"Resolving {len(items)} packages together...")
            rc = run_pip(
                self.python_exe,
                self.log_path,
                ["install", "--progress-bar", "raw", *items],
            )
            if rc == 0:
                for item in items:
                    self._record_success(PACKAGE, item)
            else:
                console.warn(
                    "Batched install failed; retrying packages individually "
                    "to isolate the problem..."
                )
                for item in items:
                    if not self.install_one(kind, item):
                        failed.append(item)
        else:
            for item in items:
                if not self.install_one(kind, item):
                    failed.append(item)

        while failed:
            console.warn(f"{len(failed)} {kind}(s) still failing.")
            if not self.interactive:
                break
            failed = self.handle_failures(kind, failed, known)
            if not failed:
                break
            again = menu.ask(
                f"Retry the {len(failed)} remaining failed {kind}(s)? [y/N] "
            ).lower()
            if not again.startswith("y"):
                break

        if kind == PACKAGE:
            self.failed_packages = list(failed)
        else:
            self.failed_extensions = list(failed)

        if failed:
            console.warn(f"Giving up on: {' '.join(failed)}")
        return failed

    # -- retry flow -------------------------------------------------------
    def handle_failures(self, kind: str, failed: Sequence[str], known: Sequence[str]) -> list[str]:
        failed = list(failed)
        still_failed: list[str] = []
        index = 0

        while index < len(failed):
            item = failed[index]
            suggestions = format_suggestions(fuzzy_suggest(item, known))

            console.plain("")
            console.err(f"Failed {kind}: {STYLE.bold}{item}{STYLE.reset}")
            if suggestions:
                console.warn(f"Did you mean: {STYLE.bold}{suggestions}{STYLE.reset}")

            remaining = len(failed) - index
            choice = menu.select_menu(
                f"What do you want to do about '{item}'?",
                [
                    f"Retry '{item}' as-is",
                    "Enter alternative(s)",
                    "Skip this one",
                    f"Skip all remaining ({remaining} left)",
                ],
            )

            if choice == 0:
                if not self.install_one(kind, item):
                    still_failed.append(item)
            elif choice == 1:
                alternatives = split_list(menu.ask(f"Alternative(s) for '{item}': "))
                if not alternatives:
                    still_failed.append(item)
                for alternative in alternatives:
                    alternative = trim(alternative)
                    if not alternative:
                        continue
                    if not self.install_one(kind, alternative):
                        still_failed.append(alternative)
            elif choice == 2:
                console.warn(f"Skipping '{item}'")
            elif choice == 3:
                console.warn(f"Skipping all remaining failed {kind}s")
                still_failed.extend(failed[index:])
                break

            index += 1

        return still_failed
