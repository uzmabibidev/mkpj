"""Discover installed editors and offer to open the finished project."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import console, menu
from .console import STYLE

__all__ = ["EDITORS", "discover_editors", "print_install_hint", "open_project_in_editor"]

# (command, display name)
EDITORS: tuple[tuple[str, str], ...] = (
    ("code", "Visual Studio Code"),
    ("codium", "VSCodium"),
    ("cursor", "Cursor"),
    ("zed", "Zed"),
    ("pycharm", "PyCharm"),
    ("pycharm-community", "PyCharm Community"),
    ("charm", "PyCharm"),
    ("nvim", "Neovim"),
    ("vim", "Vim"),
    ("emacs", "Emacs"),
)

_TERMINAL_EDITORS = {"vim", "nvim", "emacs", "nano"}


def discover_editors() -> list[tuple[str, str]]:
    return [(cmd, name) for cmd, name in EDITORS if shutil.which(cmd)]


def print_install_hint(command: str, name: str) -> None:
    console.warn(f"Command '{command}' ({name}) is valid, but not installed on your system.")
    console.plain("")
    console.info(f"Installation Guide for {name} ({command}):")

    if command in ("pycharm", "pycharm-community", "charm"):
        console.plain(f"{STYLE.bold}Ubuntu / Linux (via Snap){STYLE.reset}")
        console.plain("  sudo snap install pycharm-community --classic")
        console.plain(f"{STYLE.bold}macOS (via Homebrew){STYLE.reset}")
        console.plain("  brew install --cask pycharm-community")
    elif command == "code":
        console.plain(f"{STYLE.bold}Ubuntu / Linux (via Snap){STYLE.reset}")
        console.plain("  sudo snap install code --classic")
        console.plain(f"{STYLE.bold}macOS (via Homebrew){STYLE.reset}")
        console.plain("  brew install --cask visual-studio-code")
    else:
        console.plain(f"{STYLE.bold}Ubuntu / Debian Linux{STYLE.reset}")
        console.plain(f"  sudo apt update && sudo apt install {command}")
        console.plain(f"{STYLE.bold}Fedora / Red Hat Linux{STYLE.reset}")
        console.plain(f"  sudo dnf install {command}")
        console.plain(f"{STYLE.bold}macOS (via Homebrew){STYLE.reset}")
        console.plain(f"  brew install {command}")
    console.plain("")


def _launch(command: str, project_dir: Path) -> None:
    if command in _TERMINAL_EDITORS:
        subprocess.call([command, str(project_dir)])
    else:
        subprocess.Popen(
            [command, str(project_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def open_project_in_editor(project_name: str, project_dir: Path) -> None:
    available = discover_editors()
    options = [f"Open in {name}" for _, name in available]
    options.append("Open with another command")
    options.append("Skip — do not open an editor")

    selection = menu.select_menu(f"Where do you want to open '{project_name}'?", options)

    # 1) An installed editor from the menu.
    if selection < len(available):
        command, name = available[selection]
        console.info(f"Opening project in {STYLE.bold}{name}{STYLE.reset}...")
        _launch(command, project_dir)
        if command not in _TERMINAL_EDITORS:
            console.info(
                f"Launching {name} in background (GUI may take a few seconds to appear)..."
            )
        console.ok(f"Opened project in {name}")
        console.ok("Python environment: .venv")
        return

    # 2) A command typed by hand.
    if selection == len(available):
        while True:
            custom = menu.ask("Enter editor command (or press Enter to skip): ").strip()
            if not custom:
                console.warn("Skipping editor launch.")
                return

            if shutil.which(custom):
                console.info(f"Opening project with {STYLE.bold}{custom}{STYLE.reset}...")
                _launch(custom, project_dir)
                console.ok(f"Opened project with {custom}")
                console.ok("Python environment: .venv")
                return

            match = next(
                ((cmd, name) for cmd, name in EDITORS if custom in (cmd, name)),
                None,
            )
            if match:
                print_install_hint(*match)
                return

            console.err(f"Command '{custom}' was not found on your system.")
            console.warn("Please re-enter a valid command name or press Enter to skip.")

    # 3) Skip.
    console.warn("Skipping editor launch.")
