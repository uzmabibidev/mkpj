"""mkpj — one-command Python project generator.

Command-line parsing is hand-rolled rather than delegated to argparse so that
messages, ordering and exit codes are part of the contract and are
pinned by the tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import console, editors, menu
from .console import STYLE, MkpjExit
from .generator import ProjectSpec, write_project_files, write_vscode
from .installer import EXTENSION, PACKAGE, Installer
from .profiles import (
    PROFILE_CHOICES,
    PROFILE_LABELS,
    ProfileState,
    apply_tooling_bundles,
    build_profile,
    default_framework,
    framework_options,
    framework_packages,
    split_dev_and_runtime,
    validate_framework,
)
from .spinner import run_command_with_spinner
from .textutil import dedupe, package_name_for, split_list, strip_item, trim

__all__ = ["main", "parse_args", "Options", "HELP_TEXT"]

_TEMP_DIR = Path(tempfile.gettempdir())
INSTALL_LOG = Path(os.environ.get("MKPJ_INSTALL_LOG", _TEMP_DIR / "mkpj_install.log"))
LARGE_DOWNLOAD_FLAG = Path(
    os.environ.get("MKPJ_LARGE_DOWNLOAD_FLAG", _TEMP_DIR / "mkpj_large_download_warned")
)

HELP_TEXT = """\
mkpj — one-command Python project generator

  * creates a project folder + .venv
  * installs dev tools / packages (pip) and VS Code extensions
  * writes standard project files (pyproject.toml, README, .gitignore, ...)
  * configures VS Code (.vscode/settings.json, extensions.json)
  * initializes git with a first commit
  * failed installs go through an interactive retry flow:
      - a non-scrolling ↑/↓ wheel menu
      - fuzzy "did you mean …" suggestions (via python difflib)
      - re-entry of alternative names (comma, space, or newline separated)

Usage:
  mkpj [project-name] [options]
  mkpj -n myapp -p fastapi,pydantic -e ms-python.python,charliermarsh.ruff

Options:
  -n, --name <name>          project name
      --profile, --type <p>  project profile: minimal, learner, developer,
                             library, cli, web, data, ml, dl, research, automation
      --framework <name>     framework for cli/web/ml/dl profiles
      --example              add an optional runnable example workflow
      --no-example           create the clean structure only
      --no-install           generate files/venv without installing packages or extensions
      --pytorch <mode>       PyTorch install mode: cpu, cuda, or skip
  -p, --packages <list>      additional Python packages (comma/space/newline separated)
  -e, --extensions <list>    additional VS Code extension IDs
  -y, --yes                  accept profile defaults and skip optional prompts
      --no-git               skip git init
      --no-vscode            skip VS Code config/extensions
      --python <bin>         Python interpreter for the venv (default: python3)
  -h, --help                 show this help

Any of -n/-p/-e can be combined with -y: -y fills in anything you didn't
pass explicitly with the defaults, so the whole thing runs non-interactively.

Requires: Python 3.9+. The VS Code CLI ("code") is optional — if missing,
extension installation is skipped automatically.
"""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
@dataclass
class Options:
    project_name: str = ""
    profile: str | None = None
    framework: str | None = None
    packages: str | None = None
    extensions: str | None = None
    example: bool | None = None
    pytorch: str | None = None
    assume_yes: bool = False
    do_git: bool = True
    do_vscode: bool = True
    do_install: bool = True
    python_bin: str = "python3"


def _require_value(option: str, values: Sequence[str], index: int) -> str:
    if index >= len(values) or values[index] == "":
        console.err(f"{option} requires a value.")
        raise MkpjExit(2)
    return values[index]


def parse_args(argv: Sequence[str]) -> Options:
    opts = Options()
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--profile", "--type"):
            opts.profile = _require_value(arg, argv, index + 1).lower()
            index += 2
        elif arg == "--framework":
            opts.framework = _require_value(arg, argv, index + 1).lower()
            index += 2
        elif arg == "--example":
            opts.example = True
            index += 1
        elif arg == "--no-example":
            opts.example = False
            index += 1
        elif arg == "--no-install":
            opts.do_install = False
            index += 1
        elif arg == "--pytorch":
            opts.pytorch = _require_value(arg, argv, index + 1).lower()
            index += 2
        elif arg in ("-n", "--name"):
            opts.project_name = _require_value(arg, argv, index + 1)
            index += 2
        elif arg in ("-p", "--packages"):
            opts.packages = _require_value(arg, argv, index + 1)
            index += 2
        elif arg in ("-e", "--extensions"):
            opts.extensions = _require_value(arg, argv, index + 1)
            index += 2
        elif arg in ("-y", "--yes"):
            opts.assume_yes = True
            index += 1
        elif arg == "--no-git":
            opts.do_git = False
            index += 1
        elif arg == "--no-vscode":
            opts.do_vscode = False
            index += 1
        elif arg == "--python":
            opts.python_bin = _require_value(arg, argv, index + 1)
            index += 2
        elif arg in ("-h", "--help"):
            sys.stdout.write(HELP_TEXT)
            raise MkpjExit(0)
        elif arg.startswith("-"):
            console.err(f"Unknown option: {arg}")
            raise MkpjExit(1)
        else:
            opts.project_name = arg
            index += 1
    return opts


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------
@dataclass
class Run:
    opts: Options
    project_name: str = ""
    pkg_name: str = ""
    project_dir: Path = field(default_factory=Path)
    state: ProfileState | None = None
    framework: str = ""
    pytorch_mode: str = ""
    example_workflow: bool = False
    py_major: int = 3
    py_minor: int = 9


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def validate_project_name(name: str) -> int | None:
    if not name:
        console.err("A project name is required.")
        return 1
    if "/" in name or name in (".", ".."):
        console.err("Project name must be a directory name, not a path.")
        return 2
    if Path(name).exists():
        console.err(f"'{name}' already exists in {Path.cwd()}.")
        return 1
    return None


def resolve_project_name(opts: Options) -> str:
    if opts.project_name:
        name = trim(opts.project_name)
        code = validate_project_name(name)
        if code is not None:
            raise MkpjExit(code)
        return name

    while True:
        name = trim(menu.ask("Project name: "))
        if validate_project_name(name) is None:
            return name
        console.plain("Please enter a different name.")


def choose_profile(opts: Options) -> str:
    if opts.profile:
        return opts.profile
    index = menu.select_menu("Choose a project profile", list(PROFILE_LABELS))
    return PROFILE_CHOICES[index]


def prompt_tooling_customization(run: Run) -> None:
    opts, state = run.opts, run.state
    if opts.assume_yes or state.name == "minimal":
        return
    console.plain("")
    if not menu.confirm("Customize tooling bundles? [y/N] "):
        return

    t = state.tooling
    t.quality = menu.choose(
        "Code quality",
        ["Ruff (default)", "Black + Pylint", "Black + Flake8", "None"],
        ["ruff", "black_pylint", "black_flake8", "none"],
    )
    t.testing = menu.choose(
        "Testing",
        ["pytest (default)", "pytest + coverage", "unittest", "None"],
        ["pytest", "pytest_cov", "unittest", "none"],
    )
    t.type_checker = menu.choose(
        "Type checking",
        ["No type checker (default)", "mypy", "Pyright"],
        ["none", "mypy", "pyright"],
    )
    if state.name in ("data", "ml", "research"):
        t.visualization = menu.choose(
            "Visualization",
            ["Matplotlib (default)", "Seaborn", "Plotly", "None"],
            ["matplotlib", "seaborn", "plotly", "none"],
        )
    if state.name in ("data", "ml"):
        t.data_frame = menu.choose(
            "DataFrame",
            ["pandas (default)", "Polars", "pandas + Polars"],
            ["pandas", "polars", "pandas_polars"],
        )
    if state.name == "library":
        t.docs = menu.choose(
            "Documentation",
            ["MkDocs (default)", "Sphinx", "None"],
            ["mkdocs", "sphinx", "none"],
        )
    t.security = menu.choose("Security", ["None (default)", "Bandit"], ["none", "bandit"])
    t.git_workflow = menu.choose(
        "Git workflow", ["None (default)", "pre-commit"], ["none", "pre_commit"]
    )


def prompt_framework(run: Run) -> str:
    options = framework_options(run.state.name)
    if options is None:
        return ""
    if run.opts.framework:
        return run.opts.framework
    if run.opts.assume_yes:
        return default_framework(run.state.name)
    prompt, labels, values, _ = options
    return menu.choose(prompt, list(labels), list(values))


def prompt_pytorch_mode(run: Run) -> str:
    if run.framework != "pytorch":
        return ""
    if run.opts.pytorch:
        return run.opts.pytorch
    if run.opts.assume_yes:
        return "cpu"
    return menu.choose(
        "PyTorch installation",
        ["CPU — easiest and smallest default", "NVIDIA CUDA", "Skip PyTorch installation"],
        ["cpu", "cuda", "skip"],
    )


def prompt_example_workflow(run: Run) -> bool:
    if run.state.name not in ("ml", "dl"):
        return False
    if run.opts.example is not None:
        return run.opts.example
    if run.opts.assume_yes:
        return False
    return menu.confirm("Generate an optional runnable example workflow? [y/N] ")


def venv_python(root: Path) -> Path:
    posix = root / ".venv" / "bin" / "python"
    if posix.exists():
        return posix
    windows = root / ".venv" / "Scripts" / "python.exe"
    if windows.exists():  # pragma: no cover - Windows only
        return windows
    return posix


def resolve_interpreter(binary: str) -> str | None:
    # Only the default falls back. Windows ships "python", not "python3"; an
    # interpreter the user named explicitly is still an error if it is absent.
    found = shutil.which(binary)
    if found:
        return found
    if binary != "python3":
        return None
    for candidate in ("python", "py"):
        found = shutil.which(candidate)
        if found:
            return found
    return sys.executable or None


def create_virtualenv(run: Run) -> Path:
    opts = run.opts
    python_path = resolve_interpreter(opts.python_bin)
    if python_path is None:
        console.err(f"'{opts.python_bin}' not found on PATH.")
        raise MkpjExit(1)

    console.info(f"Creating virtual environment (.venv) with {opts.python_bin}")
    rc = run_command_with_spinner(
        "Setting up .venv",
        [python_path, "-m", "venv", ".venv"],
        log_path=INSTALL_LOG,
        cwd=run.project_dir,
    )
    if rc != 0:
        console.err("Failed to create the virtual environment.")
        console.err(f"See {INSTALL_LOG} for details.")
        raise MkpjExit(1)

    activate = run.project_dir / ".venv" / "bin" / "activate"
    activate_win = run.project_dir / ".venv" / "Scripts" / "activate"
    if not activate.exists() and not activate_win.exists():
        console.err("Virtual environment was created without an activation script.")
        console.err(f"See {INSTALL_LOG} for details.")
        raise MkpjExit(1)

    python_exe = venv_python(run.project_dir)

    if opts.do_install:
        rc = run_command_with_spinner(
            "Upgrading pip",
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
            log_path=INSTALL_LOG,
        )
        if rc != 0:
            console.warn("Could not upgrade pip; continuing with the existing pip.")
    else:
        console.muted("Skipping Python package installation (--no-install)")

    console.ok("Virtual environment ready")
    return python_exe


_VERSION_PROBE = "import sys; print(sys.version_info.major, sys.version_info.minor)"


def interpreter_version(python_exe: Path) -> tuple:
    output = subprocess.check_output([str(python_exe), "-c", _VERSION_PROBE], text=True).split()
    return int(output[0]), int(output[1])


def init_git(project_dir: Path) -> None:
    if shutil.which("git") is None:
        console.warn("'git' not found on PATH — skipping git init.")
        return
    console.info("Initializing git repository...")
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "cwd": str(project_dir)}
    subprocess.call(["git", "init", "-q"], **quiet)
    subprocess.call(["git", "add", "-A"], **quiet)
    rc = subprocess.call(["git", "commit", "-q", "-m", "Initial commit from mkpj"], **quiet)
    if rc != 0:
        console.warn("Nothing to commit or git user not configured")
    console.ok("Git repository initialized")


# ---------------------------------------------------------------------------
# Final status block
# ---------------------------------------------------------------------------
def _status_line(label: str, value: str) -> None:
    sys.stderr.write(f"  {label:<16} {value}\n")


def _status_list(label: str, items: Sequence[str]) -> None:
    _status_line(label, " ".join(items) if items else "none")


def print_env_status(run: Run, installer: Installer) -> None:
    project_dir = run.project_dir
    python_exe = venv_python(project_dir)

    console.plain("")
    sys.stderr.write(f"{STYLE.cyan}{STYLE.bold}📊 ENVIRONMENT STATUS{STYLE.reset}\n")
    sys.stderr.write(f"{STYLE.dim}{'━' * 68}{STYLE.reset}\n")
    _status_line("Project", run.project_name)
    _status_line("Location", str(project_dir))
    _status_line("Python", str(python_exe))

    if os.access(python_exe, os.X_OK):
        _status_line(".venv", "✓ Created")
        try:
            version = subprocess.check_output(
                [str(python_exe), "--version"], text=True, stderr=subprocess.STDOUT
            ).strip()
            if version:
                _status_line("Python version", version)
        except (OSError, subprocess.CalledProcessError):
            pass
    else:
        _status_line(".venv", "✗ Missing")

    _status_list("Packages", installer.installed_packages)
    _status_list("Extensions", installer.installed_extensions)

    candidates = [
        ".vscode/settings.json",
        ".vscode/extensions.json",
        ".gitignore",
        ".env.example",
        "pyproject.toml",
        "README.md",
        f"src/{run.pkg_name}/__init__.py",
        f"src/{run.pkg_name}/main.py",
        "tests/__init__.py",
        "tests/test_main.py",
        "examples/baseline.py",
        "examples/README.md",
    ]
    existing = [f for f in candidates if (project_dir / f).exists()]
    _status_line("Files", " ".join(existing) if existing else "none")

    if (project_dir / ".git").is_dir():
        try:
            branch = subprocess.check_output(
                ["git", "-C", str(project_dir), "branch", "--show-current"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            branch = ""
        _status_line("Git", f"✓ initialized ({branch or 'main'} branch)")
    else:
        _status_line("Git", "✗ not initialized")

    sys.stderr.write(f"{STYLE.dim}{'━' * 68}{STYLE.reset}\n")
    _status_line("Run", f"{python_exe} {project_dir}/src/{run.pkg_name}/main.py")

    if installer.failed_packages:
        _status_line("Failed packages", " ".join(installer.failed_packages))
    if installer.failed_extensions:
        _status_line("Failed extensions", " ".join(installer.failed_extensions))


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def run_mkpj(argv: Sequence[str]) -> int:
    opts = parse_args(argv)
    run = Run(opts=opts)

    INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    INSTALL_LOG.write_text("", encoding="utf-8")
    try:
        LARGE_DOWNLOAD_FLAG.unlink()
    except (FileNotFoundError, OSError):
        pass

    console.print_banner()

    # 1. Name -----------------------------------------------------------
    run.project_name = resolve_project_name(opts)
    run.pkg_name = package_name_for(run.project_name)

    # 2. Profile + tooling ----------------------------------------------
    profile_name = choose_profile(opts)
    try:
        run.state = build_profile(profile_name, run.pkg_name)
    except KeyError:
        console.err(f"Unknown project profile: {profile_name}")
        raise MkpjExit(1) from None

    prompt_tooling_customization(run)

    run.framework = prompt_framework(run)
    if not validate_framework(run.state.name, run.framework):
        console.err(f"Unsupported framework '{run.framework}' for profile '{run.state.name}'.")
        raise MkpjExit(2)

    run.pytorch_mode = prompt_pytorch_mode(run)
    if run.framework == "pytorch" and run.pytorch_mode not in ("cpu", "cuda", "skip"):
        console.err(
            f"Unsupported PyTorch mode '{run.pytorch_mode}'. Use cpu, cuda, or skip."
        )
        raise MkpjExit(2)

    for package in framework_packages(run.state.name, run.framework, run.pytorch_mode):
        run.state.add_package(package)

    run.example_workflow = prompt_example_workflow(run)
    apply_tooling_bundles(run.state)

    console.step(f"Profile: {run.state.name}")
    console.muted(f"Default tools: {run.state.summary}")

    # 3. Directories -----------------------------------------------------
    console.info(f"Creating project '{STYLE.cyan}{STYLE.bold}{run.project_name}{STYLE.reset}'")
    root = Path(run.project_name)
    if opts.do_vscode:
        (root / ".vscode").mkdir(parents=True, exist_ok=True)
    for directory in run.state.dirs:
        (root / directory).mkdir(parents=True, exist_ok=True)
    run.project_dir = root.resolve()

    # 4. Virtual environment ---------------------------------------------
    python_exe = create_virtualenv(run)
    run.py_major, run.py_minor = interpreter_version(python_exe)

    if run.framework == "pytorch" and run.pytorch_mode != "skip":
        if (run.py_major, run.py_minor) < (3, 10):
            console.err(
                "PyTorch requires Python 3.10 or newer; selected interpreter is "
                f"Python {run.py_major}.{run.py_minor}."
            )
            console.err("Use --python with a Python 3.10+ interpreter, or choose --pytorch skip.")
            raise MkpjExit(2)

    installer = Installer(
        python_exe=str(python_exe),
        log_path=INSTALL_LOG,
        interactive=not opts.assume_yes,
    )

    # 5. Packages ---------------------------------------------------------
    dev_packages, runtime_packages = split_dev_and_runtime(run.state.packages)
    if opts.packages is not None:
        extra_dev, extra_runtime = split_dev_and_runtime(split_list(opts.packages))
        dev_packages = dedupe(dev_packages + extra_dev)
        runtime_packages = dedupe(runtime_packages + extra_runtime)

    install_dev = list(dev_packages)
    install_runtime = list(runtime_packages)

    if opts.do_install:
        if run.framework == "pytorch" and run.pytorch_mode != "skip":
            installer.install_pytorch(run.pytorch_mode)
            # torch is installed on its own above (its index is a separate flag).
            install_dev = strip_item(install_dev, "torch")
            install_runtime = strip_item(install_runtime, "torch")
        if install_dev:
            console.info("Installing development packages...")
            installer.install_batch(PACKAGE, install_dev)
        else:
            console.muted("No development packages selected.")
    else:
        console.muted("Skipping package installation (--no-install)")

    # Additional packages are asked for only after the bundle finished
    # installing, so nobody answers a prompt and then waits on an unrelated
    # install.
    if opts.packages is None and not opts.assume_yes:
        typed = split_list(menu.ask_list("Additional packages (press Enter for none)", ""))
        runtime_packages = dedupe(runtime_packages + typed)
        install_runtime = dedupe(install_runtime + typed)

    if opts.do_install and install_runtime:
        console.info("Installing extra packages...")
        installer.install_batch(PACKAGE, install_runtime)

    # 6. Project files -----------------------------------------------------
    requires_python = ">=3.9"
    if run.framework == "pytorch" and run.pytorch_mode != "skip":
        requires_python = ">=3.10"

    extensions = list(run.state.extensions)
    if opts.extensions is not None:
        extensions += split_list(opts.extensions)
    extensions = [e for e in extensions if e]

    spec = ProjectSpec(
        project_name=run.project_name,
        pkg_name=run.pkg_name,
        profile=run.state.name,
        files_mode=run.state.files_mode,
        framework=run.framework,
        tooling=run.state.tooling,
        root=run.project_dir,
        dev_packages=dev_packages,
        runtime_packages=runtime_packages,
        extensions=extensions,
        py_major=run.py_major,
        py_minor=run.py_minor,
        requires_python=requires_python,
        example_workflow=run.example_workflow,
    )

    console.info(f"Writing {run.state.name} project structure...")
    write_project_files(spec)
    console.success("Project structure written")

    # 7. VS Code -----------------------------------------------------------
    if opts.do_vscode:
        console.info("Writing VS Code configuration...")
        write_vscode(spec)
        console.ok("VS Code configuration written")

        if opts.do_install and shutil.which("code"):
            if extensions:
                console.info("Installing VS Code extensions...")
                installer.install_batch(EXTENSION, extensions)
        elif opts.do_install:
            console.warn("'code' CLI not found on PATH — skipping extension installation.")
            console.warn(
                "Install the VS Code CLI if you want mkpj to install extensions automatically."
            )
        else:
            console.muted("Skipping VS Code extension installation (--no-install)")
    else:
        console.warn("Skipping VS Code setup (--no-vscode)")

    # 8. Git ----------------------------------------------------------------
    if opts.do_git:
        init_git(run.project_dir)
    else:
        console.warn("Skipping git init (--no-git)")

    # 9. Status + editor -----------------------------------------------------
    print_env_status(run, installer)

    console.plain("")
    if opts.assume_yes:
        console.muted("Skipping editor selection (--yes).")
    else:
        editors.open_project_in_editor(run.project_name, run.project_dir)

    console.plain("")
    console.ok(f"{STYLE.bold}{run.project_name} is ready.{STYLE.reset}")
    console.plain(
        f"{STYLE.dim}  Editor selection can be run after creation; "
        f"VS Code config points to .venv.{STYLE.reset}"
    )
    console.plain(f"{STYLE.dim}  Virtual environment: {run.project_dir}/.venv{STYLE.reset}")
    return 0


def main(argv: list[str] | None = None) -> int:
    console.configure_output_encoding()
    console.refresh_style()
    console.install_signal_handlers()
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        return run_mkpj(args)
    except MkpjExit as exit_signal:
        code = exit_signal.code
        return int(code) if isinstance(code, int) else 0
    except KeyboardInterrupt:
        console.restore_terminal()
        console.plain("")
        console.warn("Interrupted by user.")
        console.warn("mkpj stopped safely.")
        return 130
    finally:
        console.restore_terminal()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
