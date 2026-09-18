# mkpj

[![CI](https://github.com/uzmabibidev/mkpj/actions/workflows/ci.yml/badge.svg)](https://github.com/uzmabibidev/mkpj/actions/workflows/ci.yml)

![mkpj demo](docs/demo.gif)

Starting a Python project means the same twenty minutes every time: make the
folder, make the venv, install ruff and pytest, write `pyproject.toml`, write
`.gitignore`, point VS Code at the right interpreter, `git init`. I got bored of
doing it by hand, so this does it in one command.

```sh
mkpj myapp
```

Eleven profiles, from `minimal` (a folder and a venv, nothing else) to `ml` and
`dl` (data directories, notebooks, scikit-learn or PyTorch). If a package fails
to install you get a menu instead of a stack trace: retry it, type a different
name, or skip it, with a "did you mean numpy?" suggestion when the name looks
like a typo.

## Installation

> [!NOTE]
> mkpj requires Python 3.9 or newer and has no runtime dependencies. The VS Code
> CLI (`code`) is optional; extension installation is skipped when it is missing.

Not on PyPI yet, so install it from the repository. Use a tool installer:

```sh
# pipx
pipx install git+https://github.com/uzmabibidev/mkpj.git

# uv
uv tool install git+https://github.com/uzmabibidev/mkpj.git
```

Or, install from a clone:

```sh
git clone https://github.com/uzmabibidev/mkpj.git
cd mkpj
pip install .
```

<details>
<summary><strong>PATH and multiple-install notes</strong></summary>

`pipx` and `uv tool` both create an isolated environment and drop a launcher
named `mkpj` into the same directory, so running both leaves one overwriting the
other. Pick one.

If the shell cannot find `mkpj` after installing, the launcher directory is not
on your `PATH`:

```sh
pipx ensurepath        # then restart the terminal
```

Check what you are actually running, especially if an older copy is lying
around:

```sh
which -a mkpj          # POSIX
where mkpj             # Windows
```

</details>

## Usage

```sh
mkpj [project-name] [options]
mkpj -n service --profile web --framework fastapi --yes
```

Any of `-n` / `-p` / `-e` combines with `-y`, so a fully non-interactive run
needs no prompts at all. Run `mkpj --help` for the full text.

<details>
<summary><strong>All options</strong></summary>

| Option | Meaning |
| --- | --- |
| `-n`, `--name <name>` | project name |
| `--profile`, `--type <p>` | `minimal`, `learner`, `developer`, `library`, `cli`, `web`, `data`, `ml`, `dl`, `research`, `automation` |
| `--framework <name>` | framework for the `cli` / `web` / `ml` / `dl` profiles |
| `--example` / `--no-example` | add (or skip) a runnable example workflow |
| `--no-install` | generate files and the venv without installing anything |
| `--pytorch <mode>` | `cpu`, `cuda` or `skip` |
| `-p`, `--packages <list>` | extra Python packages (comma / space / newline separated) |
| `-e`, `--extensions <list>` | extra VS Code extension IDs |
| `-y`, `--yes` | accept profile defaults, skip optional prompts |
| `--no-git`, `--no-vscode` | skip `git init` / VS Code setup |
| `--python <bin>` | interpreter used for the venv (default `python3`) |

Exit codes: `2` for usage errors (missing option value, unsupported framework or
PyTorch mode, a path instead of a name), `1` for unknown options and unusable
project names, `130` for Ctrl-C.

</details>

## Downloads

```text
  numpy
  ████████████████████░░░░░░░░   72.4%
  7.5/10.4 MB   2.5 MB/s   ETA 00:01
```

Both halves of a size pair always share one unit, wheels pip already has are
marked `(cached)` instead of being redrawn as downloads, and the first artifact
over 250 MB warns once per run. Full pip output goes to `mkpj_install.log` in
the system temp directory.

| Key | Action |
| --- | --- |
| `↑` / `↓`, `k` / `j` | move the wheel (wraps at both ends) |
| `Enter` | confirm |
| `Ctrl-C` | abort safely: the cursor is restored and mkpj stops immediately |

Without a TTY (CI, pipes) menus fall back to a numbered prompt instead of
blocking on key reads that can never arrive.

## Development

```sh
pip install -e ".[dev]"
pytest                   # fast + generation suites
pytest -m "not slow"     # unit tests only, ~1s
pytest --run-network     # real pip installs
pytest --run-heavy       # + PyTorch (multi-GB download)
```

```text
src/mkpj/
├── cli.py         argument parsing and the top-level flow
├── console.py     palette, log helpers, cursor and signal handling
├── menu.py        ↑/↓ wheel menu and prompts
├── spinner.py     braille spinner
├── pipdriver.py   pip runner with live download progress
├── installer.py   batched installs + interactive retry flow
├── suggest.py     "did you mean …" via difflib
├── profiles.py    profile catalog and tooling bundles
├── generator.py   generated project templates
└── editors.py     editor discovery and launch
```

CI runs lint plus the fast and generation suites on Linux, macOS and Windows
against Python 3.9, 3.11 and 3.13, and installs the built wheel to check the
`mkpj` console script end to end. The package-installation suites run nightly.

<details>
<summary><strong>Test layout</strong></summary>

| File | Covers |
| --- | --- |
| `tests/test_generation.py` | structure, config, flags, frameworks (`--no-install`) |
| `tests/test_packages.py` | real pip installs, batched-resolution guard |
| `tests/test_cli.py` | flag parsing, help text, exit codes |
| `tests/test_templates.py` | profile resolution and file rendering |
| `tests/test_installer.py` | batching and the retry/fuzzy flow, with a fake pip |
| `tests/test_pytorch.py` | the PyTorch install path and the dependency it declares |
| `tests/test_ui.py` | key decoding, wheel movement, spinner, progress parsing |
| `tests/test_interactive.py` | the real TUI, driven through a pty |

</details>

<details>
<summary><strong>Design notes: things that were not obvious</strong></summary>

Linux, macOS and Windows are all supported and covered in CI. The venv
interpreter, the `.vscode` interpreter path and the generated README follow the
platform layout (`bin/python` or `Scripts\python.exe`), key reading uses
`termios` on POSIX and `msvcrt` on Windows behind one abstraction, and nothing
shells out to a Unix utility.

**One pip call per batch, not one per package.** Installing packages one at a
time lets pip re-resolve against whatever the previous call pinned. I watched it
walk pytest back from 8.4 to 6.1 trying to satisfy a constraint from a package
installed thirty seconds earlier. One call, every constraint visible at once.

**PyTorch is installed on its own.** Its CPU and CUDA wheels live on a separate
index that has to be passed as its own flag, so it cannot ride along in the
batch. It is still written to `pyproject.toml`; installing it elsewhere is no
reason to leave it out of the dependencies. Picking `cpu` or `cuda` also raises
the generated `requires-python` to `>=3.10`, because PyTorch dropped 3.9.

**The menu uses cbreak, not raw mode.** Raw mode also turns off the terminal's
newline translation, so every line the menu drew started where the last one
ended and the whole thing walked diagonally down the screen. cbreak gives the
same single-key reads and leaves output alone.

**Output is forced to UTF-8.** On Windows a redirected stdout defaults to
cp1252, which has no `↑`, so `mkpj --help > notes.txt` died with a
`UnicodeEncodeError`. The console was fine; only redirection broke.

</details>

## License

MIT
