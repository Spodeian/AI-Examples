"""Thin production wrapper around the `aix` Rust repository toolchain (`crates/aix`).

Ensures `aix` is compiled in release mode (`cargo build --release -p aix`) before
invoking it. Supports both `aix` subcommands directly (`check`, `sync`, `status`,
`run-notebooks`, `test`, `scaffold`, `weights`, `ci`, `verify`, `help`) and legacy
`verify_repo.py` flags (`--check-symmetry`, `--check-links`, `--check-semantics`,
`--run`, `--timeout`, `--parallel`, `--no-parallel`, `--workers`, `--check-candle`,
`--check-kernel`, `--path`), which automatically route through `aix verify`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

AIX_SUBCOMMANDS = {
    "check",
    "sync",
    "status",
    "next",
    "run-notebooks",
    "test",
    "scaffold",
    "notebook",
    "weights",
    "verify",
    "ci",
    "doctor",
    "clean",
    "completions",
    "help",
    "--help",
    "-h",
}


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, Path.cwd().resolve()):
        cur = candidate
        while not (cur / "Cargo.toml").is_file() and cur.parent != cur:
            cur = cur.parent
        if (cur / "Cargo.toml").is_file():
            return cur
    raise RuntimeError("Could not locate workspace Cargo.toml")


def aix_binary_path(repo_root: Path) -> Path:
    exe = "aix.exe" if os.name == "nt" else "aix"
    return repo_root / "target" / "release" / exe


def needs_rebuild(repo_root: Path, bin_path: Path) -> bool:
    if not bin_path.is_file():
        return True
    bin_mtime = bin_path.stat().st_mtime
    watch_paths = [
        repo_root / "Cargo.toml",
        repo_root / "crates" / "aix" / "Cargo.toml",
        *sorted((repo_root / "crates" / "aix" / "src").rglob("*.rs")),
    ]
    return any(p.is_file() and p.stat().st_mtime > bin_mtime for p in watch_paths)


def ensure_release_binary(repo_root: Path) -> Path:
    bin_path = aix_binary_path(repo_root)
    if needs_rebuild(repo_root, bin_path):
        cargo = shutil.which("cargo") or "cargo"
        env = os.environ.copy()
        env.setdefault("RUSTC_WRAPPER", "")
        res = subprocess.run(
            [cargo, "build", "--release", "--quiet", "-p", "aix"],
            cwd=str(repo_root),
            env=env,
            check=False,
        )
        if res.returncode != 0 or not bin_path.is_file():
            raise SystemExit(res.returncode or 1)
    return bin_path


def main() -> int:
    repo_root = find_repo_root()
    bin_path = ensure_release_binary(repo_root)

    argv = sys.argv[1:]
    if argv and argv[0] in AIX_SUBCOMMANDS:
        cmd = [str(bin_path), *argv]
    else:
        cmd = [str(bin_path), "verify", *argv]

    env = os.environ.copy()
    env.setdefault("AIX_PYTHON", sys.executable)
    res = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
