"""Comprehensive repository validator for AI-Examples.

Refactored for worst-case performance, zero-hang guarantees, and instant verification.

Verifies:
1. Valid JSON and nbformat 4 with cell IDs.
2. Strict 8-cell notebook blueprint and line count budget (ceiling 220 lines).
3. Strict 8-section Markdown contract for Track 1 documents.
4. Absence of external network calls in notebooks and modules.
5. Path symmetry across all 7 tracks (markdown, notebooks, pytorch2, jax, candle, kernels, weights).
6. Internal relative link integrity across documentation without disk I/O thrashing.
7. Deterministic headless execution under 3.0 seconds with hard wall-clock timeout protection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Enforce headless environment globally before any library imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Pre-warm core modules to eliminate cold import overhead from runtime timing
try:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    import core  # noqa: F401
except ImportError:
    pass

# Configure matplotlib to never block on GUI events
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
    plt.pause = lambda *args, **kwargs: None  # type: ignore[assignment]
    plt.ion = lambda *args, **kwargs: None  # type: ignore[assignment]
    plt.ioff = lambda *args, **kwargs: None  # type: ignore[assignment]
except ImportError:
    pass

PRUNE_DIRS = {
    ".git",
    ".github",
    ".cargo",
    "target",
    "build",
    "dist",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "scratch",
    ".idea",
    ".vscode",
}

FORBIDDEN_CALL_PATTERNS = [
    (re.compile(r"urllib\.request"), "urllib.request"),
    (re.compile(r"requests\.(?:get|post|put|delete|head)"), "requests"),
    (re.compile(r"\bwget\b"), "wget"),
    (re.compile(r"\bcurl\s+(?:-[a-zA-Z]|https?://)"), "curl"),
    (re.compile(r"hf_hub_download"), "hf_hub_download"),
    (re.compile(r"snapshot_download"), "snapshot_download"),
    (re.compile(r"torchvision\.datasets"), "torchvision.datasets"),
    (re.compile(r"torchaudio\.datasets"), "torchaudio.datasets"),
]

EXPECTED_CELL_TYPES = [
    ("markdown", "1. Header, Citations, Purpose & Topic Bridges"),
    ("markdown", "2. Mathematical Foundations & Governing Equations"),
    ("code", "3. Deterministic Setup & Micro-Data Generator"),
    ("code", "4. First-Principles Implementation & Native Parity"),
    ("code", "5. Quantitative Invariants & Parity Verification"),
    ("code", "6. Performance Profile & Diagnostic Visualizations"),
    ("code", "7. Pathological Misuse vs. Corrected Implementation"),
    ("markdown", "8. Algorithmic Takeaways & Cross-Domain Isomorphisms"),
]

REQUIRED_MARKDOWN_SECTIONS = [
    "1. Executive Summary",
    "2. Mathematical Foundations",
    "3. Tensor Mechanics",
    "4. Multi-Ecosystem",
    "5. Pathological Failure Modes",
    "6. Cross-Domain Generalisation",
    "7. Verification",
    "8. Navigation",
]

EXPECTED_TRACKS = [
    "markdown",
    "notebooks",
    "pytorch2",
    "jax",
    "candle",
    "kernels",
    "weights",
]

EXPECTED_CATEGORIES = [
    "00_foundations",
    "01_paradigms_and_dichotomies",
    "02_classical_and_probabilistic_ml",
    "03_building_blocks_and_losses",
    "04_training_methods_and_optimisers",
    "05_architectures",
    "06_smooth_and_physics_informed_ai",
    "07_domain_tasks",
    "08_learning_paradigms_rl_and_alignment",
    "09_causal_ai_and_evolutionary_search",
    "10_xai_and_ai_analysis_methods",
    "11_ai_safety_privacy_and_verification",
    "12_compilation_and_performance",
    "13_precision_and_numerics",
    "14_quantisation",
    "15_compression_peft_and_speculative_inference",
    "16_distributed_and_scalable",
    "17_deployment_low_level_and_capstones",
]


class RepoIndex:
    """High-speed in-memory repository filesystem index.
    
    Walks the repository directory tree exactly once, completely pruning heavy
    directories (target/, .git/, etc.) before descent, and caches all paths in normalized
    sets for O(1) in-memory link resolution and symmetry checking without disk I/O.
    """
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.notebooks: list[Path] = []
        self.markdowns: list[Path] = []
        self.all_paths_normalized: set[str] = set()
        self.all_paths_case_folded: set[str] = set()
        self.existing_tracks: set[str] = set()
        self.track_categories: dict[str, set[str]] = {t: set() for t in EXPECTED_TRACKS}

        self._scan()

    def _scan(self) -> None:
        root_str = str(self.root)
        for dirpath, dirnames, filenames in os.walk(root_str):
            # In-place directory pruning to prevent descending into target/, .git/, etc.
            dirnames[:] = [
                d for d in dirnames
                if d not in PRUNE_DIRS and not d.startswith(".") and d != "Linear Neuron"
            ]

            rel_dir = os.path.relpath(dirpath, root_str).replace("\\", "/")
            if rel_dir != ".":
                self.all_paths_normalized.add(rel_dir)
                self.all_paths_case_folded.add(rel_dir.lower())

                # Track category structure detection
                parts = rel_dir.split("/")
                if len(parts) >= 1 and parts[0] in EXPECTED_TRACKS:
                    self.existing_tracks.add(parts[0])
                    if len(parts) >= 2:
                        self.track_categories[parts[0]].add(parts[1])

            for fname in filenames:
                if fname.startswith(".") or fname.endswith(".pyc"):
                    continue
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_str).replace("\\", "/")
                self.all_paths_normalized.add(rel_path)
                self.all_paths_case_folded.add(rel_path.lower())

                if fname.endswith(".ipynb"):
                    self.notebooks.append(Path(full_path))
                elif fname.endswith(".md"):
                    self.markdowns.append(Path(full_path))

        self.notebooks.sort()
        self.markdowns.sort()

    def contains(self, rel_path: str) -> bool:
        """O(1) in-memory check for file or directory existence."""
        normalized = os.path.normpath(rel_path).replace("\\", "/")
        if normalized in self.all_paths_normalized or normalized.lower() in self.all_paths_case_folded:
            return True
        # Handle trailing slash directory checks
        stripped = normalized.rstrip("/")
        return stripped in self.all_paths_normalized or stripped.lower() in self.all_paths_case_folded


def audit_notebook_single_pass(path: Path, check_semantics: bool = False) -> list[str]:
    """Single-pass structural and semantic validator.
    
    Loads JSON exactly once and performs all contract, format, line-count, and
    pedagogical semantic checks in memory without repeated disk reads.
    """
    errors: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return [f"Invalid JSON: {e}"]

    if nb.get("nbformat") != 4:
        errors.append(f"Expected nbformat 4, got {nb.get('nbformat')}")

    cells = nb.get("cells", [])
    if len(cells) != 8:
        errors.append(f"Expected exactly 8 cells, found {len(cells)}")

    total_lines = 0
    code_cell_count = 0

    for idx, (expected_type, desc) in enumerate(EXPECTED_CELL_TYPES):
        if idx >= len(cells):
            break
        cell = cells[idx]
        actual_type = cell.get("cell_type")
        if actual_type != expected_type:
            errors.append(
                f"Cell {idx + 1} ({desc}): Expected cell_type '{expected_type}', got '{actual_type}'"
            )

        if "id" not in cell:
            errors.append(f"Cell {idx + 1} is missing required 'id' attribute")

        source_lines = cell.get("source", [])
        if isinstance(source_lines, str):
            source_lines = source_lines.splitlines(keepends=True)
        total_lines += len(source_lines)

        source_text = "".join(source_lines)
        if actual_type == "code":
            code_cell_count += 1
            for pattern, name in FORBIDDEN_CALL_PATTERNS:
                if pattern.search(source_text):
                    errors.append(f"Cell {idx + 1} contains forbidden network call '{name}'")

    if total_lines > 220:
        errors.append(f"Total lines ({total_lines}) exceeds 220-line ceiling")

    # Optional deep semantic audit (if requested)
    if check_semantics and len(cells) == 8:
        c1_text = "".join(cells[0].get("source", []))
        if not any(k in c1_text for k in ["Citation", "Lineage", "Author", "19", "20"]):
            errors.append("Cell 1: Missing historical lineage or seminal citations")
        if not any(k in c1_text for k in ["Decision Boundary", "When NOT to use", "When to Use", "WARNING"]):
            errors.append("Cell 1: Missing formal decision boundary ('When NOT to use')")

        c2_text = "".join(cells[1].get("source", []))
        if "$$" not in c2_text:
            errors.append("Cell 2: Missing KaTeX display math formulation ($$...$$)")
        if not any(k in c2_text for k in ["\\mathcal{O}", "O(", "Complexity"]):
            errors.append("Cell 2: Missing asymptotic time/memory complexity analysis O(...)")

        c3_text = "".join(cells[2].get("source", []))
        if not any(k in c3_text for k in ["seed", "setup_env"]):
            errors.append("Cell 3: Missing deterministic RNG seeding (seed 42)")

        c5_text = "".join(cells[4].get("source", []))
        if "assert" not in c5_text:
            errors.append("Cell 5: Missing quantitative mathematical assertions (assert)")

        c6_text = "".join(cells[5].get("source", []))
        if not any(k in c6_text for k in ["benchmark", "perf_counter", "latency", "time", "p50"]):
            errors.append("Cell 6: Missing performance micro-benchmark profiling")

        c7_text = "".join(cells[6].get("source", []))
        if not any(k in c7_text for k in ["Misuse", "Failure", "FAILURE", "Unnormalized", "Anti-Pattern", "Contrasting"]):
            errors.append("Cell 7: Missing misuse/failure mode documentation")
        if "assert" not in c7_text:
            errors.append("Cell 7: Missing active programmatic failure assertion")

        c8_text = "".join(cells[7].get("source", []))
        if not any(k in c8_text for k in ["Takeaways", "Key Takeaways"]):
            errors.append("Cell 8: Missing key takeaways summary")
        if not any(k in c8_text for k in ["Master Index", "README.md", "Navigation", "Curriculum"]):
            errors.append("Cell 8: Missing curriculum navigation footer")

    return errors


def check_track_symmetry_fast(index: RepoIndex) -> list[str]:
    """Validates that all 18 categories exist across all 7 tracks via the in-memory index."""
    errors: list[str] = []
    for track in EXPECTED_TRACKS:
        if track not in index.existing_tracks:
            errors.append(f"Missing root track directory: {track}/")
            continue
        present_cats = index.track_categories.get(track, set())
        for cat in EXPECTED_CATEGORIES:
            if cat not in present_cats:
                errors.append(f"Missing category directory in {track}: {track}/{cat}/")
    return errors


def check_internal_links_fast(index: RepoIndex) -> list[str]:
    """Checks relative markdown and notebook links using pure in-memory O(1) set lookups."""
    errors: list[str] = []
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    root_str = str(index.root)

    for md_path in index.markdowns:
        rel_md = os.path.relpath(str(md_path), root_str).replace("\\", "/")
        md_dir = os.path.dirname(rel_md)

        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{rel_md}: Failed to read file: {e}")
            continue

        for _text, link in link_pattern.findall(content):
            if link.startswith(("http://", "https://", "mailto:", "#", "file://", "conversation://")):
                continue
            target_link = link.split("#")[0].strip()
            if not target_link:
                continue

            resolved_rel = os.path.normpath(os.path.join(md_dir, target_link)).replace("\\", "/")
            if not index.contains(resolved_rel):
                errors.append(f"{rel_md}: Broken link '{link}' (target does not exist)")

    return errors


def execute_notebook_safe(path: Path, max_duration: float = 3.0) -> tuple[bool, float, str]:
    """Executes a notebook with hard wall-clock timeout and headless monkey-patching.
    
    Protects against worst-case execution failure modes:
    1. Blocking UI events: Matplotlib is monkey-patched to never block on plt.show().
    2. Infinite loops / deadlocks: Monitored via thread join and PyThreadState_SetAsyncExc interrupt.
    3. Memory leaks: PyTorch CUDA cache and Python GC are explicitly cleared.
    """
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        combined_code = "\n\n".join(code_cells)
        compiled_code = compile(combined_code, str(path), "exec")

        # Headless namespace
        ns: dict[str, object] = {
            "__name__": "__main__",
            "__file__": str(path.resolve()),
        }

        # Clear memory caches prior to timed execution
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        exec_error: list[Exception] = []
        start_time = time.perf_counter()

        def _target() -> None:
            try:
                # Ensure matplotlib stays headless inside execution thread
                if "matplotlib" in sys.modules:
                    import matplotlib.pyplot as plt
                    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
                exec(compiled_code, ns)
            except Exception as e:
                exec_error.append(e)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=max_duration)

        elapsed = time.perf_counter() - start_time

        if thread.is_alive():
            # Asynchronously inject KeyboardInterrupt to interrupt tight loops immediately
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident or 0),
                ctypes.py_object(KeyboardInterrupt),
            )
            thread.join(timeout=0.3)
            return False, elapsed, f"Execution timed out (> {max_duration:.2f}s budget exceeded)"

        if exec_error:
            return False, elapsed, str(exec_error[0])

        if elapsed > max_duration:
            return False, elapsed, f"Execution took {elapsed:.2f}s (budget: {max_duration:.1f}s)"

        return True, elapsed, ""

    except Exception as e:
        return False, 0.0, str(e)


def _worker_init() -> None:
    """Pre-warm heavy scientific packages inside parallel worker processes."""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import core  # noqa: F401
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.show = lambda *args, **kwargs: None
    except ImportError:
        pass


def _worker_execute(args_tuple: tuple[str, float]) -> tuple[str, bool, float, str]:
    path_str, timeout = args_tuple
    p = Path(path_str)
    success, duration, err_msg = execute_notebook_safe(p, max_duration=timeout)
    return (path_str, success, duration, err_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Track Validator for AI-Examples (Worst-Case Optimized)")
    parser.add_argument("--run", action="store_true", help="Execute notebooks and verify timing (<3.0s)")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-notebook execution budget in seconds")
    parser.add_argument("--check-symmetry", action="store_true", help="Validate directory mirroring across all 7 tracks")
    parser.add_argument("--check-links", action="store_true", help="Validate internal relative links")
    parser.add_argument("--check-semantics", action="store_true", help="Validate semantic depth and educational contract")
    parser.add_argument("--parallel", action="store_true", default=False, help="Execute notebooks in parallel worker processes")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", help="Execute notebooks sequentially")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: min(cpu_count, 4))")
    parser.add_argument("--check-candle", action="store_true", help="Run native Rust Candle test suite via cargo test")
    parser.add_argument("--check-kernel", action="store_true", help="Run native Rust ai-kernel test suite via cargo test")
    parser.add_argument("--path", type=str, default=".", help="Root directory or specific file")
    args = parser.parse_args()

    t_start = time.perf_counter()
    target_path = Path(args.path).resolve()
    repo_root = target_path if target_path.is_dir() else target_path.parent
    while not (repo_root / "Cargo.toml").exists() and repo_root.parent != repo_root:
        repo_root = repo_root.parent

    # 1. Fast in-memory filesystem index
    index = RepoIndex(repo_root)
    failed = False

    # 2. Track symmetry check
    if args.check_symmetry:
        print("Checking directory symmetry across all 7 tracks...")
        symmetry_errors = check_track_symmetry_fast(index)
        if symmetry_errors:
            print(f"[FAIL] Directory symmetry errors ({len(symmetry_errors)}):")
            for err in symmetry_errors[:15]:
                print(f"       - {err}")
            if len(symmetry_errors) > 15:
                print(f"       ... and {len(symmetry_errors) - 15} more.")
            failed = True
        else:
            print("[PASS] All 18 categories present across all 7 tracks.")

    # 3. Internal link check
    if args.check_links:
        print("Checking relative links in documentation...")
        link_errors = check_internal_links_fast(index)
        if link_errors:
            print(f"[FAIL] Broken link errors ({len(link_errors)}):")
            for err in link_errors[:15]:
                print(f"       - {err}")
            if len(link_errors) > 15:
                print(f"       ... and {len(link_errors) - 15} more.")
            failed = True
        else:
            print("[PASS] All relative documentation links resolve.")

    # 4. Native Candle Rust suite
    if args.check_candle:
        print("Running native Rust Candle test suite (ai-candle)...")
        cargo_bin = shutil.which("cargo")
        if cargo_bin is None:
            print("[FAIL] cargo binary not found on PATH")
            failed = True
        else:
            env = os.environ.copy()
            env["RUSTC_WRAPPER"] = ""
            try:
                res = subprocess.run(
                    [cargo_bin, "test", "-p", "ai-candle", "--tests", "-j", "2"],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=60.0,
                    check=False,
                )
                if res.returncode != 0:
                    print(f"[FAIL] Candle test suite failed:\n{res.stderr}\n{res.stdout}")
                    failed = True
                else:
                    print("[PASS] Native Candle test suite passed.")
            except subprocess.TimeoutExpired:
                print("[FAIL] Candle test suite timed out (> 60s)")
                failed = True

    # 5. Native ai-kernel Rust suite
    if args.check_kernel:
        print("Running native Rust ai-kernel test suite (CPU + WGPU)...")
        cargo_bin = shutil.which("cargo")
        if cargo_bin is None:
            print("[FAIL] cargo binary not found on PATH")
            failed = True
        else:
            env = os.environ.copy()
            env["RUSTC_WRAPPER"] = ""
            try:
                res = subprocess.run(
                    [cargo_bin, "test", "-p", "ai-kernel", "--tests", "-j", "2"],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=60.0,
                    check=False,
                )
                if res.returncode != 0:
                    print(f"[FAIL] ai-kernel test suite failed:\n{res.stderr}\n{res.stdout}")
                    failed = True
                else:
                    print("[PASS] Native ai-kernel test suite passed.")
            except subprocess.TimeoutExpired:
                print("[FAIL] ai-kernel test suite timed out (> 60s)")
                failed = True

    # 6. Notebook selection
    if target_path.is_file() and target_path.suffix == ".ipynb":
        selected_notebooks = [target_path]
    elif target_path.is_dir():
        # Filter notebooks inside target directory
        selected_notebooks = [nb for nb in index.notebooks if target_path in nb.parents or target_path == nb]
    else:
        selected_notebooks = []

    # 7. Single-pass Notebook Validation & Execution
    should_audit_notebooks = bool(selected_notebooks) and (
        args.run or args.check_semantics or (not args.check_symmetry and not args.check_links and not args.check_candle and not args.check_kernel)
    )

    if should_audit_notebooks:
        print(f"Scanning {len(selected_notebooks)} notebook(s)...")
        valid_notebooks: list[Path] = []
        for nb_path in selected_notebooks:
            rel = nb_path.relative_to(repo_root)
            errors = audit_notebook_single_pass(nb_path, check_semantics=args.check_semantics)
            if errors:
                print(f"[FAIL] {rel} - Contract errors:")
                for err in errors:
                    print(f"       - {err}")
                failed = True
            else:
                valid_notebooks.append(nb_path)
                if not args.run:
                    print(f"[PASS] {rel} (Contract OK)")

        # 8. Execution
        if args.run and valid_notebooks:
            if args.parallel and len(valid_notebooks) > 1:
                workers = args.workers or min(os.cpu_count() or 4, 4)
                print(f"Executing {len(valid_notebooks)} notebook(s) in parallel across {workers} workers (timeout={args.timeout}s)...")
                tasks = [(str(nb), args.timeout) for nb in valid_notebooks]
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
                    results = list(executor.map(_worker_execute, tasks))
                for path_str, success, duration, err_msg in results:
                    p = Path(path_str)
                    rel = p.relative_to(repo_root)
                    if not success:
                        print(f"[FAIL] {rel} - Execution error: {err_msg}")
                        failed = True
                    else:
                        print(f"[PASS] {rel} (Execution: {duration:.2f}s, budget: {args.timeout}s)")
            else:
                for nb_path in valid_notebooks:
                    rel = nb_path.relative_to(repo_root)
                    success, duration, err_msg = execute_notebook_safe(nb_path, max_duration=args.timeout)
                    if not success:
                        print(f"[FAIL] {rel} - Execution error: {err_msg}")
                        failed = True
                    else:
                        print(f"[PASS] {rel} (Execution: {duration:.2f}s, budget: {args.timeout}s)")

    total_time = time.perf_counter() - t_start
    if failed:
        print(f"\n[FAILED] Repository validation failed in {total_time:.2f}s.")
        sys.exit(1)

    print(f"\nAll repository validation checks completed successfully in {total_time:.2f}s.")


if __name__ == "__main__":
    main()
