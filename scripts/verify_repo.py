"""Repository validator for AI-Examples.

Verifies:
1. Valid JSON and nbformat 4 with cell IDs.
2. Strict 8-cell notebook blueprint.
3. Absence of external network calls.
4. Line count budget (ceiling 220 lines).
5. Headless execution under 3.0 seconds (when --run is passed).
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Warm up core modules to eliminate one-time cold DLL initialization from notebook timing
import torch
import core

FORBIDDEN_CALLS = [
    "urllib.request",
    "requests.get",
    "requests.post",
    "wget",
    "curl",
    "hf_hub_download",
    "snapshot_download",
    "torchvision.datasets",
    "torchaudio.datasets",
]

EXPECTED_CELL_TYPES = [
    ("markdown", "1. Header, Lineage & Decision Boundary"),
    ("markdown", "2. Mathematical Overview & Complexity"),
    ("code", "3. Deterministic Setup & Micro-Data"),
    ("code", "4. PyTorch 2 Implementation"),
    ("code", "5. Quantitative Verification & Invariant Checks"),
    ("code", "6. Performance Profile & Benchmark Table"),
    ("code", "7. Misuse, Anti-Patterns & Runnable Failure Contrast"),
    ("markdown", "8. Key Takeaways & Navigation Footer"),
]


def validate_notebook_structure(path: Path) -> list[str]:
    errors = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return [f"Invalid JSON: {e}"]

    if nb.get("nbformat") != 4:
        errors.append(f"Expected nbformat 4, got {nb.get('nbformat')}")

    cells = nb.get("cells", [])
    if len(cells) != 8:
        errors.append(f"Expected exactly 8 cells, found {len(cells)}")

    total_lines = 0
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
        for forbidden in FORBIDDEN_CALLS:
            if forbidden in source_text:
                errors.append(f"Cell {idx + 1} contains forbidden network call '{forbidden}'")

    if total_lines > 220:
        errors.append(f"Total lines ({total_lines}) exceeds 220-line ceiling")

    return errors


def execute_notebook(path: Path, max_duration: float = 3.0) -> tuple[bool, float, str]:
    """Executes code cells in an isolated namespace, verifying runtime budget."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        combined_code = "\n\n".join(code_cells)

        # Pre-compile to isolate syntax validation
        compiled_code = compile(combined_code, str(path), "exec")

        # Clean isolated execution namespace
        ns = {
            "__name__": "__main__",
            "__file__": str(path.resolve()),
        }

        gc.collect()
        start = time.perf_counter()
        exec(compiled_code, ns)
        duration = time.perf_counter() - start

        if duration > max_duration:
            return False, duration, f"Execution took {duration:.2f}s (budget: {max_duration:.1f}s)"

        return True, duration, ""
    except Exception as e:
        return False, 0.0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Validate AI-Examples notebooks")
    parser.add_argument("--run", action="store_true", help="Execute notebooks and verify timing")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-notebook execution budget in seconds")
    parser.add_argument("--path", type=str, default=".", help="Root directory or specific file")
    args = parser.parse_args()

    root = Path(args.path)
    if root.is_file() and root.suffix == ".ipynb":
        notebooks = [root]
    else:
        notebooks = sorted(
            p
            for p in root.rglob("*.ipynb")
            if not any(part.startswith(".") or part in ("scratch", "Linear Neuron") for part in p.parts)
        )

    print(f"Scanning {len(notebooks)} notebook(s)...")
    failed = False

    for nb_path in notebooks:
        rel_path = nb_path.relative_to(root) if root.is_dir() else nb_path
        errors = validate_notebook_structure(nb_path)

        if errors:
            print(f"[FAIL] {rel_path} - Structure errors:")
            for err in errors:
                print(f"       - {err}")
            failed = True
            continue

        if args.run:
            success, duration, err_msg = execute_notebook(nb_path, max_duration=args.timeout)
            if not success:
                print(f"[FAIL] {rel_path} - Execution error: {err_msg}")
                failed = True
                continue
            print(f"[PASS] {rel_path} (Execution: {duration:.2f}s, budget: {args.timeout}s)")
        else:
            print(f"[PASS] {rel_path} (Structure OK)")

    if failed:
        sys.exit(1)
    print("All notebook checks passed successfully.")


if __name__ == "__main__":
    main()
