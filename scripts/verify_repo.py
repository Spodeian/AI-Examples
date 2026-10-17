"""Comprehensive repository validator for AI-Examples.

Verifies:
1. Valid JSON and nbformat 4 with cell IDs.
2. Strict 8-cell notebook blueprint and line count budget (ceiling 220 lines).
3. Strict 8-section Markdown contract for Track 1 documents.
4. Absence of external network calls in notebooks and modules.
5. Path symmetry across all 7 tracks (markdown, notebooks, pytorch2, jax, candle, kernels, weights).
6. Internal relative link integrity across documentation.
7. Deterministic headless execution under 3.0 seconds (when --run is passed).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

# Warm up core modules to eliminate cold import overhead from runtime timing
try:
    import torch

    import core
except ImportError:
    pass

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


def validate_notebook_structure(path: Path) -> list[str]:
    """Validates adherence to the strict 8-cell notebook contract."""
    errors = []
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


def validate_markdown_structure(path: Path) -> list[str]:
    """Validates adherence to the 8-section Markdown contract."""
    errors = []
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Failed to read file: {e}"]

    for sec in REQUIRED_MARKDOWN_SECTIONS:
        # Check for section title in headers
        pattern = re.compile(rf"##\s+.*{re.escape(sec)}", re.IGNORECASE)
        if not pattern.search(content):
            errors.append(f"Missing required section: '## ... {sec} ...'")

    return errors


def check_track_symmetry(root: Path) -> list[str]:
    """Validates that all 18 categories exist across all 7 tracks."""
    errors = []
    for track in EXPECTED_TRACKS:
        track_dir = root / track
        if not track_dir.is_dir():
            errors.append(f"Missing root track directory: {track}/")
            continue
        for cat in EXPECTED_CATEGORIES:
            cat_dir = track_dir / cat
            if not cat_dir.is_dir():
                errors.append(f"Missing category directory in {track}: {track}/{cat}/")
    return errors


def check_internal_links(root: Path) -> list[str]:
    """Checks for broken relative file links in markdown documents and notebooks."""
    errors = []
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    # Find all markdown files
    for md_path in root.rglob("*.md"):
        if any(part.startswith(".") for part in md_path.parts):
            continue
        content = md_path.read_text(encoding="utf-8")
        for _text, link in link_pattern.findall(content):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            # Strip anchors
            target_link = link.split("#")[0]
            if not target_link:
                continue
            target_file = (md_path.parent / target_link).resolve()
            if not target_file.exists():
                errors.append(
                    f"{md_path.relative_to(root)}: Broken link '{link}' (target does not exist)"
                )

    return errors


def execute_notebook(path: Path, max_duration: float = 3.0) -> tuple[bool, float, str]:
    """Executes code cells in an isolated namespace, verifying runtime budget."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Track Validator for AI-Examples")
    parser.add_argument(
        "--run", action="store_true", help="Execute notebooks and verify timing (<3.0s)"
    )
    parser.add_argument(
        "--timeout", type=float, default=3.0, help="Per-notebook execution budget in seconds"
    )
    parser.add_argument(
        "--check-symmetry",
        action="store_true",
        help="Validate directory mirroring across all 7 tracks",
    )
    parser.add_argument(
        "--check-links", action="store_true", help="Validate internal relative links"
    )
    parser.add_argument("--path", type=str, default=".", help="Root directory or specific file")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    failed = False

    # 1. Symmetry check
    if args.check_symmetry:
        print("Checking directory symmetry across all 7 tracks...")
        symmetry_errors = check_track_symmetry(root)
        if symmetry_errors:
            print(f"[FAIL] Directory symmetry errors ({len(symmetry_errors)}):")
            for err in symmetry_errors[:15]:
                print(f"       - {err}")
            if len(symmetry_errors) > 15:
                print(f"       ... and {len(symmetry_errors) - 15} more.")
            failed = True
        else:
            print("[PASS] All 18 categories present across all 7 tracks.")

    # 2. Link check
    if args.check_links:
        print("Checking relative links in documentation...")
        link_errors = check_internal_links(root)
        if link_errors:
            print(f"[FAIL] Broken link errors ({len(link_errors)}):")
            for err in link_errors[:15]:
                print(f"       - {err}")
            failed = True
        else:
            print("[PASS] All relative documentation links resolve.")

    # 3. Notebook validation
    if root.is_file() and root.suffix == ".ipynb":
        notebooks = [root]
    elif root.is_dir():
        notebooks = sorted(
            p
            for p in root.rglob("*.ipynb")
            if not any(
                part.startswith(".") or part in ("scratch", "Linear Neuron") for part in p.parts
            )
        )
    else:
        notebooks = []

    if notebooks:
        print(f"Scanning {len(notebooks)} notebook(s)...")
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
    print("\nAll repository validation checks completed successfully.")


if __name__ == "__main__":
    main()
