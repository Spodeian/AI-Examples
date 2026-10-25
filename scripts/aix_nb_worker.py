"""Isolated single-notebook worker invoked by `aix run-notebooks`.

Pre-warms heavy scientific imports (`numpy`, `torch`, `core`, `matplotlib` headless)
before starting the execution timer so the 3.0s notebook budget measures pure cell
runtime while the parent `aix` process enforces a hard OS-level wall-clock kill.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _prewarm() -> None:
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import core  # noqa: F401
    except ImportError:
        pass
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


def main() -> int:
    if len(sys.argv) < 2:
        print("AIX_RESULT " + json.dumps({"ok": False, "elapsed": 0.0, "error": "missing notebook path"}))
        return 2

    nb_path = Path(sys.argv[1]).resolve()
    repo_root = Path.cwd().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("AIX_RESULT " + json.dumps({"ok": False, "elapsed": 0.0, "error": f"invalid JSON: {exc}"}))
        return 1

    _prewarm()

    ns: dict[str, object] = {"__name__": "__main__", "__file__": str(nb_path)}
    t0 = time.perf_counter()
    ok = True
    err = ""

    for idx, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        code_str = "".join(src) if isinstance(src, list) else str(src)
        if not code_str.strip():
            continue
        try:
            compiled = compile(code_str, f"{nb_path.name}#cell_{idx}", "exec")
            exec(compiled, ns)
        except Exception as exc:
            ok = False
            tb = traceback.extract_tb(exc.__traceback__)
            line_info = f" (line {tb[-1].lineno})" if tb else ""
            err = f"Cell {idx}{line_info}: {type(exc).__name__}: {exc}"
            traceback.print_exc()
            break

    elapsed = time.perf_counter() - t0
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    print("AIX_RESULT " + json.dumps({"ok": ok, "elapsed": elapsed, "error": err}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
