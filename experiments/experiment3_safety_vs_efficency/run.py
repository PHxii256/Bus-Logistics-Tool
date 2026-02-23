"""
experiments/experiment3_safety_vs_efficency/run.py
===================================================
Thin launcher for Experiment 3 – Safety vs Efficiency.

Each unique meta config produces a deterministic 8-char subfolder so runs
with different parameters are never overwritten.  The comparison map *and*
the exact meta used are saved together in the subfolder.

Usage
-----
    # Run with the default meta.json in this folder
    python experiments/experiment3_safety_vs_efficency/run.py

    # Run with a custom meta
    python experiments/experiment3_safety_vs_efficency/run.py --meta path/to/meta.json

Output structure
----------------
    experiments/experiment3_safety_vs_efficency/
        meta.json               ← default / template config
        run.py                  ← this file
        a1b2c3d4/               ← hash of the meta config used
            meta.json           ← copy of the config for reproducibility
            comparison_map.html ← interactive Folium map
        e5f6a7b8/               ← another run with different settings
            meta.json
            comparison_map.html
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

# ── ensure repo root is importable ────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, os.pardir, os.pardir))
_COMPARISON_DIR = os.path.join(_ROOT, "experiments", "comparison")

for _p in (_ROOT, _COMPARISON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_comments(obj):
    """Recursively remove ``_comment`` keys so comment-only edits
    don't produce a new experiment hash."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _meta_hash(meta_path: str) -> str:
    """Return an 8-char hex digest of the canonical (comment-stripped) meta."""
    with open(meta_path, encoding="utf-8") as f:
        raw = json.load(f)
    canonical = json.dumps(
        _strip_comments(raw), sort_keys=True, separators=(",", ":")
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 – Safety vs Efficiency comparison runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--meta",
        default=os.path.join(_DIR, "meta.json"),
        help=(
            "Path to a meta.json config file.  "
            "Defaults to meta.json in this folder."
        ),
    )
    args = parser.parse_args()

    meta_path = os.path.abspath(args.meta)
    if not os.path.isfile(meta_path):
        sys.exit(f"[ERROR] meta file not found: {meta_path}")

    # ── compute hash → create run subfolder ───────────────────────────────────
    h       = _meta_hash(meta_path)
    run_dir = os.path.join(_DIR, h)
    os.makedirs(run_dir, exist_ok=True)

    dest_meta   = os.path.join(run_dir, "meta.json")
    output_path = os.path.join(run_dir, "comparison_map.html")

    # copy meta into the run folder (idempotent; overwrites same content)
    shutil.copy2(meta_path, dest_meta)

    print("Experiment 3 – Safety vs Efficiency")
    print(f"  Config hash  : {h}")
    print(f"  Run folder   : {run_dir}")
    print(f"  Meta source  : {meta_path}")
    print(f"  Output map   : {output_path}")
    print()

    # ── delegate to the comparison engine ─────────────────────────────────────
    from run_comparison import run as _run_comparison
    _run_comparison(
        meta_path=dest_meta,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
