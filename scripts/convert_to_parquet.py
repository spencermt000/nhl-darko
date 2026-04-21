"""
convert_to_parquet.py — Two jobs:

1. CONVERT: Reads existing large CSV files in output/ and writes .parquet equivalents.
2. PATCH:   Rewrites pipeline Python scripts to use read_parquet / to_parquet
            instead of read_csv / to_csv for those files.

Usage:
    python scripts/convert_to_parquet.py convert   # convert existing CSVs → parquet
    python scripts/convert_to_parquet.py patch      # patch scripts in-place
    python scripts/convert_to_parquet.py all        # both steps
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent
OUTPUT = BASE / "output"

# ── Files to migrate (CSV → Parquet) ─────────────────────────────────────────
# Only pipeline-internal files ≥ ~1 MB.
# Dashboard-facing deploy/aggregated outputs stay as CSV.
MIGRATE = [
    "v2_clean_pbp",
    "v2_penalties",
    "v2_box_prior",
    "v2_rapm_by_season",
    "v2_final_ratings_by_season",
    "v2_gar_by_season",
    "v2_gar_pooled",
    "v3_rolling_rapm",
    "v4_epm_raw_per_game",
    "v4_daily_epm",
    "v4_bpm_player_seasons",
    "v4_season_war",
    "v5_daily_ratings",       # full (deploy version stays CSV)
    "v5_composite_player_seasons",
    "v5_season_war",
    "v5_daily_war",
    "v6_carry_forward",
    "v6_team_game_ratings",
    "ifinish_by_season",
    "win_shares_by_season",
    "v3_rolling_rapm_latest",
    "v2_final_ratings",
]

# ── Pipeline scripts to patch ─────────────────────────────────────────────────
SCRIPTS = [
    "prep/build_dataset.py",
    "rapm/rapm_bayesian.py",
    "rapm/rolling_rapm.py",
    "rapm/pp_pk_rapm.py",
    "rapm/bootstrap_rapm.py",
    "bpr/box_prior.py",
    "bpr/bpm.py",
    "bpr/epm.py",
    "bpr/gar.py",
    "bpr/learn_weights.py",
    "bpr/composite_v2.py",
    "bpr/composite_v4.py",
    "bpr/carry_forward.py",
    "bpr/daily.py",
    "bpr/win_shares.py",
    "team/team_ratings.py",
    "viz/viz_mcdavid.py",
    "dashboard/streamlit_app.py",
    "scripts/predictiveness_v2.py",
]


def cmd_convert():
    """Convert existing CSVs to parquet."""
    for stem in MIGRATE:
        csv_path = OUTPUT / f"{stem}.csv"
        pq_path = OUTPUT / f"{stem}.parquet"
        if not csv_path.exists():
            print(f"  SKIP (no CSV): {stem}.csv")
            continue
        if pq_path.exists() and pq_path.stat().st_mtime >= csv_path.stat().st_mtime:
            print(f"  OK  (up to date): {stem}.parquet")
            continue
        size_mb = csv_path.stat().st_size / 1_048_576
        print(f"  Converting ({size_mb:.0f} MB): {stem}.csv → .parquet", end="", flush=True)
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_parquet(pq_path, index=False)
        pq_mb = pq_path.stat().st_size / 1_048_576
        print(f"  → {pq_mb:.0f} MB  ({100*pq_mb/size_mb:.0f}% of CSV size)")
    print("Convert step done.")


def _build_replacement_map():
    """Build regex patterns for CSV → parquet substitution in source code."""
    patterns = []
    for stem in MIGRATE:
        csv_rel = f"output/{stem}.csv"
        pq_rel = f"output/{stem}.parquet"
        patterns.append((csv_rel, pq_rel))
    return patterns


def _patch_file(path: Path, replacements):
    """Apply CSV→parquet substitutions and fix read/write calls in a Python file."""
    original = path.read_text()
    text = original

    for csv_rel, pq_rel in replacements:
        if csv_rel not in text:
            continue

        # Replace the path string itself
        text = text.replace(f'"{csv_rel}"', f'"{pq_rel}"')
        text = text.replace(f"'{csv_rel}'", f"'{pq_rel}'")

    # Rewrite read_csv(...parquet...) → read_parquet(...)
    # Handle usecols= → columns= only inside read_parquet calls
    def fix_read_call(m):
        call = m.group(0)
        # Change function name
        call = call.replace("pd.read_csv(", "pd.read_parquet(", 1)
        # usecols → columns
        call = re.sub(r'\busecols\b', 'columns', call)
        # Remove unsupported kwargs
        call = re.sub(r',?\s*low_memory\s*=\s*\w+', '', call)
        call = re.sub(r',?\s*dtype\s*=\s*\{[^}]*\}', '', call)
        return call

    # Match read_csv calls that now reference a .parquet path
    text = re.sub(
        r'pd\.read_csv\([^)]*\.parquet[^)]*\)',
        fix_read_call,
        text,
        flags=re.DOTALL,
    )

    # Rewrite to_csv(...parquet...) → to_parquet(...)
    def fix_write_call(m):
        call = m.group(0)
        call = call.replace(".to_csv(", ".to_parquet(", 1)
        return call

    text = re.sub(
        r'\.to_csv\([^)]*\.parquet[^)]*\)',
        fix_write_call,
        text,
        flags=re.DOTALL,
    )

    if text == original:
        return False

    path.write_text(text)
    return True


def cmd_patch():
    """Patch pipeline scripts to use parquet paths."""
    replacements = _build_replacement_map()
    patched = 0
    for rel in SCRIPTS:
        p = BASE / rel
        if not p.exists():
            print(f"  SKIP (not found): {rel}")
            continue
        changed = _patch_file(p, replacements)
        if changed:
            print(f"  PATCHED: {rel}")
            patched += 1
        else:
            print(f"  no changes: {rel}")
    print(f"Patch step done. {patched} files updated.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("convert", "patch", "all"):
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd in ("convert", "all"):
        cmd_convert()
    if cmd in ("patch", "all"):
        cmd_patch()


if __name__ == "__main__":
    main()
