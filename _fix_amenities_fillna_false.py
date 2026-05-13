"""
Replace cell 14 (`fill_amenities_by_year`) in exploring_2_ivan_changes.ipynb
with a fillna(False) implementation.

Bug being fixed
---------------
9 of 12 amenity columns in the raw CSV have zero False values
(empty cell encodes 'amenity absent', not 'unknown'). The old
cohort-mode imputation flipped every empty to True for those
columns, producing post-OHE columns like `has_pool_True == 1`
for all 21,594 surviving rows when the raw CSV had only 5,171
True rows.

Snapshot the notebook (`cp …ipynb ….ipynb.prev`) before running.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "exploring_2_ivan_changes.ipynb"

NEW_SOURCE = '''\
def fill_amenities_by_year(df: pd.DataFrame,
                           amenity_cols: list[str] | None = None,
                           year_col: str = "built_year",
                           bin_width: int = 5) -> pd.DataFrame:
    """
    Cast boolean amenity flags to true `bool` dtype, treating an empty
    raw CSV cell as `False`.

    Why fillna(False), not cohort-mode imputation
    ---------------------------------------------
    Inspection of the raw `houses_Madrid.csv` shows that 9 of the 12
    amenity columns have **zero `False` values** in the source:

        has_pool, has_garden, has_balcony, has_storage_room,
        is_accessible, has_green_zones, has_terrace, has_ac,
        has_fitted_wardrobes

    For these columns the listing only records positive presence;
    empty = "amenity not present", not "unknown". Cohort-mode
    imputation would compute the cohort mode of {True, NaN}, find
    `True`, and flip every empty cell to True — producing a column
    that is `has_pool == True` everywhere and a post-`get_dummies`
    output of `has_pool_True == 1` for all rows.

    The remaining three columns (`has_lift`, `is_exterior`,
    `has_central_heating`) do have genuine `False` rows in the raw
    data, but we apply the same `fillna(False)` rule uniformly per
    the project decision: treat absence as "amenity not present".

    Side effect: the output amenity columns end up as **dtype=bool**,
    which `pd.get_dummies` skips by default — so the downstream
    pipeline keeps `has_pool` as a single boolean feature instead of
    creating an `has_pool_True` dummy.

    The `year_col` and `bin_width` parameters are retained for
    signature compatibility with the previous cohort-mode version
    but are no longer used.

    Parameters
    ----------
    df : pd.DataFrame
    amenity_cols : list[str] or None
        Columns to coerce. Defaults to AMENITY_COLS global.
    year_col, bin_width
        Unused; kept for signature compatibility.

    Returns
    -------
    pd.DataFrame
        Copy with amenity columns cast to bool, NaN → False.
    """
    if amenity_cols is None:
        amenity_cols = AMENITY_COLS

    df = df.copy()
    present_cols = [c for c in amenity_cols if c in df.columns]

    if not present_cols:
        print("[fill_amenities_by_year] No amenity columns found — skipping.")
        return df

    total_filled = 0
    for col in present_cols:
        n_nan = int(df[col].isna().sum())
        total_filled += n_nan

        # Object-dtype columns hold the raw CSV strings 'True'/'False';
        # coerce those to real bool before fillna so the column doesn't
        # end up mixed-typed.
        df[col] = df[col].replace({"True": True, "False": False})
        df[col] = df[col].fillna(False)
        df[col] = df[col].astype(bool)

    print(f"[fill_amenities_by_year] Filled {total_filled} NaNs with False "
          f"across {len(present_cols)} amenity columns "
          f"(cast to bool; get_dummies will skip them).\\n")
    return df
'''


def main() -> int:
    try:
        compile(NEW_SOURCE, "<cell_14>", "exec")
    except SyntaxError as exc:
        print(f"SyntaxError in NEW_SOURCE: {exc}", file=sys.stderr)
        return 1

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    target_idx = 14
    cell = cells[target_idx]
    old_src = "".join(cell["source"])
    if "def fill_amenities_by_year" not in old_src:
        print(
            f"Cell {target_idx} does not look like fill_amenities_by_year. "
            f"Aborting. First 80 chars: {old_src[:80]!r}",
            file=sys.stderr,
        )
        return 1

    cell["source"] = NEW_SOURCE.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"OK -- rewrote cell {target_idx} (fill_amenities_by_year).")
    print(f"     Old source: {len(old_src)} chars")
    print(f"     New source: {len(NEW_SOURCE)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
