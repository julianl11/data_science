"""
Insert a markdown rationale cell immediately before cell 14
(`fill_amenities_by_year`) in exploring_2_ivan_changes.ipynb.

Snapshot the notebook (`cp …ipynb ….ipynb.prev`) before running.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "exploring_2_ivan_changes.ipynb"

MD_SRC = """\
### Why `fillna(False)` instead of cohort-mode imputation

The raw `houses_Madrid.csv` was scraped from Idealista, where each row is a
**sales advertisement** — listings function as marketing material in a
premium real-estate market. The economic incentive is therefore asymmetric:

- An amenity that **exists** is almost always listed, because it justifies a
  higher asking price (pool, garden, terrace, AC, fitted wardrobes, …).
- An amenity that is **absent** is simply not mentioned. The portal has no
  UI for "this apartment has no pool"; the seller has no reason to record it.

Inspection of the raw values confirms this: 9 of the 12 amenity columns
(`has_pool`, `has_garden`, `has_balcony`, `has_storage_room`,
`is_accessible`, `has_green_zones`, `has_terrace`, `has_ac`,
`has_fitted_wardrobes`) contain **zero `False` rows**. The data only ever
records `True` or empty. Under those semantics, "empty" is the implicit
negative class, not a missing measurement.

We therefore replace the original cohort-mode imputation with
`fillna(False)` across all amenity columns (function below). Mode
imputation on a True-or-empty distribution would compute mode = `True` and
flip every empty cell to `True` — producing the failure mode we observed
pre-fix, where post-pipeline `has_pool_True` was `1` for every one of the
21,594 surviving rows even though the raw CSV had only 5,171 True listings.

The three columns with genuine `False` rows in the source (`has_lift`,
`is_exterior`, `has_central_heating`) are treated the same way for
consistency. The marketing-material reading still applies: a building with
a lift, or one explicitly classified as exterior, would advertise it; an
unlisted entry is closer to "amenity not present" than to "we don't know".
"""


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    target_idx = 14
    src = "".join(cells[target_idx].get("source", []))
    if "def fill_amenities_by_year" not in src:
        print(
            f"Cell {target_idx} does not look like fill_amenities_by_year. "
            f"Aborting. First 80 chars: {src[:80]!r}",
            file=sys.stderr,
        )
        return 1

    md_cell = {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": MD_SRC.splitlines(keepends=True),
    }
    cells.insert(target_idx, md_cell)

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"OK -- inserted markdown rationale at index {target_idx}; "
          f"total cells now {len(cells)} (was {len(cells)-1}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
