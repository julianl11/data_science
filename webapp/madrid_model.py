"""Inference for best_lgbm.pkl (the LGBMRegressor from exploring_2_ivan_changes.ipynb).

The training pipeline (cells 10-24) does `pd.get_dummies(drop_first=True)` then
`clean_feature_names`, fits on `log1p(buy_price)`. We CANNOT replay get_dummies at
inference: on a single row each categorical has one value, so `drop_first` deletes it
and every category collapses to its reference level. Instead we build the feature
vector deterministically against the model's own `feature_name_` (230 names), which is
exactly equivalent and single-row safe. Reference categories => all dummies 0, which is
what training produced for the dropped level.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

# cell 10 constants
AMENITY_COLS = [
    "has_ac", "has_fitted_wardrobes", "has_lift", "is_exterior", "has_pool",
    "has_terrace", "has_garden", "has_balcony", "has_storage_room",
    "is_accessible", "has_green_zones", "has_central_heating",
]
ORIENTATION_COLS = [
    "is_orientation_north", "is_orientation_south",
    "is_orientation_east", "is_orientation_west",
]
# extra raw boolean/flag columns the model also consumes
EXTRA_BOOL_COLS = [
    "has_individual_heating", "has_parking", "is_new_development",
    "is_renewal_needed", "is_exact_address_hidden", "is_floor_under",
]
NUMERIC_COLS = ["sq_mt_built", "n_rooms", "n_bathrooms", "built_year"]
# raw categorical -> OHE prefix used by get_dummies (prefix == column name)
OHE_COLS = ["subtitle", "floor", "house_type_id", "energy_certificate",
            "district_id", "addr_street_type"]


def _clean(name: str) -> str:
    """clean_feature_names (cell 20), applied to a single already-joined name."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", str(name))).strip("_")


def encode_address(raw_address: str | None) -> dict[str, float | str]:
    """Mirror of encode_address (cell 19) for one address string."""
    addr = raw_address.strip() if isinstance(raw_address, str) else ""
    m = re.match(
        r"^(Calle|Avenida|Paseo|Plaza|Carretera|Camino|Cuesta|Ronda|"
        r"Glorieta|Travesía|Callejón|Sector|Urbanización|Vía)",
        addr, flags=re.IGNORECASE,
    )
    street_type = m.group(1).title() if m else "Other"
    has_number = 1 if re.search(r",?\s*\d+", addr) else 0
    num_m = re.search(r",\s*(\d+)", addr)
    bucket = 0.0
    if num_m:
        n = float(num_m.group(1))
        edges = [0, 10, 30, 60, 100, np.inf]
        for i in range(len(edges) - 1):
            if edges[i] < n <= edges[i + 1]:
                bucket = float(i + 1)
                break
    return {"addr_street_type": street_type,
            "addr_has_number": has_number,
            "addr_number_bucket": bucket}


def load_model(path: str | Path):
    return joblib.load(path)


def model_feature_names(model) -> list[str]:
    fn = getattr(model, "feature_name_", None)
    if fn is None:
        fn = getattr(model, "feature_names_in_", None)
    if fn is None and hasattr(model, "booster_"):
        fn = model.booster_.feature_name()
    return [str(x) for x in fn]


def build_X(records: list[dict], feature_names: list[str]) -> pd.DataFrame:
    """Build the model design matrix from raw-input dicts.

    Each record may contain: sq_mt_built, n_rooms, n_bathrooms, built_year,
    raw_address, district_id, subtitle, floor, house_type_id, energy_certificate,
    and any AMENITY/ORIENTATION/EXTRA bool flags (truthy => set).
    """
    feat_set = set(feature_names)
    rows = []
    for rec in records:
        v = {f: 0 for f in feature_names}

        for col in NUMERIC_COLS:
            if col in feat_set and rec.get(col) is not None:
                v[col] = float(rec[col])

        addr = encode_address(rec.get("raw_address"))
        for col in ("addr_has_number", "addr_number_bucket"):
            if col in feat_set:
                v[col] = addr[col]

        # boolean flags: training emitted either `<col>_True` (bool dtype) or
        # plain `<col>` (numeric). Match whichever the model actually has.
        for col in AMENITY_COLS + ORIENTATION_COLS + EXTRA_BOOL_COLS:
            on = 1 if rec.get(col) else 0
            t = f"{col}_True"
            if t in feat_set:
                v[t] = on
            elif col in feat_set:
                v[col] = on

        # one-hot categoricals: feature == clean(f"{prefix}_{value}")
        cat_values = {
            "subtitle": rec.get("subtitle"),
            "floor": rec.get("floor"),
            "house_type_id": rec.get("house_type_id"),
            "energy_certificate": rec.get("energy_certificate"),
            "district_id": rec.get("district_id"),
            "addr_street_type": addr["addr_street_type"],
        }
        for prefix, value in cat_values.items():
            if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
                continue
            fname = _clean(f"{prefix}_{value}")
            if fname in feat_set:
                v[fname] = 1
        rows.append(v)

    return pd.DataFrame(rows, columns=feature_names)


def predict_eur(model, records: list[dict]) -> np.ndarray:
    X = build_X(records, model_feature_names(model))
    pred_log = model.predict(X)
    return np.expm1(pred_log).clip(min=0)
