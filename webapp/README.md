# Madrid Asking-Price Appraisal — Web Simulation

A local web app that turns the trained `best_lgbm.pkl` model into an interactive
asking-price estimator. The user clicks a point on a map of Madrid to set the
district, answers a short multiple-choice form, and gets a price estimate with a
± confidence band.

Everything runs locally and uses only free/open-source components — no paid APIs,
no keys, no accounts. Map tiles come from OpenStreetMap (free, no key) and the
district boundaries are a bundled public-domain GeoJSON.

---

## 1. What it does (end to end)

1. **Map** — Leaflet map of Madrid with the 21 official district polygons drawn
   on top. Clicking anywhere resolves the point to a district
   (point-in-polygon).
2. **District → form** — once a district is set, a short form appears. A
   **required** neighborhood dropdown (populated from the dataset for that
   district) pins down the strongest location signal.
3. **Predict** — the answers are turned into the exact 230-feature vector the
   model expects, the model predicts `log1p(price)`, and the result is converted
   back to euros.
4. **Interval** — the estimate is shown with a band of
   `estimate × (1 ± MAPE)`, where **MAPE is fixed at 14.58 %** (the model's
   mean absolute percentage error — the metric used in the report to justify
   the final-model selection). It is the constant `MAPE_PCT` in `app.py`.

Only inputs the model actually uses are asked. Price/leakage fields
(`buy_price`, `buy_price_by_area`, `rent_price`, …) are **never** requested —
they are dropped by the training pipeline and would not exist for a real unseen
listing.

---

## 2. The model

- **File:** `../best_lgbm.pkl` (project root, one level above `webapp/`).
- **Type:** `lightgbm.sklearn.LGBMRegressor`, 230 named input features,
  `n_estimators=25000`.
- **Target:** trained on `log1p(buy_price)`, so predictions are log-space and
  inverted with `expm1`.
- **Origin:** the `exploring_2_ivan_changes.ipynb` pipeline
  (`prepare` / `encode_address` / `clean_feature_names`, one-hot encoding with
  `drop_first=True`).

### Why a custom encoder instead of replaying the notebook pipeline

The training pipeline does `pd.get_dummies(drop_first=True)`. On a **single**
listing every categorical column has exactly one value, so `drop_first` deletes
it and every category silently collapses to its reference level — the model
would ignore district, property type, etc.

To avoid this, `madrid_model.py` builds the feature row **deterministically
against the model's own `feature_name_`**:

- numeric features (`sq_mt_built`, `n_rooms`, `n_bathrooms`, `built_year`,
  `addr_has_number`, `addr_number_bucket`) are passed through;
- boolean flags are matched to whichever form training produced
  (`<col>_True` dummy *or* a plain numeric column);
- one-hot categoricals (`subtitle`, `floor`, `house_type_id`,
  `energy_certificate`, `district_id`, `addr_street_type`) set the single
  column `clean("<prefix>_<value>")` to 1 if it exists; the reference/unknown
  level is represented as all-zeros — exactly what training produced.

This is mathematically equivalent to the training encoding and is correct for a
single row.

---

## 3. Which questions are asked, and why

| Form field | Raw column(s) | Role in the model |
|---|---|---|
| Map click | `neighborhood_id` → `district_id` | One-hot `district_id_*` |
| Neighborhood * | `subtitle` | One-hot `subtitle_*` (strongest location signal) |
| Built area m² * | `sq_mt_built` | Numeric (dominant predictor) |
| Bathrooms * | `n_bathrooms` | Numeric |
| Rooms * | `n_rooms` | Numeric (1–7) |
| Build year | `built_year` | Numeric (defaults to district median if blank) |
| Property type | `house_type_id` | One-hot |
| Floor | `floor` | One-hot |
| Energy certificate | `energy_certificate` | One-hot |
| Street type | `raw_address` | One-hot `addr_street_type_*` + number features |
| Amenities chips | `has_*`, `is_exterior`, `is_accessible` | Boolean dummies |
| Characteristics chips | `is_new_development`, `is_renewal_needed` | Boolean dummies |
| Orientation chips | `is_orientation_*` | Boolean dummies |

`*` = required. The neighborhood dropdown shows only the neighborhood **name** —
no €/m² figure is displayed or sent (this model does not use a per-neighborhood
price index).

---

## 4. File structure

```
Project_housing_prices_madrid/
├── best_lgbm.pkl                 # the trained model (NOT created by this app)
├── data/houses_Madrid.csv        # dataset — drives form options
└── webapp/
    ├── app.py                    # FastAPI backend (endpoints, startup, MAPE)
    ├── madrid_model.py           # model load + deterministic feature encoder
    ├── lookups.py                # district/neighborhood data + point-in-polygon
    ├── districts.geojson         # 21 Madrid district polygons (bundled, free)
    ├── requirements.txt          # Python dependencies
    ├── run.ps1                   # one-command launcher
    ├── README.md                 # this file
    └── static/
        ├── index.html            # page layout
        ├── app.js                # map, form building, fetch calls
        └── styles.css            # dark theme
```

- **`app.py`** — loads the model + reference data once at startup, exposes
  the API, serves the frontend. Starts cleanly even if `best_lgbm.pkl` is
  missing (predictions disabled until it exists).
- **`madrid_model.py`** — `load_model`, `model_feature_names`, the
  `build_X` / `predict_eur` deterministic encoder, and the `encode_address`
  mirror of the notebook.
- **`lookups.py`** — parses `houses_Madrid.csv` for district names,
  per-district neighborhoods (`subtitle`), default build years, and the
  categorical option lists; loads `districts.geojson` and resolves
  `(lat, lon) → district_num` with Shapely.

---

## 5. Setup

> **Environment note:** the machine's Python 3.12 install is missing the venv
> launchers, so a virtualenv cannot be created. The app is run with the global
> interpreter:
> `C:\Users\IVANL\AppData\Local\Programs\Python\Python312\python.exe`

Install dependencies once:

```powershell
& "C:\Users\IVANL\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r webapp\requirements.txt
```

`scikit-learn` is pinned to `1.5.2` so the pickled model stays loadable.

---

## 6. Running the simulation

### Option A — launcher script

```powershell
powershell webapp\run.ps1
```

Then open <http://localhost:8000>.

### Option B — manual

```powershell
cd webapp
& "C:\Users\IVANL\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### Using the page

1. Wait for the map to load (district polygons appear in blue; grey = a district
   with no dataset listings).
2. **Click a point** inside Madrid, or click a district polygon directly. The
   banner confirms the detected district.
3. Pick a **neighborhood** (required — the strongest location signal).
4. Fill **Built area (m²)**, **Bathrooms** and **Rooms** (required); fill any
   other fields you know — blanks fall back to sensible defaults.
5. Toggle the amenity / characteristic / orientation chips that apply.
6. Click **Estimate asking price**. You get a central estimate and a
   `lower – upper` band with the model's MAPE.

---

## 7. API reference

Base URL: `http://localhost:8000`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web page |
| `GET` | `/api/districts` | District GeoJSON + `district_num → name` map |
| `GET` | `/api/options` | Form metadata (types, floors, certs, neighborhoods, MAPE, `model_loaded`) |
| `POST` | `/api/predict` | Returns the estimate + interval |

**`POST /api/predict` body** (`subtitle`, `sq_mt_built`, `n_rooms` and
`n_bathrooms` are required; provide either `district_num` or `lat`+`lon`):

```json
{
  "district_num": 15,
  "subtitle": "Barrio de Salamanca, Madrid",
  "sq_mt_built": 110,
  "n_bathrooms": 2,
  "n_rooms": 3,
  "built_year": 1970,
  "house_type_id": "HouseType 1: Pisos",
  "floor": "3",
  "energy_certificate": "C",
  "street_type": "Calle",
  "flags": ["has_lift", "has_ac", "is_exterior"]
}
```

**Response:**

```json
{
  "point_estimate": 647730,
  "lower": 553291,
  "upper": 742169,
  "mape_pct": 14.58,
  "district_num": 15,
  "district_name": "Salamanca",
  "note": null
}
```

`503` is returned if `best_lgbm.pkl` is not present; `422` if a clicked point is
outside the Madrid districts or if `subtitle` is missing/empty.

---

## 8. Replacing / retraining the model

The app does not train anything. To use a new model, overwrite
`../best_lgbm.pkl` and restart the server. Requirements:

- It must be loadable with `joblib.load` and expose feature names via
  `feature_name_` / `feature_names_in_` / `booster_.feature_name()`.
- It must predict `log1p(buy_price)` (the app applies `expm1`).
- If the **feature set or names change**, the encoder still aligns to the new
  `feature_name_`, but only for the column families it knows (numeric / boolean
  / the six one-hot prefixes). New feature families would need handling added in
  `madrid_model.build_X`.

MAPE stays fixed at 14.58 % — edit `MAPE_PCT` in `app.py` if the new model
has a different reported error.

---

## 9. Notes & caveats

- **Neighborhood is required.** `subtitle_*` is one of the model's strongest
  signals, so the form blocks submission until a neighborhood is picked. This
  avoids the silent bias where a blank dropdown fell back to the dropped
  reference barrio (not a district average).
- **Districts with no dataset listings** (e.g. San Blas) are drawn grey and
  have no neighborhoods. Clicking one is rejected outright (it never becomes
  the selected district) — the model could not meaningfully estimate it anyway.
- **MAPE** is fixed at 14.58 % (`MAPE_PCT` in `app.py`), used only to size
  the interval. Change that constant to resize the band.
- **Map tiles** require internet (OpenStreetMap, free, no key). Everything else
  works offline; the GeoJSON is bundled.
- The interval is symmetric in percentage terms
  (`estimate × (1 ± MAPE/100)`), not a calibrated prediction interval.
