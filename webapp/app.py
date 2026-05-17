"""FastAPI backend for the Madrid asking-price appraisal estimator.

Serves best_lgbm.pkl. The form only collects inputs the model actually uses
(no price/leakage fields — those are dropped by the training pipeline).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from madrid_model import (
    AMENITY_COLS, ORIENTATION_COLS, EXTRA_BOOL_COLS,
    load_model, model_feature_names, predict_eur,
)
from lookups import LocationData

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent
MODEL_PATH = ROOT / "best_lgbm.pkl"
STATIC_DIR = WEBAPP_DIR / "static"

# Fixed reported model error used to size the ± interval. MAPE = 14.58% is the
# metric used in the report to justify the final-model selection.
MAPE_PCT = 14.58

app = FastAPI(title="Madrid Asking-Price Appraisal")
STATE: dict = {}

# user-facing labels for the boolean flags the model uses
BOOL_LABELS = {
    "has_lift": "Lift", "has_ac": "Air conditioning", "has_parking": "Parking",
    "has_terrace": "Terrace", "has_balcony": "Balcony", "has_pool": "Pool",
    "has_garden": "Garden", "has_storage_room": "Storage room",
    "has_fitted_wardrobes": "Fitted wardrobes", "has_green_zones": "Green zones",
    "has_central_heating": "Central heating", "has_individual_heating": "Individual heating",
    "is_exterior": "Exterior", "is_accessible": "Accessible",
    "is_new_development": "New development", "is_renewal_needed": "Needs renovation",
    "is_exact_address_hidden": "Exact address hidden", "is_floor_under": "Below street level",
}
ORIENT_LABELS = {
    "is_orientation_north": "North", "is_orientation_south": "South",
    "is_orientation_east": "East", "is_orientation_west": "West",
}


@app.on_event("startup")
def _startup() -> None:
    loc = LocationData()
    STATE["loc"] = loc
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
        STATE["model"] = model
        STATE["n_features"] = len(model_feature_names(model))
        STATE["mape"] = MAPE_PCT
    else:
        STATE["model"] = None
        STATE["mape"] = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/districts")
def districts() -> JSONResponse:
    loc: LocationData = STATE["loc"]
    return JSONResponse({
        "geojson": loc.geojson_text(),
        "district_name": {str(k): v for k, v in loc.district_name.items()},
        "perimeter": loc.perimeter_geojson(),
    })


@app.get("/api/options")
def options() -> JSONResponse:
    loc: LocationData = STATE["loc"]
    return JSONResponse({
        "model_loaded": STATE["model"] is not None,
        "mape_pct": STATE["mape"],
        "house_types": loc.house_types,
        "default_house_type": loc.default_house_type,
        "floors": loc.floors,
        "energy_certificates": loc.energy_certificates,
        "street_types": loc.street_types,
        "amenities": [{"key": k, "label": BOOL_LABELS.get(k, k)}
                      for k in AMENITY_COLS + ["has_parking", "has_individual_heating"]],
        "flags": [{"key": k, "label": BOOL_LABELS.get(k, k)}
                  for k in ["is_new_development", "is_renewal_needed"]],
        "orientations": [{"key": k, "label": ORIENT_LABELS[k]} for k in ORIENTATION_COLS],
        "districts": {str(n): loc.district_name[n] for n in sorted(loc.district_name)},
        "neighborhoods": {str(n): loc.neighborhood_options(n) for n in loc.district_name},
        "builtyear_by_district": {str(k): v for k, v in loc.builtyear_by_district.items()},
        "builtyear_global": loc.builtyear_global,
    })


class PredictRequest(BaseModel):
    lat: float | None = None
    lon: float | None = None
    district_num: int | None = None
    subtitle: str = Field(..., min_length=1)
    sq_mt_built: float = Field(..., gt=0)
    n_bathrooms: float = Field(..., gt=0)
    n_rooms: int = Field(..., ge=1)
    built_year: int | None = None
    house_type_id: str | None = None
    floor: str | None = None
    energy_certificate: str | None = None
    street_type: str | None = None
    flags: list[str] = []  # truthy amenity/flag/orientation keys


@app.post("/api/predict")
def predict(req: PredictRequest) -> JSONResponse:
    if STATE["model"] is None:
        raise HTTPException(503, "Model not loaded — best_lgbm.pkl is not present yet.")
    loc: LocationData = STATE["loc"]

    district_num = req.district_num
    district_name = None
    note = None
    if district_num is None:
        if req.lat is None or req.lon is None:
            raise HTTPException(400, "Provide either district_num or lat/lon.")
        resolved = loc.resolve_point(req.lat, req.lon)
        if resolved is None:
            raise HTTPException(422, "That point is outside the Madrid districts.")
        district_num = resolved.get("district_num")
        district_name = resolved.get("district_name")
        note = resolved.get("message")
    else:
        district_name = loc.district_name.get(district_num)

    built_year = req.built_year
    if built_year is None:
        built_year = loc.builtyear_by_district.get(district_num, loc.builtyear_global)

    rec = {
        "sq_mt_built": req.sq_mt_built,
        "n_bathrooms": req.n_bathrooms,
        "n_rooms": req.n_rooms,
        "built_year": built_year,
        "district_id": district_num,
        "subtitle": req.subtitle,
        "floor": req.floor,
        "house_type_id": req.house_type_id or loc.default_house_type,
        "energy_certificate": req.energy_certificate,
        "raw_address": f"{req.street_type or 'Calle'} sin número",
    }
    flagset = set(req.flags)
    for c in AMENITY_COLS + ORIENTATION_COLS + EXTRA_BOOL_COLS:
        rec[c] = c in flagset

    try:
        point = float(predict_eur(STATE["model"], [rec])[0])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Prediction failed: {exc}")

    mape = STATE["mape"]
    resp = {
        "point_estimate": round(point),
        "district_num": district_num,
        "district_name": district_name,
        "note": note,
        "mape_pct": round(mape, 2) if mape is not None else None,
    }
    if mape is not None:
        resp["lower"] = round(point * (1 - mape / 100))
        resp["upper"] = round(point * (1 + mape / 100))
    return JSONResponse(resp)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
