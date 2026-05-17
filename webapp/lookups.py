"""Reference data derived from data/houses_Madrid.csv + districts.geojson.

Drives the form (only inputs the model actually uses) and resolves a clicked
map point to a district number.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape, mapping
from shapely.ops import unary_union

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent
DATA_PATH = ROOT / "data" / "houses_Madrid.csv"
GEOJSON_PATH = WEBAPP_DIR / "districts.geojson"

DISTRICT_NAME_ALIASES = {
    "moncloa": "Moncloa-Aravaca",
    "fuencarral": "Fuencarral-El Pardo",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


class LocationData:
    def __init__(self) -> None:
        df = pd.read_csv(
            DATA_PATH,
            usecols=["neighborhood_id", "subtitle", "built_year",
                     "house_type_id", "floor", "energy_certificate"],
        )
        d_num = df["neighborhood_id"].astype(str).str.extract(r"District\s+(\d+)", flags=re.I)[0]
        df = df.assign(_district=pd.to_numeric(d_num, errors="coerce"))

        self.district_name: dict[int, str] = {}
        self._norm_to_num: dict[str, int] = {}
        for s in df["neighborhood_id"].dropna().unique():
            m = re.search(r"District\s+(\d+):\s*(.+?)\s*$", str(s), flags=re.I)
            if m:
                num, name = int(m.group(1)), m.group(2).strip()
                self.district_name.setdefault(num, name)
                self._norm_to_num.setdefault(_norm(name), num)
        for alias_norm, geo_name in DISTRICT_NAME_ALIASES.items():
            if alias_norm in self._norm_to_num:
                self._norm_to_num[_norm(geo_name)] = self._norm_to_num[alias_norm]

        # neighborhoods (subtitle) per district, with display label
        self.neighborhoods: dict[int, list[dict]] = {}
        sub = df.dropna(subset=["_district", "subtitle"])
        for dn, grp in sub.groupby("_district"):
            items = sorted(grp["subtitle"].astype(str).unique())
            self.neighborhoods[int(dn)] = [
                {"subtitle": s, "label": re.sub(r",\s*Madrid$", "", s)} for s in items
            ]

        # default built_year per district + global (median)
        by = df.dropna(subset=["built_year"])
        self.builtyear_by_district = {
            int(dn): int(round(g["built_year"].median()))
            for dn, g in by.dropna(subset=["_district"]).groupby("_district")
        }
        self.builtyear_global = int(round(df["built_year"].median())) if df["built_year"].notna().any() else 1965

        # categorical option lists straight from the dataset
        self.house_types = sorted(df["house_type_id"].dropna().unique())
        self.default_house_type = df["house_type_id"].mode().iloc[0]
        self.floors = sorted(df["floor"].dropna().astype(str).unique(),
                             key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
        self.energy_certificates = sorted(df["energy_certificate"].dropna().unique())
        self.street_types = ["Calle", "Avenida", "Paseo", "Plaza", "Carretera",
                             "Camino", "Cuesta", "Ronda", "Glorieta", "Travesía", "Other"]

        # geojson polygons -> dataset district_num
        gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
        self._geo: list[tuple[int | None, object]] = []
        for feat in gj["features"]:
            num = self._norm_to_num.get(_norm(feat["properties"].get("name", "")))
            self._geo.append((num, shape(feat["geometry"])))

    def geojson_text(self) -> str:
        gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
        for feat in gj["features"]:
            num = self._norm_to_num.get(_norm(feat["properties"].get("name", "")))
            feat["properties"]["district_num"] = num
            feat["properties"]["has_data"] = num is not None
        return json.dumps(gj)

    def perimeter_geojson(self) -> dict:
        """Outer boundary of all districts dissolved into one (city perimeter)."""
        union = unary_union([g for _, g in self._geo])
        return {"type": "Feature", "properties": {},
                "geometry": mapping(union.boundary)}

    def resolve_point(self, lat: float, lon: float) -> dict | None:
        pt = Point(lon, lat)
        for d_num, geom in self._geo:
            if geom.contains(pt):
                if d_num is None:
                    return {"district_num": None, "district_name": None,
                            "message": "This district has no listings in the dataset; "
                                       "the model treats it as unknown."}
                return {"district_num": d_num,
                        "district_name": self.district_name.get(d_num)}
        return None

    def neighborhood_options(self, district_num: int) -> list[dict]:
        return self.neighborhoods.get(district_num, [])
