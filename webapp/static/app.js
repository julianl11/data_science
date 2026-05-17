"use strict";

const EUR = new Intl.NumberFormat("es-ES", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const $ = (id) => document.getElementById(id);

let MAP, GEOLAYER, SELECTED_LAYER = null;
let OPTIONS = null;
let SELECTED = { district_num: null, district_name: null };
let HAS_RESULT = false;

async function init() {
  MAP = L.map("map", { scrollWheelZoom: true }).setView([40.4168, -3.7038], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(MAP);

  const [dRes, oRes] = await Promise.all([
    fetch("/api/districts").then((r) => r.json()),
    fetch("/api/options").then((r) => r.json()),
  ]);
  OPTIONS = oRes;

  if (!OPTIONS.model_loaded) {
    const w = $("model-warning");
    w.textContent = "best_lgbm.pkl is not present yet — the form works but predictions are disabled until the model file is ready.";
    w.classList.remove("hidden");
  }
  buildForm();

  const geo = JSON.parse(dRes.geojson);
  GEOLAYER = L.geoJSON(geo, {
    style: (f) => f.properties.has_data
      ? { color: "#3da5ff", weight: 2, opacity: 0.9, fill: false }
      : { color: "#3da5ff", weight: 1.5, opacity: 0.4, fill: false },
    onEachFeature: (f, layer) => {
      layer.bindTooltip(f.properties.name, { sticky: true });
      layer.on("click", () => { selectFromLayer(layer); });
    },
  }).addTo(MAP);

  L.geoJSON(dRes.perimeter, {
    interactive: false,
    style: { color: "#3da5ff", weight: 2, opacity: 1, fill: false },
  }).addTo(MAP);

  const bounds = GEOLAYER.getBounds();
  MAP.fitBounds(bounds);
  const fitZoom = MAP.getBoundsZoom(bounds);
  MAP.setMinZoom(fitZoom);
  MAP.setMaxBounds(bounds.pad(0.05));
  MAP.options.maxBoundsViscosity = 1.0;

  MAP.on("click", (e) => resolvePoint(e.latlng.lat, e.latlng.lng));
}

function resolvePoint(lat, lon) {
  let found = null;
  GEOLAYER.eachLayer((layer) => {
    if (!found && layer.feature && leafletContains(layer, lat, lon)) found = layer;
  });
  if (!found) { setBanner("That point is outside the Madrid districts.", false); return; }
  selectFromLayer(found);
}

function selectFromLayer(layer) {
  const f = layer.feature;
  const num = f.properties.district_num;
  const list = (OPTIONS.neighborhoods && OPTIONS.neighborhoods[String(num)]) || [];
  if (list.length === 0) {
    setBanner(`${f.properties.name}: no neighbourhoods in the dataset — this district can't be estimated.`, false);
    return;
  }
  setSelected(num, f.properties.name);
  if (SELECTED_LAYER) GEOLAYER.resetStyle(SELECTED_LAYER);
  layer.setStyle({ weight: 4, color: "#7ef7b0", fill: true, fillColor: "#7ef7b0", fillOpacity: 0.28 });
  SELECTED_LAYER = layer;
  MAP.flyToBounds(layer.getBounds(), { padding: [20, 20], maxZoom: 14, duration: 0.8 });
}

function leafletContains(layer, lat, lon) {
  const pt = L.latLng(lat, lon);
  if (!layer.getBounds().contains(pt)) return false;
  const g = layer.feature.geometry;
  const polys = g.type === "MultiPolygon" ? g.coordinates : [g.coordinates];
  return polys.some((poly) => pointInRing(lon, lat, poly[0]) &&
    !poly.slice(1).some((hole) => pointInRing(lon, lat, hole)));
}

function pointInRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
    if (((yi > y) !== (yj > y)) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function setSelected(num, name) {
  SELECTED = { district_num: num, district_name: name };
  $("district-display").value = num == null ? `${name} (no listings)` : name;
  setBanner(num == null
    ? `${name}: no listings in the dataset — treated as unknown district.`
    : `District set: ${name}`, true);

  const sel = $("subtitle");
  sel.innerHTML = '<option value="">— select a neighborhood —</option>';
  const list = (OPTIONS.neighborhoods && OPTIONS.neighborhoods[String(num)]) || [];
  for (const nb of list) {
    const o = document.createElement("option");
    o.value = nb.subtitle; o.textContent = nb.label;
    sel.appendChild(o);
  }
  const by = OPTIONS.builtyear_by_district[String(num)] || OPTIONS.builtyear_global;
  $("built_year").placeholder = `default ${by}`;
  $("appraisal-form").classList.remove("disabled");
  refreshSubmit();
}

function setBanner(text, active) {
  const b = $("district-banner");
  b.textContent = text;
  b.className = "banner " + (active ? "active" : "muted");
}

function fillSelect(id, values, keepFirst) {
  const el = $(id);
  if (!keepFirst) el.innerHTML = "";
  for (const v of values) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    el.appendChild(o);
  }
}

function addChip(group, key, label) {
  const d = document.createElement("div");
  d.className = "chip"; d.textContent = label; d.dataset.key = key;
  d.addEventListener("click", () => d.classList.toggle("on"));
  $(group).appendChild(d);
}

function buildForm() {
  fillSelect("house_type_id", OPTIONS.house_types, false);
  $("house_type_id").value = OPTIONS.default_house_type;
  fillSelect("floor", OPTIONS.floors, true);
  fillSelect("energy_certificate", OPTIONS.energy_certificates, true);
  fillSelect("street_type", OPTIONS.street_types, false);
  for (const a of OPTIONS.amenities) addChip("amenities", a.key, a.label);
  for (const f of OPTIONS.flags) addChip("flags", f.key, f.label);
  for (const o of OPTIONS.orientations) addChip("orientations", o.key, o.label);
  $("sq_mt_built").addEventListener("input", refreshSubmit);
  $("n_bathrooms").addEventListener("change", refreshSubmit);
  $("subtitle").addEventListener("change", refreshSubmit);
  $("n_rooms").addEventListener("change", refreshSubmit);
  $("appraisal-form").addEventListener("submit", onSubmit);
  $("result-close").addEventListener("click", closeResult);
  $("result-modal").addEventListener("click", (e) => {
    if (e.target.id === "result-modal") closeResult();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeResult();
  });
}

function chosenChips() {
  return [...document.querySelectorAll(".chip.on")].map((c) => c.dataset.key);
}

function renderResult(prefix, data) {
  $(prefix + "district").textContent = data.district_name
    ? `${data.district_name}${data.mape_pct != null ? ` · MAPE ±${data.mape_pct}%` : ""}`
    : "Unknown district";
  $(prefix + "point").textContent = EUR.format(data.point_estimate);
  $(prefix + "band").textContent = (data.lower != null)
    ? `Likely range: ${EUR.format(data.lower)} — ${EUR.format(data.upper)}`
    : "(interval unavailable)";
  $(prefix + "note").textContent =
    (data.note ? data.note + " " : "") +
    (data.mape_pct != null
      ? "Interval = estimate × (1 ± MAPE), the model's mean absolute % error (the metric used to select the final model)."
      : "");
}

function closeResult() {
  $("result-modal").classList.remove("show");
  if (HAS_RESULT) $("result-inline").classList.remove("hidden");
}

function refreshSubmit() {
  const ok = OPTIONS.model_loaded && SELECTED.district_name &&
    $("subtitle").value !== "" && $("n_rooms").value !== "" &&
    parseFloat($("sq_mt_built").value) > 0 && $("n_bathrooms").value !== "";
  $("submit-btn").disabled = !ok;
}

async function onSubmit(ev) {
  ev.preventDefault();
  $("error").classList.add("hidden");
  $("result-modal").classList.remove("show");
  $("result-inline").classList.add("hidden");
  $("submit-btn").disabled = true;
  $("submit-btn").textContent = "Estimating…";

  const body = {
    district_num: SELECTED.district_num,
    subtitle: $("subtitle").value || null,
    sq_mt_built: parseFloat($("sq_mt_built").value),
    n_bathrooms: parseFloat($("n_bathrooms").value),
    n_rooms: $("n_rooms").value ? parseInt($("n_rooms").value, 10) : null,
    built_year: $("built_year").value ? parseInt($("built_year").value, 10) : null,
    house_type_id: $("house_type_id").value || null,
    floor: $("floor").value || null,
    energy_certificate: $("energy_certificate").value || null,
    street_type: $("street_type").value || null,
    flags: chosenChips(),
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");

    renderResult("result-", data);
    renderResult("resinline-", data);
    HAS_RESULT = true;
    $("result-inline").classList.add("hidden");
    $("result-modal").classList.add("show");
  } catch (e) {
    $("error").textContent = e.message;
    $("error").classList.remove("hidden");
  } finally {
    $("submit-btn").textContent = "Estimate asking price";
    refreshSubmit();
  }
}

init();
