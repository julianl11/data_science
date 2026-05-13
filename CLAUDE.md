# CLAUDE.md — Project guide for future Claude sessions

Madrid housing-price regression project. Goal: predict `buy_price` from Idealista/Fotocasa listings using an end-to-end ML pipeline, evaluated against a neighbourhood-mean baseline. Group project; remote is `https://github.com/julianl11/data_science`.

## Working notebook

**`exploring_2.ipynb`** is the live working notebook (~71 cells, 43 code / 28 markdown). All work happens here.

`exploring.ipynb` is a static copy of the pre-restructure version, kept for diffing against `exploring_2.ipynb`. Don't edit it.

### Section structure

| # | Section | What's in it |
|---|---|---|
| §1 | General Information | Domain, two research questions, success criteria (R² ≥ 0.80, MAPE < 7.5 %, baseline = neighbourhood mean) |
| §2 | Dataset Description and Collection | Source (Idealista/Fotocasa via Kaggle), `read_csv`, `df.describe()`, `df.info()` |
| §3 | Data Analysis | The heart of the notebook |
| §3.1 | Pre-split preprocessing | Constants (`AMENITY_COLS`, `ORIENTATION_COLS`, `REFERENCE_DUMMIES`, `COLS_TO_DROP_FINAL`), leakage note, helper functions, pipeline run + inspection |
| §3.2 | EDA (post-cleaning) | `plot_target_distribution`, `plot_correlation_heatmap` — defs in cell 32, calls in cell 33 |
| §3.3 | Train/test split + post-split preprocessing | `build_pipeline` (`ColumnTransformer` w/ skewed-vs-non-skewed continuous, binary pass-through, `VarianceThreshold`), `np.log1p(y)`, 80/20 holdout fit on train only |
| §3.4 | ML algorithms | 10-fold CV of DT + RF (cell 40), then XGB + LightGBM (cell 41) |
| §3.5 | Evaluation and diagnostics | Feature importance + single-feature R² + correlation (cell 43); overfit diagnostics with Fig 1, Fig 2, Fig 3 (cell 45) |
| §3.6 | Statistical comparison — paired t-test | 10-fold paired `scipy.stats.ttest_rel` across DT/RF/XGB/LGBM (cell 49) + box-strip plot |
| §4 | Results | (§4.1 was removed; subsections renumbered down) |
| §4.1 | Neural network exploration | PyTorch residual MLP with `RobustScaler`, early-stopping val split |
| §4.2 | Model selection — nested CV | Ridge/ElasticNet/GBR/XGB/LGB, 5×3 folds, `RandomizedSearchCV` |
| §4.3 | Final model — test evaluation | XGBoost refit on full train with tuned `best_params`, evaluated once on held-out test |
| §4.4 | Learning curve | Train vs CV MAE vs training-set size |
| §4.5 | Model interpretation | PDP + SHAP |
| §5 | Conclusion | Scattered notes (over/underfit defs, pipeline diagram, baseline scores, pre/postprocessing taxonomy) |

A `sys.exit()` at cell 56 is a kill-switch separating exploration (§4.1, §4.2) from the final-model section. Comment it out to run §4.3+.

## Preprocessing pipeline

Defined inline in §3.1 (cells 12–24). Actual order in `run_pipeline` (cell 24):

1. **`filter_built_year`** — clip rows by `built_year` (default ≥ 1850).
2. **`fill_amenities_by_year`** — group-impute `AMENITY_COLS` by build-year decade cohort (era-dependent prevalence).
3. **`curate_df`** — drop columns whose effective missing ratio > threshold. **Binary-aware**: 0s count as missing only for non-binary columns. Binary columns (`dtype bool` or values ⊆ {0,1,True,False}) are protected so `is_orientation_*` and `has_*` flags survive.
4. **`fill_orientation`** — mode-fill `is_orientation_*` flags.
5. **`prepare`** — drop `COLS_TO_DROP_FINAL`, parse `neighborhood_id` strings into integer `neighborhood_id` + `district_id`, one-hot encode `district_id` → `district_id_*`, OHE other categoricals, drop reference-category dummies (`REFERENCE_DUMMIES`), split → `(X, y)`. The €/m² value embedded in the raw string is **not** extracted by this path.
6. **`drop_correlated`** — drop one of every pair with `|ρ| > threshold` (kept the higher-target-correlation one).
7. **`clean_feature_names`** — sanitise column names for LightGBM (no spaces, parens, accents).

Call site: `X, y = run_pipeline(df, missing_threshold=0.75, corr_threshold=0.9, min_built_year=1850, devmode=True)`.

**Dead code in cell 23**: `encode_neighborhood` defines a richer neighborhood parser (extracts `neighborhood_price_m2`, `neighborhood_rank`, `district_num`, `district_<name>_*` OHE) but is **never called** by `run_pipeline`. Cell 22's `prepare` does its own minimal inline parsing instead. If you want the €/m² feature, wire `encode_neighborhood` in or move its logic into `prepare`.

## Known issues and conventions

- **`parking_price` is leakage-prone**: removed automatically by `curate_df` with `missing_threshold=0.75` (high zero/NaN ratio). Flagged by the single-feature R² check in §3.5.
- **`sq_mt_built` is the dominant (non-leakage) predictor**: RF importance ≈ 0.77 and single-feature train R² ≈ 0.78 in §3.5 cell 43. That is just price ∝ size and is expected, not leakage. The model's ~0.94 CV R² ceiling is mostly carried by this one feature.
- **No `neighborhood_price_m2` in `X_train`**: the €/m² value is embedded in the raw `neighborhood_id` strings by Idealista/Fotocasa, but the live pipeline (`prepare` in cell 22) only extracts integer IDs and drops the rest. The `encode_neighborhood` parser that would extract €/m² lives in cell 23 but is dead code — see note above.
- **Validation curves all show optima at boundaries**: model is at its performance ceiling on this data; tuning won't unlock much more. See §3.5 cell 45.
- **`max_features` panel in Fig 3 shows train R² ≈ 0.99 across the entire range**: not a bug. With `max_depth=None` and `min_samples_leaf=1`, trees grow until each leaf has one training row → memorisation, regardless of `max_features`. `max_features` only controls which columns each split considers, not tree growth.
- **Python 3.14.4 + sklearn warning silencer**: cell 45 has a `warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")` to silence the dozens of `delayed`/`Parallel` UserWarnings emitted when `n_jobs != 1`. Benign — sklearn-internal pattern that hasn't caught up to Python 3.14.
- **`IPython.display.Image` instead of `plt.show()`** in cell 45: matplotlib's inline backend can silently fail on Python 3.14. Each `fig.savefig(...)` writes a PNG to disk and `display(Image(...))` reads it back — backend-agnostic.

## Data and outputs

- **Input:** `./data/houses_Madrid.csv` (21,742 listings × 58 raw columns). Path is relative, so the notebook must run from the project root.
- **Generated figures** (re-created on each cell run):
  - `fig1_overfit_folds.png` — per-fold train/CV R² + gap bars + box plots
  - `fig2_learning_curves.png` — score vs training-set size
  - `fig3_validation_curves.png` — 2×2 grid: DT max_depth, RF min_samples_leaf (log-scale), RF max_features, RF max_depth
  - `fig_model_comparison_paired_ttest.png` — §3.6 box+strip of per-fold MAE
  - `final_learning_curve.png` — §4.4 learning curve for the tuned XGBoost
  - `nn_diagnostics.png` — §4.1 PyTorch NN loss/scatter/residuals
  - `fig1_pdp.png`, `fig2a_shap_beeswarm.png`, `fig2b_shap_importance.png`, `fig2c_shap_dependence.png`, `fig2d_shap_waterfall.png` — §4.5 PDP + SHAP

## Environment

- **Python:** 3.14.4 via pyenv on WSL: Ubuntu (the path inside notebook is `./data/houses_Madrid.csv` relative; works the same on Windows native).
- **Key deps:** pandas, numpy, scikit-learn, matplotlib, scipy, xgboost, lightgbm, torch, shap, regex.
- **Python 3.14 caveat:** newer than sklearn's tested matrix — expect benign `UserWarning`s about `joblib.delayed`. Silencer is already in place for §3.5.

## Editing conventions (set during this project)

- **Do not modify or reorder existing cells** unless the user explicitly asks. Insert new markdown cells at section boundaries.
- **All edits via Python scripts**, not Jupyter UI, so the file is reproducibly modifiable from the terminal. Helper scripts live in the project root with names like `_add_paired_ttest.py`, `_expand_fig3.py`, `_silence_and_display.py`, `_fix_curate_df.py`, `_restructure_section3_4.py`. They are single-use and deletable.
- **Snapshot before bulk edits**: `cp exploring_2.ipynb exploring_2.ipynb.prev` so the user can diff in VS Code (right-click both → "Compare with").
- **Two persistent backups in the repo:**
  - `exploring_2.ipynb.bak` — pre-restructure original (46 cells)
  - `exploring_2.ipynb.prev` — state before the most recent edit

## Repo layout

```
Project_housing_prices_madrid/
├── exploring_2.ipynb           # working notebook (71 cells)
├── exploring_2.ipynb.bak       # pre-restructure original (46 cells)
├── exploring_2.ipynb.prev      # snapshot before the most recent edit
├── exploring.ipynb             # pre-restructure copy for diffing
├── data/houses_Madrid.csv      # input dataset
├── fig*.png, nn_diagnostics.png, …  # generated figures
├── README.md                   # team-facing readme (project background)
├── .claude/settings.local.json # Claude Code per-project settings
├── _*.py                       # one-shot edit scripts (deletable)
└── CLAUDE.md                   # this file
```
