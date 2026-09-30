# Spatiotemporal Acer Pollen Forecasting

Code for the paper's 7-day-ahead forecasts of maple (*Acer*) pollen at 15 US
stations. It compares two LightGBM models, veg+meteo (weather + MODIS
NDVI/EVI) and meteo (weather only), against two benchmarks: climatology and
7-day persistence.

## Pipeline

Run the steps in this order:

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `1_data_organization.R` | Smooths daily pollen, downloads Daymet weather, loads MODIS NDVI/EVI and builds lagged features with a 7-day forecast buffer | `data/processed/pollen_weather_smoothed.csv` |
| 1b | `1b_static_features.R` | Adds station elevation | updates `pollen_weather_smoothed.csv`, `station_static_features.csv` |
| 2 | `2_feature_selection.R` | Log-transforms the target and splits by year | `data/processed/train_data_base.csv` (2003–2017), `test_data_base.csv` (2018–2022) |
| 3 | `benchmarks/benchmark_climatology_model.R` | Day-of-year climatology benchmark | `benchmarks/results/climatology/` |
| 3 | `benchmarks/benchmark_persistence_model.R` | Persistence benchmark (7, 14 and 30 days) | `benchmarks/results/persistence/` |
| 4 | `base_models/lightgbm/tune_lightgbm.R` | Bayesian optimization of the pooled LightGBM hyperparameters for both models | `base_models/lightgbm/best_params.csv`, `tuning_results_*.csv` |
| 5 | `base_models/lightgbm/train_lightgbm.R` | Trains the veg+meteo model | `base_models/lightgbm/results/` |
| 5 | `base_models/lightgbm/train_lightgbm_no_veg.R` | Trains the meteo model | `base_models/lightgbm/results_no_veg/` |
| 6 | `base_models/lightgbm/paper_visualizations.R` | Builds all paper figures | `base_models/lightgbm/results/paper_figures.pdf` |

```sh
cd all_code
Rscript 1_data_organization.R
Rscript 1b_static_features.R
Rscript 2_feature_selection.R
Rscript benchmarks/benchmark_climatology_model.R
Rscript benchmarks/benchmark_persistence_model.R
Rscript base_models/lightgbm/tune_lightgbm.R
Rscript base_models/lightgbm/train_lightgbm.R
Rscript base_models/lightgbm/train_lightgbm_no_veg.R
Rscript base_models/lightgbm/paper_visualizations.R
```

## Setup

### Input data

Scripts read raw inputs from `data/raw/`. That folder isn't stored here; in the
original project it is a symlink to the project's `data/raw`

- `2023_data.csv`: daily pollen counts (uses the `Acer`, `Date` and `Station.ID` columns)
- `station_locations.csv`: station `id`, `lat` and `lon`
- `Pollen-Stations-v2-MOD13A1-061-results.csv`: MODIS Terra NDVI/EVI (AppEEARS export)
- `asdf-MYD13A1-061-results.csv`: MODIS Aqua NDVI/EVI (AppEEARS export)

### Network access

- **Daymet** (step 1) through `daymetr::download_daymet`
- **USGS Elevation Point Query Service** (step 1b) through `elevatr::get_elev_point(src = "epqs")`

### R packages

```r
install.packages(c(
  "dplyr", "tidyr", "tibble", "readr", "stringr", "lubridate", "janitor", "tidyverse",
  "daymetr", "ptw", "imputeTS", "elevatr", "sf",
  "lightgbm", "ggplot2", "ggpubr", "gridExtra", "scales", "usmap",
  "rBayesianOptimization"   # tune_lightgbm.R
))
```

## Method

### Forecast setup

- **Target:** daily *Acer* pollen smoothed with a Whittaker smoother
  (`ptw::whit2`, λ = 50) within each station-year, modelled as `log(1 + Acer)`.
  Predictions are converted back with a Duan smearing correction.
- **Horizon:** every predictor window ends 7 days before the target day
  (`FORECAST_BUFFER <- 7`).
- **Split:** train 2003–2014, validation 2015–2017, test 2018–2022.

### Features

Both models use the same 25 features. The veg+meteo model adds 6 more (31 in
total).

| Group | Features |
|---|---|
| Pollen | `acer_lag_1week`, `acer_lag_1month`, `acer_lag_3month` (window means ending at t−7); `acer_lag_7d` (value at t−7, the persistence input); `acer_slope_7d` (t−7 minus t−14) |
| Weather (Daymet) | `tmin`, `tmax`, `prcp`, `srad`, `vp` and `swe`, each as 1-week, 1-month and 3-month means ending at t−7 |
| Static | `photoperiod`, `elevation_m` |
| Vegetation (veg+meteo only) | `ndvi` and `evi`, each as 1-week, 1-month and 3-month means ending at t−7 (MODIS 16-day composites, gap-filled to daily) |

### Models

`tune_lightgbm.R` runs Bayesian optimization (Gaussian-process
surrogate, `rBayesianOptimization`, expected improvement). It uses the same
search space, budget (10 random starting points + 30 guided steps) and seed for
both models. Each candidate is scored by validation RMSE on 2015–2017 (log
scale):

| Parameter | Search range | veg+meteo | meteo |
|---|---|---|---|
| `num_leaves` | 8–255 | 85 | 25 |
| `min_data_in_leaf` | 5–200 | 13 | 156 |
| `feature_fraction` | 0.3–1.0 | 0.66 | 0.55 |
| `bagging_fraction` | 0.5–1.0 | 0.84 | 0.97 |
| `lambda_l1` | 10⁻³–10 (log) | 0.001 | 7.3 |
| `lambda_l2` | 10⁻³–10 (log) | 0.086 | 0.036 |

| Parameter | LightGBM veg+meteo | LightGBM meteo |
|---|---|---|
| `num_leaves` | 53 | 229 |
| `min_data_in_leaf` | 7 | 128 |
| `feature_fraction` | 0.51 | 0.39 |
| `bagging_fraction` | 0.96 | 0.98 |
| `bagging_freq` | 5 | 5 |
| `lambda_l1` | 0.007 | 10.0 |
| `lambda_l2` | 7.8 | 0.059 |
| `learning_rate` | 0.05 | 0.05 |
| `max_depth` | −1 | −1 |
| Rounds | early stopping on 2015–2017 (patience 100, up to 3,000) | early stopping on 2015–2017 (patience 100, up to 3,000) |
| Trees per seed | 288–1,090 | 268–715 |
| Seeds averaged | 5 | 5 |
| Objective | regression (MSE) on log(1 + pollen) | regression (MSE) on log(1 + pollen) |
| Predictors | 34 (26 shared + 6 NDVI/EVI + lat/lon) | 28 (26 shared + lat/lon) |
| Validation RMSE (log, 2015–2017) | 0.441 | 0.453 |


Fixed for both models: learning rate 0.05, bagging every 5 iterations.

**Training.** Each script reads its model's `location == "pooled"` row from
`best_params.csv`. It trains with early stopping on 2015–2017 (patience 100, up
to 3,000 rounds) and averages predictions over 5 seeds. Without
`best_params.csv` it falls back to 63 leaves, `min_data_in_leaf` 10, feature
fraction 0.6 and bagging 0.8.

With `POOLED <- FALSE` the scripts instead fit one model per station, using the
hard-coded defaults in each script.

### Benchmarks

- Climatology: the mean log pollen for each station and day of year over
  2003–2017.
- Persistence: the smoothed value observed exactly 7 days earlier (14- and
  30-day versions are also saved).

Both are scored on the full test set.

### Evaluation

`paper_visualizations.R` scores all four models on the same test rows: rows
that every model predicted, at stations with non-zero pollen. It prints the row
count when it runs. Figures 1a and 2a report RMSE, MAE and R² on the
`log1p` scale. The parity plots, time series and maps use grains/m³.

## Results

Test years 2018–2022, 7,358 shared rows. Raw scale in grains/m³, log scale is
`log1p`:

| Model | RMSE | MAE | R² | log RMSE | log MAE | log R² |
|---|---|---|---|---|---|---|
| **LightGBM veg+meteo** | **14.01** | **3.31** | **0.859** | **0.398** | **0.194** | **0.911** |
| LightGBM meteo | 17.36 | 3.96 | 0.784 | 0.416 | 0.221 | 0.902 |
| Climatology | 36.15 | 7.37 | 0.063 | 0.708 | 0.498 | 0.717 |
| Persistence (7-day) | 23.86 | 6.07 | 0.592 | 0.595 | 0.399 | 0.800 |

On pollen-season days (above 1 grain/m³), log R² is 0.759 for veg+meteo and
0.744 for meteo.

Veg+meteo has the lower error at 9 of 14 active stations and in all 5 test
years. It also has the lower log RMSE in 99% of 1,000 station-bootstrap
resamples. Part of the gap comes from tuning: the meteo model's tuned settings
did slightly better on validation but worse on the test years than its untuned
defaults (test log RMSE 0.416 vs 0.403).
  `.model` files from earlier runs next to the current `pooled_seed*.model`
  files. `model_metadata.rds` records which setup is current.
