library(dplyr)
library(readr)
library(ggplot2)
library(ggpubr)

set.seed(42)

OUTPUT_DIR <- "benchmarks/results/climatology"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Same train/test files as LightGBM, full test set, so every model is scored on the same rows
train_data <- read_csv("data/processed/train_data_base.csv", show_col_types = FALSE)
test_data_full <- read_csv("data/processed/test_data_base.csv", show_col_types = FALSE)

test_data <- test_data_full

inverse_log_transform <- function(x) exp(x) - 1
duan_smearing_transform <- function(pred_log, smearing_factor) exp(pred_log) * smearing_factor - 1

rmse_fun <- function(pred, actual) sqrt(mean((pred - actual)^2, na.rm = TRUE))
mae_fun <- function(pred, actual) mean(abs(pred - actual), na.rm = TRUE)
r2_fun <- function(pred, actual) {
  ss_res <- sum((actual - pred)^2, na.rm = TRUE)
  ss_tot <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  1 - (ss_res / ss_tot)
}

climatology <- train_data %>%
  group_by(lat, lon, doy) %>%
  summarise(
    climatology_pred_log = mean(Acer, na.rm = TRUE),
    climatology_sd = sd(Acer, na.rm = TRUE),
    n_years = n(),
    .groups = "drop"
  )

train_with_clim <- train_data %>%
  left_join(climatology %>% select(lat, lon, doy, climatology_pred_log), by = c("lat", "lon", "doy")) %>%
  filter(!is.na(climatology_pred_log))

smearing_factor <- mean(exp(train_with_clim$Acer - train_with_clim$climatology_pred_log), na.rm = TRUE)

test_with_pred <- test_data %>%
  left_join(climatology %>% select(lat, lon, doy, climatology_pred_log), by = c("lat", "lon", "doy")) %>%
  filter(!is.na(climatology_pred_log)) %>%
  mutate(
    predicted_log = climatology_pred_log,
    predicted = duan_smearing_transform(predicted_log, smearing_factor),
    actual = inverse_log_transform(Acer)
  )

overall_rmse <- rmse_fun(test_with_pred$predicted, test_with_pred$actual)
overall_mae <- mae_fun(test_with_pred$predicted, test_with_pred$actual)
overall_r2 <- r2_fun(test_with_pred$predicted, test_with_pred$actual)

overall_metrics <- tibble(
  Model = "Climatology",
  RMSE = overall_rmse,
  MAE = overall_mae,
  R2 = overall_r2,
  n_predictions = nrow(test_with_pred)
)

metrics_by_location <- test_with_pred %>%
  group_by(lat, lon) %>%
  summarise(
    RMSE = rmse_fun(predicted, actual),
    MAE = mae_fun(predicted, actual),
    R2 = r2_fun(predicted, actual),
    n_obs = n(),
    .groups = "drop"
  ) %>%
  arrange(RMSE)

metrics_by_year <- test_with_pred %>%
  group_by(year) %>%
  summarise(
    RMSE = rmse_fun(predicted, actual),
    MAE = mae_fun(predicted, actual),
    R2 = r2_fun(predicted, actual),
    n_obs = n(),
    .groups = "drop"
  )

test_predictions <- test_with_pred %>%
  select(lat, lon, year, doy, actual, predicted)

write_csv(climatology %>% rename(climatology_pred = climatology_pred_log), file.path(OUTPUT_DIR, "climatology_model.csv"))
write_csv(test_predictions, file.path(OUTPUT_DIR, "test_predictions.csv"))
write_csv(overall_metrics, file.path(OUTPUT_DIR, "overall_metrics.csv"))
write_csv(metrics_by_location, file.path(OUTPUT_DIR, "metrics_by_location.csv"))
write_csv(metrics_by_year, file.path(OUTPUT_DIR, "metrics_by_year.csv"))
