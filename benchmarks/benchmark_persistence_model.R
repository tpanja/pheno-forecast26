suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(ggplot2)
  library(ggpubr)
})

set.seed(42)

OUTPUT_DIR <- "benchmarks/results/persistence"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Same train/test files as LightGBM, full test set, so every model is scored on the same rows
train_data <- read_csv("data/processed/train_data_base.csv", show_col_types = FALSE)
test_data_full <- read_csv("data/processed/test_data_base.csv", show_col_types = FALSE)

train_data <- train_data %>% mutate(date = as.Date(date))
test_data_full <- test_data_full %>% mutate(date = as.Date(date))

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

# One value per location-date (duplicate records averaged) so the lag join can't multiply rows
all_data <- bind_rows(train_data, test_data_full) %>%
  group_by(lat, lon, date) %>%
  summarise(Acer = mean(Acer), .groups = "drop")

time_offsets <- c(30, 14, 7)
all_results <- list()
all_metrics_loc <- list()
all_metrics_year <- list()

for (offset_days in time_offsets) {

  train_with_lag <- train_data %>%
    mutate(prev_date = date - days(offset_days)) %>%
    left_join(
      all_data %>% rename(prev_date = date, persistence_pred_log = Acer),
      by = c("lat", "lon", "prev_date")
    ) %>%
    filter(!is.na(persistence_pred_log))

  smearing_factor <- mean(exp(train_with_lag$Acer - train_with_lag$persistence_pred_log), na.rm = TRUE)

  test_with_persistence <- test_data %>%
    mutate(prev_date = date - days(offset_days)) %>%
    left_join(
      all_data %>% rename(prev_date = date, persistence_pred_log = Acer),
      by = c("lat", "lon", "prev_date")
    ) %>%
    filter(!is.na(persistence_pred_log)) %>%
    mutate(
      predicted = duan_smearing_transform(persistence_pred_log, smearing_factor),
      actual = inverse_log_transform(Acer)
    )

  overall_rmse <- rmse_fun(test_with_persistence$predicted, test_with_persistence$actual)
  overall_mae <- mae_fun(test_with_persistence$predicted, test_with_persistence$actual)
  overall_r2 <- r2_fun(test_with_persistence$predicted, test_with_persistence$actual)

  model_name <- sprintf("Persistence_%dday", offset_days)

  metrics_loc <- test_with_persistence %>%
    group_by(lat, lon) %>%
    summarise(
      RMSE = rmse_fun(predicted, actual),
      MAE = mae_fun(predicted, actual),
      R2 = r2_fun(predicted, actual),
      n_obs = n(),
      .groups = "drop"
    ) %>%
    mutate(Model = model_name, Offset_Days = offset_days)

  metrics_year <- test_with_persistence %>%
    group_by(year) %>%
    summarise(
      RMSE = rmse_fun(predicted, actual),
      MAE = mae_fun(predicted, actual),
      R2 = r2_fun(predicted, actual),
      n_obs = n(),
      .groups = "drop"
    ) %>%
    mutate(Model = model_name, Offset_Days = offset_days)

  preds_formatted <- test_with_persistence %>%
    select(lat, lon, year, doy, actual, predicted) %>%
    mutate(Model = model_name, Offset_Days = offset_days)

  all_results[[model_name]] <- preds_formatted
  all_metrics_loc[[model_name]] <- metrics_loc
  all_metrics_year[[model_name]] <- metrics_year
}

combined_predictions <- bind_rows(all_results)
combined_metrics_loc <- bind_rows(all_metrics_loc)
combined_metrics_year <- bind_rows(all_metrics_year)

write_csv(combined_predictions, file.path(OUTPUT_DIR, "test_predictions.csv"))
write_csv(combined_metrics_loc, file.path(OUTPUT_DIR, "metrics_by_location.csv"))
write_csv(combined_metrics_year, file.path(OUTPUT_DIR, "metrics_by_year.csv"))

overall_summary <- combined_predictions %>%
  group_by(Model, Offset_Days) %>%
  summarise(
    RMSE = rmse_fun(predicted, actual),
    MAE = mae_fun(predicted, actual),
    R2 = r2_fun(predicted, actual),
    n_predictions = n(),
    .groups = "drop"
  )

write_csv(overall_summary, file.path(OUTPUT_DIR, "overall_metrics.csv"))

print(overall_summary)
