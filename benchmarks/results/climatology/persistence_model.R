suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(janitor)
})

set.seed(42)

csv_path <- "../../data/processed/pollen_weather_smoothed.csv"

df <- read.csv(csv_path) %>%
  clean_names() %>%
  select(-acer_orig)

if ("acer" %in% names(df)) df <- df %>% dplyr::rename(Acer = acer)

if ("Date" %in% names(df)) names(df)[names(df) == "Date"] <- "date"

if (!"year" %in% names(df)) {
  df <- df %>% mutate(year = year(as.Date(date)))
}

df <- df %>%
  mutate(
    date = as.Date(date),
    doy  = yday(date)
  )

target_years <- 2003:2022

df_filtered <- df %>%
  dplyr::filter(year %in% target_years)

df_filtered_lags <- df_filtered %>%
  group_by(lat, lon) %>%
  arrange(lat, lon, year, doy) %>%
  mutate(Acer_lag = sapply(1:n(), function(i) {
    target_doy <- doy[i]
    if(target_doy > 30) {
      lag_data <- doy >= (target_doy - 30) & doy < target_doy
    } else {
      lag_data <- (doy >= (target_doy - 30 + 365) | doy < target_doy)
    }
    if(sum(lag_data) > 0 && sum(!is.na(Acer[lag_data])) > 0) {
      mean(Acer[lag_data], na.rm = TRUE)
    } else {
      NA_real_
    }
  })) %>%
  ungroup() %>%
  dplyr::filter(!is.na(Acer_lag))

df_filtered_lags_selected <- df_filtered_lags[, !grepl("lag0", names(df_filtered_lags))]

num_df <- df_filtered_lags_selected %>% select(where(is.numeric))
cor_mat <- cor(num_df, use = "complete.obs")

acer_cors <- abs(cor_mat["Acer", ])
acer_cors <- acer_cors[names(acer_cors) != "Acer"]

best_predictor <- names(which.max(acer_cors))

keep_cols <- c("date", "lat", "lon", "year", "doy", "Acer", best_predictor)
remaining_cols <- setdiff(colnames(cor_mat), keep_cols)

for(col in remaining_cols) {
  is_redundant <- FALSE
  for(kept in keep_cols) {
    if(kept %in% colnames(cor_mat) && kept != "Acer" && abs(cor_mat[col, kept]) > 0.8) {
      is_redundant <- TRUE
      break
    }
  }
  if(!is_redundant) keep_cols <- c(keep_cols, col)
}

df_filtered_lags_selected <- df_filtered_lags_selected[, keep_cols]

years <- 2003:2022
n_years <- length(years)
n_train_years <- floor(n_years * 0.75)
train_years <- years[1:n_train_years]
test_years <- years[(n_train_years + 1):n_years]

train_data <- df_filtered_lags_selected %>% dplyr::filter(year %in% train_years)
test_data <- df_filtered_lags_selected %>% dplyr::filter(year %in% test_years)

glimpse(df_filtered_lags_selected)

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

rmse_fun <- function(pred, actual) sqrt(mean((pred - actual)^2, na.rm = TRUE))
mae_fun <- function(pred, actual) mean(abs(pred - actual), na.rm = TRUE)
r2_fun <- function(pred, actual) {
  ss_res <- sum((actual - pred)^2, na.rm = TRUE)
  ss_tot <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  1 - (ss_res / ss_tot)
}

test_with_persistence <- test_data %>%
  dplyr::group_by(lat, lon) %>%
  dplyr::arrange(lat, lon, date) %>%
  dplyr::mutate(
    date_lag7 = date - 7,
    persistence_pred = NA_real_
  ) %>%
  dplyr::ungroup()

all_data <- bind_rows(train_data, test_data) %>%
  dplyr::select(lat, lon, date, Acer)

test_with_persistence <- test_with_persistence %>%
  dplyr::left_join(
    all_data %>% dplyr::rename(Acer_7day_ago = Acer),
    by = c("lat" = "lat", "lon" = "lon", "date_lag7" = "date")
  ) %>%
  dplyr::mutate(persistence_pred = Acer_7day_ago) %>%
  dplyr::filter(!is.na(persistence_pred))

overall_rmse <- rmse_fun(test_with_persistence$persistence_pred, test_with_persistence$Acer)
overall_mae <- mae_fun(test_with_persistence$persistence_pred, test_with_persistence$Acer)
overall_r2 <- r2_fun(test_with_persistence$persistence_pred, test_with_persistence$Acer)

overall_metrics <- tibble(
  Model = "Persistence_7day",
  RMSE = overall_rmse,
  MAE = overall_mae,
  R2 = overall_r2,
  n_predictions = nrow(test_with_persistence)
)

metrics_by_location <- test_with_persistence %>%
  dplyr::group_by(lat, lon) %>%
  dplyr::summarise(
    RMSE = rmse_fun(persistence_pred, Acer),
    MAE = mae_fun(persistence_pred, Acer),
    R2 = r2_fun(persistence_pred, Acer),
    n_obs = dplyr::n(),
    .groups = "drop"
  ) %>%
  dplyr::arrange(RMSE)

metrics_by_year <- test_with_persistence %>%
  dplyr::group_by(year) %>%
  dplyr::summarise(
    RMSE = rmse_fun(persistence_pred, Acer),
    MAE = mae_fun(persistence_pred, Acer),
    R2 = r2_fun(persistence_pred, Acer),
    n_obs = dplyr::n(),
    .groups = "drop"
  )

print(metrics_by_year)

clim_metrics <- readr::read_csv("../climatology/overall_metrics.csv",
                                show_col_types = FALSE)

comparison <- bind_rows(
  clim_metrics,
  overall_metrics
) %>%
  dplyr::mutate(
    RMSE_diff = RMSE - first(RMSE),
    MAE_diff = MAE - first(MAE),
    R2_diff = R2 - first(R2)
  )

print(comparison)

dir.create("../persistence", showWarnings = FALSE)

readr::write_csv(test_with_persistence, "../persistence/test_predictions.csv")
readr::write_csv(overall_metrics, "../persistence/overall_metrics.csv")
readr::write_csv(metrics_by_location, "../persistence/metrics_by_location.csv")
readr::write_csv(metrics_by_year, "../persistence/metrics_by_year.csv")
readr::write_csv(comparison, "../persistence/model_comparison.csv")

p1 <- ggplot(test_with_persistence, aes(x = Acer, y = persistence_pred)) +
  geom_point(alpha = 0.3, color = "#E63946") +
  geom_abline(slope = 1, intercept = 0, color = "#2E86AB", linetype = "dashed", linewidth = 1) +
  labs(title = "7-Day Persistence Model: Actual vs Predicted",
       subtitle = sprintf("RMSE: %.4f | MAE: %.4f | R²: %.4f",
                          overall_rmse, overall_mae, overall_r2),
       x = "Actual Acer", y = "Predicted Acer (7-day lag)") +
  theme_minimal(base_size = 12)

ggsave("../persistence/actual_vs_predicted.png", p1, width = 8, height = 6, dpi = 300)

clim_by_year <- readr::read_csv("../climatology/metrics_by_year.csv",
                                show_col_types = FALSE)

year_comparison <- bind_rows(
  clim_by_year %>% dplyr::mutate(Model = "Climatology"),
  metrics_by_year %>% dplyr::mutate(Model = "Persistence_7day")
)

p2 <- ggplot(year_comparison, aes(x = year, y = RMSE, color = Model, group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Climatology" = "#2E86AB", "Persistence_7day" = "#E63946")) +
  labs(title = "Model Comparison: RMSE by Test Year",
       x = "Year", y = "RMSE") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("../persistence/rmse_comparison_by_year.png", p2, width = 8, height = 5, dpi = 300)

sample_locs <- test_with_persistence %>%
  dplyr::distinct(lat, lon) %>%
  dplyr::slice_head(n = 3)

sample_data <- test_with_persistence %>%
  dplyr::semi_join(sample_locs, by = c("lat", "lon")) %>%
  dplyr::filter(year == min(year))

if (nrow(sample_data) > 0) {
  p3 <- ggplot(sample_data, aes(x = doy)) +
    geom_line(aes(y = Acer, color = "Actual"), linewidth = 0.9) +
    geom_line(aes(y = persistence_pred, color = "Persistence"), linewidth = 0.9) +
    scale_color_manual(values = c("Actual" = "#2E86AB", "Persistence" = "#E63946")) +
    facet_wrap(~ paste0("Lat: ", round(lat, 2), ", Lon: ", round(lon, 2)), ncol = 1) +
    labs(title = sprintf("Sample Predictions - Year %d (7-day Persistence)", min(sample_data$year)),
         x = "Day of Year", y = "Acer Pollen Count", color = "") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom")

  ggsave("../persistence/sample_predictions.png", p3, width = 8, height = 9, dpi = 300)
}

p4 <- ggplot(comparison, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.4f", RMSE)), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("Climatology" = "#2E86AB", "Persistence_7day" = "#E63946")) +
  labs(title = "Baseline Model Comparison: RMSE",
       x = "", y = "RMSE") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave("../persistence/model_comparison_bar.png", p4, width = 7, height = 5, dpi = 300)

residuals_data <- bind_rows(
  test_with_persistence %>%
    dplyr::mutate(residual = Acer - persistence_pred, Model = "Persistence_7day"),
  readr::read_csv("../climatology/test_predictions.csv", show_col_types = FALSE) %>%
    dplyr::mutate(residual = Acer - climatology_pred, Model = "Climatology")
)

p5 <- ggplot(residuals_data, aes(x = residual, fill = Model)) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed", linewidth = 1) +
  scale_fill_manual(values = c("Climatology" = "#2E86AB", "Persistence_7day" = "#E63946")) +
  labs(title = "Residuals Distribution Comparison",
       x = "Residual (Actual - Predicted)", y = "Count") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("../persistence/residuals_comparison.png", p5, width = 9, height = 5, dpi = 300)
