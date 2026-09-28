set.seed(42)
suppressPackageStartupMessages({
  library(ggplot2)
  library(ggpubr)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lightgbm)
  library(sf)
  library(usmap)
  library(scales)
})

r2_trans <- trans_new(
  name      = "r2_log",
  transform = function(x) ifelse(x >= 0, x, -log1p(-x)),
  inverse   = function(x) ifelse(x >= 0, x, 1 - exp(-x)),
  breaks    = function(x) {
    raw <- c(-100, -25, -10, -4, -1, 0, 0.25, 0.5, 0.75, 1)
    raw[raw >= min(x, na.rm = TRUE) & raw <= max(x, na.rm = TRUE)]
  }
)

OUTPUT_DIR <- "base_models/lightgbm/results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

lgb_metrics_loc <- read_csv("base_models/lightgbm/results/metrics_by_location.csv", show_col_types = FALSE) %>%
  mutate(Model = "LightGBM_veg+meteo")
lgb_metrics_year <- read_csv("base_models/lightgbm/results/metrics_by_year.csv", show_col_types = FALSE) %>%
  mutate(Model = "LightGBM_veg+meteo")
lgb_preds <- read_csv("base_models/lightgbm/results/test_predictions.csv", show_col_types = FALSE) %>%
  mutate(Model = "LightGBM_veg+meteo")

lgb_noveg_metrics_loc  <- NULL
lgb_noveg_metrics_year <- NULL
lgb_noveg_preds <- read_csv("base_models/lightgbm/results_no_veg/test_predictions.csv", show_col_types = FALSE) %>%
  mutate(Model = "LightGBM_meteo")

clim_preds <- read_csv("benchmarks/results/climatology/test_predictions.csv", show_col_types = FALSE) %>%
  mutate(Model = "Climatology")

pers_preds_raw <- read_csv("benchmarks/results/persistence/test_predictions.csv", show_col_types = FALSE)
if ("Offset_Days" %in% names(pers_preds_raw)) {
  pers_preds <- pers_preds_raw %>%
    filter(Offset_Days == 7) %>%
    dplyr::select(-any_of(c("Model", "Offset_Days"))) %>%
    mutate(Model = "Persistence (7-day)")
} else {
  pers_preds <- pers_preds_raw %>%
    mutate(Model = "Persistence (7-day)")
}

active_locs <- lgb_preds %>%
  group_by(lat, lon) %>%
  summarise(nonzero = sum(actual > 0, na.rm = TRUE), .groups = "drop") %>%
  filter(nonzero > 0)

# Score every model on the same rows: keep only test rows (at active locations)
# that all four models produced a prediction for
key_cols <- c("lat", "lon", "year", "doy")
common_keys <- list(lgb_preds, lgb_noveg_preds, clim_preds, pers_preds) %>%
  lapply(function(df) distinct(df, across(all_of(key_cols)))) %>%
  Reduce(function(a, b) inner_join(a, b, by = key_cols), .) %>%
  semi_join(active_locs, by = c("lat", "lon"))

lgb_preds       <- lgb_preds       %>% semi_join(common_keys, by = key_cols)
lgb_noveg_preds <- lgb_noveg_preds %>% semi_join(common_keys, by = key_cols)
clim_preds      <- clim_preds      %>% semi_join(common_keys, by = key_cols)
pers_preds      <- pers_preds      %>% semi_join(common_keys, by = key_cols)

cat(sprintf("Scoring all models on %d common test rows (LightGBM %d, meteo %d, climatology %d, persistence %d)\n",
            nrow(common_keys), nrow(lgb_preds), nrow(lgb_noveg_preds), nrow(clim_preds), nrow(pers_preds)))

rmse_fun <- function(pred, actual) sqrt(mean((pred - actual)^2, na.rm = TRUE))
mae_fun  <- function(pred, actual) mean(abs(pred - actual), na.rm = TRUE)
r2_fun   <- function(pred, actual) {
  ss_res <- sum((actual - pred)^2, na.rm = TRUE)
  ss_tot <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  1 - (ss_res / ss_tot)
}
rmse_log <- function(pred, actual) rmse_fun(log1p(pred), log1p(actual))
mae_log  <- function(pred, actual) mae_fun(log1p(pred),  log1p(actual))
r2_log   <- function(pred, actual) r2_fun(log1p(pred),   log1p(actual))

clim_metrics_loc <- clim_preds %>%
  group_by(lat, lon) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "Climatology")
clim_metrics_year <- clim_preds %>%
  group_by(year) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "Climatology")

pers_metrics_loc <- pers_preds %>%
  group_by(lat, lon) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "Persistence (7-day)")
pers_metrics_year <- pers_preds %>%
  group_by(year) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "Persistence (7-day)")

lgb_metrics_loc <- lgb_preds %>%
  group_by(lat, lon) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "LightGBM_veg+meteo")
lgb_noveg_metrics_loc <- lgb_noveg_preds %>%
  group_by(lat, lon) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "LightGBM_meteo")
lgb_metrics_year <- lgb_preds %>%
  group_by(year) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "LightGBM_veg+meteo")
lgb_noveg_metrics_year <- lgb_noveg_preds %>%
  group_by(year) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), n_obs = n(), .groups = "drop") %>%
  mutate(Model = "LightGBM_meteo")

all_metrics_loc  <- bind_rows(lgb_metrics_loc, lgb_noveg_metrics_loc, clim_metrics_loc, pers_metrics_loc)
all_metrics_year <- bind_rows(lgb_metrics_year, lgb_noveg_metrics_year, clim_metrics_year, pers_metrics_year)

model_order <- c("LightGBM_veg+meteo", "LightGBM_meteo", "Climatology", "Persistence (7-day)")
all_metrics_loc  <- all_metrics_loc  %>% mutate(Model = factor(Model, levels = model_order))
all_metrics_year <- all_metrics_year %>% mutate(Model = factor(Model, levels = model_order))

pdf_file <- file.path(OUTPUT_DIR, "paper_figures.pdf")
pdf(pdf_file, width = 12, height = 9)

metrics_loc  <- all_metrics_loc  %>% mutate(aggregation = "by location")
metrics_year <- all_metrics_year %>% mutate(aggregation = "by year")
metrics_combined <- bind_rows(
  metrics_loc  %>% select(Model, RMSE, MAE, R2, aggregation),
  metrics_year %>% select(Model, RMSE, MAE, R2, aggregation)
)

p_rmse <- ggplot(metrics_combined %>% filter(!is.na(RMSE)), aes(x = Model, y = RMSE, fill = aggregation)) +
  geom_boxplot(position = position_dodge(0.8), outlier.shape = 21, outlier.size = 2) +
  stat_summary(fun = median, geom = "text", aes(label = sprintf("%.2f", after_stat(y)), group = aggregation),
               position = position_dodge(0.8), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("by location" = "#66c2a5", "by year" = "#fc8d62"), name = "") +
  coord_cartesian(ylim = c(0, 1.5)) +
  labs(title = "RMSE", y = "RMSE (log scale)") +
  theme_pubr(base_size = 11) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = "top",
        plot.margin = ggplot2::margin(t = 5, r = 5, b = 5, l = 15))

p_mae <- ggplot(metrics_combined %>% filter(!is.na(MAE)), aes(x = Model, y = MAE, fill = aggregation)) +
  geom_boxplot(position = position_dodge(0.8), outlier.shape = 21, outlier.size = 2) +
  stat_summary(fun = median, geom = "text", aes(label = sprintf("%.2f", after_stat(y)), group = aggregation),
               position = position_dodge(0.8), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("by location" = "#66c2a5", "by year" = "#fc8d62"), name = "") +
  coord_cartesian(ylim = c(0, 1.5)) +
  labs(title = "MAE", y = "MAE (log scale)") +
  theme_pubr(base_size = 11) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = "top",
        plot.margin = ggplot2::margin(t = 5, r = 5, b = 5, l = 15))

r2_medians <- metrics_combined %>%
  filter(!is.na(R2), R2 > -200) %>%
  group_by(Model, aggregation) %>%
  summarise(y = median(R2, na.rm = TRUE),
            whisker_top = max(R2[R2 <= quantile(R2, 0.75) + 1.5 * IQR(R2)]), .groups = "drop") %>%
  mutate(vjust_val = ifelse(
    (Model == "LightGBM_veg+meteo" & aggregation == "by year") |
    (Model == "Persistence (7-day)" & aggregation == "by year"), 1.5, -0.5
  ),
  # LightGBM_meteo by-year median sits inside a crowded box: print it above the whisker instead
  label_y = ifelse(Model == "LightGBM_meteo" & aggregation == "by year", whisker_top, y))

p_r2 <- ggplot(metrics_combined %>% filter(!is.na(R2), R2 > -200), aes(x = Model, y = R2, fill = aggregation)) +
  geom_boxplot(position = position_dodge(0.8), outlier.shape = 21, outlier.size = 2) +
  geom_text(data = r2_medians, aes(x = Model, y = label_y, label = sprintf("%.2f", y),
            group = aggregation, vjust = vjust_val),
            position = position_dodge(0.8), size = 3, inherit.aes = FALSE) +
  scale_fill_manual(values = c("by location" = "#66c2a5", "by year" = "#fc8d62"), name = "") +
  coord_cartesian(ylim = c(0, 1.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "R²", y = "R² (log scale)") +
  theme_pubr(base_size = 11) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = "top",
        plot.margin = ggplot2::margin(t = 5, r = 5, b = 5, l = 15))

# Centered panel titles leave the top-left corner free for the A/B/C labels
center_title <- theme(plot.title = element_text(hjust = 0.5))
p_fig1a <- ggarrange(p_rmse + center_title, p_mae + center_title, p_r2 + center_title,
                     ncol = 3, nrow = 1, common.legend = TRUE, legend = "top", labels = c("A", "B", "C"), label.x = 0.10)
p_fig1a <- annotate_figure(p_fig1a,
                            top = text_grob("Figure 1a: Model Performance Comparison",
                                           face = "bold", size = 14))
print(p_fig1a)

common_cols <- c("lat", "lon", "year", "doy", "actual", "predicted", "Model")

all_preds <- bind_rows(
  lgb_preds       %>% dplyr::select(any_of(common_cols)),
  lgb_noveg_preds %>% dplyr::select(any_of(common_cols)),
  clim_preds      %>% dplyr::select(any_of(common_cols)),
  pers_preds      %>% dplyr::select(any_of(common_cols))
) %>% mutate(Model = factor(Model, levels = model_order))

r2_by_model <- all_preds %>%
  group_by(Model) %>%
  summarise(R2 = r2_fun(predicted, actual), .groups = "drop")

parity_lim <- range(c(all_preds$actual, all_preds$predicted), na.rm = TRUE)
label_x    <- parity_lim[1] + 0.02 * diff(parity_lim)
label_y    <- parity_lim[2] - 0.05 * diff(parity_lim)

# One panel per model (same axes) so each can carry an A-D label
parity_sample <- all_preds %>% sample_frac(0.5)
parity_panel <- function(model_name) {
  ggplot(filter(parity_sample, Model == model_name), aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.3, size = 0.8, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "black", linewidth = 0.8) +
    geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed", linewidth = 0.8) +
    geom_text(data = filter(r2_by_model, Model == model_name), aes(x = label_x, y = label_y, label = sprintf("R² = %.3f", R2)),
              inherit.aes = FALSE, hjust = 0, vjust = 1, size = 3.5, fontface = "bold") +
    coord_cartesian(xlim = parity_lim, ylim = parity_lim) +
    labs(title = model_name, x = "Observed (grains/m³)", y = "Predicted (grains/m³)") +
    theme_pubr(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5))
}
parity_models <- levels(all_preds$Model)
p_fig1b <- ggarrange(plotlist = lapply(parity_models, parity_panel), ncol = 2, nrow = 2,
                     labels = LETTERS[seq_along(parity_models)], label.x = 0.06)
p_fig1b <- annotate_figure(p_fig1b,
                           top = text_grob("Figure 1b: Observed vs Predicted Pollen Concentration",
                                           face = "bold", size = 14))
print(p_fig1b)

lgb_loc_year <- lgb_preds %>%
  group_by(lat, lon, year) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), .groups = "drop") %>%
  mutate(Model = "LightGBM_veg+meteo")
lgb_noveg_loc_year <- lgb_noveg_preds %>%
  group_by(lat, lon, year) %>%
  summarise(RMSE = rmse_log(predicted, actual), MAE = mae_log(predicted, actual), R2 = r2_log(predicted, actual), .groups = "drop") %>%
  mutate(Model = "LightGBM_meteo")

loc_year_metrics <- bind_rows(lgb_loc_year, lgb_noveg_loc_year) %>%
  mutate(Model = factor(Model, levels = c("LightGBM_veg+meteo", "LightGBM_meteo")))

lat_order <- loc_year_metrics %>% distinct(lat, lon) %>% arrange(desc(lat)) %>%
  mutate(loc_label = sprintf("%.2f°N, %.2f°W", lat, abs(lon)))
loc_year_metrics <- loc_year_metrics %>%
  inner_join(lat_order, by = c("lat", "lon")) %>%
  mutate(loc_label = factor(loc_label, levels = lat_order$loc_label))

lgb_box_colors <- c("LightGBM_veg+meteo" = "#2ecc71", "LightGBM_meteo" = "#3498db")

p_loc_rmse <- ggplot(loc_year_metrics %>% filter(!is.na(RMSE)), aes(x = loc_label, y = RMSE, fill = Model)) +
  geom_boxplot(position = position_dodge(0.8), outlier.shape = 21, outlier.size = 1.5) +
  scale_fill_manual(values = lgb_box_colors, name = "", labels = c("LightGBM_veg+meteo", "LightGBM_meteo")) +
  labs(title = "RMSE", y = "RMSE (log scale)") +
  theme_pubr(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.title.x = element_blank(), legend.position = "top")

p_loc_mae <- ggplot(loc_year_metrics %>% filter(!is.na(MAE)), aes(x = loc_label, y = MAE, fill = Model)) +
  geom_boxplot(position = position_dodge(0.8), outlier.shape = 21, outlier.size = 1.5) +
  scale_fill_manual(values = lgb_box_colors, name = "", labels = c("LightGBM_veg+meteo", "LightGBM_meteo")) +
  labs(title = "MAE", y = "MAE (log scale)") +
  theme_pubr(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.title.x = element_blank(), legend.position = "top")

p_loc_r2 <- ggplot(loc_year_metrics %>% filter(!is.na(R2), R2 > -200), aes(x = loc_label, y = R2, fill = Model)) +
  geom_boxplot(position = position_dodge(0.8), outlier.shape = 21, outlier.size = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_fill_manual(values = lgb_box_colors, name = "", labels = c("LightGBM_veg+meteo", "LightGBM_meteo")) +
  scale_y_continuous(trans = r2_trans) +
  labs(title = "R²", y = "R² (log scale)") +
  theme_pubr(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.title.x = element_blank(), legend.position = "top")

fig2a_caption <- "Boxplots show yearly variation across 2018-2022 (metrics on log scale)"
print(annotate_figure(p_loc_rmse,
  top    = text_grob("Figure 2a: RMSE by Location (sorted by latitude, north to south)", face = "bold", size = 14),
  bottom = text_grob(fig2a_caption, face = "plain", size = 10, color = "gray40")))
print(annotate_figure(p_loc_mae,
  top    = text_grob("Figure 2a: MAE by Location (sorted by latitude, north to south)", face = "bold", size = 14),
  bottom = text_grob(fig2a_caption, face = "plain", size = 10, color = "gray40")))
print(annotate_figure(p_loc_r2,
  top    = text_grob("Figure 2a: R² by Location (sorted by latitude, north to south)", face = "bold", size = 14),
  bottom = text_grob(fig2a_caption, face = "plain", size = 10, color = "gray40")))

sample_locations <- tibble(
  lat = c(31.51, 35.55, 41.12),
  lon = c(-97.23, -97.41, -95.94)
)

sample_ts <- lgb_preds %>%
  filter(year == 2018) %>%
  mutate(lat_r = round(lat, 2), lon_r = round(lon, 2)) %>%
  semi_join(sample_locations %>% mutate(lat_r = round(lat, 2), lon_r = round(lon, 2)), by = c("lat_r", "lon_r")) %>%
  mutate(location_label = sprintf("%.4f° %s, %.4f° %s",
    abs(lat), ifelse(lat >= 0, "N", "S"),
    abs(lon), ifelse(lon >= 0, "E", "W")))

ts_metrics <- sample_ts %>%
  group_by(location_label) %>%
  summarise(
    R2   = r2_fun(predicted, actual),
    RMSE = rmse_fun(predicted, actual),
    MAE  = mae_fun(predicted, actual),
    .groups = "drop"
  ) %>%
  mutate(metric_label = sprintf("R² = %.3f\nRMSE = %.1f\nMAE = %.1f", R2, RMSE, MAE))

# One panel per sample location (own y scale) so each can carry an A/B/C label
ts_panel <- function(loc) {
  ggplot(filter(sample_ts, location_label == loc), aes(x = doy)) +
    geom_point(aes(y = actual, color = "Observed"), size = 1.5, alpha = 0.8) +
    geom_line(aes(y = predicted, color = "Predicted"), linewidth = 0.8, linetype = "solid") +
    geom_text(data = filter(ts_metrics, location_label == loc), aes(x = Inf, y = Inf, label = metric_label),
              inherit.aes = FALSE, hjust = 1.05, vjust = 1.2, size = 3.2, fontface = "plain") +
    scale_color_manual(values = c("Observed" = "black", "Predicted" = "red"), name = "") +
    # Same day-of-year range in every panel so the x axes line up
    coord_cartesian(xlim = c(1, 366)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 366)) +
    labs(title = loc, x = "Day of Year", y = "Pollen (grains/m³)") +
    theme_pubr(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5))
}
ts_locations <- sort(unique(sample_ts$location_label))
# align = "v" lines up the panels even though their y tick labels differ in width
p_fig2c <- ggarrange(plotlist = lapply(ts_locations, ts_panel), ncol = 1, align = "v",
                     labels = LETTERS[seq_along(ts_locations)], label.x = 0.03, common.legend = TRUE, legend = "bottom")
p_fig2c <- annotate_figure(p_fig2c,
                           top = text_grob("Figure 2c: Sample Prediction Time Series (LightGBM)",
                                           face = "bold", size = 14))
print(p_fig2c)

map_theme <- theme(
  panel.background = element_rect(fill = "white"),
  panel.grid.major = element_line(linetype = "dashed", color = "gray70", linewidth = 0.3),
  axis.line        = element_line(color = "gray40"),
  axis.text        = element_text(size = 9, color = "gray30"),
  axis.ticks       = element_line(color = "gray40"),
  axis.title       = element_text(size = 10, color = "gray30")
)

graticule <- st_graticule(
  x = c(-125, 24, -67, 50),
  crs = usmap_crs(),
  datum = st_crs(4326),
  lon = seq(-120, -70, 10),
  lat = seq(25, 50, 5)
)

veg_metrics_orig <- lgb_preds %>%
  group_by(lat, lon) %>%
  summarise(R2_veg   = r2_fun(predicted, actual),
            RMSE_veg = rmse_fun(predicted, actual),
            MAE_veg  = mae_fun(predicted, actual), .groups = "drop")
nov_metrics_orig <- lgb_noveg_preds %>%
  group_by(lat, lon) %>%
  summarise(R2_noveg   = r2_fun(predicted, actual),
            RMSE_noveg = rmse_fun(predicted, actual),
            MAE_noveg  = mae_fun(predicted, actual), .groups = "drop")

set.seed(42)
veg_improvement <- veg_metrics_orig %>%
  inner_join(nov_metrics_orig, by = c("lat", "lon")) %>%
  mutate(
    R2_improvement   = ifelse(is.na(R2_veg)   | is.na(R2_noveg),   0, R2_veg   - R2_noveg),
    RMSE_improvement = ifelse(is.na(RMSE_veg) | is.na(RMSE_noveg), 0, RMSE_veg - RMSE_noveg),
    MAE_improvement  = ifelse(is.na(MAE_veg)  | is.na(MAE_noveg),  0, MAE_veg  - MAE_noveg)
  ) %>%
  mutate(lon_j = lon + runif(n(), -0.2, 0.2), lat_j = lat + runif(n(), -0.15, 0.15))

veg_sf <- veg_improvement %>%
  select(lon = lon_j, lat = lat_j, R2_improvement, RMSE_improvement, MAE_improvement) %>%
  usmap_transform(input_names = c("lon", "lat"))

r2_fill_palette    <- scale_fill_gradient2(low = "red", mid = "white", high = "darkgreen", midpoint = 0, name = "Change", na.value = "grey90")
error_fill_palette <- scale_fill_gradient2(low = "darkgreen", mid = "white", high = "red", midpoint = 0, name = "Change", na.value = "grey90")

p_map_r2 <- plot_usmap(exclude = c("AK", "HI"), fill = "grey90", color = NA) +
  geom_sf(data = graticule, linetype = "dashed", color = "gray70", linewidth = 0.3) +
  geom_sf(data = veg_sf, aes(fill = R2_improvement), size = 6, alpha = 0.9, shape = 21, color = "black", stroke = 0.5) +
  r2_fill_palette + labs(title = "Figure 3b: R² Improvement (original scale)", x = "Longitude", y = "Latitude") +
  map_theme + theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(p_map_r2)

p_map_rmse <- plot_usmap(exclude = c("AK", "HI"), fill = "grey90", color = NA) +
  geom_sf(data = graticule, linetype = "dashed", color = "gray70", linewidth = 0.3) +
  geom_sf(data = veg_sf, aes(fill = RMSE_improvement), size = 6, alpha = 0.9, shape = 21, color = "black", stroke = 0.5) +
  error_fill_palette + labs(title = "Figure 3b: RMSE Change (original scale)", x = "Longitude", y = "Latitude") +
  map_theme + theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(p_map_rmse)

p_map_mae <- plot_usmap(exclude = c("AK", "HI"), fill = "grey90", color = NA) +
  geom_sf(data = graticule, linetype = "dashed", color = "gray70", linewidth = 0.3) +
  geom_sf(data = veg_sf, aes(fill = MAE_improvement), size = 6, alpha = 0.9, shape = 21, color = "black", stroke = 0.5) +
  error_fill_palette + labs(title = "Figure 3b: MAE Change (original scale)", x = "Longitude", y = "Latitude") +
  map_theme + theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(p_map_mae)

shap_data <- read_csv("data/processed/pollen_weather_smoothed.csv", show_col_types = FALSE)
shap_data <- shap_data %>% rename(date = Date)

if ("lat" %in% names(shap_data) && "lon" %in% names(shap_data)) {
  shap_data <- shap_data %>% mutate(location = paste0("Lat:", round(lat, 2), "_Lon:", round(lon, 2)))
}

exclude_cols <- c("date", "Date", "Acer", "Acer_orig", "year", "lat", "lon", "doy", "location", "photoperiod",
                  "Acer_lag_1week", "Acer_lag_1month", "Acer_lag_3month")
predictors <- setdiff(names(shap_data), exclude_cols)
predictors <- predictors[!grepl("^Acer", predictors)]

X    <- as.matrix(shap_data[, predictors, drop = FALSE])
y    <- shap_data$Acer
doys <- shap_data$doy

dtrain <- lgb.Dataset(data = X, label = y)
params <- list(
  objective = "regression", metric = "rmse",
  num_leaves = 31, learning_rate = 0.1, max_depth = -1,
  min_data_in_leaf = 20, feature_fraction = 0.8,
  bagging_fraction = 0.8, bagging_freq = 5, verbosity = -1
)
model <- lgb.train(params = params, data = dtrain, nrounds = 300, verbose = -1)

shap_values <- predict(model, X, type = "contrib")
shap_matrix <- shap_values[, seq_along(predictors), drop = FALSE]
colnames(shap_matrix) <- predictors

extract_lag_type <- function(feature_name) {
  if (grepl("1week|_1week",   feature_name, ignore.case = TRUE)) return("1-week")
  if (grepl("1month|_1month", feature_name, ignore.case = TRUE)) return("1-month")
  if (grepl("3month|_3month", feature_name, ignore.case = TRUE)) return("3-month")
  return("other")
}

extract_predictor_group <- function(feature_name) {
  if (grepl("tmin", feature_name, ignore.case = TRUE)) return("tmin")
  if (grepl("tmax", feature_name, ignore.case = TRUE)) return("tmax")
  if (grepl("prcp", feature_name, ignore.case = TRUE)) return("prcp")
  if (grepl("srad", feature_name, ignore.case = TRUE)) return("srad")
  if (grepl("vp_|vp\\.", feature_name, ignore.case = TRUE)) return("vp")
  if (grepl("swe",  feature_name, ignore.case = TRUE)) return("swe")
  if (grepl("ndvi", feature_name, ignore.case = TRUE)) return("ndvi")
  if (grepl("evi",  feature_name, ignore.case = TRUE)) return("evi")
  if (grepl("acer", feature_name, ignore.case = TRUE)) return("acer")
  return("other")
}

DOY_BIN_SIZE <- 16
doy_bins     <- seq(1, 365, by = DOY_BIN_SIZE)

assign_doy_bin <- function(doy) {
  bin_idx <- findInterval(doy, doy_bins, rightmost.closed = TRUE)
  doy_bins[pmax(1, bin_idx)]
}

shap_df         <- as.data.frame(shap_matrix)
shap_df$doy     <- doys
shap_df$doy_bin <- sapply(doys, assign_doy_bin)

shap_long <- shap_df %>%
  pivot_longer(cols = -c(doy, doy_bin), names_to = "feature", values_to = "shap_value") %>%
  mutate(
    predictor_group = sapply(feature, extract_predictor_group),
    lag_type        = sapply(feature, extract_lag_type)
  ) %>%
  filter(lag_type != "other")

shap_by_group <- shap_long %>%
  group_by(doy_bin, predictor_group, lag_type) %>%
  summarise(
    mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
    mean_shap     = mean(shap_value, na.rm = TRUE),
    .groups = "drop"
  )

plot_groups     <- c("evi", "tmax", "tmin", "srad", "prcp", "ndvi")
FORECAST_BUFFER <- 7
LAG_1WEEK       <- 7
LAG_1MONTH      <- 30
LAG_3MONTH      <- 90

shap_heatmap_data <- shap_by_group %>%
  filter(predictor_group %in% plot_groups, lag_type != "other") %>%
  mutate(
    lag_days = case_when(
      lag_type == "1-week"  ~ FORECAST_BUFFER + LAG_1WEEK,
      lag_type == "1-month" ~ FORECAST_BUFFER + LAG_1MONTH,
      lag_type == "3-month" ~ FORECAST_BUFFER + LAG_3MONTH
    )
  ) %>%
  filter(!is.na(lag_days)) %>%
  group_by(predictor_group) %>%
  mutate(phi = (mean_abs_shap - min(mean_abs_shap, na.rm = TRUE)) /
               (max(mean_abs_shap, na.rm = TRUE) - min(mean_abs_shap, na.rm = TRUE) + 1e-10)) %>%
  ungroup()

group_importance_order <- shap_heatmap_data %>%
  group_by(predictor_group) %>%
  summarise(overall_shap = mean(mean_abs_shap, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(overall_shap)) %>%
  pull(predictor_group)

lag_labels <- c("1-week" = "1wk", "1-month" = "1mo", "3-month" = "3mo")
lag_order  <- c("1-week", "1-month", "3-month")
doy_ticks  <- seq(1, 365, by = DOY_BIN_SIZE)

shap_heatmap_data <- shap_heatmap_data %>%
  mutate(predictor_group = factor(predictor_group, levels = group_importance_order),
         lag_type        = factor(lag_type, levels = lag_order))

p_fig5a <- ggplot(shap_heatmap_data, aes(x = lag_type, y = doy_bin, fill = phi)) +
  geom_tile(height = DOY_BIN_SIZE) +
  facet_wrap(~ predictor_group, ncol = 3) +
  scale_fill_viridis_c(option = "viridis", name = "phi", limits = c(0, 1), breaks = c(0.25, 0.50, 0.75)) +
  scale_x_discrete(labels = lag_labels, expand = c(0, 0)) +
  scale_y_continuous(breaks = doy_ticks, expand = c(0, 0)) +
  labs(title = "Figure 5a: SHAP Importance by DOY and Lag", x = "lag", y = "doy") +
  theme_minimal(base_size = 10) +
  theme(
    strip.text        = element_text(size = 10, hjust = 0.5),
    strip.background  = element_blank(),
    panel.grid        = element_blank(),
    panel.background  = element_rect(fill = "white", color = NA),
    axis.text         = element_text(size = 8),
    axis.title        = element_text(size = 10),
    legend.position   = "right",
    legend.key.height = unit(1.5, "cm"),
    legend.key.width  = unit(0.4, "cm"),
    panel.spacing     = unit(0.3, "lines")
  )
print(p_fig5a)

shap_by_doy <- shap_long %>%
  filter(predictor_group %in% plot_groups) %>%
  group_by(doy_bin, predictor_group) %>%
  summarise(mean_abs_shap = mean(abs(shap_value), na.rm = TRUE), .groups = "drop")

p_fig5c <- ggplot(shap_by_doy, aes(x = doy_bin, y = mean_abs_shap, color = predictor_group)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  scale_color_brewer(palette = "Set1", name = "Predictor") +
  labs(title = "Figure 5c: Feature Importance by Day of Year",
       x = "Day of Year",
       y = "Mean |SHAP|") +
  theme_pubr(base_size = 11) +
  theme(legend.position = "right")
print(p_fig5c)

p_map_r2_small <- plot_usmap(exclude = c("AK", "HI"), fill = "grey90", color = NA) +
  geom_sf(data = graticule, linetype = "dashed", color = "gray70", linewidth = 0.3) +
  geom_sf(data = veg_sf, aes(fill = R2_improvement), size = 4, alpha = 0.9, shape = 21, color = "black", stroke = 0.4) +
  r2_fill_palette + labs(title = "R² Change", x = "Longitude", y = "Latitude") +
  map_theme + theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
                    axis.title = element_text(size = 11),
                    legend.key.size = unit(0.4, "cm"), legend.text = element_text(size = 7),
                    legend.title = element_text(size = 8))

p_map_rmse_small <- plot_usmap(exclude = c("AK", "HI"), fill = "grey90", color = NA) +
  geom_sf(data = graticule, linetype = "dashed", color = "gray70", linewidth = 0.3) +
  geom_sf(data = veg_sf, aes(fill = RMSE_improvement), size = 4, alpha = 0.9, shape = 21, color = "black", stroke = 0.4) +
  error_fill_palette + labs(title = "RMSE Change", x = "Longitude", y = "Latitude") +
  map_theme + theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
                    axis.title = element_text(size = 11),
                    legend.key.size = unit(0.4, "cm"), legend.text = element_text(size = 7),
                    legend.title = element_text(size = 8))

p_map_mae_small <- plot_usmap(exclude = c("AK", "HI"), fill = "grey90", color = NA) +
  geom_sf(data = graticule, linetype = "dashed", color = "gray70", linewidth = 0.3) +
  geom_sf(data = veg_sf, aes(fill = MAE_improvement), size = 4, alpha = 0.9, shape = 21, color = "black", stroke = 0.4) +
  error_fill_palette + labs(title = "MAE Change", x = "Longitude", y = "Latitude") +
  map_theme + theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
                    axis.title = element_text(size = 11),
                    legend.key.size = unit(0.4, "cm"), legend.text = element_text(size = 7),
                    legend.title = element_text(size = 8))

# Tags anchored just outside each map's top-left corner (the fixed-aspect maps
# leave empty space in their cells, so ggarrange's cell-corner labels float away)
map_tag <- function(letter) {
  list(labs(tag = letter),
       theme(plot.tag.location = "panel", plot.tag.position = c(0, 1),
             plot.tag = element_text(face = "bold", size = 14, hjust = 0.4, vjust = -1)))
}
p_maps_combined <- ggarrange(p_map_r2_small + map_tag("A"), p_map_rmse_small + map_tag("B"),
                             p_map_mae_small + map_tag("C"), ncol = 3, nrow = 1)
p_maps_combined <- annotate_figure(p_maps_combined,
  top    = text_grob("Figure 3b: Vegetation Contribution Maps (veg+meteo vs meteo-only, original scale)", face = "bold", size = 13),
  bottom = text_grob("Green = veg model is better. Red = veg model is worse.", size = 9, color = "gray40"))
print(p_maps_combined)

dev.off()
