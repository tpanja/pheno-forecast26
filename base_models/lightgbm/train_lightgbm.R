set.seed(42)
suppressPackageStartupMessages({
  library(lightgbm)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(stringr)
  library(gridExtra)
  library(grid)
})

OUTPUT_DIR <- "base_models/lightgbm/results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

num_leaves <- 63
learning_rate <- 0.05
max_depth <- -1
min_data_in_leaf <- 10
feature_fraction <- 0.6
bagging_fraction <- 0.8
bagging_freq <- 5
num_iterations <- 500

# Pooled mode: one model for all stations with lat/lon as features, early
# stopping on 2015-2017 and predictions averaged over POOLED_SEEDS seeds. These
# settings are identical in train_lightgbm.R and train_lightgbm_no_veg.R, so the
# two models differ only in the NDVI/EVI features. POOLED <- FALSE trains the
# per-station models below instead
POOLED <- TRUE
POOLED_SEEDS <- 5
POOLED_MAX_ROUNDS <- 3000
POOLED_EARLY_STOP <- 100
pooled_params <- list(objective = "regression", metric = "rmse", num_leaves = 63, learning_rate = 0.05, max_depth = -1, min_data_in_leaf = 10, feature_fraction = 0.6, bagging_fraction = 0.8, bagging_freq = 5, lambda_l1 = 0, lambda_l2 = 0, verbosity = -1)

# Tuned hyperparameters from tune_lightgbm.R (best_params.csv). Rows with
# location == "pooled" configure pooled mode; per-station mode uses rows matching
# each station and falls back to the defaults above
TUNED_MODEL <- "veg+meteo"
tuned_file <- "base_models/lightgbm/best_params.csv"
tuned_params <- if (file.exists(tuned_file)) dplyr::filter(readr::read_csv(tuned_file, show_col_types = FALSE), model == TUNED_MODEL) else NULL
station_params <- function(loc) {
  defaults <- list(num_leaves = num_leaves, learning_rate = learning_rate, min_data_in_leaf = min_data_in_leaf, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction, bagging_freq = bagging_freq, num_iterations = num_iterations, lambda_l1 = 0, lambda_l2 = 0)
  row <- if (!is.null(tuned_params)) dplyr::filter(tuned_params, location == loc) else NULL
  if (is.null(row) || nrow(row) == 0) return(defaults)
  as.list(row[1, names(defaults)])
}

train_data <- readr::read_csv("data/processed/train_data_base.csv", show_col_types = FALSE)
test_data_full <- readr::read_csv("data/processed/test_data_base.csv", show_col_types = FALSE)

has_location <- "lat" %in% names(train_data) && "lon" %in% names(train_data)
if (has_location) {
  train_data <- train_data %>% mutate(location = paste0("Lat:", round(lat, 2), "_Lon:", round(lon, 2)))
  test_data_full <- test_data_full %>% mutate(location = paste0("Lat:", round(lat, 2), "_Lon:", round(lon, 2)))
}

predictors <- setdiff(names(train_data), c("date", "Acer", "year", "lat", "lon", "doy", if (has_location) "location"))

rmse_fun <- function(pred, actual) sqrt(mean((pred - actual)^2, na.rm = TRUE))
mae_fun <- function(pred, actual) mean(abs(pred - actual), na.rm = TRUE)
r2_fun <- function(pred, actual) { ss_res <- sum((actual - pred)^2, na.rm = TRUE); ss_tot <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE); 1 - (ss_res / ss_tot) }
inverse_log_transform <- function(x) exp(x) - 1
duan_smearing_transform <- function(pred_log, smearing_factor) exp(pred_log) * smearing_factor - 1

set.seed(42)
val_data   <- dplyr::filter(train_data, year >= 2015, year <= 2017)
train_data <- dplyr::filter(train_data, year <= 2014)
test_data  <- test_data_full

train_lgb <- function(train_data, val_data, predictors, num_leaves = 31, learning_rate = 0.1, max_depth = -1, min_data_in_leaf = 20, feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, num_iterations = 300, verbose = 1, lambda_l1 = 0, lambda_l2 = 0) {
  X_train <- as.matrix(train_data[, predictors, drop = FALSE]); y_train <- train_data$Acer
  X_val <- as.matrix(val_data[, predictors, drop = FALSE]); y_val <- val_data$Acer
  dtrain <- lgb.Dataset(data = X_train, label = y_train)
  dval <- lgb.Dataset(data = X_val, label = y_val, reference = dtrain)
  params <- list(objective = "regression", metric = "rmse", num_leaves = num_leaves, learning_rate = learning_rate, max_depth = max_depth, min_data_in_leaf = min_data_in_leaf, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction, bagging_freq = bagging_freq, lambda_l1 = lambda_l1, lambda_l2 = lambda_l2, verbosity = -1)
  model <- lgb.train(params = params, data = dtrain, nrounds = num_iterations, valids = list(train = dtrain, val = dval), verbose = ifelse(verbose > 0, 1, -1))
  ev <- model$record_evals
  list(model = model, train_history = unlist(ev$train$rmse$eval), val_history = unlist(ev$val$rmse$eval), best_iter = model$best_iter)
}
predict_lgb <- function(model_obj, data, predictors) predict(model_obj$model, as.matrix(data[, predictors, drop = FALSE]))

model_list <- list(); imp_list <- list(); smearing_factors <- list(); params_used <- list()

if (POOLED) {
  # Bayesian-optimized pooled settings from tune_lightgbm.R (same search for both
  # models); the defaults above are used if tuning has not been run
  tuned_pooled <- if (!is.null(tuned_params)) dplyr::filter(tuned_params, location == "pooled") else NULL
  pooled_is_tuned <- !is.null(tuned_pooled) && nrow(tuned_pooled) > 0
  if (pooled_is_tuned) {
    tp <- as.list(tuned_pooled[1, ])
    pooled_params <- modifyList(pooled_params, list(num_leaves = as.integer(tp$num_leaves), learning_rate = tp$learning_rate, min_data_in_leaf = as.integer(tp$min_data_in_leaf), feature_fraction = tp$feature_fraction, bagging_fraction = tp$bagging_fraction, bagging_freq = as.integer(tp$bagging_freq), lambda_l1 = tp$lambda_l1, lambda_l2 = tp$lambda_l2))
  }
  pooled_predictors <- c(predictors, "lat", "lon")
  X_of <- function(d) as.matrix(d[, pooled_predictors, drop = FALSE])
  dtrain <- lgb.Dataset(data = X_of(train_data), label = train_data$Acer)
  dval <- lgb.Dataset(data = X_of(val_data), label = val_data$Acer, reference = dtrain)
  for (s in seq_len(POOLED_SEEDS)) {
    key <- paste0("pooled_seed", s)
    m <- lgb.train(params = modifyList(pooled_params, list(seed = s)), data = dtrain, nrounds = POOLED_MAX_ROUNDS, valids = list(train = dtrain, val = dval), early_stopping_rounds = POOLED_EARLY_STOP, verbose = -1)
    ev <- m$record_evals
    model_list[[key]] <- list(model = m, train_history = unlist(ev$train$rmse$eval), val_history = unlist(ev$val$rmse$eval), best_iter = m$best_iter)
    this_imp <- lgb.importance(model = m); this_imp$location <- key; imp_list[[key]] <- this_imp
    params_used[[key]] <- tibble::as_tibble(pooled_params[c("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2")]) %>%
      dplyr::mutate(location = "pooled", tuned = pooled_is_tuned, seed = s, num_iterations = m$best_iter)
  }
  # predict() uses each model's best (early-stopped) iteration
  predict_pooled <- function(d) Reduce(`+`, lapply(model_list, function(mo) predict(mo$model, X_of(d)))) / length(model_list)
  train_preds_log <- predict_pooled(train_data)
  global_smearing_factor <- mean(exp(train_data$Acer - train_preds_log), na.rm = TRUE)
  smearing_factors[["pooled"]] <- global_smearing_factor
  pred_frame <- function(d, pred_log) tibble(location = d$location, lat = d$lat, lon = d$lon, year = d$year, doy = d$doy, actual_log = d$Acer, predicted_log = pred_log, actual = inverse_log_transform(d$Acer), predicted = duan_smearing_transform(pred_log, global_smearing_factor))
  train_preds_df <- pred_frame(train_data, train_preds_log)
  val_preds_df <- pred_frame(val_data, predict_pooled(val_data))
  preds_df <- pred_frame(test_data, predict_pooled(test_data))
  min_len <- min(sapply(model_list, function(mo) length(mo$val_history)))
  avg_train_hist <- Reduce("+", lapply(model_list, function(mo) as.numeric(mo$train_history[1:min_len]))) / length(model_list)
  avg_val_hist <- Reduce("+", lapply(model_list, function(mo) as.numeric(mo$val_history[1:min_len]))) / length(model_list)
} else if (has_location) {
  locations <- sort(unique(train_data$location))
  all_train_preds <- list(); all_val_preds <- list(); all_preds <- list()
  all_train_hist <- list(); all_val_hist <- list()

  for (loc in locations) {
    train_loc <- dplyr::filter(train_data, location == loc)
    val_loc <- dplyr::filter(val_data, location == loc)
    test_loc <- dplyr::filter(test_data, location == loc)
    if (nrow(train_loc) == 0 || nrow(val_loc) == 0 || nrow(test_loc) == 0) next

    p <- station_params(loc)
    params_used[[loc]] <- tibble::as_tibble(p) %>% dplyr::mutate(location = loc, tuned = !is.null(tuned_params) && loc %in% tuned_params$location)
    mdl <- train_lgb(train_loc, val_loc, predictors, p$num_leaves, p$learning_rate, max_depth, p$min_data_in_leaf, p$feature_fraction, p$bagging_fraction, p$bagging_freq, p$num_iterations, verbose = 0, lambda_l1 = p$lambda_l1, lambda_l2 = p$lambda_l2)
    model_list[[loc]] <- mdl

    train_preds_log <- predict_lgb(mdl, train_loc, predictors)
    smearing_factor <- mean(exp(train_loc$Acer - train_preds_log), na.rm = TRUE)
    smearing_factors[[loc]] <- smearing_factor

    all_train_preds[[loc]] <- tibble(location = loc, lat = train_loc$lat, lon = train_loc$lon, year = train_loc$year, doy = train_loc$doy, actual_log = train_loc$Acer, predicted_log = train_preds_log, actual = inverse_log_transform(train_loc$Acer), predicted = duan_smearing_transform(train_preds_log, smearing_factor))
    val_preds_log <- predict_lgb(mdl, val_loc, predictors)
    all_val_preds[[loc]] <- tibble(location = loc, lat = val_loc$lat, lon = val_loc$lon, year = val_loc$year, doy = val_loc$doy, actual_log = val_loc$Acer, predicted_log = val_preds_log, actual = inverse_log_transform(val_loc$Acer), predicted = duan_smearing_transform(val_preds_log, smearing_factor))
    test_preds_log <- predict_lgb(mdl, test_loc, predictors)
    all_preds[[loc]] <- tibble(location = loc, lat = test_loc$lat, lon = test_loc$lon, year = test_loc$year, doy = test_loc$doy, actual_log = test_loc$Acer, predicted_log = test_preds_log, actual = inverse_log_transform(test_loc$Acer), predicted = duan_smearing_transform(test_preds_log, smearing_factor))

    all_train_hist[[loc]] <- mdl$train_history; all_val_hist[[loc]] <- mdl$val_history
    this_imp <- tryCatch(lgb.importance(model = mdl$model), error = function(e) NULL)
    if (!is.null(this_imp) && nrow(this_imp) > 0) { this_imp$location <- loc; imp_list[[loc]] <- this_imp }
  }
  train_preds_df <- bind_rows(all_train_preds); val_preds_df <- bind_rows(all_val_preds); preds_df <- bind_rows(all_preds)
  global_smearing_factor <- mean(unlist(smearing_factors), na.rm = TRUE)
  valid_train_hist <- Filter(function(x) !is.null(x) && length(x) > 0, all_train_hist)
  valid_val_hist <- Filter(function(x) !is.null(x) && length(x) > 0, all_val_hist)
  if (length(valid_train_hist) > 0 && length(valid_val_hist) > 0) {
    min_len <- min(sapply(valid_train_hist, length), sapply(valid_val_hist, length))
    avg_train_hist <- Reduce("+", lapply(valid_train_hist, function(v) as.numeric(v[1:min_len]))) / length(valid_train_hist)
    avg_val_hist <- Reduce("+", lapply(valid_val_hist, function(v) as.numeric(v[1:min_len]))) / length(valid_val_hist)
  } else { avg_train_hist <- numeric(0); avg_val_hist <- numeric(0) }
} else {
  mdl <- train_lgb(train_data, val_data, predictors, num_leaves, learning_rate, max_depth, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq, num_iterations, verbose = 1)
  train_preds_log <- predict_lgb(mdl, train_data, predictors)
  global_smearing_factor <- mean(exp(train_data$Acer - train_preds_log), na.rm = TRUE)
  train_preds_df <- tibble(year = train_data$year, doy = train_data$doy, actual_log = train_data$Acer, predicted_log = train_preds_log, actual = inverse_log_transform(train_data$Acer), predicted = duan_smearing_transform(train_preds_log, global_smearing_factor))
  if ("lat" %in% names(train_data)) { train_preds_df$lat <- train_data$lat; train_preds_df$lon <- train_data$lon }
  val_preds_log <- predict_lgb(mdl, val_data, predictors)
  val_preds_df <- tibble(year = val_data$year, doy = val_data$doy, actual_log = val_data$Acer, predicted_log = val_preds_log, actual = inverse_log_transform(val_data$Acer), predicted = duan_smearing_transform(val_preds_log, global_smearing_factor))
  if ("lat" %in% names(val_data)) { val_preds_df$lat <- val_data$lat; val_preds_df$lon <- val_data$lon }
  test_preds_log <- predict_lgb(mdl, test_data, predictors)
  preds_df <- tibble(year = test_data$year, doy = test_data$doy, actual_log = test_data$Acer, predicted_log = test_preds_log, actual = inverse_log_transform(test_data$Acer), predicted = duan_smearing_transform(test_preds_log, global_smearing_factor))
  if ("lat" %in% names(test_data)) { preds_df$lat <- test_data$lat; preds_df$lon <- test_data$lon }
  avg_train_hist <- as.numeric(mdl$train_history); avg_val_hist <- as.numeric(mdl$val_history)
}

val_rmse <- rmse_fun(val_preds_df$predicted, val_preds_df$actual)
val_mae <- mae_fun(val_preds_df$predicted, val_preds_df$actual)
val_r2 <- r2_fun(val_preds_df$predicted, val_preds_df$actual)
overall_rmse <- rmse_fun(preds_df$predicted, preds_df$actual)
overall_mae <- mae_fun(preds_df$predicted, preds_df$actual)
overall_r2 <- r2_fun(preds_df$predicted, preds_df$actual)

# Hyperparameters actually used by each station's model (tuned or defaults)
if (has_location && length(params_used) > 0) {
  best_model_params <- dplyr::bind_rows(params_used) %>%
    dplyr::mutate(max_depth = max_depth, loss_function = "regression (MSE)", duan_smearing_factor = unlist(smearing_factors[location])) %>%
    dplyr::relocate(location, tuned)
} else {
  best_model_params <- tibble::tibble(
    parameter = c("num_leaves", "learning_rate", "max_depth", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "bagging_freq", "num_iterations", "loss_function", "duan_smearing_factor"),
    value = c(as.character(num_leaves), as.character(learning_rate), as.character(max_depth), as.character(min_data_in_leaf), as.character(feature_fraction), as.character(bagging_fraction), as.character(bagging_freq), as.character(num_iterations), "regression (MSE)", as.character(round(global_smearing_factor, 6)))
  )
}
readr::write_csv(best_model_params, file.path(OUTPUT_DIR, "best_model_parameters.csv"))
readr::write_csv(train_preds_df, file.path(OUTPUT_DIR, "train_predictions.csv"))
readr::write_csv(val_preds_df, file.path(OUTPUT_DIR, "validation_predictions.csv"))
readr::write_csv(preds_df, file.path(OUTPUT_DIR, "test_predictions.csv"))

MODEL_DIR <- file.path(OUTPUT_DIR, "saved_models")
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)
if (has_location) {
  for (loc in names(model_list)) {
    safe_name <- gsub("[^A-Za-z0-9_.-]", "_", loc)
    lgb.save(model_list[[loc]]$model, file.path(MODEL_DIR, paste0(safe_name, ".model")))
  }
}
saveRDS(list(pooled = POOLED, smearing_factors = smearing_factors, predictors = if (POOLED) pooled_predictors else predictors, has_location = has_location, global_smearing_factor = global_smearing_factor), file.path(MODEL_DIR, "model_metadata.rds"))

results_by_year <- preds_df %>% group_by(year) %>%
  summarise(RMSE = rmse_fun(predicted, actual), MAE = mae_fun(predicted, actual), R2 = r2_fun(predicted, actual), n_obs = n(), mean_actual = mean(actual, na.rm = TRUE), mean_predicted = mean(predicted, na.rm = TRUE), .groups = "drop")
readr::write_csv(results_by_year, file.path(OUTPUT_DIR, "metrics_by_year.csv"))

if (has_location) {
  results_by_location <- preds_df %>% group_by(location, lat, lon) %>%
    summarise(RMSE = rmse_fun(predicted, actual), MAE = mae_fun(predicted, actual), R2 = r2_fun(predicted, actual), n_obs = n(), .groups = "drop") %>% arrange(RMSE)
  readr::write_csv(results_by_location, file.path(OUTPUT_DIR, "metrics_by_location.csv"))
}

pdf_file <- file.path(OUTPUT_DIR, "lightgbm_report.pdf")
pdf(pdf_file, width = 11, height = 8.5)

par(mar = c(2, 2, 2, 2)); plot.new()
text(0.5, 0.85, "LightGBM (Standard)", cex = 2.2, font = 2)
text(0.5, 0.77, "Acer Pollen Prediction - Original Scale", cex = 1.8, font = 2)
text(0.5, 0.70, "(Duan Smearing Back-transformation)", cex = 1.4, col = "darkblue")
text(0.5, 0.58, if (POOLED) sprintf("Pooled model (all stations), %d-seed average, early stopping%s", POOLED_SEEDS, if (pooled_is_tuned) ", Bayesian-optimized" else "") else if (!is.null(tuned_params)) "Per-station hyperparameters from best_params.csv" else sprintf("num_leaves=%d | lr=%.2f | iterations=%d", num_leaves, learning_rate, num_iterations), cex = 1.1)
text(0.5, 0.48, sprintf("Duan Smearing Factor: %.4f", global_smearing_factor), cex = 1.2, col = "darkgreen", font = 2)
text(0.5, 0.35, "Test Performance (Original Scale):", cex = 1.4, font = 2)
text(0.5, 0.29, sprintf("RMSE: %.2f | MAE: %.2f | R²: %.4f", overall_rmse, overall_mae, overall_r2), cex = 1.2)

max_val <- max(c(preds_df$actual, preds_df$predicted), na.rm = TRUE) * 1.1
p1 <- ggplot(preds_df, aes(x = actual, y = predicted)) + geom_point(alpha = 0.4, color = "#2E86AB", size = 2) + geom_abline(slope = 1, intercept = 0, color = "#E63946", linetype = "dashed", linewidth = 1.2) + geom_smooth(method = "lm", se = FALSE, color = "#F4A261", linewidth = 0.8) + labs(title = "Actual vs Predicted (Original Scale)", subtitle = sprintf("RMSE: %.2f | MAE: %.2f | R²: %.4f", overall_rmse, overall_mae, overall_r2), x = "Actual Acer Pollen (grains/m³)", y = "Predicted Acer Pollen (grains/m³)") + coord_cartesian(xlim = c(0, max_val), ylim = c(0, max_val)) + theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold", size = 14), plot.margin = ggplot2::margin(t = 10, r = 15, b = 10, l = 10))
print(p1)

p2 <- ggplot(results_by_year, aes(x = factor(year), y = RMSE)) + geom_col(fill = "#2E86AB", alpha = 0.8, width = 0.7) + geom_text(aes(label = sprintf("%.1f", RMSE)), vjust = -0.5, size = 3.5) + labs(title = "RMSE by Year (Original Scale)", x = "Year", y = "RMSE (pollen grains/m³)") + theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold", size = 14), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
print(p2)

p3a <- ggplot(results_by_year, aes(x = factor(year), y = MAE)) + geom_col(fill = "#E9C46A", alpha = 0.8, width = 0.7) + geom_text(aes(label = sprintf("%.1f", MAE)), vjust = -0.5, size = 3) + labs(title = "MAE by Year", x = "Year", y = "MAE") + theme_minimal(base_size = 10) + theme(plot.title = element_text(face = "bold", size = 12), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
p3b <- ggplot(results_by_year, aes(x = factor(year), y = R2)) + geom_col(fill = "#2A9D8F", alpha = 0.8, width = 0.7) + geom_text(aes(label = sprintf("%.3f", R2)), vjust = -0.5, size = 3) + labs(title = "R² by Year", x = "Year", y = "R²") + theme_minimal(base_size = 10) + theme(plot.title = element_text(face = "bold", size = 12), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, 1))
gridExtra::grid.arrange(p3a, p3b, ncol = 2, top = grid::textGrob("Performance Metrics by Year", gp = grid::gpar(fontsize = 14, fontface = "bold")))

if (has_location) { sample_locs <- preds_df %>% dplyr::distinct(location, lat, lon) %>% dplyr::slice_head(n = 3); sample_data <- preds_df %>% dplyr::semi_join(sample_locs, by = c("location", "lat", "lon")) %>% dplyr::filter(year == min(year)) } else { sample_data <- preds_df %>% dplyr::filter(year == min(year)) }
if (nrow(sample_data) > 0) {
  sample_long <- sample_data %>% tidyr::pivot_longer(cols = c(actual, predicted), names_to = "type", values_to = "pollen")
  p4 <- ggplot(sample_long, aes(x = doy, y = pollen, color = type)) + geom_line(linewidth = 1) + scale_color_manual(values = c("actual" = "#264653", "predicted" = "#2E86AB"), labels = c("actual" = "Actual", "predicted" = "Predicted")) + labs(title = sprintf("Sample Predictions - Year %d (Original Scale)", min(sample_data$year)), x = "Day of Year", y = "Acer Pollen (grains/m³)", color = "") + theme_minimal(base_size = 12) + theme(legend.position = "bottom", plot.title = element_text(face = "bold", size = 14))
  if (has_location) p4 <- p4 + facet_wrap(~ paste0("Lat: ", round(lat, 2), ", Lon: ", round(lon, 2)), ncol = 1, scales = "free_y")
  print(p4)
}

residuals_orig <- preds_df$actual - preds_df$predicted
p5 <- ggplot(tibble(residual = residuals_orig), aes(x = residual)) + geom_histogram(bins = 40, fill = "#2E86AB", alpha = 0.7, color = "white") + geom_vline(xintercept = 0, color = "#E63946", linetype = "dashed", linewidth = 1.2) + labs(title = "Residuals Distribution (Original Scale)", subtitle = sprintf("Mean: %.2f | SD: %.2f | Median: %.2f", mean(residuals_orig, na.rm = TRUE), sd(residuals_orig, na.rm = TRUE), median(residuals_orig, na.rm = TRUE)), x = "Residual (pollen grains/m³)", y = "Count") + theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold", size = 14))
print(p5)

if (length(avg_train_hist) > 0 && length(avg_val_hist) > 0) {
  loss_df <- tibble(iteration = seq_along(avg_train_hist), Training = avg_train_hist, Validation = avg_val_hist) %>% tidyr::pivot_longer(cols = c(Training, Validation), names_to = "Set", values_to = "RMSE")
  p6 <- ggplot(loss_df, aes(x = iteration, y = RMSE, color = Set, linetype = Set)) + geom_line(linewidth = 1) + scale_color_manual(values = c("Training" = "#2E86AB", "Validation" = "#E63946")) + scale_linetype_manual(values = c("Training" = "solid", "Validation" = "dashed")) + labs(title = "Training vs Validation RMSE Curve", x = "Iteration", y = "RMSE") + theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold", size = 14), legend.position = "bottom")
  print(p6)
}

if (has_location && length(imp_list) > 0) {
  agg_imp <- bind_rows(imp_list) %>% group_by(Feature) %>% summarise(mean_gain = mean(Gain, na.rm = TRUE), .groups = "drop") %>% arrange(desc(mean_gain)) %>% dplyr::slice_head(n = 15)
  p7 <- ggplot(agg_imp, aes(x = reorder(Feature, mean_gain), y = mean_gain)) + geom_col(fill = "#2E86AB", alpha = 0.85) + coord_flip() + labs(title = "Top 15 Feature Importance (Avg Gain)", x = "", y = "Mean Importance") + theme_minimal(base_size = 11) + theme(plot.title = element_text(face = "bold", size = 13))
  print(p7)
} else if (!has_location && exists("mdl")) {
  importance <- lgb.importance(model = mdl$model)
  top_features <- head(importance %>% arrange(desc(Gain)), 15)
  p7 <- ggplot(top_features, aes(x = reorder(Feature, Gain), y = Gain)) + geom_col(fill = "#2E86AB", alpha = 0.8) + coord_flip() + labs(title = "Top 15 Feature Importance (Gain)", x = "", y = "Importance") + theme_minimal(base_size = 11) + theme(plot.title = element_text(face = "bold", size = 13))
  print(p7)
}

par(mar = c(2, 2, 3, 2)); plot.new()
text(0.5, 0.95, "Summary Statistics", cex = 1.6, font = 2)
text(0.5, 0.82, "Test Set Performance (Original Scale)", cex = 1.3, font = 2)
text(0.5, 0.76, sprintf("RMSE: %.2f pollen grains/m³", overall_rmse), cex = 1.1)
text(0.5, 0.71, sprintf("MAE: %.2f pollen grains/m³", overall_mae), cex = 1.1)
text(0.5, 0.66, sprintf("R²: %.4f", overall_r2), cex = 1.1)
text(0.5, 0.54, "Validation Performance", cex = 1.3, font = 2)
text(0.5, 0.48, sprintf("RMSE: %.2f | MAE: %.2f | R²: %.4f", val_rmse, val_mae, val_r2), cex = 1.1)
text(0.5, 0.36, sprintf("Duan Smearing Factor: %.4f", global_smearing_factor), cex = 1.2, col = "darkgreen", font = 2)

dev.off()
