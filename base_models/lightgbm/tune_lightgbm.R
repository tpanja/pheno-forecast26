set.seed(42)
suppressPackageStartupMessages({
  library(lightgbm)
  library(dplyr)
  library(readr)
  library(rBayesianOptimization)
})

# Bayesian optimization (Gaussian-process surrogate, rBayesianOptimization) of the
# pooled LightGBM model used by train_lightgbm.R / train_lightgbm_no_veg.R.
# Both feature sets get the same search space, budget and seed; each candidate is
# scored by validation RMSE (log scale, 2015-2017) with early stopping choosing the
# number of rounds.

OUTPUT_DIR <- "base_models/lightgbm"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

all_train <- read_csv("data/processed/train_data_base.csv", show_col_types = FALSE)

train_pool <- all_train %>% filter(year <= 2014)
val_pool   <- all_train %>% filter(year >= 2015, year <= 2017)

# Pooled model: station identity enters through lat/lon
base_exclude <- c("date", "Acer", "year", "doy")
pred_veg    <- setdiff(names(train_pool), base_exclude)
pred_noveg  <- pred_veg[!grepl("evi|ndvi", pred_veg, ignore.case = TRUE)]

LEARNING_RATE <- 0.05
MAX_ROUNDS    <- 3000L
EARLY_STOP    <- 100L
INIT_POINTS   <- 10L
N_ITER        <- 30L

# lambda_l1 / lambda_l2 are searched on a log10 scale
bounds <- list(
  num_leaves       = c(8L, 255L),
  min_data_in_leaf = c(5L, 200L),
  feature_fraction = c(0.3, 1.0),
  bagging_fraction = c(0.5, 1.0),
  log10_lambda_l1  = c(-3, 1),
  log10_lambda_l2  = c(-3, 1)
)

make_params <- function(num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction,
                        log10_lambda_l1, log10_lambda_l2) {
  list(objective = "regression", metric = "rmse", learning_rate = LEARNING_RATE,
       num_leaves = as.integer(num_leaves), min_data_in_leaf = as.integer(min_data_in_leaf),
       feature_fraction = feature_fraction, bagging_fraction = bagging_fraction, bagging_freq = 5L,
       lambda_l1 = 10^log10_lambda_l1, lambda_l2 = 10^log10_lambda_l2,
       max_depth = -1L, seed = 42L, verbosity = -1L)
}

run_bayes_opt <- function(predictors, model_label) {
  X_train <- as.matrix(train_pool[, predictors, drop = FALSE])
  X_val   <- as.matrix(val_pool[, predictors, drop = FALSE])

  score_fn <- function(num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction,
                       log10_lambda_l1, log10_lambda_l2) {
    params <- make_params(num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction,
                          log10_lambda_l1, log10_lambda_l2)
    # Datasets are rebuilt per candidate because min_data_in_leaf affects binning
    dtrain <- lgb.Dataset(X_train, label = train_pool$Acer, params = list(feature_pre_filter = FALSE))
    dval   <- lgb.Dataset(X_val, label = val_pool$Acer, reference = dtrain)
    m <- lgb.train(params, dtrain, nrounds = MAX_ROUNDS, valids = list(val = dval),
                   early_stopping_rounds = EARLY_STOP, verbose = -1)
    # rBayesianOptimization maximizes, so return negative validation RMSE
    list(Score = -m$best_score, Pred = m$best_iter)
  }

  set.seed(42)
  opt <- BayesianOptimization(score_fn, bounds = bounds, init_points = INIT_POINTS, n_iter = N_ITER,
                              acq = "ei", eps = 0.0, verbose = FALSE)

  history <- opt$History %>%
    mutate(val_rmse = -Value, num_iterations = as.integer(unlist(opt$Pred)), model = model_label) %>%
    select(-Value)
  best <- history %>% slice_min(val_rmse, n = 1, with_ties = FALSE)
  cat(sprintf("%s: best validation RMSE %.4f after %d evaluations\n", model_label, best$val_rmse, nrow(history)))

  best_row <- with(best, tibble(
    model            = model_label,
    location         = "pooled",
    num_leaves       = as.integer(num_leaves),
    learning_rate    = LEARNING_RATE,
    num_iterations   = num_iterations,
    min_data_in_leaf = as.integer(min_data_in_leaf),
    feature_fraction = feature_fraction,
    bagging_fraction = bagging_fraction,
    bagging_freq     = 5L,
    lambda_l1        = 10^log10_lambda_l1,
    lambda_l2        = 10^log10_lambda_l2,
    val_rmse         = val_rmse
  ))
  list(best = best_row, history = history)
}

res_veg   <- run_bayes_opt(pred_veg,   "veg+meteo")
res_noveg <- run_bayes_opt(pred_noveg, "meteo-only")

write_csv(res_veg$history,   file.path(OUTPUT_DIR, "tuning_results_veg.csv"))
write_csv(res_noveg$history, file.path(OUTPUT_DIR, "tuning_results_noveg.csv"))

best_params <- bind_rows(res_veg$best, res_noveg$best)
write_csv(best_params, file.path(OUTPUT_DIR, "best_params.csv"))

print(as.data.frame(best_params), digits = 4)
