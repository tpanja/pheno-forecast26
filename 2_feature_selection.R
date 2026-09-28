library(dplyr)
library(lubridate)
library(janitor)
library(tidyr)
library(tibble)

set.seed(42)

input_file <- readline("Enter the input file name: ")

if (input_file == "edm") {
  df <- read.csv("data/processed/pollen_weather_edm.csv") %>%
    clean_names() %>%
    select(-any_of(c("acer_orig"))) %>%
    na.omit()
} else {
   # acer_slope_7d is NA when there is no sample in the week before t-14; LightGBM
   # handles missing values natively, so keep those rows
   df <- read.csv("data/processed/pollen_weather_smoothed.csv") %>%
    clean_names() %>%
    select(-any_of(c("acer_orig"))) %>%
    drop_na(-any_of("acer_slope_7d"))
}

if ("acer" %in% names(df)) df <- df %>% rename(Acer = acer)

if (input_file == "edm") {
  N_LAGS_TO_KEEP <- 6

  lag_pattern <- sprintf("_lag(%s)$", paste(sprintf("%02d", 1:N_LAGS_TO_KEEP), collapse = "|"))
  non_lag_cols <- names(df)[!grepl("_lag\\d{2}$", names(df))]
  lag_cols_to_keep <- names(df)[grepl(lag_pattern, names(df))]

  df <- df %>% select(all_of(c(non_lag_cols, lag_cols_to_keep)))

  if (!("date" %in% names(df))) stop("Expected a 'date' column after clean_names().")
}

df_selected <- df %>%
  mutate(
    date = as.Date(date),
    year = if ("year" %in% names(df)) year else year(date),
    doy  = yday(date)
  )

years <- 2003:2022
n_years <- length(years)
n_train_years <- floor(n_years * 0.75)

train_years <- years[1:n_train_years]
test_years  <- years[(n_train_years + 1):n_years]

df_selected <- df_selected %>%
  mutate(Acer = log(1 + Acer))

train_data <- df_selected %>% filter(year %in% train_years)
test_data  <- df_selected %>% filter(year %in% test_years)

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

if (input_file == "edm") {
  train_file <- "data/processed/train_data_edm.csv"
  test_file  <- "data/processed/test_data_edm.csv"
} else {
  train_file <- "data/processed/train_data_base.csv"
  test_file  <- "data/processed/test_data_base.csv"
}

write.csv(train_data, train_file, row.names = FALSE)
write.csv(test_data,  test_file,  row.names = FALSE)
