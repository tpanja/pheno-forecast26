library(dplyr)
library(daymetr)
library(lubridate)
library(tidyr)
library(janitor)
library(tibble)

set.seed(42)

FORECAST_BUFFER <- 7

LAG_1WEEK  <- 7
LAG_1MONTH <- 30
LAG_3MONTH <- 90

df <- read.csv("data/raw/2023_data.csv")
station_locations <- read.csv("data/raw/station_locations.csv")

df_filtered <- df %>%
  dplyr::select(Acer, Date, Station.ID) %>%
  dplyr::filter(!is.na(Acer))

df_filtered_w_location <- df_filtered %>%
  left_join(
    station_locations %>% dplyr::select(id, lat, lon),
    by = c("Station.ID" = "id")
  ) %>%
  dplyr::select(-Station.ID) %>%
  mutate(
    Date = as.Date(Date),
    year = year(Date),
    doy  = yday(Date)
  ) %>%
  drop_na()

years <- 2003:2022
n_observations <- 10

valid_sites <- df_filtered_w_location %>%
  dplyr::filter(year %in% years) %>%
  dplyr::count(lat, lon, year, name = "n") %>%
  complete(lat, lon, year = years, fill = list(n = 0)) %>%
  group_by(lat, lon) %>%
  summarize(ok = all(n >= n_observations), .groups = "drop") %>%
  dplyr::filter(ok) %>%
  dplyr::select(lat, lon)

df_min_per_year <- df_filtered_w_location %>%
  dplyr::filter(year %in% years) %>%
  inner_join(valid_sites, by = c("lat", "lon"))

library(ptw)

df_smoothed <- df_min_per_year %>%
  group_by(lat, lon, year) %>%
  arrange(doy) %>%
  mutate(Acer_smooth = whit2(Acer, lambda = 50)) %>%
  mutate(Acer_smooth = pmax(Acer_smooth, 0)) %>%
  ungroup()

saveRDS(df_smoothed, "data/processed/cache_df_smoothed.rds")

# Smoothed Acer on `target` (same-day duplicates averaged); if the station did
# not sample that day, the most recent value within the preceding `max_back` days
acer_on_date <- function(dates, values, target, max_back = LAG_1WEEK) {
  idx <- which(dates <= target & dates >= target - max_back & !is.na(values))
  if (length(idx) == 0) return(NA_real_)
  latest <- max(dates[idx])
  mean(values[idx][dates[idx] == latest])
}

df_filtered_lags <- df_smoothed %>%
  group_by(lat, lon) %>%
  arrange(lat, lon, year, doy) %>%
  mutate(
    date_idx = as.Date(paste(year, doy), format = "%Y %j"),

    Acer_lag_1week = sapply(seq_len(n()), function(i) {
      target_date <- date_idx[i]
      window_start <- target_date - (FORECAST_BUFFER + LAG_1WEEK)
      window_end   <- target_date - FORECAST_BUFFER
      window_idx <- (date_idx >= window_start & date_idx <= window_end)
      if (sum(window_idx) > 0 && sum(!is.na(Acer_smooth[window_idx])) > 0) {
        mean(Acer_smooth[window_idx], na.rm = TRUE)
      } else NA_real_
    }),

    Acer_lag_1month = sapply(seq_len(n()), function(i) {
      target_date <- date_idx[i]
      window_start <- target_date - (FORECAST_BUFFER + LAG_1MONTH)
      window_end   <- target_date - FORECAST_BUFFER
      window_idx <- (date_idx >= window_start & date_idx <= window_end)
      if (sum(window_idx) > 0 && sum(!is.na(Acer_smooth[window_idx])) > 0) {
        mean(Acer_smooth[window_idx], na.rm = TRUE)
      } else NA_real_
    }),

    Acer_lag_3month = sapply(seq_len(n()), function(i) {
      target_date <- date_idx[i]
      window_start <- target_date - (FORECAST_BUFFER + LAG_3MONTH)
      window_end   <- target_date - FORECAST_BUFFER
      window_idx <- (date_idx >= window_start & date_idx <= window_end)
      if (sum(window_idx) > 0 && sum(!is.na(Acer_smooth[window_idx])) > 0) {
        mean(Acer_smooth[window_idx], na.rm = TRUE)
      } else NA_real_
    }),

    # Point value at t - FORECAST_BUFFER (the persistence forecast input) and its
    # change over the preceding week
    Acer_lag_7d = sapply(seq_len(n()), function(i) {
      acer_on_date(date_idx, Acer_smooth, date_idx[i] - FORECAST_BUFFER)
    }),

    Acer_slope_7d = Acer_lag_7d - sapply(seq_len(n()), function(i) {
      acer_on_date(date_idx, Acer_smooth, date_idx[i] - FORECAST_BUFFER - LAG_1WEEK)
    })
  ) %>%
  dplyr::select(-date_idx) %>%
  ungroup() %>%
  dplyr::filter(!is.na(Acer_lag_1week) | !is.na(Acer_lag_1month) | !is.na(Acer_lag_3month))

df_filtered_lags$Date <- as.Date(df_filtered_lags$Date)
unique_locations <- df_filtered_lags[!duplicated(df_filtered_lags[c("lat", "lon")]), c("lat", "lon")]
unique_locations$site_id <- seq_len(nrow(unique_locations))
years_needed <- unique(year(df_filtered_lags$Date))

all_weather_list <- list()
counter <- 1

for (i in seq_len(nrow(unique_locations))) {
  for (yr in years_needed) {
    weather_data <- NULL
    for (attempt in 1:5) {
      weather_data <- tryCatch(
        download_daymet(
          site     = paste0("site_", unique_locations$site_id[i], "_", yr),
          lat      = unique_locations$lat[i],
          lon      = unique_locations$lon[i],
          start    = yr,
          end      = yr,
          internal = TRUE,
          silent   = TRUE
        ),
        error = function(e) {
          Sys.sleep(10)
          NULL
        }
      )
      if (!is.null(weather_data)) break
    }
    if (is.null(weather_data)) stop(sprintf("Daymet download failed after 5 attempts: loc %d year %d", i, yr))
    weather_data$data$lat <- unique_locations$lat[i]
    weather_data$data$lon <- unique_locations$lon[i]
    all_weather_list[[counter]] <- weather_data$data
    counter <- counter + 1
  }
}

all_weather <- do.call(rbind, all_weather_list)
all_weather$Date <- as.Date(paste(all_weather$year, all_weather$yday), format = "%Y %j")
saveRDS(all_weather, "data/processed/cache_all_weather.rds")

weather_vars <- c("tmin..deg.c.", "tmax..deg.c.", "prcp..mm.day.",
                  "srad..W.m.2.", "vp..Pa.", "swe..kg.m.2.")

calculate_buffered_weather_lags <- function(obs_date, lat_val, lon_val, weather_data,
                                            buffer = FORECAST_BUFFER,
                                            lag_1w = LAG_1WEEK,
                                            lag_1m = LAG_1MONTH,
                                            lag_3m = LAG_3MONTH) {
  loc_weather <- weather_data[weather_data$lat == lat_val & weather_data$lon == lon_val, ]
  loc_weather <- loc_weather[order(loc_weather$Date), ]
  result <- list()

  window_end <- obs_date - buffer

  window_mean <- function(var, lag_days) {
    window_start <- obs_date - (buffer + lag_days)
    in_window <- loc_weather[loc_weather$Date >= window_start & loc_weather$Date <= window_end, ]
    if (nrow(in_window) > 0 && sum(!is.na(in_window[[var]])) > 0) {
      mean(in_window[[var]], na.rm = TRUE)
    } else NA_real_
  }

  for (var in weather_vars) {
    result[[paste0(var, "_1week_avg")]]  <- window_mean(var, lag_1w)
    result[[paste0(var, "_1month_avg")]] <- window_mean(var, lag_1m)
    result[[paste0(var, "_3month_avg")]] <- window_mean(var, lag_3m)
  }

  data.frame(result)
}

lag_results <- vector("list", nrow(df_filtered_lags))

for (i in seq_len(nrow(df_filtered_lags))) {
  if (i %% 100 == 0) cat("Processing observation", i, "of", nrow(df_filtered_lags), "\n")
  lag_results[[i]] <- calculate_buffered_weather_lags(
    df_filtered_lags$Date[i], df_filtered_lags$lat[i], df_filtered_lags$lon[i], all_weather
  )
}
lag_df <- do.call(rbind, lag_results)

calculate_photoperiod <- function(lat, doy) {
  phi_rad <- lat * pi / 180
  declination_deg <- -23.45 * cos(2 * pi * (doy + 10) / 365)
  declination_rad <- declination_deg * pi / 180
  a <- -tan(phi_rad) * tan(declination_rad)
  a <- pmax(pmin(a, 1), -1)
  (24 / pi) * acos(a)
}

final_df <- df_filtered_lags %>%
  mutate(Acer_orig = Acer, Acer = Acer_smooth) %>%
  dplyr::select(-Acer_smooth) %>%
  bind_cols(lag_df) %>%
  mutate(photoperiod = calculate_photoperiod(lat, doy))

mod13 <- read.csv("data/raw/Pollen-Stations-v2-MOD13A1-061-results.csv",
                  stringsAsFactors = FALSE) %>%
  mutate(
    date = as.Date(Date),
    lat  = as.numeric(Latitude),
    lon  = as.numeric(Longitude)
  ) %>%
  dplyr::filter(
    MOD13A1_061__500m_16_days_pixel_reliability %in% c(0, 1),
    MOD13A1_061__500m_16_days_NDVI > -3000,
    MOD13A1_061__500m_16_days_EVI  > -3000
  ) %>%
  mutate(
    ndvi = MOD13A1_061__500m_16_days_NDVI * 0.0001,
    evi  = MOD13A1_061__500m_16_days_EVI  * 0.0001
  ) %>%
  dplyr::select(lat, lon, date, ndvi, evi)

myd13 <- read.csv("data/raw/asdf-MYD13A1-061-results.csv",
                  stringsAsFactors = FALSE) %>%
  mutate(
    date = as.Date(Date),
    lat  = as.numeric(Latitude),
    lon  = as.numeric(Longitude),
    ndvi = MYD13A1_061__500m_16_days_NDVI * 0.0001,
    evi  = MYD13A1_061__500m_16_days_EVI  * 0.0001
  ) %>%
  dplyr::select(lat, lon, date, ndvi, evi)

mod13 <- bind_rows(mod13, myd13) %>% arrange(lat, lon, date)

library(imputeTS)

VI_LAMBDA <- 50
VI_MAXGAP <- Inf

whittaker_smoothing_filling <- function(x, lambda, maxgap = Inf, minseg = 2) {
  x <- imputeTS::na_replace(x, fill = -9999, maxgap = maxgap)
  w <- (x != -9999)

  max_id <- 0
  done <- FALSE
  while (!done) {
    v_non_na <- which(!is.na(x[(max_id + 1):length(x)]))
    if (length(v_non_na) == 0) {
      done <- TRUE
    } else {
      min_id <- min(v_non_na) + max_id
      v_na <- which(is.na(x[min_id:length(x)]))
      if (length(v_na) == 0) {
        max_id <- length(x)
        done <- TRUE
      } else {
        max_id <- min(v_na) - 1 + (min_id - 1)
      }
      if ((max_id - min_id + 1) < minseg) {
        x[min_id:max_id] <- -9999
      } else {
        x[min_id:max_id] <- ptw::whit1(x[min_id:max_id], lambda, w[min_id:max_id])
      }
    }
  }
  x[x == -9999] <- NA
  x
}

mod13 <- mod13 %>%
  group_by(lat, lon) %>%
  group_modify(~ {
    obs <- .x %>%
      group_by(date) %>%
      summarise(ndvi = mean(ndvi, na.rm = TRUE),
                evi  = mean(evi,  na.rm = TRUE), .groups = "drop")
    daily <- tibble(date = seq(min(obs$date), max(obs$date), by = "day")) %>%
      left_join(obs, by = "date")
    daily$ndvi <- whittaker_smoothing_filling(daily$ndvi, lambda = VI_LAMBDA, maxgap = VI_MAXGAP)
    daily$evi  <- whittaker_smoothing_filling(daily$evi,  lambda = VI_LAMBDA, maxgap = VI_MAXGAP)
    daily
  }) %>%
  ungroup() %>%
  arrange(lat, lon, date)
saveRDS(mod13, "data/processed/cache_mod13_daily.rds")

modis_window_avg <- function(obs_date, lat_val, lon_val, modis_df, value_cols,
                             buffer       = FORECAST_BUFFER,
                             lag_days,
                             max_lookback = 32,
                             tol          = 1e-6) {

  window_end   <- obs_date - buffer
  window_start <- window_end - lag_days

  loc <- modis_df[
    abs(modis_df$lat - lat_val) < tol &
      abs(modis_df$lon - lon_val) < tol, ]

  out <- setNames(rep(NA_real_, length(value_cols)), value_cols)
  if (nrow(loc) == 0) return(as.data.frame(t(out)))

  loc <- loc[order(loc$date), ]
  in_window <- loc[loc$date >= window_start & loc$date <= window_end, ]

  if (nrow(in_window) > 0) {
    for (vc in value_cols) out[vc] <- mean(in_window[[vc]], na.rm = TRUE)
  } else {
    recent <- loc[loc$date <= window_end & loc$date >= (window_end - max_lookback), ]
    if (nrow(recent) > 0) {
      best <- recent[which.max(recent$date), ]
      for (vc in value_cols) out[vc] <- best[[vc]]
    }
  }

  as.data.frame(t(out))
}

mod13_vars <- c("ndvi", "evi")
n_obs <- nrow(final_df)

modis_lag_list <- vector("list", n_obs)

for (i in seq_len(n_obs)) {
  if (i %% 500 == 0) cat("  MODIS lag: observation", i, "of", n_obs, "\n")

  obs_date <- as.Date(final_df$Date[i])
  lat_val  <- final_df$lat[i]
  lon_val  <- final_df$lon[i]

  m13_1w <- modis_window_avg(obs_date, lat_val, lon_val, mod13, mod13_vars,
                             buffer = FORECAST_BUFFER, lag_days = LAG_1WEEK,
                             max_lookback = 32)
  m13_1m <- modis_window_avg(obs_date, lat_val, lon_val, mod13, mod13_vars,
                             buffer = FORECAST_BUFFER, lag_days = LAG_1MONTH,
                             max_lookback = 32)
  m13_3m <- modis_window_avg(obs_date, lat_val, lon_val, mod13, mod13_vars,
                             buffer = FORECAST_BUFFER, lag_days = LAG_3MONTH,
                             max_lookback = 32)

  names(m13_1w) <- paste0(mod13_vars, "_1week_avg")
  names(m13_1m) <- paste0(mod13_vars, "_1month_avg")
  names(m13_3m) <- paste0(mod13_vars, "_3month_avg")

  modis_lag_list[[i]] <- bind_cols(m13_1w, m13_1m, m13_3m)
}

modis_lag_df <- do.call(rbind, modis_lag_list)
final_df <- bind_cols(final_df, modis_lag_df)

final_df <- final_df %>% mutate(across(!Date, as.numeric))

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
write.csv(final_df, "data/processed/pollen_weather_smoothed.csv", row.names = FALSE)

print(head(final_df[, c("Date", "Acer", "lat", "lon", "doy", "photoperiod")]))

edm_choice <- toupper(trimws(readline()))

if (edm_choice %in% c("Y", "ALL")) {
  N_LAGS <- 6
  LAG_PERIOD <- 8

  edm_weather_vars <- c("tmin..deg.c.", "tmax..deg.c.", "prcp..mm.day.",
                        "srad..W.m.2.", "vp..Pa.", "swe..kg.m.2.")
  edm_veg_vars <- c("ndvi", "evi")
  edm_pollen_var <- "Acer_smooth"
  
  all_edm_vars <- c(edm_weather_vars, edm_veg_vars, edm_pollen_var)
  
  base_df <- df_smoothed %>%
    dplyr::select(lat, lon, Date, year, doy, Acer, Acer_smooth) %>%
    group_by(lat, lon, Date) %>%
    slice_max(Acer_smooth, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    arrange(lat, lon, Date)
  
  locations <- base_df %>% distinct(lat, lon)
  edm_results <- list()
  
  for (loc_i in seq_len(nrow(locations))) {
    loc_lat <- locations$lat[loc_i]
    loc_lon <- locations$lon[loc_i]
    
    df_loc <- base_df %>% 
      filter(lat == loc_lat, lon == loc_lon) %>% 
      arrange(Date)
    
    weather_loc <- all_weather %>%
      filter(lat == loc_lat, lon == loc_lon) %>%
      arrange(Date)
    
    modis_loc <- mod13 %>%
      filter(abs(lat - loc_lat) < 1e-6, abs(lon - loc_lon) < 1e-6) %>%
      arrange(date)
    
    if (nrow(df_loc) < 100) {
      next
    }
    
    n_new_cols <- length(all_edm_vars) * N_LAGS
    lag_matrix <- matrix(NA_real_, nrow = nrow(df_loc), ncol = n_new_cols)
    lag_names <- character(n_new_cols)
    
    col_idx <- 1
    
    for (var in edm_weather_vars) {
      for (lag_num in 1:N_LAGS) {
        lag_name <- sprintf("%s_lag%02d", gsub("\\.\\.|\\.\\.", "_", var), lag_num)
        lag_names[col_idx] <- lag_name

        for (row in seq_len(nrow(df_loc))) {
          obs_date <- df_loc$Date[row]
          window_end <- obs_date - FORECAST_BUFFER - (lag_num - 1) * LAG_PERIOD
          window_start <- window_end - (LAG_PERIOD - 1)

          period_data <- weather_loc %>%
            filter(Date >= window_start, Date <= window_end)

          if (nrow(period_data) > 0 && sum(!is.na(period_data[[var]])) > 0) {
            lag_matrix[row, col_idx] <- mean(period_data[[var]], na.rm = TRUE)
          }
        }
        col_idx <- col_idx + 1
      }
    }
    
    for (var in edm_veg_vars) {
      for (lag_num in 1:N_LAGS) {
        lag_name <- sprintf("%s_lag%02d", var, lag_num)
        lag_names[col_idx] <- lag_name
        
        for (row in seq_len(nrow(df_loc))) {
          obs_date <- df_loc$Date[row]
          window_end <- obs_date - FORECAST_BUFFER - (lag_num - 1) * LAG_PERIOD
          window_start <- window_end - (LAG_PERIOD - 1)
          
          period_data <- modis_loc %>%
            filter(date >= window_start, date <= window_end)
          
          if (nrow(period_data) > 0 && sum(!is.na(period_data[[var]])) > 0) {
            lag_matrix[row, col_idx] <- mean(period_data[[var]], na.rm = TRUE)
          }
        }
        col_idx <- col_idx + 1
      }
    }
    
    for (lag_num in 1:N_LAGS) {
      lag_name <- sprintf("Acer_lag%02d", lag_num)
      lag_names[col_idx] <- lag_name

      for (row in seq_len(nrow(df_loc))) {
        obs_date <- df_loc$Date[row]
        window_end <- obs_date - FORECAST_BUFFER - (lag_num - 1) * LAG_PERIOD
        window_start <- window_end - (LAG_PERIOD - 1)

        period_data <- df_loc %>%
          filter(Date >= window_start, Date <= window_end)

        if (nrow(period_data) > 0 && sum(!is.na(period_data$Acer_smooth)) > 0) {
          lag_matrix[row, col_idx] <- mean(period_data$Acer_smooth, na.rm = TRUE)
        }
      }
      col_idx <- col_idx + 1
    }
    
    colnames(lag_matrix) <- lag_names
    
    df_loc_edm <- df_loc %>%
      mutate(Acer = Acer_smooth,
             photoperiod = calculate_photoperiod(lat, doy)) %>%
      dplyr::select(lat, lon, Date, year, doy, Acer, photoperiod) %>%
      bind_cols(as_tibble(lag_matrix))
    
    edm_results[[loc_i]] <- df_loc_edm
    
    if (loc_i %% 5 == 0 || loc_i == nrow(locations)) {
    }
  }
  
  edm_df <- bind_rows(edm_results)
  edm_df <- edm_df %>% na.omit()
  
  write.csv(edm_df, "data/processed/pollen_weather_edm.csv", row.names = FALSE)
  
  if (edm_choice == "ALL") {
  }

} else {
}
