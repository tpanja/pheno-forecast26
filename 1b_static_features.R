library(elevatr)
library(sf)
library(dplyr)

input_file <- "data/processed/pollen_weather_smoothed.csv"

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file,
       "\n  Run 1_data_organization.R first to generate it.")
}

final_df <- read.csv(input_file, stringsAsFactors = FALSE)
final_df$Date <- as.Date(final_df$Date)

static_col_names <- c("elevation_m")

all_possible_static <- c("clim_zone_code", "elevation_m",
                         "prism_ann_tmean", "prism_growing_tmean",
                         "prism_ann_ppt", "prism_summer_ppt", "prism_winter_ppt",
                         "prism_ppt_seasonality",
                         "prism_gdd_base5", "prism_gdd_base10")
existing_static <- intersect(names(final_df), all_possible_static)
if (length(existing_static) > 0) {
  final_df <- final_df[, !(names(final_df) %in% existing_static)]
}

stations <- final_df %>%
  distinct(lat, lon) %>%
  arrange(lat, lon)

n_stations <- nrow(stations)

stations_sf <- st_as_sf(stations, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
elev_sf     <- get_elev_point(stations_sf, src = "epqs")
stations$elevation_m <- elev_sf$elevation

static_features <- stations %>%
  dplyr::select(lat, lon, elevation_m)

final_df <- final_df %>%
  left_join(static_features, by = c("lat", "lon"))

final_df <- final_df %>% mutate(across(!Date, as.numeric))

na_cols <- names(final_df)[sapply(final_df, function(x) all(is.na(x)))]
if (length(na_cols) > 0) {
  final_df <- final_df[, !(names(final_df) %in% na_cols)]
}

write.csv(final_df, input_file, row.names = FALSE)

write.csv(stations, "data/processed/station_static_features.csv", row.names = FALSE)

for (col in static_col_names) {
}
print(head(final_df[, c("Date", "Acer", "lat", "lon", "elevation_m")]))
