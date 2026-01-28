#!/usr/bin/env Rscript
######################### Calculation of monthly heatwave metrics (spatial) #########################


########################################
# Setup
########################################

# Load required packages
library(ncdf4)
library(lubridate)

# Working directory
setwd("/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/")

# Settings
zone <- "zone1"                    # adjust if needed
ens  <- Sys.getenv("ENS")           # ensemble member from environment

# Define paths
hmd_base_dir <- "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output"
out_base     <- file.path(hmd_base_dir, zone, "metrics_monthly")

# Output subdirectories
vars <- c(
  "abs_occurrence",
  "mean_duration",
  "mean_magnitude",
  "max_magnitude",
  "mask"
)

for (v in vars) {
  dir.create(file.path(out_base, v), recursive = TRUE, showWarnings = FALSE)
}

# Input NetCDF (daily HMD)
input_nc <- file.path(
  hmd_base_dir, zone,
  paste0("hmd_", zone, "_", ens, "_medewsa_daily_1993-2014.nc")
)


## Helper function: extract heatwave events
get_events_full <- function(hmd_ts) {

  is_hw <- hmd_ts > 0
  r <- rle(is_hw)

  ends   <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1

  data.frame(
    start    = starts[r$values],
    end      = ends[r$values],
    duration = r$lengths[r$values]
  )
}


########################################
# Read NetCDF metadata
########################################

nc <- nc_open(input_nc)

hmd  <- ncvar_get(nc, "hmd")   # [lon, lat, time]
lon  <- ncvar_get(nc, "lon")
lat  <- ncvar_get(nc, "lat")
time <- ncvar_get(nc, "time")

time_units <- ncatt_get(nc, "time", "units")$value
nc_close(nc)

# Convert time axis to Date
dates_daily <- as.Date(sub("days since ", "", time_units)) + time

# Monthly reference dates (mid-month)
dates_monthly <- seq(
  from = as.Date("1993-01-15"),
  to   = as.Date("2014-12-15"),
  by   = "month"
)

ntime <- length(dates_monthly)

# Reorder HMD to [lat, lon, time]
hmd <- aperm(hmd, c(2, 1, 3))

nlat <- dim(hmd)[1]
nlon <- dim(hmd)[2]


## Initialize output arrays
abs_occurrence <- array(NA_real_, dim = c(nlat, nlon, ntime))
mean_duration  <- array(NA_real_, dim = c(nlat, nlon, ntime))
mean_magnitude <- array(NA_real_, dim = c(nlat, nlon, ntime))
max_magnitude  <- array(NA_real_, dim = c(nlat, nlon, ntime))

# Static land–sea mask (1 = valid grid cell, 0 = invalid / ocean)
valid_mask <- array(0L, dim = c(nlat, nlon))


########################################
# Grid loop
########################################

for (i in seq_len(nlat)) {
  for (j in seq_len(nlon)) {

    ts <- hmd[i, j, ]

    # Skip fully invalid grid cells (e.g. ocean)
    if (all(is.na(ts))) next

    # Mark grid cell as valid (land)
    valid_mask[i, j] <- 1

    # Detect heatwave events
    events <- get_events_full(ts)
    if (nrow(events) == 0) next

    event_start_dates <- dates_daily[events$start]

    for (t in seq_len(ntime)) {

      yr <- year(dates_monthly[t])
      mo <- month(dates_monthly[t])

      # Events starting in this month
      ev_t <- events[
        year(event_start_dates) == yr &
        month(event_start_dates) == mo,
      ]

      if (nrow(ev_t) == 0) next

      abs_occurrence[i, j, t] <- round(nrow(ev_t), 0)
      mean_duration[i, j, t] <- mean(ev_t$duration)

      idx <- unlist(mapply(seq, ev_t$start, ev_t$end, SIMPLIFY = FALSE))
      vals <- ts[idx]
      mean_magnitude[i, j, t] <- mean(vals, na.rm = TRUE)
      max_magnitude[i, j, t]  <- max(vals,  na.rm = TRUE)
    }
  }
}


########################################
# Write NetCDF outputs
########################################

# Define dimensions
lon_dim <- ncdim_def("lon", "degrees_east", lon)
lat_dim <- ncdim_def("lat", "degrees_north", lat)
time_dim <- ncdim_def(
  "time",
  "days since 1993-01-15",
  as.numeric(dates_monthly - as.Date("1993-01-15")),
  unlim = FALSE
)

# Generic writer function
write_var <- function(arr, name, units) {

  out_nc <- file.path(
    out_base, name,
    paste0("hmd_", name, "_monthly_", zone, "_", ens, "_1993-2014.nc")
  )

  var_def <- ncvar_def(
    name, units,
    list(lon_dim, lat_dim, time_dim),
    missval = NA,
    prec = "float"
  )

  nc_out <- nc_create(out_nc, var_def)
  ncvar_put(nc_out, name, aperm(arr, c(2, 1, 3)))  # back to [lon, lat, time]
  nc_close(nc_out)
}

# Write metrics
write_var(abs_occurrence, "abs_occurrence", "count")
write_var(mean_duration,  "mean_duration",  "days")
write_var(mean_magnitude, "mean_magnitude", "HMD")
write_var(max_magnitude,  "max_magnitude",  "HMD")


# Write static land–sea mask
mask_nc <- file.path(
  out_base, "mask",
  paste0("land_mask_", zone, ".nc")
)

if (!file.exists(mask_nc)) {

  mask_var <- ncvar_def(
    "land_mask",
    "1 = valid grid cell (land), 0 = invalid (ocean)",
    list(lon_dim, lat_dim),
    missval = NA,
    prec = "byte"
  )

  nc_mask <- nc_create(mask_nc, mask_var)
  ncvar_put(nc_mask, "land_mask", t(valid_mask))
  nc_close(nc_mask)
}

message("Finished ensemble ", ens)