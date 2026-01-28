#!/usr/bin/env Rscript
######################### PET calculation (spatial) #########################


########################################
# Setup
########################################

# Load packages
library(ncdf4)
library(SPEI)
library(lubridate)

# Set working directory
setwd("/work/bb1478/Celine/MedEWSa/hazard_analysis/SPEI/")

# Define ensembles
ensembles <- sprintf("ens%02d", 1:25)


########################################
# Loop over ensembles
########################################

for (ensemble in ensembles) {

  cat("Processing ensemble:", ensemble, "\n")
 
  # Define paths
  in_nc <- file.path(
    "data/Input",
    paste0(ensemble, "_cal_daily_common_0p5.nc")
  )

  out_nc <- file.path(
    "data/PET/",
    paste0("pet_", ensemble, "_daily_1993-2014.nc")
  )

  # Read input data
  nc <- nc_open(in_nc)

  lon  <- ncvar_get(nc, "lon")
  lat  <- ncvar_get(nc, "lat")
  time <- ncvar_get(nc, "time")
  time_units <- ncatt_get(nc, "time", "units")$value

  tx <- ncvar_get(nc, "tx")   # [lon, lat, time]
  tn <- ncvar_get(nc, "tn")
  pr <- ncvar_get(nc, "pr")

  nc_close(nc)

  # Time conversion
  origin <- as.Date(substr(sub("^days since ", "", time_units), 1, 10))
  dates <- origin + time

  nlon  <- length(lon)
  nlat  <- length(lat)
  ntime <- length(time)

  # Prepare output array
  pet <- array(NA_real_, dim = c(nlon, nlat, ntime))

    
  ## PET computation (serial)
  cat("Starting PET computation...\n")

  for (j in seq_len(nlat)) {
    cat("Latitude", j, "of", nlat, "\n")

    for (i in seq_len(nlon)) {

      tmin <- tn[i, j, ]
      tmax <- tx[i, j, ]
      prec <- pr[i, j, ]

      if (all(is.na(tmin)) || all(is.na(tmax))) next

      pet[i, j, ] <- hargreaves(
        Tmin = tmin,
        Tmax = tmax,
        lat  = lat[j]
      )
    }
  }

  cat("PET computation finished\n")


  ########################################
  # Write NetCDF output
  ########################################

  lon_dim  <- ncdim_def("lon", "degrees_east", lon)
  lat_dim  <- ncdim_def("lat", "degrees_north", lat)
  time_dim <- ncdim_def("time", time_units, time)

  pet_var <- ncvar_def(
    "pet", "mm/day",
    dim = list(lon_dim, lat_dim, time_dim),
    missval = NA, prec = "float"
  )

  nc_out <- nc_create(out_nc, pet_var)
  ncvar_put(nc_out, pet_var, pet)
  nc_close(nc_out)

  cat("Output written to:", out_nc, "\n")
}