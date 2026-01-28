######################### Calculation of the SPEI (spatial) ###############################


########################################
# Setup
########################################

# Load packages
library(ncdf4)
library(SPEI)
library(zoo)

# Set working directory
setwd("/work/bb1478/Celine/MedEWSa/hazard_analysis/SPEI/")

# Define ensembles
ensembles <- sprintf("ens%02d", 1:25)


########################################
# Loop over ensembles
########################################

for (ensemble in ensembles) {

  cat("Processing ensemble:", ensemble, "\n")

  # Define input paths
  pet_file <- file.path(
    "data/PET",
    paste0("pet_", ensemble, "_daily_1993-2014.nc")
  )

  pr_file <- file.path(
    "data/Input",
    paste0(ensemble, "_cal_daily_common_0p5.nc")
  )

  # Define output path
  out_nc <- file.path(
    "data/SPEI",
    paste0("spei_", ensemble, "_monthly_1993-2014.nc")
  )

  # Open NetCDF
  nc_pr  <- nc_open(pr_file)
  nc_pet <- nc_open(pet_file)

  pr_array  <- ncvar_get(nc_pr,  "pr")
  pet_array <- ncvar_get(nc_pet, "pet")

  lon <- ncvar_get(nc_pr, "lon")
  lat <- ncvar_get(nc_pr, "lat")

  time_raw <- ncvar_get(nc_pr, "time")
  origin <- sub("days since ", "", nc_pr$dim$time$units)
  dates <- as.Date(time_raw, origin = origin)

  # Checks
  stopifnot(all(dim(pr_array) == dim(pet_array)))


  ## Water balance
  balance_array <- pr_array - pet_array

    
  ## Monthly reference time
  monthly_time <- seq(
    from = as.Date("1993-01-01"),
    to   = as.Date("2014-12-01"),
    by   = "month"
  )

  monthly_spei_array <- array(
    NA,
    dim = c(length(lon), length(lat), length(monthly_time))
  )

    
  ## SPEI calculation per grid cell
  for (i in seq_along(lon)) {
    for (j in seq_along(lat)) {

      series <- balance_array[i, j, ]
      if (all(is.na(series))) next

      zoo_series <- zoo(series, dates)

      monthly <- tryCatch({
        aggregate(zoo_series, as.yearmon, sum, na.rm = TRUE)
      }, error = function(e) NULL)

      if (is.null(monthly)) next

      start_year  <- as.numeric(format(start(monthly), "%Y"))
      start_month <- as.numeric(format(start(monthly), "%m"))

      ts_series <- ts(
        coredata(monthly),
        start = c(start_year, start_month),
        frequency = 12
      )

      spei_res <- tryCatch({
        spei(ts_series, scale = 1)
      }, error = function(e) NULL)

      if (is.null(spei_res)) next

      spei_vals <- as.numeric(spei_res$fitted)
      spei_vals[!is.finite(spei_vals)] <- NA

      spei_dates <- seq(
        from = as.Date(paste(start_year, start_month, "01", sep = "-")),
        by   = "month",
        length.out = length(spei_vals)
      )

      idx <- match(spei_dates, monthly_time)
      ok  <- which(!is.na(idx) & !is.na(spei_vals))

      if (length(ok) > 0) {
        monthly_spei_array[i, j, idx[ok]] <- spei_vals[ok]
      }
    }
  }


  ## Create NetCDF
  time_nc <- as.numeric(monthly_time - as.Date("1993-01-02"))

  dim_lon  <- ncdim_def("lon",  "degrees_east",  lon)
  dim_lat  <- ncdim_def("lat",  "degrees_north", lat)
  dim_time <- ncdim_def("time", "days since 1993-01-02", time_nc, unlim = FALSE)

  var_spei <- ncvar_def(
    name = "spei",
    units = "1",
    dim = list(dim_lon, dim_lat, dim_time),
    missval = NA_real_,
    longname = "Standardized Precipitation Evapotranspiration Index (scale 1)",
    prec = "float"
  )

  nc_out <- nc_create(out_nc, vars = list(var_spei))

  ncvar_put(nc_out, "spei", monthly_spei_array)

  # Global attributes
  ncatt_put(nc_out, 0, "title", "Monthly SPEI (scale=1)")
  ncatt_put(nc_out, 0, "source", "pr - PET (daily aggregated to monthly)")
  ncatt_put(nc_out, 0, "institution", "MedEWSa")
  ncatt_put(nc_out, 0, "period", "1993-2014")

  # Close files
  nc_close(nc_out)
  nc_close(nc_pr)
  nc_close(nc_pet)

  cat("Finished ensemble:", ensemble, "\n")
}