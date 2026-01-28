#!/usr/bin/env Rscript
######################### 05_hmd_monthly_sum_blocks.R #########################
# This script creates "monthly HMD" from the daily HMD file.
# For every grid point, it sums daily HMD values within each month.
# Because the grid is large, it processes latitude blocks and writes temporary NetCDF files per block.

library(ncdf4)
library(lubridate)
library(rslurm)

# This block defines the main run settings.
# We keep the zone fixed (Style 1) and we take the ensemble name from the environment.
# This matches the way your other HPC scripts are run (ENS=ens01, ENS=ens02, ...).
zone <- "zone1"
ens  <- Sys.getenv("ENS")
if (ens == "") stop("Environment variable ENS is not set (e.g., ENS=ens01).")

# This block defines input and output paths.
# Input is the merged daily HMD file (1993–2014).
# Output will be monthly HMD sums in a dedicated folder, written first as blocks and later merged.
hmd_base_dir <- "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output"

input_nc <- file.path(
  hmd_base_dir, zone,
  paste0("hmd_", zone, "_", ens, "_medewsa_daily_1993-2014.nc")
)

out_dir <- file.path(hmd_base_dir, zone, "monthly_sum", "tmp_blocks", ens)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

output_prefix <- file.path(
  hmd_base_dir, zone, "monthly_sum",
  paste0("hmd_monthlysum_", zone, "_", ens, "_1993-2014")
)
dir.create(dirname(output_prefix), recursive = TRUE, showWarnings = FALSE)

# This block reads only the metadata (lon/lat/time) from the input file.
# We do this “cheaply” to avoid loading the full variable.
# Then we create a monthly grouping (year-month) to know which daily indices belong to each month.
nc <- nc_open(input_nc)
lon <- ncvar_get(nc, "lon")
lat <- ncvar_get(nc, "lat")
time <- ncvar_get(nc, "time")
time_units <- ncatt_get(nc, "time", "units")$value
nc_close(nc)

if (!grepl("^days since", time_units)) stop("Unknown time unit: ", time_units)
origin_str <- sub("^days since ", "", time_units)
origin <- as.Date(substr(origin_str, 1, 10))
dates_daily <- origin + time

    

# This block defines the monthly reference axis to match the SPEI file exactly.
# SPEI uses monthly timestamps on the 1st of each month and the NetCDF axis "days since 1993-01-02".
# We reproduce the same monthly_time, units and numeric time values here, so CSI can match them 1:1.
monthly_time <- seq(
  from = as.Date("1993-01-01"),
  to   = as.Date("2014-12-01"),
  by   = "month"
)

months <- format(monthly_time, "%Y-%m")
nmon <- length(months)

month_id <- format(dates_daily, "%Y-%m")

# For each target month, store the daily indices belonging to it
month_groups <- lapply(months, function(m) which(month_id == m))
names(month_groups) <- months

# NetCDF time axis MUST match SPEI output
time_month_units <- "days since 1993-01-02"
time_month_vals  <- as.numeric(monthly_time - as.Date("1993-01-02"))











nlon <- length(lon)
nlat <- length(lat)
ntime <- length(time)

# This block defines how we split the spatial grid.
# Each job processes a chunk of latitude rows, which keeps memory reasonable on HPC.
block_size <- 40
lat_blocks <- seq(1, nlat, by = block_size)

# This is the worker function executed by each Slurm job.
# It reads only a latitude block from the daily HMD data.
# Then for each month, it sums the daily values within that month and writes a temporary NetCDF file.
process_block <- function(lat_start) {
  library(ncdf4)
  
  lat_end <- min(lat_start + block_size - 1, length(lat))
  nlat_b <- lat_end - lat_start + 1
  
  nc_in <- nc_open(input_nc)
  hmd_block <- ncvar_get(nc_in, "hmd",
                         start = c(1, lat_start, 1),
                         count = c(nlon, nlat_b, ntime))
  nc_close(nc_in)
  
  out_block <- array(NA_real_, dim = c(nlon, nlat_b, nmon))
  
  for (m in seq_len(nmon)) {
    idx <- month_groups[[m]]
    if (length(idx) == 0) next
    out_block[,,m] <- apply(hmd_block[,,idx, drop = FALSE], c(1,2), function(v) {
      if (all(is.na(v))) NA_real_ else sum(v, na.rm = TRUE)
    })
  }
  
  
  tmp_nc <- file.path(out_dir, paste0(basename(output_prefix), "_block_", lat_start, ".nc"))
  
  lon_dim  <- ncdim_def("lon", "degrees_east", lon)
  lat_dim  <- ncdim_def("lat", "degrees_north", lat[lat_start:lat_end])
  time_dim <- ncdim_def("time", time_month_units, time_month_vals, unlim = FALSE)
  
  var_def <- ncvar_def("hmd_monthly_sum", "HMD (monthly sum of daily HMD)",
                       dim = list(lon_dim, lat_dim, time_dim),
                       missval = NA_real_, prec = "float")
  
  nc_out <- nc_create(tmp_nc, vars = list(var_def))
  ncvar_put(nc_out, "hmd_monthly_sum", out_block)
  ncatt_put(nc_out, 0, "title", "Monthly HMD = sum of daily HMD")
  ncatt_put(nc_out, 0, "source_daily", basename(input_nc))
  ncatt_put(nc_out, 0, "zone", zone)
  ncatt_put(nc_out, 0, "ensemble", ens)
  nc_close(nc_out)
  
  paste0("Monthly HMD sum block ", lat_start, "-", lat_end, " done")
}

# This block submits the worker function to Slurm using rslurm.
# Each latitude block becomes one Slurm job.
# After all jobs finish, you run the separate merge script to create a single monthly NetCDF file.
params <- data.frame(lat_start = lat_blocks)

sopt <- list(
  partition = "compute",
  account   = "bb1201",
  `mail-type` = "FAIL",
  time = "2:00:00",
  mem  = "0G"
)

jobname <- paste0("HMDmonSum-", ens, "-", as.numeric(Sys.time()))
cat("[INFO] Submitting job: ", jobname, "\n")

sjob <- slurm_apply(
  f = process_block,
  params = params,
  jobname = jobname,
  global_objects = c("input_nc","out_dir","output_prefix","lon","lat","time",
                     "time_units","dates_daily","month_groups","months","nmon",
                     "time_month_vals","time_month_units",
                     "nlon","nlat","ntime","block_size","zone","ens"),
  
  nodes = nrow(params),
  cpus_per_node = 1,
  slurm_options = sopt,
  submit = TRUE,
  preschedule_cores = FALSE
)

cat("[INFO] Submitted monthly HMD-sum jobs for ", ens, "\n")
