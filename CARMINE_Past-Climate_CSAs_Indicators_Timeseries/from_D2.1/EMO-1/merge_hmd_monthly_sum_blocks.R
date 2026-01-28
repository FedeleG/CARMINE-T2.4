#!/usr/bin/env Rscript
######################### 06_merge_hmd_monthly_sum_blocks.R #########################
# This script merges the temporary monthly HMD blocks created by 05_hmd_monthly_sum_blocks.R.
# Each block contains a subset of latitude rows; this script stitches them back into a full grid.
# The result is one monthly NetCDF file for the chosen zone and ensemble.

library(ncdf4)

# This block defines the run settings and where to find the block files.
# ENS is taken from the environment, like your other scripts.
# We merge everything found in the tmp_blocks folder for this ensemble.
zone <- "zone1"
ens  <- Sys.getenv("ENS")
if (ens == "") stop("Environment variable ENS is not set (e.g., ENS=ens01).")

hmd_base_dir <- "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output"

block_dir <- file.path(hmd_base_dir, zone, "monthly_sum", "tmp_blocks", ens)
out_file  <- file.path(hmd_base_dir, zone, "monthly_sum",
                       paste0("hmd_monthlysum_", zone, "_", ens, "_1993-2014.nc"))
dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)

# This block finds all block files and sorts them by their latitude start index.
# Sorting is important to ensure blocks are written into the correct location in the final file.
block_files <- list.files(block_dir, pattern = "_block_\\d+\\.nc$", full.names = TRUE)
if (length(block_files) == 0) stop("No block files found: ", block_dir)

block_files <- block_files[order(as.numeric(gsub(".*_block_(\\d+)\\.nc", "\\1", block_files)))]
cat("[INFO] Found ", length(block_files), " block files\n")

# This block reads lon/time metadata from the first block.
# We also collect the full latitude vector by scanning all blocks (some blocks may be flipped).
nc0 <- nc_open(block_files[1])
lon <- ncvar_get(nc0, "lon")
time <- ncvar_get(nc0, "time")
time_units <- ncatt_get(nc0, "time", "units")$value
nc_close(nc0)

lat_all <- c()
for (f in block_files) {
  nc <- nc_open(f)
  lat_b <- ncvar_get(nc, "lat")
  if (lat_b[1] > lat_b[length(lat_b)]) lat_b <- rev(lat_b)
  lat_all <- c(lat_all, lat_b)
  nc_close(nc)
}
lat_all <- sort(unique(lat_all))

# This block defines the output NetCDF dimensions and creates the final file.
# The variable name matches the one written in the block files: hmd_monthly_sum.
lon_dim  <- ncdim_def("lon", "degrees_east", lon)
lat_dim  <- ncdim_def("lat", "degrees_north", lat_all)
time_dim <- ncdim_def("time", time_units, time, unlim = FALSE)

var_def <- ncvar_def("hmd_monthly_sum", "HMD (monthly sum of daily HMD)",
                     dim = list(lon_dim, lat_dim, time_dim),
                     missval = NA_real_, prec = "float")

nc_out <- nc_create(out_file, vars = list(var_def))

# This block loops through each block file, aligns latitude orientation if needed,
# and writes the data into the correct latitude indices in the final output.
for (f in block_files) {
  cat("[INFO] Merging: ", f, "\n")
  nc_in <- nc_open(f)
  lat_b <- ncvar_get(nc_in, "lat")
  dat   <- ncvar_get(nc_in, "hmd_monthly_sum")
  nc_close(nc_in)
  
  if (lat_b[1] > lat_b[length(lat_b)]) {
    lat_b <- rev(lat_b)
    dat <- dat[, length(lat_b):1, , drop = FALSE]
  }
  
  lat_idx <- match(lat_b, lat_all)
  ncvar_put(nc_out, "hmd_monthly_sum", dat,
            start = c(1, min(lat_idx), 1),
            count = c(length(lon), length(lat_idx), length(time)))
}

nc_close(nc_out)
cat("[SUCCESS] Wrote: ", out_file, "\n")
