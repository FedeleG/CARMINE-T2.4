#!/usr/bin/env Rscript
######################### 08_merge_csi_blocks.R #########################
# This script merges the temporary CSI blocks created by 07_csi_pca_blocks.R.
# Each block contains CSI for a subset of latitude rows plus the PCA coefficients (a,b),
# and additionally a CSI extreme mask (CSI > p90) plus the per-cell p90 threshold.
# The output is one final CSI NetCDF file for the chosen zone and ensemble.

library(ncdf4)

# This block defines run settings and block locations.
# ENS comes from the environment, consistent with the other scripts.
zone <- "zone1"
ens  <- Sys.getenv("ENS")
if (ens == "") stop("Environment variable ENS is not set (e.g., ENS=ens01).")

hmd_base_dir <- "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output"

block_dir <- file.path(hmd_base_dir, zone, "CSI", "tmp_blocks", ens)
out_file  <- file.path(hmd_base_dir, zone, "CSI",
                       paste0("csi_", zone, "_", ens, "_monthly_1993-2014.nc"))
dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)

# This block collects all CSI block files and sorts them by latitude start index.
# Sorting ensures the merge writes blocks into the correct latitude range.
block_files <- list.files(block_dir, pattern = "_block_\\d+\\.nc$", full.names = TRUE)
if (length(block_files) == 0) stop("No block files found: ", block_dir)

block_files <- block_files[order(as.numeric(gsub(".*_block_(\\d+)\\.nc", "\\1", block_files)))]
cat("[INFO] Found ", length(block_files), " CSI block files\n")

# This block reads lon/time metadata from the first block and builds the full latitude vector.
# We also handle cases where individual blocks store latitude in descending order.
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

# This block creates the final output NetCDF and defines the variables:
# - CSI time series
# - PCA loadings a and b
# - CSI extreme mask (CSI > p90)
# - p90 threshold used per grid cell
lon_dim  <- ncdim_def("lon", "degrees_east", lon)
lat_dim  <- ncdim_def("lat", "degrees_north", lat_all)
time_dim <- ncdim_def("time", time_units, time, unlim = FALSE)

v_csi <- ncvar_def("csi", "1", list(lon_dim, lat_dim, time_dim), missval = NA_real_, prec = "float")
v_a   <- ncvar_def("pca_a", "1", list(lon_dim, lat_dim), missval = NA_real_, prec = "float")
v_b   <- ncvar_def("pca_b", "1", list(lon_dim, lat_dim), missval = NA_real_, prec = "float")
v_ext <- ncvar_def("csi_extreme_p90", "1", list(lon_dim, lat_dim, time_dim), missval = NA_real_, prec = "byte")
v_thr <- ncvar_def("csi_p90", "1", list(lon_dim, lat_dim), missval = NA_real_, prec = "float")

nc_out <- nc_create(out_file, vars = list(v_csi, v_a, v_b, v_ext, v_thr))

# This block loops over each temporary file and writes into the output.
for (f in block_files) {
  cat("[INFO] Merging: ", f, "\n")
  nc_in <- nc_open(f)
  lat_b <- ncvar_get(nc_in, "lat")
  csi_b <- ncvar_get(nc_in, "csi")
  a_b   <- ncvar_get(nc_in, "pca_a")
  b_b   <- ncvar_get(nc_in, "pca_b")
  ext_b <- ncvar_get(nc_in, "csi_extreme_p90")
  thr_b <- ncvar_get(nc_in, "csi_p90")
  nc_close(nc_in)
  
  if (lat_b[1] > lat_b[length(lat_b)]) {
    lat_b <- rev(lat_b)
    csi_b <- csi_b[, length(lat_b):1, , drop = FALSE]
    a_b   <- a_b[,   length(lat_b):1, drop = FALSE]
    b_b   <- b_b[,   length(lat_b):1, drop = FALSE]
    ext_b <- ext_b[, length(lat_b):1, , drop = FALSE]
    thr_b <- thr_b[, length(lat_b):1, drop = FALSE]
  }
  
  lat_idx <- match(lat_b, lat_all)
  
  ncvar_put(nc_out, "csi", csi_b,
            start = c(1, min(lat_idx), 1),
            count = c(length(lon), length(lat_idx), length(time)))
  
  ncvar_put(nc_out, "pca_a", a_b,
            start = c(1, min(lat_idx)),
            count = c(length(lon), length(lat_idx)))
  
  ncvar_put(nc_out, "pca_b", b_b,
            start = c(1, min(lat_idx)),
            count = c(length(lon), length(lat_idx)))
  
  ncvar_put(nc_out, "csi_extreme_p90", ext_b,
            start = c(1, min(lat_idx), 1),
            count = c(length(lon), length(lat_idx), length(time)))
  
  ncvar_put(nc_out, "csi_p90", thr_b,
            start = c(1, min(lat_idx)),
            count = c(length(lon), length(lat_idx)))
}

nc_close(nc_out)
cat("[SUCCESS] Wrote: ", out_file, "\n")
