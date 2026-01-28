#!/usr/bin/env Rscript
######################### Merging of HMD Blocks ###############################


########################################
# Setup
########################################

# Load required packages
library(ncdf4)

# Set working directory
setwd("/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/")

# Define paths
ensemble <- "ens01" # Define ensemble
file_dir  <- file.path("/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output/zone1/tmp_blocks", ensemble) # Adjust to corresponding zone
out_dir   <- "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output/zone1" # Adjust to corresponding zone
output_file <- file.path(out_dir, paste0("hmd_zone1_", ensemble, "_medewsa_daily_1993-2014.nc")) # Adjust to corresponding zone

# Find all block files
block_files <- list.files(file_dir, pattern = "_block_\\d+\\.nc$", full.names = TRUE)
if (length(block_files) == 0) stop("No block files found in folder: ", file_dir)

# Sort by latitude start index in filename
block_files <- block_files[order(as.numeric(gsub(".*_block_(\\d+)\\.nc", "\\1", block_files)))]
cat("[INFO] Found block files:\n")
print(block_files)


## Read metadata from first block
nc0 <- nc_open(block_files[1]) 
lon <- ncvar_get(nc0, "lon") 
time <- ncvar_get(nc0, "time") 
time_units <- ncatt_get(nc0, "time", "units")$value 
nlon <- length(lon) 
ntime <- length(time) 
nc_close(nc0)


## Collect all latitude values
lat_all <- c()
for (f in block_files) {
  nc <- nc_open(f)
  lat_block <- ncvar_get(nc, "lat")
  # Flip latitude if descending
  if (lat_block[1] > lat_block[length(lat_block)]) lat_block <- rev(lat_block)
  lat_all <- c(lat_all, lat_block)
  nc_close(nc)
}
lat_all <- sort(unique(lat_all))
nlat <- length(lat_all)


## Define dimensions and create output NetCDF
lon_dim  <- ncdim_def("lon", "degrees_east", lon)
lat_dim  <- ncdim_def("lat", "degrees_north", lat_all)
time_dim <- ncdim_def("time", time_units, time)
hmd_var  <- ncvar_def("hmd", "unitless", list(lon_dim, lat_dim, time_dim),
                      missval = NA, prec = "float")

cat("[INFO] Creating final NetCDF file:", output_file, "\n")
nc_out <- nc_create(output_file, list(hmd_var))


########################################
# Merge each block into the final file
########################################

for (f in block_files) {
  cat("[INFO] Processing block:", f, "\n")
  
  nc_in <- nc_open(f)
  lat_block <- ncvar_get(nc_in, "lat")
  hmd_block <- ncvar_get(nc_in, "hmd")  # [lon, lat, time]
  nc_close(nc_in)
  
  # Flip block if latitude is descending
  if (lat_block[1] > lat_block[length(lat_block)]) {
    lat_block <- rev(lat_block)
    hmd_block <- hmd_block[, length(lat_block):1, , drop = FALSE]
  }
  
  # Find position in full latitude grid
  lat_idx <- match(lat_block, lat_all)
  
  # Write the block
  ncvar_put(nc_out, "hmd", hmd_block,
            start = c(1, min(lat_idx), 1),
            count = c(length(lon), length(lat_idx), ntime))
}

# Close final NetCDF
nc_close(nc_out)
cat("[SUCCESS] All blocks merged successfully!\n")