
#!/usr/bin/env Rscript
######################### 07_csi_pca_blocks.R #########################
# This script computes the CSI (Combined Stress Index) using PCA between monthly HMD and monthly SPEI.
# For each grid point, we standardize HMD and SPEI over time (correlation-based PCA), then take PC1 as CSI.
# Additionally, we define CSI extreme months as CSI > 90th percentile (per grid cell, over the full time series).
# To handle large grids, the computation is done by latitude blocks and written as temporary NetCDF files.

library(ncdf4)
library(rslurm)

# This block defines the run settings.
# We keep the zone fixed (Style 1) and read the ensemble name from the environment.
zone <- "zone1"
ens  <- Sys.getenv("ENS")
if (ens == "") stop("Environment variable ENS is not set (e.g., ENS=ens01).")

# This block defines the input files for CSI.
# HMD is the monthly sum file produced by the other scripts.
# SPEI is the monthly file already produced by your SPEI script.
hmd_base_dir <- "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output"

hmd_mon_nc <- file.path(
  hmd_base_dir, zone, "monthly_sum",
  paste0("hmd_monthlysum_", zone, "_", ens, "_1993-2014.nc")
)

spei_nc <- file.path(
  "/work/bb1478/Celine/MedEWSa/hazard_analysis/SPEI/data/SPEI",
  paste0("spei_", ens, "_monthly_1993-2014.nc")
)

# This block sets up output folders for temporary CSI blocks.
# Later, a merge script will combine them into one final CSI NetCDF.
out_dir <- file.path(hmd_base_dir, zone, "CSI", "tmp_blocks", ens)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_prefix <- file.path(
  hmd_base_dir, zone, "CSI",
  paste0("csi_", zone, "_", ens, "_monthly_1993-2014")
)
dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)

# This block reads metadata from both inputs and checks they match.
# We stop immediately if lon/lat/time do not align, because CSI would be wrong otherwise.
nc_h <- nc_open(hmd_mon_nc)
lon <- ncvar_get(nc_h, "lon")
lat <- ncvar_get(nc_h, "lat")
time <- ncvar_get(nc_h, "time")
tu_h <- ncatt_get(nc_h, "time", "units")$value
nc_close(nc_h)

nc_s <- nc_open(spei_nc)
lon_s <- ncvar_get(nc_s, "lon")
lat_s <- ncvar_get(nc_s, "lat")
time_s <- ncvar_get(nc_s, "time")
tu_s <- ncatt_get(nc_s, "time", "units")$value
nc_close(nc_s)

stopifnot(length(lon) == length(lon_s), all(lon == lon_s))
stopifnot(length(lat) == length(lat_s), all(lat == lat_s))
stopifnot(length(time) == length(time_s), all(time == time_s))
stopifnot(tu_h == tu_s)

nlon <- length(lon)
nlat <- length(lat)
ntime <- length(time)

# This helper function computes CSI for one grid cell.
# We standardize the two time series (so PCA is based on correlation).
# We then take PC1 and enforce a stable sign so CSI increases with heat (pca_a > 0).
pca_csi_cell <- function(h, s) {
  ok <- is.finite(h) & is.finite(s)
  if (sum(ok) < 10) return(list(csi = rep(NA_real_, length(h)), a = NA_real_, b = NA_real_))
  
  H <- as.numeric(scale(h[ok]))
  S <- as.numeric(scale(s[ok]))
  
  p <- prcomp(cbind(H, S), center = FALSE, scale. = FALSE)
  a <- p$rotation[1,1]
  b <- p$rotation[2,1]
  csi_ok <- a * H + b * S
  
  if (is.finite(a) && a < 0) { a <- -a; b <- -b; csi_ok <- -csi_ok }
  
  csi <- rep(NA_real_, length(h))
  csi[ok] <- csi_ok
  list(csi = csi, a = a, b = b)
}

# This helper computes a per-grid-cell percentile threshold on CSI.
# Returns NA if not enough finite values.
csi_threshold <- function(csi, p = 0.9, min_n = 10) {
  ok <- is.finite(csi)
  if (sum(ok) < min_n) return(NA_real_)
  as.numeric(quantile(csi[ok], probs = p, na.rm = TRUE, names = FALSE, type = 7))
}

# This block defines how we split the spatial grid.
# Each Slurm job handles one latitude block, keeping memory use manageable.
block_size <- 40
lat_blocks <- seq(1, nlat, by = block_size)

# This is the worker function for CSI computation.
# It reads monthly HMD and monthly SPEI for one latitude block, computes CSI per grid cell,
# computes CSI extreme mask (CSI > p90), and writes everything to a temporary NetCDF file.
process_block <- function(lat_start) {
  library(ncdf4)
  
  lat_end <- min(lat_start + block_size - 1, length(lat))
  nlat_b <- lat_end - lat_start + 1
  
  nc_h <- nc_open(hmd_mon_nc)
  hmd_b <- ncvar_get(nc_h, "hmd_monthly_sum",
                     start = c(1, lat_start, 1),
                     count = c(nlon, nlat_b, ntime))
  nc_close(nc_h)
  
  nc_s <- nc_open(spei_nc)
  spei_b <- ncvar_get(nc_s, "spei",
                      start = c(1, lat_start, 1),
                      count = c(nlon, nlat_b, ntime))
  nc_close(nc_s)
  
  csi_b     <- array(NA_real_, dim = c(nlon, nlat_b, ntime))
  a_b       <- array(NA_real_, dim = c(nlon, nlat_b))
  b_b       <- array(NA_real_, dim = c(nlon, nlat_b))
  csi_p90_b <- array(NA_real_, dim = c(nlon, nlat_b))      # threshold per cell
  csi_ext_b <- array(0L,      dim = c(nlon, nlat_b, ntime)) # 0/1 extreme mask
  
  for (j in seq_len(nlat_b)) {
    for (i in seq_len(nlon)) {
      h <- hmd_b[i, j, ]
      s <- spei_b[i, j, ]
      
      res <- pca_csi_cell(h, s)
      csi_b[i, j, ] <- res$csi
      a_b[i, j] <- res$a
      b_b[i, j] <- res$b
      
      thr <- csi_threshold(res$csi, p = 0.9, min_n = 10)
      csi_p90_b[i, j] <- thr
      
      if (is.finite(thr)) {
        okc <- is.finite(res$csi)
        csi_ext_b[i, j, okc] <- as.integer(res$csi[okc] > thr)
      }
    }
  }
  
  tmp_nc <- file.path(out_dir, paste0(basename(out_prefix), "_block_", lat_start, ".nc"))
  
  lon_dim  <- ncdim_def("lon", "degrees_east", lon)
  lat_dim  <- ncdim_def("lat", "degrees_north", lat[lat_start:lat_end])
  time_dim <- ncdim_def("time", tu_h, time, unlim = FALSE)
  
  v_csi <- ncvar_def("csi", "1", list(lon_dim, lat_dim, time_dim),
                     missval = NA_real_, prec = "float")
  v_a   <- ncvar_def("pca_a", "1", list(lon_dim, lat_dim),
                     missval = NA_real_, prec = "float")
  v_b   <- ncvar_def("pca_b", "1", list(lon_dim, lat_dim),
                     missval = NA_real_, prec = "float")
  v_ext <- ncvar_def("csi_extreme_p90", "1", list(lon_dim, lat_dim, time_dim),
                     missval = NA_real_, prec = "byte")
  v_thr <- ncvar_def("csi_p90", "1", list(lon_dim, lat_dim),
                     missval = NA_real_, prec = "float")
  
  nc_out <- nc_create(tmp_nc, vars = list(v_csi, v_a, v_b, v_ext, v_thr))
  
  ncvar_put(nc_out, "csi", csi_b)
  ncvar_put(nc_out, "pca_a", a_b)
  ncvar_put(nc_out, "pca_b", b_b)
  ncvar_put(nc_out, "csi_extreme_p90", csi_ext_b)
  ncvar_put(nc_out, "csi_p90", csi_p90_b)
  
  ncatt_put(nc_out, 0, "title", "CSI = PC1 of PCA(HMD_monthly_sum, SPEI-1) based on correlation")
  ncatt_put(nc_out, 0, "note", "Variables standardized per grid cell over time; PC1 sign enforced so pca_a>0.")
  ncatt_put(nc_out, 0, "note_extremes", "Extreme months defined as CSI > 90th percentile per grid cell (computed over full CSI time series).")
  ncatt_put(nc_out, 0, "zone", zone)
  ncatt_put(nc_out, 0, "ensemble", ens)
  nc_close(nc_out)
  
  paste0("CSI block ", lat_start, "-", lat_end, " done")
}

# This block submits the CSI worker function to Slurm using rslurm.
# Each latitude block becomes one job. After all jobs are complete, run the merge script (08).
params <- data.frame(lat_start = lat_blocks)

sopt <- list(
  partition = "compute",
  account   = "bb1201",
  `mail-type` = "FAIL",
  time = "2:00:00",
  mem  = "0G"
)

jobname <- paste0("CSI-", ens, "-", as.numeric(Sys.time()))
cat("[INFO] Submitting job: ", jobname, "\n")

sjob <- slurm_apply(
  f = process_block,
  params = params,
  jobname = jobname,
  global_objects = c("zone","ens","hmd_mon_nc","spei_nc","out_dir","out_prefix",
                     "lon","lat","time","tu_h","nlon","nlat","ntime","block_size",
                     "pca_csi_cell","csi_threshold"),
  nodes = nrow(params),
  cpus_per_node = 1,
  slurm_options = sopt,
  submit = TRUE,
  preschedule_cores = FALSE
)

cat("[INFO] Submitted CSI jobs for ", ens, "\n")
