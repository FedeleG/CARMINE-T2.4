#!/usr/bin/env Rscript
######################### Calculation of HMD (spatial) ###############################


########################################
# Setup
########################################

# Load packages
library(ncdf4)
library(heatwaveR)
library(lubridate)
library(rslurm)

# Set working directory
setwd("/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/")

# Define settings
zone <- "zone1" # Adjust to corresponding zone
ensemble <- "ens01" # Define ensemble

# Define paths
file_dir <- file.path("/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/Tx", zone)

out_dir <- file.path("/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output",
                     zone, "tmp_blocks", ensemble)

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Input filename
input_nc <- file.path( file_dir,
  paste0("bc_medewsa_cal_tx_", zone, "_", ensemble, "_daily_1993-2014.nc")
)

# Output filename
output_nc <- file.path(
  "/work/bb1478/Celine/MedEWSa/hazard_analysis/HMD/data/hmd_output", zone,
  paste0("hmd_", zone, "_", ensemble, "_medewsa_daily_1993-2014")
)


########################################
# HMD Function with NA handling
########################################

hwmd <- function(x, time.ts, climatology = NULL, smoothWindow = 15,
                 deseas = TRUE, seas.vary = TRUE,
                 minDuration = 3, maxGap = 1,
                 detrend.var = NULL, tresh = 0.9,
                 raw = FALSE, time.frame = NULL,
                 only.climatology = FALSE, joinAcrossGaps = TRUE,
                 only.hmd = TRUE, cumulate = c("none", "monthly", "weekly"),
                 cum.fct = sum) {
  
  if (!is.numeric(x)) stop("x needs to be numeric!")
  nn <- length(x)
  cumulate <- match.arg(cumulate)
  smoothWindow.half <- (smoothWindow - 1) / 2
  if (smoothWindow.half != floor(smoothWindow.half))
    stop("SmoothWindow must be an odd number!")
  
  # Skip completely NA series
  if (all(is.na(x))) return(rep(NA, nn))
  
  # Climatology setup
  if (is.null(climatology)) {
    use.clim <- FALSE
    climatology <- as.character(c(time.ts[1], time.ts[nn]))
    refindex <- seq_along(x)
  } else {
    use.clim <- TRUE
    if (is.null(time.frame)) {
      start.ref <- which(time.ts == climatology[1])
      end.ref <- which(time.ts == climatology[2])
    } else {
      start.ref <- which(time.frame$year == climatology[1])[1]
      end.ref <- which(time.frame$year == climatology[2])[1]
    }
    refindex <- start.ref:end.ref
  }
  
  # Year sequence for seasonal climatology
  if (is.null(time.frame)) {
    nyears <- length(unique(lubridate::year(time.ts)))
  } else {
    nyears <- length(unique(time.frame$year))
  }
  day.seq <- rep(1:365, nyears)
  
  # Deseasonalization
  if (is.null(time.frame)) {
    seas.frame <- data.frame(t = time.ts, temp = x)
    seas.clim <- heatwaveR::ts2clm(seas.frame, climatologyPeriod = climatology)
    if (deseas) {
      x <- seas.clim$temp - seas.clim$seas
      time.ts <- seas.clim$t
      if (!is.null(detrend.var)) x <- residuals(lm(x ~ detrend.var))
      seas.frame <- data.frame(t = time.ts, temp = x)
      seas.clim <- heatwaveR::ts2clm(seas.frame, climatologyPeriod = climatology)
    }
  }
  if (!is.null(detrend.var)) x <- residuals(lm(x ~ detrend.var))
  
  # Remove February 29th
  index29 <- if (is.null(time.frame)) {
    which(lubridate::month(time.ts) == 2 & lubridate::day(time.ts) == 29)
  } else numeric()
  
  if (length(index29) != 0) {
    x <- x[-index29]
    time.ts <- time.ts[-index29]
    if (is.null(time.frame)) seas.clim <- seas.clim[-index29, ]
    nn <- length(x)
    refindex <- refindex[!refindex %in% index29]
  }
  
  # Climatology / quantiles
  if (seas.vary) {
    quant.fct <- function(day.use) {
      day.index <- intersect(which(day.seq == day.use), refindex)
      day.index14 <- unlist(lapply(day.index, function(d) {
        (d - smoothWindow.half):(d + smoothWindow.half)
      }))
      day.index14 <- day.index14[day.index14 >= 1 & day.index14 <= nn]
      quantile(x[day.index14], c(0.1, 0.25, 0.5, 0.75, tresh), na.rm = TRUE)
    }
    
    clim.fct <- function(xclim) rep(xclim, nyears)
    day.seq.use <- if (is.null(time.frame)) 1:365 else 1:sum(time.frame$year == time.frame$year[1])
    quant.seq <- sapply(day.seq.use, quant.fct)
    quant.clims <- apply(quant.seq, 1, clim.fct)
  } else {
    quant.seq <- quantile(x[refindex], c(0.1, 0.25, 0.5, 0.75, tresh), na.rm = TRUE)
    quant.clims <- t(as.matrix(quant.seq))
  }
  
  if (only.climatology) return(quant.clims)
  
  # HMD calculation
  hmd <- (x - quant.clims[,2]) / (quant.clims[,4] - quant.clims[,2])
  if (raw) return(hmd)
  
  seas.clim$seas <- rep(0, length(x))
  seas.clim$thresh <- quant.clims[,5]
  
  hmd.detect.frame <- heatwaveR::detect_event(seas.clim, minDuration = minDuration, maxGap = maxGap)
  event_info <- hmd.detect.frame$event
  
  hw.detect <- rep(0, length(x))
  for (i in seq_len(nrow(event_info))) {
    hw.detect[event_info$index_start[i]:event_info$index_end[i]] <- 1
  }
  
  if (cumulate != "none") {
    time.char <- paste0(lubridate::year(time.ts), "-", lubridate::month(time.ts))
    time.fac <- factor(time.char, levels = unique(time.char))
    hmd.cum <- tapply(hmd * hw.detect, time.fac, cum.fct)
  } else {
    hmd.cum <- hmd * hw.detect
  }
  
  if (only.hmd) return(hmd.cum)
  else {
    list(
      hmd.cum = hmd.cum,
      hmd = hmd,
      climatology = seas.clim$seas,
      quant = quant.clims,
      event = hw.detect,
      hw.metric = hmd.detect.frame
    )
  }
}


########################################
# Read NetCDF metadata
########################################

nc_in <- nc_open(input_nc)
lon <- ncvar_get(nc_in, "lon")
lat <- ncvar_get(nc_in, "lat")
time_nc <- ncvar_get(nc_in, "time")
time_units <- ncatt_get(nc_in, "time", "units")$value
nc_close(nc_in)

# Convert time units to Dates
if (grepl("^days since", time_units)) {
  origin_str <- sub("^days since ", "", time_units)
  origin <- as.Date(substr(origin_str, 1, 10))
  dates <- origin + time_nc
} else stop("Unknown time unit")

nlon <- length(lon)
nlat <- length(lat)
block_size <- 40
lat_blocks <- seq(1, nlat, by = block_size)


########################################
# Worker function for block processing
########################################
process_block <- function(lat_start) {
  
  # Start snowfall cluster
  library(snowfall)
  sfSetMaxCPUs(number= 120)
  sfInit(parallel = TRUE, cpus = 40, type = "SOCK")
  
  # Load required packages on all workers
  sfLibrary(heatwaveR)
  sfLibrary(lubridate)

  # Ensure output directory exists on compute node
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
    
  # Determine latitude range for this block
  lat_end <- min(lat_start + block_size - 1, length(lat))

  # Read the block of temperature data
  nc_in <- nc_open(input_nc)
  tx_block <- ncvar_get(nc_in, "tx",
                        start = c(1, lat_start, 1),
                        count = c(nlon, lat_end - lat_start + 1, length(time_nc)))
  nc_close(nc_in)

  # Reorder dimensions to [lat, lon, time]
  tx_block <- aperm(tx_block, c(2,1,3))
  hmd_block <- array(NA, dim = c(lat_end - lat_start + 1, nlon, length(dates)))

  # Export required variables to workers
  sfExport("hwmd", "dates", "tx_block", "lat_end", "lat_start", "nlon")

  # Prepare coordinate indices for parallel execution
  coords <- expand.grid(i = 1:(lat_end - lat_start + 1), j = 1:nlon)

  # Compute HMD for each grid cell in parallel
  hmd_list <- sfLapply(1:nrow(coords), function(idx) {
    i <- coords$i[idx]
    j <- coords$j[idx]
    x <- tx_block[i, j, ]
    if (all(is.na(x))) return(rep(NA, length(dates)))
    hwmd(x, dates,
         climatology = c(min(dates), max(dates)),
         cumulate = "none",
         smoothWindow = 5,
         only.hmd = TRUE)
  })

  # Reassemble the results into the block array
  for (k in seq_along(hmd_list)) {
    i <- coords$i[k]
    j <- coords$j[k]
    hmd_block[i, j, ] <- hmd_list[[k]]
  }

  # Stop snowfall cluster
  sfStop()

  # Write temporary NetCDF for this block
  tmp_nc <- file.path(out_dir, paste0(basename(output_nc), "_block_", lat_start, ".nc"))
  lon_dim  <- ncdim_def("lon", "degrees_east", lon)
  lat_dim  <- ncdim_def("lat", "degrees_north", lat[lat_start:lat_end])
  time_dim <- ncdim_def("time", time_units, as.numeric(time_nc))
  hmd_var <- ncvar_def("hmd", "unitless",
                       dim = list(lon_dim, lat_dim, time_dim),
                       missval = NA, prec = "float")

  nc_out <- nc_create(tmp_nc, vars = list(hmd_var))
  ncvar_put(nc_out, "hmd", aperm(hmd_block, c(2,1,3)))
  nc_close(nc_out)

  return(paste0("Block ", lat_start, "-", lat_end, " finished"))
}


########################################
# Submit block processing jobs with Slurm
########################################

# Define only the varying parameter
params <- data.frame(lat_start = lat_blocks)

# Slurm options
sopt <- list(
  partition = 'compute',
  account   = 'bb1201',
  'mail-type' = 'ALL',
  mem = "0G",
  constraint = "1024G",
  time = "8:00:00"
)

# Generate a unique job name
jobname <- paste0("HMD-", as.numeric(Sys.time()))
cat("Jobname created:", jobname, "\n")

# Submit jobs
sjob <- slurm_apply(
  f = process_block,          # Worker function
  params = params,            # Only lat_start varies
  jobname = jobname,
  global_objects = c("input_nc","output_nc","out_dir",
                     "lon","lat","time_nc","time_units",
                     "dates","block_size","nlon","hwmd"),
  nodes = nrow(params),
  cpus_per_node = 1,
  slurm_options = sopt,
  submit = TRUE,
  preschedule_cores = FALSE
)

cat("[INFO] Slurm job submitted\n")