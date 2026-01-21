import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
from matplotlib.ticker import MaxNLocator

# =========================================================
# CONFIGURATION
# =========================================================

CONFIG = {
    "era5-2km": dict(start_year=1989, end_year=2018),
    "eu-cordex-11": dict(
        hist_start_year=1981, hist_end_year=2010,
        proj_start_year=2021, proj_end_year=2050,
        scenario="rcp26"
    ),
    "emo": dict(start_year=1991, end_year=2020),
    "cerra": dict(start_year=1981, end_year=2010),
    "eobs": dict(start_year=1981, end_year=2010)
}

FUA_MAPPING = {
    "Prague": "Praha",
    "Leipzig": "Leipzig",
    "Funen-Odense": "Odense",
    "Athens": "Athina",
    "Barcelona": "Barcelona",
    "Bologna": "Bologna",
    "Brasov": "Brasov",
    "Birmingham": "West Midlands urban area"
}

# =========================================================
# SELECT DATASET AND PILOT AREA --> CONFIGURE HERE <--
# =========================================================

dataset_name = "emo"
pilotarea = "Barcelona"
CSA = FUA_MAPPING.get(pilotarea, pilotarea)
indicator = "rr"
var_name = "RR" #indicator.upper()
cmap = "Blues"

# =========================================================
# PARAMS EXTRACTION
# =========================================================

if dataset_name not in CONFIG:
    raise ValueError(f"Dataset '{dataset_name}' not recognized. Available: {list(CONFIG.keys())}")

params = CONFIG[dataset_name]
start_year       = params.get("start_year")
end_year         = params.get("end_year")
hist_start_year  = params.get("hist_start_year")
hist_end_year    = params.get("hist_end_year")
proj_start_year  = params.get("proj_start_year")
proj_end_year    = params.get("proj_end_year")
scenario         = params.get("scenario")

if dataset_name == "eu-cordex-11":
    start_year = hist_start_year
    end_year = proj_end_year

# =========================================================
# PATHS
# =========================================================

base = f"{pilotarea}/INDICATORS"
FUA_SHP = "shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"

# =========================================================
# FILENAME BUILDER
# =========================================================

def construct_filename(pilotarea, dataset_name, indicator, start_year, end_year,
                      hist_start_year=None, hist_end_year=None,
                      proj_start_year=None, proj_end_year=None,
                      scenario=None):
    pilot_lower = pilotarea.lower()
    if dataset_name in ["cerra", "eobs"]:
        return f"{pilot_lower}_{dataset_name}_{indicator}_eu_{start_year}_{end_year}.nc"
    elif dataset_name == "era5-2km":
        return f"{pilot_lower}_{dataset_name}_{indicator}_{start_year}{end_year}.nc"
    elif dataset_name == "emo":
        return f"{pilot_lower}_{dataset_name}_{indicator}{start_year}{end_year}.nc"
    elif dataset_name == "eu-cordex-11":
        return (
            f"{pilot_lower}_{dataset_name}_{indicator}_"
            f"{hist_start_year}-{hist_end_year}_"
            f"{proj_start_year}-{proj_end_year}_{scenario}.nc"
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

# =========================================================
# TITLE + SAVEFIG BUILDER
# =========================================================

def get_title_and_savefig(pilotarea, dataset_name, var_name, start_year, end_year,
                          hist_start_year=None, hist_end_year=None,
                          proj_start_year=None, proj_end_year=None,
                          scenario=None, suffix=""):

    if dataset_name == "eu-cordex-11":
        longitude = "lon"
        latitude = "lat"
        title = (
            f"Average Map of {var_name} from {hist_start_year}-{hist_end_year} & "
            f"{proj_start_year}-{proj_end_year}\n"
            f"over {pilotarea} in {dataset_name} ({scenario})"
        )
        savefig_name = (
            f"{pilotarea}_{var_name}_{dataset_name}_"
            f"{hist_start_year}-{hist_end_year}_"
            f"{proj_start_year}-{proj_end_year}_{scenario}_{suffix}.png"
        )
    elif dataset_name in ["era5-2km", "emo"]:
        longitude = "lon"
        latitude = "lat"
        title = (
            f"Average Map of {var_name} from {start_year} to {end_year}\n"
            f"over {pilotarea} in {dataset_name}"
        )
        savefig_name = f"{pilotarea}_{var_name}_{dataset_name}_{start_year}-{end_year}_{suffix}.png"
    elif dataset_name in ["cerra", "eobs"]:
        longitude = "longitude"
        latitude = "latitude"
        title = (
            f"Average Map of {var_name} from {start_year} to {end_year}\n"
            f"over {pilotarea} in {dataset_name}"
        )
        savefig_name = f"{pilotarea}_{var_name}_{dataset_name}_{start_year}-{end_year}_{suffix}.png"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return title, savefig_name, longitude, latitude

# =========================================================
# LOAD DATASET
# =========================================================

filename = construct_filename(
    pilotarea, dataset_name, indicator,
    start_year, end_year,
    hist_start_year, hist_end_year,
    proj_start_year, proj_end_year,
    scenario
)

file = f"{base}/{filename}"

if not os.path.exists(file):
    raise FileNotFoundError(f"File does not exist: {file}")

print(f"Loading: {file}")
ds = xr.open_dataset(file, decode_timedelta=True)
print(f"Using FUA_NAME: '{CSA}' for pilotarea '{pilotarea}'")
if pilotarea == "Birmingham":
    warnings.warn("Birmingham uses approximate FUA_NAME 'West Midlands urban area'")

# =========================================================
# TIME UNIT CONVERSION
# =========================================================

def convert_to_days(arr):
    unit = np.datetime_data(arr.dtype)[0]
    to_days = {"ns":1/(1e9*3600*24), "us":1/(1e6*3600*24),
               "ms":1/(1e3*3600*24), "s":1/(3600*24),
               "m":1/(60*24), "h":1/24, "D":1.0}
    if unit not in to_days:
        raise ValueError(f"Unsupported timedelta unit: {unit}")
    return arr.astype("float64") * to_days[unit]

# =========================================================
# FUNCTION TO PLOT ONE VARIABLE
# =========================================================

def plot_var(ds, var_name_plot, suffix):
    title, savefig_name, longitude, latitude = get_title_and_savefig(
        pilotarea, dataset_name, var_name_plot, start_year, end_year,
        hist_start_year, hist_end_year, proj_start_year, proj_end_year,
        scenario, suffix=suffix
    )

    var = ds[var_name_plot]
    lon_vals = ds[longitude].values
    lat_vals = ds[latitude].values

    # Convert 1D lon/lat to 2D if needed
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
    else:
        lon_2d, lat_2d = lon_vals, lat_vals

    if np.issubdtype(var.dtype, np.timedelta64):
        data = convert_to_days(var.data)
    else:
        data = var.values
    data = np.squeeze(data)

    # Compute extent
    mask = ~np.isnan(data)
    lon_valid = np.where(lon_2d[mask] > 180, lon_2d[mask] - 360, lon_2d[mask])
    lat_valid = lat_2d[mask]
    lon_min, lon_max = lon_valid.min(), lon_valid.max()
    lat_min, lat_max = lat_valid.min(), lat_valid.max()
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1

    # Load FUA
    try:
        fua_gdf = gpd.read_file(FUA_SHP).to_crs(epsg=4326)
        fua_shape = fua_gdf[fua_gdf["FUA_NAME"] == CSA]
        fua_found = not fua_shape.empty
    except Exception as e:
        print(f"Warning: Could not load FUA shapefile: {e}")
        fua_shape = None
        fua_found = False

    # Plot
    plt.figure(figsize=(12,10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Gridlines with max 4 ticks
    gl = ax.gridlines(draw_labels=True, dms=False, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter(number_format='.1f')
    gl.yformatter = LatitudeFormatter(number_format='.1f')
    gl.xlocator = MaxNLocator(4)
    gl.ylocator = MaxNLocator(4)

    # Plot data
    lon_plot = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
    im = ax.pcolormesh(
        lon_plot, lat_2d, data,
        cmap=cmap,
        vmin=np.nanpercentile(data,5),
        vmax=np.nanpercentile(data,95),
        transform=ccrs.PlateCarree()
    )

    # Plot FUA boundary
    if fua_found:
        fua_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=1, transform=ccrs.PlateCarree())

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(f"{var_name_plot} (days)", fontsize=12)

    # Set extent
    ax.set_extent([lon_min-lon_pad, lon_max+lon_pad,
                   lat_min-lat_pad, lat_max+lat_pad], crs=ccrs.PlateCarree())

    # Axis labels
    ax.set_xlabel("Longitude [degrees]")
    ax.set_ylabel("Latitude [degrees]")

    plt.title(title, fontsize=16)
    plt.savefig(savefig_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {savefig_name}")

# =========================================================
# PLOT VARIABLES
# =========================================================

if dataset_name == "eu-cordex-11":
    # Ensemble mean
    plot_var(ds, var_name, "ensmean")
    # Ensemble std
    var_name_std = f"{var_name}_STD"
    if var_name_std in ds:
        plot_var(ds, var_name_std, "ensstd")
    else:
        print(f"Warning: {var_name_std} not found in dataset.")
else:
    plot_var(ds, var_name, "value")
