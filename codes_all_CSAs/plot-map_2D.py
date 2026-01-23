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
# USER CONFIG
# =========================================================

dataset_name = "cerra"
pilotarea = "Barcelona"
CSA = FUA_MAPPING.get(pilotarea, pilotarea)

indicator = "rr"
var_name = "RR"
cmap = "Blues"

# =========================================================
# PARAMS EXTRACTION
# =========================================================

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

base = f"/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/{pilotarea}/INDICATORS"
FUA_SHP = "/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"

# =========================================================
# HELPERS
# =========================================================

def construct_filename(pilotarea, dataset_name, indicator, start_year, end_year,
                       hist_start_year=None, hist_end_year=None,
                       proj_start_year=None, proj_end_year=None,
                       scenario=None):

    p = pilotarea.lower()

    if dataset_name in ["cerra", "eobs"]:
        return f"{p}_{dataset_name}_{indicator}_eu_{start_year}_{end_year}.nc"
    if dataset_name == "era5-2km":
        return f"{p}_{dataset_name}_{indicator}_{start_year}{end_year}.nc"
    if dataset_name == "emo":
        return f"{p}_{dataset_name}_{indicator}{start_year}{end_year}.nc"
    if dataset_name == "eu-cordex-11":
        return (
            f"{p}_{dataset_name}_{indicator}_"
            f"{hist_start_year}-{hist_end_year}_"
            f"{proj_start_year}-{proj_end_year}_{scenario}.nc"
        )

    raise ValueError(f"Unknown dataset {dataset_name}")


def get_title_and_savefig(pilotarea, dataset_name, var_name,
                          start_year, end_year,
                          hist_start_year=None, hist_end_year=None,
                          proj_start_year=None, proj_end_year=None,
                          scenario=None, suffix=""):

    if dataset_name == "eu-cordex-11":
        lon, lat = "lon", "lat"
        title = (
            f"Average Map of {var_name} "
            f"({hist_start_year}-{hist_end_year} & "
            f"{proj_start_year}-{proj_end_year}, {scenario})\n"
            f"{pilotarea}"
        )
        fname = f"{pilotarea}_{var_name}_{dataset_name}_{scenario}_{suffix}.png"

    elif dataset_name in ["cerra", "eobs"]:
        lon, lat = "longitude", "latitude"
        title = f"Average Map of {var_name} ({start_year}-{end_year})\n{pilotarea}"
        fname = f"{pilotarea}_{var_name}_{dataset_name}_{start_year}-{end_year}_{suffix}.png"

    else:
        lon, lat = "lon", "lat"
        title = f"Average Map of {var_name} ({start_year}-{end_year})\n{pilotarea}"
        fname = f"{pilotarea}_{var_name}_{dataset_name}_{start_year}-{end_year}_{suffix}.png"

    return title, fname, lon, lat


def convert_to_days(arr):
    unit = np.datetime_data(arr.dtype)[0]
    conv = {
        "ns": 1/(1e9*3600*24),
        "us": 1/(1e6*3600*24),
        "ms": 1/(1e3*3600*24),
        "s":  1/(3600*24),
        "m":  1/(60*24),
        "h":  1/24,
        "D":  1.0
    }
    return arr.astype("float64") * conv[unit]


def get_extent_from_shape(gdf, pad_frac=0.05):
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * pad_frac
    dy = (maxy - miny) * pad_frac
    return [minx - dx, maxx + dx, miny - dy, maxy + dy]

# =========================================================
# LOAD DATA
# =========================================================

file = f"{base}/{construct_filename(
    pilotarea, dataset_name, indicator,
    start_year, end_year,
    hist_start_year, hist_end_year,
    proj_start_year, proj_end_year,
    scenario)}"

if not os.path.exists(file):
    raise FileNotFoundError(file)

print(f"Loading {file}")
ds = xr.open_dataset(file)

if pilotarea == "Birmingham":
    warnings.warn("Using approximate FUA name for Birmingham")

# =========================================================
# PLOTTING
# =========================================================

def plot_var(ds, var_name_plot, suffix):

    title, savefig_name, lon_name, lat_name = get_title_and_savefig(
        pilotarea, dataset_name, var_name_plot,
        start_year, end_year,
        hist_start_year, hist_end_year,
        proj_start_year, proj_end_year,
        scenario, suffix
    )

    lon = ds[lon_name].values
    lat = ds[lat_name].values

    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat

    data = ds[var_name_plot].values
    if np.issubdtype(data.dtype, np.timedelta64):
        data = convert_to_days(data)
    data = np.squeeze(data)

    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

    # Load FUA
    try:
        fua = gpd.read_file(FUA_SHP).to_crs(epsg=4326)
        fua = fua[fua["FUA_NAME"] == CSA]
        fua_found = not fua.empty
    except Exception as e:
        print(e)
        fua_found = False

    if fua_found:
        extent = get_extent_from_shape(fua)
    else:
        extent = [
            np.nanmin(lon2d), np.nanmax(lon2d),
            np.nanmin(lat2d), np.nanmax(lat2d)
        ]

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter(".1f")
    gl.yformatter = LatitudeFormatter(".1f")
    gl.xlocator = MaxNLocator(4)
    gl.ylocator = MaxNLocator(4)

    im = ax.pcolormesh(
        lon2d, lat2d, data,
        cmap=cmap,
        vmin=np.nanpercentile(data, 5),
        vmax=np.nanpercentile(data, 95),
        transform=ccrs.PlateCarree()
    )

    if fua_found:
        fua.boundary.plot(
            ax=ax, linewidth=1.2, edgecolor="black",
            transform=ccrs.PlateCarree()
        )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    cbar = plt.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(f"{var_name_plot} (days)")

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    plt.title(title, fontsize=15)

    plt.savefig(savefig_name, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved {savefig_name}")

# =========================================================
# RUN
# =========================================================

if dataset_name == "eu-cordex-11":
    plot_var(ds, var_name, "ensmean")
    if f"{var_name}_STD" in ds:
        plot_var(ds, f"{var_name}_STD", "ensstd")
else:
    plot_var(ds, var_name, "value")                                                       
