import os
import xarray as xr
import numpy as np
import geopandas as gpd
import shapely.vectorized as sv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box

# =========================================================
# USER SETTINGS
# =========================================================

# This code works on data input test included here: https://github.com/FedeleG/CARMINE-T2.4/blob/main/2601_EURO_CORDEX_testing_data/CARMINE_CORDEX_indicators_ensemble-mean-stdev.zip
# Additional indicators will be provided soon

# Pilot area name
pilotarea = "Bologna"

# Mapping from pilot area names to FUA names in shapefile
FUA_MAPPING = {
    "Prague": "Praha",
    "Leipzig": "Leipzig",
    "Funen-Odense": "Odense",
    "Athens": "Athina",
    "Barcelona": "Barcelona",
    "Bologna": "Bologna",
    "Brasov": "Brasov",
    "Birmingham": "West Midlands urban area",
}
CSA = FUA_MAPPING.get(pilotarea, pilotarea)  # Use mapped name if available

# Climate indicator settings
indicator = "tg"      # Name of indicator in the workflow
var_in_nc = "TG"      # Variable name in NetCDF
unit = "degreeC"               # Units for plotting

# RCP scenarios to process
scenarios = ["26", "45", "85"]

# Baseline and future periods
baseline = "1981-2010"
future   = "2036-2065"

# Paths to be modified
root_path = f".../CARMINE/CORDEX/CSA"   # Base path for NetCDF files
FUA_SHP   = f".../CARMINE-T2.4/shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"  # FUA shapefile path

# Map projection
projection = ccrs.PlateCarree()

# =========================================================
# FUNCTIONS
# =========================================================

def mask_fua_with_buffer(data, lon, lat, geometry, buffer_deg=1.0):
    """
    Mask a curvilinear grid (lon, lat) using a FUA polygon with a buffer.
    
    Parameters
    ----------
    data : xarray.DataArray
        The data to mask (ensemble mean or std)
    lon, lat : xarray.DataArray
        Longitude and latitude arrays matching data
    geometry : shapely Polygon
        The FUA polygon geometry
    buffer_deg : float
        Buffer distance in degrees around polygon
    
    Returns
    -------
    masked_data : xarray.DataArray
        Data masked outside buffered FUA
    """
    # Create buffered polygon
    geom_buffered = geometry.buffer(buffer_deg)
    # Create mask using shapely.vectorized contains
    mask = sv.contains(geom_buffered, lon.values, lat.values)
    return data.where(mask)  # Mask values outside polygon

def plot_panel(ax, lon, lat, data, title, cmap, norm, fua_geom, fua_bounds):
    """
    Plot a single map panel with FUA overlay and domain cropped to FUA + buffer.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes with cartopy projection
    lon, lat : xarray.DataArray
        Longitude and latitude
    data : xarray.DataArray
        Data to plot
    title : str
        Panel title
    cmap : matplotlib colormap
        Colormap
    norm : matplotlib.colors.Normalize
        Normalization for color scale
    fua_geom : shapely Polygon
        FUA geometry for overlay
    fua_bounds : tuple
        Bounds of FUA polygon (xmin, ymin, xmax, ymax)
    """
    # Plot data
    mesh = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm,
        shading="auto"
    )

    # Crop map to buffered FUA bounds
    xmin, ymin, xmax, ymax = fua_bounds
    buffer = 1.0  # degrees buffer around FUA
    ax.set_extent([xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer], crs=ccrs.PlateCarree())

    # Add coastlines and borders
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # Overlay FUA polygon boundary
    gpd.GeoSeries([fua_geom], crs="EPSG:4326").plot(
        ax=ax, edgecolor="black", facecolor="none", linewidth=1
    )

    # Set panel title
    ax.set_title(title, fontsize=12)
    return mesh

# =========================================================
# LOAD FUA
# =========================================================

# Read shapefile and select FUA for the pilot area
fua_gdf = gpd.read_file(FUA_SHP).to_crs(4326)
fua_sel = fua_gdf[fua_gdf["FUA_NAME"] == CSA]

if fua_sel.empty:
    raise ValueError(f"FUA not found: {CSA}")

# Extract geometry and bounds
fua_geom = fua_sel.geometry.iloc[0]
fua_bounds = fua_sel.total_bounds  # xmin, ymin, xmax, ymax

# =========================================================
# LOAD ALL SCENARIOS
# =========================================================

# Lists to store masked data for each scenario
ens_mean_list = []
ens_std_list  = []
lon = lat = None

# Loop over scenarios
for scenario in scenarios:
    # Construct NetCDF file path
    file_path = os.path.join(
        root_path,
        CSA,
        f"{CSA}_EU-CORDEX-11_YEAR_{indicator}_{baseline}_{future}_RCP{scenario}.nc"
    )

    # Check file existence
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        ens_mean_list.append(None)
        ens_std_list.append(None)
        continue

    # Load dataset
    print(f"Loading {os.path.basename(file_path)}")
    ds = xr.open_dataset(file_path)

    # Extract ensemble mean and std
    ens_mean = ds[var_in_nc]
    ens_std  = ds[f"{var_in_nc}_STD"]

    # Get lon/lat once
    if lon is None:
        lon = ds["lon"]
        lat = ds["lat"]

    # Apply FUA mask with buffer
    ens_mean_m = mask_fua_with_buffer(ens_mean, lon, lat, fua_geom, buffer_deg=1.0)
    ens_std_m  = mask_fua_with_buffer(ens_std, lon, lat, fua_geom, buffer_deg=1.0)

    # Append masked data to lists
    ens_mean_list.append(ens_mean_m)
    ens_std_list.append(ens_std_m)

# =========================================================
# PLOT ALL SCENARIOS IN ONE FIGURE
# =========================================================

# Set up figure: 2 rows (mean, std) x N scenarios
n_rows = 2
n_cols = len(scenarios)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(6 * n_cols, 10),
    subplot_kw={"projection": projection},
    squeeze=False
)

# Loop over scenarios for plotting
for col, scenario in enumerate(scenarios):
    ens_mean_m = ens_mean_list[col]
    ens_std_m  = ens_std_list[col]

    if ens_mean_m is None:
        continue

    # Compute colorbar limits based on masked region
    mean_vmin = float(np.nanmin(ens_mean_m))
    mean_vmax = float(np.nanmax(ens_mean_m))
    std_vmin  = float(np.nanmin(ens_std_m))
    std_vmax  = float(np.nanmax(ens_std_m))

    # Colormap and normalization
    mean_cmap = plt.get_cmap("hot_r")
    mean_norm = mcolors.Normalize(vmin=mean_vmin, vmax=mean_vmax)

    std_cmap = plt.get_cmap("magma_r")
    std_norm = mcolors.Normalize(vmin=std_vmin, vmax=std_vmax)

    # Row 1: Ensemble mean
    m1 = plot_panel(
        axes[0, col], lon, lat, ens_mean_m,
        f"Ensemble Mean RCP{scenario}",
        mean_cmap, mean_norm,
        fua_geom, fua_bounds
    )

    # Row 2: Ensemble std
    m2 = plot_panel(
        axes[1, col], lon, lat, ens_std_m,
        f"Ensemble Std RCP{scenario}",
        std_cmap, std_norm,
        fua_geom, fua_bounds
    )

    # Add colorbars for each panel
    for ax, mesh, label in zip([axes[0, col], axes[1, col]], [m1, m2], [f"[{unit}]", f"[{unit}]"]):
        cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.04)
        cb.set_label(label)

# Set main figure title
plt.suptitle(
    f"{CSA} - {indicator}\nEURO-CORDEX differences {future} vs {baseline}",
    fontsize=16
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
out_file = f"UNCERTAINTY_{CSA}_{indicator}_all_RCP.png"
plt.savefig(out_file, dpi=300)
plt.show()
plt.close(fig)

print(f"Saved {out_file}")
