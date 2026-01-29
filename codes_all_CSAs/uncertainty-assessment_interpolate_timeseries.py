import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.vectorized as sv
import os
from pathlib import Path
import zipfile
import shutil
import gc

# =========================================================
# CONFIGURATION
# =========================================================
pilotarea = "Leipzig"  # Name of the pilot city/area
FUA_MAPPING = {           # Map pilot area names to official FUA names in shapefile
    "Prague": "Praha",
    "Leipzig": "Leipzig",
    "Funen-Odense": "Odense",
    "Athens": "Athina",
    "Barcelona": "Barcelona",
    "Bologna": "Bologna",
    "Brasov": "Brasov",
    "Birmingham": "West Midlands urban area",
}
CSA = FUA_MAPPING.get(pilotarea, pilotarea)  # Use mapped FUA name or fallback to pilotarea

indicator = "RRx1d"      # Climate indicator of interest 
var_in_nc = "RRx1d"      # Name of variable inside NetCDF files

# Datasets to be used and their valid year ranges
DATASETS = {
    "CERRA": dict(start_year=1985, end_year=2020),
    "EOBS": dict(start_year=1950, end_year=2024),
    "EMO1": dict(start_year=1990, end_year=2024),
    "ERA5": dict(start_year=1980, end_year=2022),
    "ERA52km": dict(start_year=1989, end_year=2018)
}

# Baseline period for reference calculations
start_baseline = 1991
end_baseline = 2020

# Base folder for NetCDF time series
path = "insert path"
base_path = (
    f"{path}/CARMINE-T2.4/"
    "CARMINE_Past-Climate_CSAs_Indicators_Timeseries/"
    #"CARMINE_Past-Climate_CSAs"
)
FUA_SHP = f"{path}/CARMINE-T2.4/shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"

# =========================================================
# UNZIPPER - TEMPORARY DIRECTORY WITHIN DATA DIR
# =========================================================

unzip_path = Path(base_path) / "_tmp_unzipped_nc"
unzip_path.mkdir(parents=True, exist_ok=True)

print("=== SELECTIVE UNZIP (per CSA + dataset + indicator selection) ===")

# Where the zips live: per CSA, first level in the CSA folder
csa_dir = Path(base_path) / pilotarea
if not csa_dir.exists():
    raise FileNotFoundError(f"CSA directory not found: {csa_dir}")

# Ensure unzip output dir exists (already created above, but safe)
unzip_path.mkdir(parents=True, exist_ok=True)

# Loop only over datasets in the user selection
for dname, p in DATASETS.items():
    # --- Expected NetCDF filename for THIS selection (same as loader logic) ---
    expected_nc = (
        f"CARMINE_{dname}_{pilotarea}_{var_in_nc}_BSL_"
        f"{start_baseline}_{end_baseline}_YY_{p['start_year']}_{p['end_year']}.nc"
    )

    # --- Identify the dataset zip ---
    zpath = csa_dir / f"{dname}.zip"
    if not zpath.exists():
        print(f"Skip {dname}: zip not found -> {zpath}")
        continue

    # --- If we already extracted it, skip ---
    out_nc = unzip_path / pilotarea / dname / expected_nc
    if out_nc.exists():
        print(f"{dname}: OK (already extracted) -> {out_nc}")
        continue

    # Make sure output subdirs exist
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    # --- STEP 2: Check whether expected file exists inside the zip ---
    with zipfile.ZipFile(zpath, "r") as z:
        members = z.namelist()

        # Many zips store files at root; some may store subfolders.
        # Match either exact name or any member ending with the expected filename.
        matches = [m for m in members if m == expected_nc or m.endswith("/" + expected_nc)]

        if len(matches) == 0:
            print(f"{dname}: NOT FOUND in zip -> {expected_nc}")
            # Optional: uncomment next line to print a short preview of members for debugging
            # print("  Members (first 10):", members[:10])
            continue

        if len(matches) > 1:
            print(f"{dname}: WARNING multiple matches for {expected_nc} in {zpath.name}")
            # Prefer the shortest path (usually root-level) to reduce surprises
            matches.sort(key=len)

        member_to_extract = matches[0]

        # --- STEP 3: Extract only that matching file ---
        # We extract to a temp location inside out_nc.parent and then rename/move to expected name if needed.
        extracted_path = Path(z.extract(member_to_extract, path=out_nc.parent))

        # If the member was stored under subfolders in the zip, normalize to the expected output filename
        if extracted_path.name != expected_nc:
            # Move/rename to the standardized location
            # If a file already exists, don't overwrite
            if out_nc.exists():
                # Clean up the just-extracted file to avoid clutter
                try:
                    extracted_path.unlink()
                except Exception:
                    pass
                print(f"{dname}: OK (already existed after extract?) -> {out_nc}")
                continue

            extracted_path.rename(out_nc)
        else:
            # It extracted directly to out_nc.parent/expected_nc
            # Ensure it sits at our standardized path (out_nc)
            if extracted_path != out_nc:
                extracted_path.rename(out_nc)

        print(f"{dname}: extracted -> {out_nc}")

print("=== SELECTIVE UNZIP DONE ===")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def pick_dataarray(ds, var_in_nc, indicator):
    """
    Selects the correct DataArray from a dataset using a list of possible names.
    If only one variable exists, it returns that one.
    Raises KeyError if no variable is found.
    """
    candidates = [var_in_nc, indicator.upper(), indicator, indicator.lower()]
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise KeyError(f"No variable found. Candidates: {candidates}")

def regrid_to_shared(da_src, lon_vals_src, lat_vals_src, shared_lon, shared_lat, method='nearest'):
    """
    Regrids a DataArray to a common grid using scipy.griddata.
    Handles both 1D and 2D coordinate arrays.
    """
    # Ensure source lon/lat arrays are numpy arrays
    lon_vals_src = np.asarray(lon_vals_src)
    lat_vals_src = np.asarray(lat_vals_src)

    # Create 2D meshgrid if lon/lat are 1D
    if lon_vals_src.ndim == 1 and lat_vals_src.ndim == 1:
        Lon_src, Lat_src = np.meshgrid(lon_vals_src, lat_vals_src)
    else:
        Lon_src, Lat_src = lon_vals_src, lat_vals_src

    # Flatten the source points and values
    points = np.column_stack([Lon_src.ravel(), Lat_src.ravel()])
    values = da_src.values.ravel()

    # Ensure target lon/lat arrays are numpy arrays
    shared_lon = np.asarray(shared_lon)
    shared_lat = np.asarray(shared_lat)
    if shared_lon.ndim == 1 and shared_lat.ndim == 1:
        Lon_t, Lat_t = np.meshgrid(shared_lon, shared_lat)
    else:
        Lon_t, Lat_t = shared_lon, shared_lat

    # Interpolate values onto target grid
    target_points = np.column_stack([Lon_t.ravel(), Lat_t.ravel()])
    interp_vals = griddata(points, values, target_points, method=method).reshape(Lon_t.shape)

    # Return as xarray DataArray with proper coordinates
    return xr.DataArray(
        interp_vals,
        dims=("latitude", "longitude"),
        coords={
            "latitude": ("latitude", Lon_t[:,0] if Lon_t.ndim==2 else shared_lat),
            "longitude": ("longitude", Lat_t[0,:] if Lat_t.ndim==2 else shared_lon)
        },
        name=da_src.name,
        attrs=da_src.attrs
    )

def mask_fua_pixels(da, lon_vals, lat_vals, fua_boundary):
    """
    Masks a DataArray so that only pixels inside the FUA polygon are kept.
    """
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon, lat = np.meshgrid(lon_vals, lat_vals)
    else:
        lon, lat = lon_vals, lat_vals
    mask_2d = sv.contains(fua_boundary, lon, lat)
    return da.where(mask_2d)

def get_fua_boundary(fua_gdf, csa_name):
    """
    Retrieves the geometry and bounding box of a given FUA from a GeoDataFrame.
    """
    fua = fua_gdf[fua_gdf["FUA_NAME"] == csa_name].to_crs(4326)
    if fua.empty:
        raise ValueError(f"FUA not found: {csa_name}")
    return fua.geometry.iloc[0], fua.total_bounds

def plot_map_with_shapefile(da, lon_vals, lat_vals, fua_boundary, fua_bounds, title, fname=None, cmap='RdYlBu_r'):
    """
    Plots a 2D DataArray over a FUA region with shapefile overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add buffer around FUA bounds for map extent
    buffer = 0.5
    extent = [fua_bounds[0]-buffer, fua_bounds[2]+buffer, fua_bounds[1]-buffer, fua_bounds[3]+buffer]
    ax.set_extent(extent)

    # Ensure coordinates are arrays and create meshgrid
    lon_vals = np.asarray(lon_vals)
    lat_vals = np.asarray(lat_vals)
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        Lon, Lat = np.meshgrid(lon_vals, lat_vals)
    else:
        Lon, Lat = lon_vals, lat_vals

    # Plot the data as a pcolormesh
    cf = ax.pcolormesh(Lon, Lat, da.values, transform=ccrs.PlateCarree(),
                       cmap=cmap, shading='auto', alpha=0.8)

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

    # Overlay FUA boundary
    gpd.GeoSeries([fua_boundary], crs='EPSG:4326').plot(
        ax=ax, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree()
    )

    # Add colorbar
    plt.colorbar(cf, ax=ax, shrink=0.8, label=da.name)

    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure if filename provided
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# =========================================================
# MAIN PROCESS
# =========================================================
print("=== FUA-MASKED UNCERTAINTY ASSESSMENT ===")

# Load FUA shapefile
fua_gdf = gpd.read_file(FUA_SHP).to_crs(4326)
fua_boundary, fua_bounds = get_fua_boundary(fua_gdf, CSA)

# Load all datasets
loaded = {}
lon_all = []
lat_all = []
for dname, p in DATASETS.items():
    # Construct expected filename
    fname = f"CARMINE_{dname}_{pilotarea}_{var_in_nc}_BSL_{start_baseline}_{end_baseline}_YY_{p['start_year']}_{p['end_year']}.nc"
    path = os.path.join(unzip_path, pilotarea, dname, fname)

    # Skip missing files
    if not os.path.exists(path):
        print(f"Skip {dname}: {path}")
        continue

    # Open dataset
    ds = xr.open_dataset(path)
    da = pick_dataarray(ds, var_in_nc, indicator).squeeze(drop=True)
    da.name = dname

    # Take mean over time if present
    if "time" in da.dims:
        da = da.mean("time", skipna=True)

    # Store loaded data
    loaded[dname] = dict(da=da, lon=ds['lon'].values, lat=ds['lat'].values)
    lon_all.append(ds['lon'].values)
    lat_all.append(ds['lat'].values)

# =========================================================
# Define shared grid across all datasets
# =========================================================
lon_min = max([np.min(lon) for lon in lon_all])
lon_max = min([np.max(lon) for lon in lon_all])
lat_min = max([np.min(lat) for lat in lat_all])
lat_max = min([np.max(lat) for lat in lat_all])
nlon = max([lon.size if lon.ndim==1 else lon.shape[1] for lon in lon_all])
nlat = max([lat.size if lat.ndim==1 else lat.shape[0] for lat in lat_all])
shared_lon = np.linspace(lon_min, lon_max, nlon)
shared_lat = np.linspace(lat_min, lat_max, nlat)

# =========================================================
# Regrid all datasets to shared grid
# =========================================================
regridded = {}
for name, data in loaded.items():
    print(f"Remapping {name} ...", end="")
    da = data['da']
    regridded[name] = regrid_to_shared(da, data['lon'], data['lat'], shared_lon, shared_lat, method='nearest')
    print(" done")

# =========================================================
# Apply FUA mask
# =========================================================
masked = {}
for name, da in regridded.items():
    masked[name] = mask_fua_pixels(da, shared_lon, shared_lat, fua_boundary)
    
# =========================================================
# Plot datasets (FUA only)
# =========================================================
for name, da in masked.items():
    plot_map_with_shapefile(
        da, shared_lon, shared_lat, fua_boundary, fua_bounds,
        title=f"{CSA} – {name.upper()}",
        fname=f"{pilotarea}_{name}_FUA_masked.png"
    )

# =========================================================
# Compute metrics between datasets
# =========================================================
names = list(masked.keys())
n = len(names)

# Initialize metric matrices
pearson = np.full((n, n), np.nan)
spearm = np.full((n, n), np.nan)
rmse_m = np.full((n, n), np.nan)
bias_m = np.full((n, n), np.nan)

# Compute pairwise metrics
for i, ref_n in enumerate(names):
    ref_vals = masked[ref_n].values.ravel()
    for j, mod_n in enumerate(names):
        if i == j:
            continue
        mod_vals = masked[mod_n].values.ravel()
        valid = np.isfinite(ref_vals) & np.isfinite(mod_vals)
        if valid.sum() < 1:
            continue
        a, b = ref_vals[valid], mod_vals[valid]
        pearson[i, j] = np.corrcoef(a, b)[0, 1]
        spearm[i, j], _ = spearmanr(a, b)
        rmse_m[i, j] = np.sqrt(np.mean((a - b) ** 2))
        bias_m[i, j] = np.mean(b - a)

# =========================================================
# Heatmap plotting function
# =========================================================
def plot_heatmap(mat, title, cmap="RdBu_r", vmin=None, vmax=None, center=0, fmt=".3f"):
    """
    Plots a heatmap of a metric matrix using seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                xticklabels=names, yticklabels=names)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Reference")
    plt.tight_layout()
    plt.show()

# =========================================================
# Plot metrics
# =========================================================
plot_heatmap(pearson, f"Pearson r ({CSA}) - {var_in_nc}")
plot_heatmap(spearm, f"Spearman rho ({CSA}) - {var_in_nc}")
plot_heatmap(rmse_m, f"RMSE ({CSA}) - {var_in_nc}", cmap="viridis_r")
vlim = np.nanmax(np.abs(bias_m))
plot_heatmap(bias_m, f"Bias ({CSA})", vmin=-vlim, vmax=vlim)

# =========================================================
# Clean up tmp_unzipped_nc (close and remove all data)
# =========================================================
for dname, d in loaded.items():
    try:
        d["da"].load()  # forces the data into memory
    except Exception as e:
        print(f"Warning: could not load {dname} into memory before cleanup: {e}")

    try:
        xr.backends.file_manager.FILE_CACHE.clear()
    except Exception as e:
        print(f"Warning: could not clear xarray FILE_CACHE: {e}")

gc.collect()

if len(loaded) > 0 and unzip_path.exists():
    shutil.rmtree(unzip_path)
    print("Cleanup successful.")
else:
    print("Cleanup skipped (no data loaded or error occurred).")


print("\nAnalysis COMPLETE!")
