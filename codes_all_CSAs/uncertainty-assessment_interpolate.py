import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from matplotlib.colors import BoundaryNorm
import os

'''
To ensure meaningful comparability among climate indicators derived from different datasets, all indicators must be computed over the same reference time period. Indicators calculated over different temporal windows may reflect not only dataset- or model-related differences, but also the effects of climate variability and long-term trends, thus limiting the robustness of any comparative analysis.

The code presented here is intended to demonstrate the functionality of the comparison workflow, using currently available indicator datasets as input. At this stage, these indicators are computed over non-uniform time periods, and the results should therefore be interpreted as a technical demonstration rather than a scientifically consistent comparison.

For robust and reproducible analyses, it is essential to rely on the underlying time series data, from which indicators can be recomputed over a common shared period across all datasets.

'''

# =========================================================
# CONFIGURATION
# =========================================================
pilotarea = "Barcelona"
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
CSA = FUA_MAPPING.get(pilotarea, pilotarea)

indicator = "rr"          # string in filenames
var_in_nc = "RR"          # variable name in NetCDF

DATASETS = {
    "cerra": dict(start_year=1981, end_year=2010),
    "eobs":  dict(start_year=1981, end_year=2010),
    "emo":   dict(start_year=1991, end_year=2020),
}

base_path = f"/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/{pilotarea}/INDICATORS"
FUA_SHP = "/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"

# =========================================================
# METRICS GUIDE HELPER
# =========================================================
def print_metrics_guide():
    print("\n" + "="*80)
    print("UNCERTAINTY METRICS GUIDE (FUA GEOMETRY MASKED)")
    print("="*80)
    print("\nPOINTWISE COMPARISONS (ONLY PIXELS INSIDE FUA):")
    print("\n1. PEARSON r: Linear correlation (-1 to +1)")
    print("   r = corrcoef(A,B)[0,1] | +1=perfect linear")
    print("\n2. SPEARMAN ?~A: Rank correlation (-1 to +1, outlier-robust)")
    print("   ?~A = spearmanr(A,B) | Pattern similarity ignoring scale")
    print("\n3. RMSE: Error magnitude")
    print("   RMSE = ?~H~Z[mean((A-B)²)] | Absolute error")
    print("\n4. BIAS: Systematic error")
    print("   Bias = mean(B-A) | +ve=B overestimates A")
    print("\nTHRESHOLDS (climate data):")
    print("   Excellent: r/?~A>0.9, RMSE<10%mean, |Bias|<5%")
    print("   Good: r/?~A>0.8")
    print("="*80 + "\n")

# =========================================================
# FILE NAMING
# =========================================================
def construct_filename(pilotarea, dataset_name, indicator, start_year, end_year):
    pilot_lower = pilotarea.lower()
    if dataset_name in ["cerra", "eobs"]:
        return f"{pilot_lower}_{dataset_name}_{indicator}_eu_{start_year}_{end_year}.nc"
    elif dataset_name == "emo":
        return f"{pilot_lower}_{dataset_name}_{indicator}{start_year}{end_year}.nc"
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

# =========================================================
# CORE HELPERS
# =========================================================
def pick_dataarray(ds, var_in_nc, indicator):
    candidates = [var_in_nc, indicator.upper(), indicator, indicator.lower()]
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise KeyError(f"No variable found. Candidates: {candidates}")

def detect_lonlat(da):
    # Prefer explicit lon/lat coord names
    for lon_name, lat_name in [("longitude", "latitude"), ("lon", "lat")]:
        if lon_name in da.coords and lat_name in da.coords:
            lon = da[lon_name]
            lat = da[lat_name]
            return lon_name, lat_name, (lon.ndim == 2 or lat.ndim == 2)
    # Fallback: infer from dims
    lon_dim = next((d for d in da.dims if d.lower() in ["longitude", "lon", "x"]), None)
    lat_dim = next((d for d in da.dims if d.lower() in ["latitude", "lat", "y"]), None)
    if lon_dim is None or lat_dim is None:
        raise ValueError(f"Cannot find lon/lat: dims={da.dims}")
    return lon_dim, lat_dim, False

def cut_bbox(da, lon_name, lat_name, minx, maxx, miny, maxy):
    lon = da[lon_name]
    lat = da[lat_name]
    if lon.ndim == 1 and lat.ndim == 1:
        return da.where(
            (lon >= minx) & (lon <= maxx) &
            (lat >= miny) & (lat <= maxy), drop=True
        )
    else:
        lonv, latv = lon.values, lat.values
        mask2d = (lonv >= minx) & (lonv <= maxx) & (latv >= miny) & (latv <= maxy)
        mask_da = xr.DataArray(mask2d, dims=lon.dims, coords=lon.coords)
        return da.where(mask_da, drop=True)

def grid_spacing_1d(da, lon_name, lat_name):
    lon = np.asarray(da[lon_name].values).astype(float)
    lat = np.asarray(da[lat_name].values).astype(float)
    if lon.ndim == 2:
        lon = lon.ravel()
    if lat.ndim == 2:
        lat = lat.ravel()
    dlon = np.nanmedian(np.abs(np.diff(np.unique(lon))))
    dlat = np.nanmedian(np.abs(np.diff(np.unique(lat))))
    return float(max(dlon, dlat))

def regrid_to_ref(da_src, lon_name, lat_name, ref_lon_1d, ref_lat_1d):
    """
    Regrid da_src (con lon/lat 1D o 2D) su griglia regolare definita da ref_lon_1d/ref_lat_1d
    usando scipy.griddata, così evitiamo i vincoli di xarray.interp sulle dimensioni.
    """
    lon = da_src[lon_name].values
    lat = da_src[lat_name].values
    vals = da_src.values

    # 1) Portiamo lon/lat/vals in 1D
    Lon, Lat = (np.meshgrid(lon, lat) if (lon.ndim == 1 and lat.ndim == 1)
                else (lon, lat))
    points = np.column_stack([Lon.ravel(), Lat.ravel()])
    values = vals.ravel()

    # 2) Griglia di destinazione (ref_lon_1d / ref_lat_1d)
    ref_lon_1d = np.asarray(ref_lon_1d).astype(float)
    ref_lat_1d = np.asarray(ref_lat_1d).astype(float)
    Lon_t, Lat_t = np.meshgrid(ref_lon_1d, ref_lat_1d)
    target_points = np.column_stack([Lon_t.ravel(), Lat_t.ravel()])

    # 3) Interpolazione
    interp_vals = griddata(points, values, target_points, method="linear")
    interp_vals = interp_vals.reshape(Lon_t.shape)

    # 4) Costruiamo un DataArray con dims (lat, lon) 1D
    out = xr.DataArray(
        interp_vals,
        dims=("latitude", "longitude"),
        coords={
            "latitude": ref_lat_1d,
            "longitude": ref_lon_1d,
        },
        name=da_src.name,
        attrs=da_src.attrs,
    )
    return out

# =========================================================
# FUA GEOMETRIC MASK
# =========================================================
def mask_fua_pixels(da, lon_name, lat_name, fua_boundary):
    lon = da[lon_name]
    lat = da[lat_name]

    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon.values, lat.values)
    else:
        Lon, Lat = lon.values, lat.values

    points = [Point(lon_val, lat_val) for lon_val, lat_val in zip(Lon.ravel(), Lat.ravel())]
    inside_fua = np.array([fua_boundary.contains(p) for p in points])

    mask_2d = inside_fua.reshape(Lon.shape)
    # assume le due ultime dims corrispondono a lon/lat grid
    dims = da.dims
    mask_da = xr.DataArray(mask_2d, dims=dims[-2:], coords={lon_name: lon, lat_name: lat})

    return da.where(mask_da)

# =========================================================
# MAPPING
# =========================================================
def get_fua_boundary(fua_gdf, csa_name):
    fua = fua_gdf[fua_gdf["FUA_NAME"] == csa_name].to_crs(4326)
    if fua.empty:
        raise ValueError(f"FUA not found: {csa_name}")
    return fua.geometry.iloc[0], fua.total_bounds

def plot_map_with_shapefile(da, lon_name, lat_name, fua_boundary, fua_bounds, title, fname=None,
                            cmap='RdYlBu_r'):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    buffer = 0.1
    extent = [fua_bounds[0]-buffer, fua_bounds[2]+buffer,
              fua_bounds[1]-buffer, fua_bounds[3]+buffer]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lon, lat = da[lon_name], da[lat_name]
    if lon.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat)
        cf = ax.pcolormesh(Lon, Lat, da.values, transform=ccrs.PlateCarree(),
                           cmap=cmap, shading='auto', alpha=0.8)
    else:
        cf = ax.pcolormesh(lon.values, lat.values, da.values, transform=ccrs.PlateCarree(),
                           cmap=cmap, shading='auto', alpha=0.8)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

    fua_geom = gpd.GeoSeries([fua_boundary], crs='EPSG:4326')
    fua_geom.plot(ax=ax, edgecolor='red', linewidth=4, facecolor='none',
                  transform=ccrs.PlateCarree(), alpha=1.0)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label(f'{da.name or indicator.upper()}', fontsize=12)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = gl.right_labels = False

    ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
print("=== FUA-MASKED UNCERTAINTY ASSESSMENT ===")
print_metrics_guide()

fua_gdf = gpd.read_file(FUA_SHP).to_crs(4326)
fua_boundary, fua_bounds = get_fua_boundary(fua_gdf, CSA)
minx, miny, maxx, maxy = fua_bounds
print(f"FUA {CSA} bbox: [{minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}]\n")

loaded = {}
meta = {}

for dname, p in DATASETS.items():
    fname = construct_filename(pilotarea, dname, indicator, p["start_year"], p["end_year"])
    path = os.path.join(base_path, fname)
    if not os.path.exists(path):
        print(f"Skip {dname}: {path}")
        continue

    ds = xr.open_dataset(path)
    da = pick_dataarray(ds, var_in_nc, indicator).squeeze(drop=True)
    da.name = dname

    if "time" in da.dims:
        da = da.mean("time", skipna=True)

    lon_name, lat_name, is_2d = detect_lonlat(da)

    da_cut = cut_bbox(da, lon_name, lat_name, minx, maxx, miny, maxy)
    n_bbox = int(da_cut.count())

    da_fua = mask_fua_pixels(da_cut, lon_name, lat_name, fua_boundary)
    n_fua = int(da_fua.count())

    print(f"{dname}: {da.size:,} ?~F~R {n_bbox:,} bbox ?~F~R {n_fua:,} FUA")

    if n_fua < 10:
        print(f"  Skip {dname}: too few FUA pixels")
        continue

    meta[dname] = dict(
        lon=lon_name,
        lat=lat_name,
        spacing=grid_spacing_1d(da_fua, lon_name, lat_name)
    )
    loaded[dname] = da_fua

if len(loaded) < 2:
    raise ValueError(f"Need ?~I?2 datasets with FUA pixels: {list(loaded)}")

print("\n--- GRID RESOLUTIONS (FUA-masked) ---")
for k in loaded:
    print(f"{k}: {meta[k]['spacing']:.4f}°")

ref_name = max(meta, key=lambda k: meta[k]["spacing"])
print(f"\nReference (coarsest): {ref_name}")

ref_da = loaded[ref_name]
ref_lon_name, ref_lat_name = meta[ref_name]["lon"], meta[ref_name]["lat"]
ref_lon = ref_da[ref_lon_name]
ref_lat = ref_da[ref_lat_name]

print("\n--- REGRIDDING (FUA-masked) ---")
regridded_fua = {}
for name, da in loaded.items():
    if name == ref_name:
        regridded_fua[name] = da
        print(f"{name}: reference ?~\~S")
    else:
        print(f"{name} ?~F~R {ref_name} ...", end="")
        out = regrid_to_ref(da, meta[name]["lon"], meta[name]["lat"], ref_lon, ref_lat)
        regridded_fua[name] = out
        print(" ?~\~S")

for name, da in regridded_fua.items():
    plot_map_with_shapefile(
        da, "longitude", "latitude", fua_boundary, fua_bounds,
        title=f"{name.upper()}",
        fname=f"{pilotarea}_{name}_FUA_masked_regridded.png"
    )

names = list(regridded_fua.keys())
print("\n--- FUA PIXEL COUNT ---")
for name in names:
    print(f"{name}: {int(regridded_fua[name].count()):,} pixels")

n = len(names)
pearson = np.full((n, n), np.nan)
spearm = np.full((n, n), np.nan)
rmse_m = np.full((n, n), np.nan)
bias_m = np.full((n, n), np.nan)

print("\n--- COMPUTING FUA-mask METRICS ---")
for i, ref_n in enumerate(names):
    ref_vals = regridded_fua[ref_n].values.ravel()
    for j, mod_n in enumerate(names):
        if i == j:
            continue
        mod_vals = regridded_fua[mod_n].values.ravel()
        valid = np.isfinite(ref_vals) & np.isfinite(mod_vals)
        if valid.sum() < 10:
            continue
        a, b = ref_vals[valid], mod_vals[valid]
        pearson[i, j] = np.corrcoef(a, b)[0, 1]
        spearm[i, j], _ = spearmanr(a, b)
        rmse_m[i, j] = np.sqrt(np.mean((a - b) ** 2))
        bias_m[i, j] = np.mean(b - a)

def plot_heatmap(mat, title, cmap="RdBu_r", vmin=None, vmax=None, center=0, fmt=".3f"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
        xticklabels=names, yticklabels=names
    )
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Reference")
    plt.tight_layout()
    plt.show()

plot_heatmap(pearson, f"Pearson r ({CSA} - {var_in_nc}")
plot_heatmap(spearm, f"Spearman ?~A ({CSA}) - {var_in_nc}")
plot_heatmap(rmse_m, f"RMSE ({CSA}) - {var_in_nc}", cmap="viridis_r")
vlim = np.nanmax(np.abs(bias_m))
plot_heatmap(bias_m, f"Bias ({CSA}", vmin=-vlim, vmax=vlim)

print("\n?~\~S?~\~S?~\~S COMPLETE: FUA GEOMETRY MASKED analysis!")
print("  ?~\~S Metrics: ONLY pixels INSIDE FUA shapefile")                                   
