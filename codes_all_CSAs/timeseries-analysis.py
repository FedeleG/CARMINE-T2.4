import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shapely.vectorized as sv
from scipy.stats import spearmanr
import os

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

indicator = "RRx1d"
var_in_nc = "RRx1d"
unit = "mm"

DATASETS = {
    "CERRA": dict(start_year=1985, end_year=2020),
    "EOBS": dict(start_year=1950, end_year=2024),
    "EMO1": dict(start_year=1990, end_year=2024),
    "ERA5": dict(start_year=1980, end_year=2022),
    "ERA2km": dict(start_year=1989, end_year=2018)
}

# Analysis period
start_year = 1991
end_year = 2020
years_common = np.arange(start_year, end_year + 1)

# Paths
path = "/work/cmcc/gf27821/CARMINE"
base_path = (
    f"{path}/CARMINE-T2.4/"
    "CARMINE_Past-Climate_CSAs_Indicators_Timeseries/"
    "CARMINE_Past-Climate_CSAs"
)
FUA_SHP = f"{path}/CARMINE-T2.4/shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def pick_dataarray(ds, var_in_nc, indicator):
    candidates = [var_in_nc, indicator.upper(), indicator, indicator.lower()]
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise KeyError(f"No variable found. Candidates: {candidates}")

def mask_fua_pixels(da, lon_vals, lat_vals, fua_boundary):
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon, lat = np.meshgrid(lon_vals, lat_vals)
    else:
        lon, lat = lon_vals, lat_vals
    mask_2d = sv.contains(fua_boundary, lon, lat)
    return da.where(mask_2d)

def get_fua_boundary(fua_gdf, csa_name):
    fua = fua_gdf[fua_gdf["FUA_NAME"] == csa_name].to_crs(4326)
    if fua.empty:
        raise ValueError(f"FUA not found: {csa_name}")
    return fua.geometry.iloc[0], fua.total_bounds

def plot_heatmap(mat, names, title, cmap="RdBu_r", center=0):
    plt.figure(figsize=(7,6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap, xticklabels=names, yticklabels=names, center=center)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =========================================================
# MAIN
# =========================================================
print("=== TEMPORAL ANALYSIS CHECK ===")

# Load FUA shapefile
fua_gdf = gpd.read_file(FUA_SHP).to_crs(4326)
fua_boundary, fua_bounds = get_fua_boundary(fua_gdf, CSA)

# Load datasets
loaded = {}
for dname, p in DATASETS.items():
    fname = f"CARMINE_{dname}_{pilotarea}_{var_in_nc}_BSL_{start_year}_{end_year}_YY_{p['start_year']}_{p['end_year']}.nc"
    fpath = os.path.join(base_path, pilotarea, dname, fname)
    if not os.path.exists(fpath):
        print(f"Skip {dname}")
        continue
    ds = xr.open_dataset(fpath)
    da = pick_dataarray(ds, var_in_nc, indicator).squeeze(drop=True)
    da.name = dname
    # Save dataset + lon/lat + year
    loaded[dname] = {
        "da": da,
        "lon": ds['lon'].values,
        "lat": ds['lat'].values,
        "year": ds['year'].values
    }

print("Loaded datasets:", list(loaded.keys()))

# =========================================================
# Apply FUA mask
# =========================================================
masked = {}
for name, data in loaded.items():
    da = data['da']
    lon = data['lon']
    lat = data['lat']
    masked[name] = mask_fua_pixels(da, lon, lat, fua_boundary)
    print(name, "masked points:", np.isfinite(masked[name]).sum())

# =========================================================
# Boxplot BEFORE spatial mean
# =========================================================
records = []
for name, da in masked.items():
    years = loaded[name]['year']
    mask_years = (years >= start_year) & (years <= end_year)
    da_sel = da.isel(time=mask_years)
    vals = da_sel.values.ravel()
    vals = vals[np.isfinite(vals)]
    records.append(pd.DataFrame({"dataset": name, "value": vals}))

df_box = pd.concat(records, ignore_index=True)
plt.figure(figsize=(9,6))
sns.boxplot(x="dataset", y="value", data=df_box)
plt.title(f"Boxplot - {CSA} - ({start_year}-{end_year})")
plt.ylabel(f"{var_in_nc} [{unit}]")
plt.xlabel("")
plt.tight_layout()
plt.show()

# =========================================================
# Spatial mean annual timeseries (1991-2020)
# =========================================================
ts_annual = {}
for name, da in masked.items():
    # Spatial mean over FUA pixels
    spatial_dims = [d for d in da.dims if d not in ['time']]
    da_mean = da.mean(dim=spatial_dims, skipna=True)

    # Create DataFrame indexed by year
    df = pd.DataFrame({
        "year": loaded[name]['year'],
        "value": da_mean.values
    }).set_index("year")

    # Reindex to common years (1991-2020), fill missing with NaN
    df = df.reindex(years_common)
    ts_annual[name] = df

# Plot annual timeseries
plt.figure(figsize=(12,6))
for name, df in ts_annual.items():
    plt.plot(df.index, df['value'], label=name)
plt.title(f"{var_in_nc} [{unit}] - ({start_year}-{end_year}) - {CSA}")
plt.xlabel("Year")
plt.ylabel(var_in_nc)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# Temporal correlations on annual mean
# =========================================================
df_corr = pd.DataFrame({name: df['value'] for name, df in ts_annual.items()}, index=years_common)

names = df_corr.columns.tolist()
n = len(names)
pearson = np.full((n, n), np.nan)
spearman = np.full((n, n), np.nan)

for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        if i == j:
            pearson[i, j] = 1
            spearman[i, j] = 1
            continue
        a, b = df_corr[ni].values, df_corr[nj].values
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() > 1:
            pearson[i, j] = np.corrcoef(a[mask], b[mask])[0, 1]
            spearman[i, j], _ = spearmanr(a[mask], b[mask])

plot_heatmap(pearson, names, f"Pearson correlation {CSA} {var_in_nc} [{unit}]")
plot_heatmap(spearman, names, f"Spearman correlation {CSA} {var_in_nc} [{unit}]")

print("=== DONE ===")
