import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

'''
To ensure meaningful comparability among climate indicators derived from different datasets, all indicators must be computed over the same reference time period. Indicators calculated over different temporal windows may reflect not only dataset- or model-related differences, but also the effects of climate variability and long-term trends, thus limiting the robustness of any comparative analysis.

The code presented here is intended to demonstrate the functionality of the comparison workflow, using currently available indicator datasets as input. At this stage, these indicators are computed over non-uniform time periods, and the results should therefore be interpreted as a technical demonstration rather than a scientifically consistent comparison.

For robust and reproducible analyses, it is essential to rely on the underlying time series data, from which indicators can be recomputed over a common shared period across all datasets.

'''

# =========================================================
# CONFIGURATION
# =========================================================

CONFIG = {
    "era5-2km": dict(start_year=1989, end_year=2018),
    "cerra": dict(start_year=1981, end_year=2010),
    "eobs": dict(start_year=1981, end_year=2010),
     "emo": dict(start_year=1991, end_year=2020)
     }

pilotarea = "Barcelona"
indicator = "rr"
var_name = "RR"

path = "/work/CARMINE"
base = f"{path}/CARMINE-T2.4/{pilotarea}/INDICATORS"

# =========================================================
# FILENAME BUILDER
# =========================================================

def construct_filename(pilotarea, dataset_name, indicator, start_year, end_year):
    p = pilotarea.lower()
    if dataset_name in ["cerra", "eobs"]:
        return f"{p}_{dataset_name}_{indicator}_eu_{start_year}_{end_year}.nc"
    elif dataset_name == "era5-2km":
        return f"{p}_{dataset_name}_{indicator}_{start_year}{end_year}.nc"
    elif dataset_name == "emo":
        return f"{p}_{dataset_name}_{indicator}{start_year}{end_year}.nc"
    else:
        raise ValueError(dataset_name)

# =========================================================
# SPATIAL COORDINATE DETECTOR
# =========================================================

def get_spatial_coords(da):
    candidates = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("rlon", "rlat"),
        ("x", "y")
    ]
    for x, y in candidates:
        if x in da.dims and y in da.dims:
            return x, y
    raise RuntimeError(f"Cannot identify spatial coords in {list(da.dims)}")

# =========================================================
# GRID RESOLUTION ESTIMATOR
# =========================================================

def grid_resolution(da):
    x, y = get_spatial_coords(da)
    dx = np.median(np.abs(np.diff(da[x].values)))
    dy = np.median(np.abs(np.diff(da[y].values)))
    return dx * dy

# =========================================================
# LOAD DATASETS
# =========================================================

loaded = {}

for name, cfg in CONFIG.items():
    fn = construct_filename(pilotarea, name, indicator,
                            cfg["start_year"], cfg["end_year"])
    path = os.path.join(base, fn)

    if not os.path.exists(path):
        print(f"Missing: {name}")
        continue

    ds = xr.open_dataset(path)

    if var_name not in ds:
        print(f"{var_name} not in {name}")
        continue

    da = ds[var_name]
    get_spatial_coords(da)

    loaded[name] = da
    print(f"Loaded {name}: {da.sizes}")

if len(loaded) < 2:
    raise RuntimeError("Not enough datasets")

# =========================================================
# FIND COARSEST GRID
# =========================================================

res = {name: grid_resolution(da) for name, da in loaded.items()}
ref_name = max(res, key=res.get)
ref_da = loaded[ref_name]

print(f"\nReference grid (coarsest): {ref_name}")

# =========================================================
# INTERPOLATE TO REFERENCE GRID
# =========================================================

interp = {}

ref_x, ref_y = get_spatial_coords(ref_da)

for name, da in loaded.items():
    if name == ref_name:
        interp[name] = da
    else:
        x, y = get_spatial_coords(da)
        interp[name] = da.interp(
            {x: ref_da[ref_x], y: ref_da[ref_y]},
            method="linear"
        )

# =========================================================
# TRUE COMMON MASK (AND of all datasets)
# =========================================================

common_mask = np.isfinite(ref_da)
for da in interp.values():
    common_mask = common_mask & np.isfinite(da)

for name in interp:
    interp[name] = interp[name].where(common_mask)

# =========================================================
# DEBUG: CHECK FIELD VARIABILITY
# =========================================================

print("\nField standard deviations after interpolation + masking:")
for name, da in interp.items():
    print(f"{name}: std = {np.nanstd(da.values):.4f}")

# =========================================================
# PAIRWISE STATISTICS
# =========================================================

def compute_stats(da1, da2):
    x = da1.values.flatten()
    y = da2.values.flatten()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    return (
        np.mean(y - x),                     # bias
        np.sqrt(np.mean((y - x) ** 2)),     # rmse
        np.corrcoef(x, y)[0, 1],            # pearson
        spearmanr(x, y)[0]                  # spearman
    )

names = list(interp.keys())
n = len(names)

bias = np.zeros((n, n))
rmse = np.zeros((n, n))
pearson = np.zeros((n, n))
spearman = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            bias[i, j] = 0.0
            rmse[i, j] = 0.0
            pearson[i, j] = 1.0
            spearman[i, j] = 1.0
        else:
            b, r, p, s = compute_stats(
                interp[names[i]],
                interp[names[j]]
            )
            bias[i, j] = b
            rmse[i, j] = r
            pearson[i, j] = p
            spearman[i, j] = s

# =========================================================
# PLOTTING
# =========================================================

def plot_heatmap(mat, title, cbar_label,
                 vmin=None, vmax=None, cmap="coolwarm"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        mat,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": cbar_label}
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

plot_heatmap(
    pearson,
    f"Spatial correlation (Pearson) ?~@~S {pilotarea}",
    "r",
    vmin=0, vmax=1
)

plot_heatmap(
    spearman,
    f"Spatial correlation (Spearman) ?~@~S {pilotarea}",
    "?~A",
    vmin=0, vmax=1
)

plot_heatmap(
    rmse,
    f"Spatial RMSE ?~@~S {pilotarea}",
    var_name,
    cmap="viridis"
)

plot_heatmap(
    bias,
    f"Spatial bias ?~@~S {pilotarea}",
    var_name,
    cmap="RdBu_r"
)                  
