import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIGURATION
# =========================================================

CONFIG = {
    "era5-2km": dict(start_year=1989, end_year=2018),
    "cerra": dict(start_year=1981, end_year=2010),
    "eobs": dict(start_year=1981, end_year=2010)
}

# USER SETTINGS
pilotarea = "Birmingham"
indicator = "cdd"
var_name = "CDD"

# PATHS
base = f"/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/{pilotarea}/INDICATORS"

# =========================================================
# FILENAME BUILDER
# =========================================================

def construct_filename(pilotarea, dataset_name, indicator, start_year, end_year):
    pilot_lower = pilotarea.lower()
    if dataset_name in ["cerra", "eobs"]:
        return f"{pilot_lower}_{dataset_name}_{indicator}_eu_{start_year}_{end_year}.nc"
    elif dataset_name == "era5-2km":
        return f"{pilot_lower}_{dataset_name}_{indicator}_{start_year}{end_year}.nc"
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

# =========================================================
# LOAD DATASETS AND KEEP FULL SPATIAL VALUES
# =========================================================

loaded_data = {}  # dataset_name -> 1D numpy array (all spatial values)

for dataset_name, params in CONFIG.items():
    start_year = params.get("start_year")
    end_year = params.get("end_year")

    try:
        filename = construct_filename(pilotarea, dataset_name, indicator, start_year, end_year)
        filepath = os.path.join(base, filename)
        if not os.path.exists(filepath):
            print(f"File not found for {dataset_name}: {filepath} -> skipping")
            continue

        ds = xr.open_dataset(filepath)
        if var_name not in ds:
            print(f"Variable '{var_name}' not in {dataset_name} -> skipping")
            continue

        da = ds[var_name]
        loaded_data[dataset_name] = da.values.flatten()  # flatten spatial dims
        print(f"Loaded {dataset_name}, total spatial points: {loaded_data[dataset_name].size}")

    except Exception as e:
        print(f"Error loading {dataset_name}: {e} -> skipping")
        continue

# =========================================================
# BOXPLOT OF SPATIAL VALUES
# =========================================================

if not loaded_data:
    raise RuntimeError("No datasets loaded")

fig, ax = plt.subplots(figsize=(8,6))
ax.boxplot([vals for vals in loaded_data.values()], labels=list(loaded_data.keys()))
ax.set_ylabel(var_name)
ax.set_title(f'Spatial distribution of {var_name} in {pilotarea} for each dataset')
plt.show()

# =========================================================
# COMPUTE SPATIAL MEANS
# =========================================================

means = np.array([vals.mean() for vals in loaded_data.values()])
datasets = list(loaded_data.keys())

# =========================================================
# PAIRWISE AGREEMENT HEATMAP
# =========================================================

n = len(datasets)
pairwise_diff = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        pairwise_diff[i,j] = abs(means[i] - means[j])

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(pairwise_diff, xticklabels=datasets, yticklabels=datasets,
            annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label':'|Difference|'})
ax.set_title(f'Pairwise agreement (absolute difference) for {pilotarea} - {indicator}')
plt.tight_layout()
plt.show()
