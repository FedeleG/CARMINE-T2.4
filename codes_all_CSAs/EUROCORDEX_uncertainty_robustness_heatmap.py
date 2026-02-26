# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:11:49 2026

@author: reinhvlr
"""

# -*- coding: utf-8 -*-
"""
EURO-CORDEX annual indicators — quick model sensitivity + robustness analysis (Spyder script)

Assumptions (based on your dataset description):
- ROOT = Cordex_Carmine
- ROOT/<CSA>/ contains NetCDFs named like:
  CARMINE_Cordex_CMIP5_Athens_mod05_RR10_BSL_1991_2020_YY_1971_2100_rcp26.nc
- One indicator variable per file (e.g., RR10, PTOT), annual time axis (1971–2100), dims (time,y,x)
- Ignore ROOT/Ensemble_Outputs for now

Outputs:
- Master robustness table (CSV):
    robustness_allCSAs_allIndicators.csv
    (contains per CSA × indicator × scenario × window:
     median_slope_decade, IQR_slope_decade,
     robustness_index, spread_ratio,
     frac_positive, frac_negative)

- Robustness heatmaps (PNG):
    Separate figures per:
        scenario × time window × unit group

    Rows: indicators
    Columns: CSAs
    Color: robustness_index (|median slope| / IQR)

    Fixed color scales tested:
        0–3
        0–5

    Saved to:
    analysis_outputs/fig_robustness_heatmaps/p

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG (edit these in Spyder)
# =============================================================================
folderlocation = Path(r"set path to your location of Cordex_Carmine")
ROOT = folderlocation / "Cordex_Carmine"          # <-- set to your Cordex_Carmine folder
#CSA = "Barcelona"                              # e.g., "Athens"
INDICATOR_CODE = "SU"                     # e.g., "RR10" or "PTOT"
SCENARIOS = ("rcp26", "rcp45", "rcp85")     # expected scenarios
SPATIAL_REDUCER = "median"                    # "mean" or "median"

# Trend windows (inclusive). Use None to skip a window.
# If you only want full period, keep just that entry.
TREND_WINDOWS = [
    ("full", 1971, 2100),
    ("near", 1981, 2010),
    ("near2", 1991, 2020),
    ("mid", 2021, 2050),
    ("mid2", 2036, 2065),
    ("late", 2071, 2100),
]

# Use all models found for CSA+indicator+scenario, or restrict:
MODEL_WHITELIST = None  # e.g., ["mod01", "mod02"] or None for all

# Where to save outputs
OUTDIR = ROOT / "analysis_outputs"
OVERWRITE = True
TABLEDIR = OUTDIR / "tables"
TABLEDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

FNAME_RE = re.compile(
    r"^CARMINE_Cordex_CMIP5_"
    r"(?P<csa>[^_]+)_"
    r"(?P<model>mod\d+)_"
    r"(?P<ind>.+?)_BSL_"          # indicator = between model and _BSL_
    r"(?P<b0>\d{4})_(?P<b1>\d{4})_"
    r"(?P<freq>[A-Za-z0-9]+)_"
    r"(?P<t0>\d{4})_(?P<t1>\d{4})_"
    r"(?P<rcp>rcp\d+)\.nc$"
)

def list_indicators(root: Path) -> List[str]:
    indicators = set()

    for csa in list_csas(root):
        csa_dir = root / csa
        for p in csa_dir.glob("*.nc"):
            meta = parse_filename(p)
            indicators.add(meta["ind"])

    return sorted(indicators)

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def run_one_csa(root: Path, csa: str, indicator_code: str) -> Optional[pd.DataFrame]:
    # 1) catalog
    df_cat = build_catalog(root, csa, indicator_code)
    df_cat = df_cat.sort_values(["rcp", "model"]).reset_index(drop=True)

    # 2) load series
    df_long, varname, units_local = load_all_series(df_cat, SPATIAL_REDUCER, SCENARIOS)

    # 3) trends
    df_slopes = analyze_trends(df_long)

    # 4) extended summary (per scenario x window)
    summary_rows = []
    for (sc, w), g in df_slopes.groupby(["scenario", "window"]):
        slopes_year = g["slope_per_year"].dropna().values
        slopes_decade = slopes_year * 10.0

        if slopes_decade.size == 0:
            continue

        median = float(np.nanmedian(slopes_decade))
        q25 = float(np.nanquantile(slopes_decade, 0.25))
        q75 = float(np.nanquantile(slopes_decade, 0.75))
        iqr = q75 - q25

        slope_min = float(np.nanmin(slopes_decade))
        slope_max = float(np.nanmax(slopes_decade))

        frac_pos = float(np.mean(slopes_decade > 0))
        frac_neg = float(np.mean(slopes_decade < 0))

        robustness_index = np.nan
        spread_ratio = np.nan
        if iqr > 0:
            robustness_index = abs(median) / iqr
        if abs(median) > 0:
            spread_ratio = iqr / abs(median)

        summary_rows.append({
            "CSA": csa,
            "indicator": indicator_code,
            "units": units_local,
            "scenario": sc,
            "window": w,
            "n_models": int(slopes_decade.size),

            "median_slope_decade": median,
            "IQR_slope_decade": iqr,
            "slope_min_decade": slope_min,
            "slope_max_decade": slope_max,

            "frac_positive": frac_pos,
            "frac_negative": frac_neg,

            "robustness_index": robustness_index,
            "spread_ratio": spread_ratio,
        })

    if not summary_rows:
        return None

    df_summary = pd.DataFrame(summary_rows).sort_values(["window", "scenario"])
    return df_summary

def list_csas(root: Path) -> List[str]:
    csas = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.lower() == "ensemble_outputs":
            continue
        csas.append(p.name)
    return sorted(csas)

def list_nc_files_for_csa(root: Path, csa: str) -> List[Path]:
    csa_dir = root / csa
    if not csa_dir.exists():
        raise FileNotFoundError(f"CSA folder not found: {csa_dir}")
    # only files directly inside CSA folder per your structure
    return sorted([p for p in csa_dir.glob("*.nc") if p.is_file()])


def parse_filename(path: Path) -> Dict[str, object]:
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    d = m.groupdict()
    # cast numeric fields
    for k in ("b0", "b1", "t0", "t1"):
        d[k] = int(d[k])
    d["path"] = str(path)
    return d


def open_indicator_series(nc_path: Path, reducer: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Returns: years (int), series (float), varname, units
    """
    ds = xr.open_dataset(
        nc_path,
        mask_and_scale=True,
        decode_cf=True,
        decode_times=False,
        decode_timedelta=False,
    )

    # pick the data variable: assume exactly one "main" variable (not coords)
    # preference: variable named same as indicator in filename, else first data_var
    data_vars = list(ds.data_vars)
    if not data_vars:
        ds.close()
        raise ValueError(f"No data variables found in {nc_path.name}")

    # choose var
    varname = data_vars[0]
    if len(data_vars) > 1:
        # try find one with dims including time,y,x; else keep first
        candidates = []
        for v in data_vars:
            da = ds[v]
            if "time" in da.dims and ("y" in da.dims or "rlat" in da.dims) and ("x" in da.dims or "rlon" in da.dims):
                candidates.append(v)
        if candidates:
            varname = candidates[0]

    da = ds[varname]

    # years coordinate: your files have year(time=130) coordinate variable
    if "year" in ds.variables and "time" in ds["year"].dims:
        years = ds["year"].values
    else:
        # fallback: time coordinate values
        years = da["time"].values

    years = np.asarray(years).astype(int)

    # spatial reduction over y/x (or rlat/rlon)
    spatial_dims = [d for d in da.dims if d.lower() in ("y", "x", "rlat", "rlon")]
    if len(spatial_dims) < 2:
        # some products might already be reduced; still return 1D
        series = da.values
    else:
        if reducer == "mean":
            series = da.mean(dim=spatial_dims, skipna=True).values
        elif reducer == "median":
            series = da.median(dim=spatial_dims, skipna=True).values
        else:
            ds.close()
            raise ValueError(f"Unknown reducer: {reducer}")

    series = np.asarray(series, dtype=float)

    units = str(da.attrs.get("units", "")).strip()
    ds.close()
    return years, series, varname, units


def linear_slope(years: np.ndarray, values: np.ndarray, y0: int, y1: int) -> float:
    """
    OLS slope in 'units per year' over [y0, y1] inclusive, ignoring NaNs.
    """
    mask = (years >= y0) & (years <= y1) & np.isfinite(values)
    x = years[mask].astype(float)
    y = values[mask].astype(float)
    if x.size < 3:
        return np.nan
    # slope, intercept
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def safe_title(indicator: str, units: str, reducer: str) -> str:
    if units:
        return f"{indicator} ({units}), spatial {reducer}"
    return f"{indicator}, spatial {reducer}"


def quantiles_over_models(mat: np.ndarray, qs=(0.25, 0.5, 0.75)) -> Dict[float, np.ndarray]:
    # mat: (n_models, n_time)
    out = {}
    for q in qs:
        out[q] = np.nanquantile(mat, q, axis=0)
    return out


# =============================================================================
# Main analysis
# =============================================================================

def build_catalog(root: Path, csa: str, indicator_code: str) -> pd.DataFrame:
    files = list_nc_files_for_csa(root, csa)
    rows = []
    for f in files:
        meta = parse_filename(f)
        if meta["csa"] != csa:
            continue
        if str(meta["ind"]) != indicator_code:
            continue
        rows.append(meta)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No files found for CSA={csa}, indicator={indicator_code} in {root/csa}")
    return df


def load_all_series(df_cat: pd.DataFrame, reducer: str, scenarios: Tuple[str, ...]) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns a long DataFrame with columns:
      model, scenario, year, value
    plus varname and units (checked for consistency).
    """
    records = []
    varnames = set()
    units_set = set()

    for _, r in df_cat.iterrows():
        sc = r["rcp"]
        if sc not in scenarios:
            continue
        model = r["model"]
        if MODEL_WHITELIST is not None and model not in MODEL_WHITELIST:
            continue

        years, series, varname, units = open_indicator_series(Path(r["path"]), reducer)
        varnames.add(varname)
        units_set.add(units)

        for y, v in zip(years, series):
            records.append({"model": model, "scenario": sc, "year": int(y), "value": float(v)})

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("After filtering, no time series records were loaded.")

    # consistency checks (soft)
    varname_final = sorted(list(varnames))[0] if varnames else ""
    units_final = sorted(list(units_set))[0] if units_set else ""
    if len(varnames) > 1:
        print(f"[warn] Multiple varnames found: {sorted(varnames)} (using '{varname_final}' for labeling)")
    if len(units_set) > 1:
        print(f"[warn] Multiple units found: {sorted(units_set)} (using '{units_final}' for labeling)")

    return df, varname_final, units_final


def analyze_trends(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model slopes for each scenario and each window.
    """
    out_rows = []
    for sc in sorted(df_long["scenario"].unique()):
        df_sc = df_long[df_long["scenario"] == sc]
        for model in sorted(df_sc["model"].unique()):
            s = df_sc[df_sc["model"] == model].sort_values("year")
            years = s["year"].to_numpy()
            vals = s["value"].to_numpy()

            for wname, y0, y1 in TREND_WINDOWS:
                slope = linear_slope(years, vals, y0, y1)
                out_rows.append({
                    "scenario": sc,
                    "model": model,
                    "window": wname,
                    "start_year": y0,
                    "end_year": y1,
                    "slope_per_year": slope
                })
    return pd.DataFrame(out_rows)


def leave_one_out_influence(df_long: pd.DataFrame, window: Tuple[str, int, int]) -> pd.DataFrame:
    """
    Influence of each model on ensemble median slope:
    - compute median series across models (per year), then slope of that median
    - recompute leaving one model out; report delta slope
    """
    wname, y0, y1 = window
    rows = []

    for sc in sorted(df_long["scenario"].unique()):
        dsc = df_long[df_long["scenario"] == sc]
        models = sorted(dsc["model"].unique())
        years_all = np.sort(dsc["year"].unique())

        # build matrix models x years
        mat = np.full((len(models), len(years_all)), np.nan, dtype=float)
        for i, m in enumerate(models):
            s = dsc[dsc["model"] == m].set_index("year")["value"]
            mat[i, :] = s.reindex(years_all).to_numpy()

        # ensemble median slope with all models
        med_all = np.nanmedian(mat, axis=0)
        slope_all = linear_slope(years_all, med_all, y0, y1)

        for i, m in enumerate(models):
            mat_loo = np.delete(mat, i, axis=0)
            med_loo = np.nanmedian(mat_loo, axis=0)
            slope_loo = linear_slope(years_all, med_loo, y0, y1)
            rows.append({
                "scenario": sc,
                "window": wname,
                "start_year": y0,
                "end_year": y1,
                "model_left_out": m,
                "slope_all_models": slope_all,
                "slope_leave_one_out": slope_loo,
                "delta_slope": slope_loo - slope_all
            })

    return pd.DataFrame(rows)

# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    if "..." in str(ROOT):
        print("[info] ROOT is still a placeholder.")
        return

    ensure_outdir(OUTDIR)

    csas = list_csas(ROOT)
    indicators = list_indicators(ROOT)

    print(f"[info] Found {len(csas)} CSAs")
    print(f"[info] Found {len(indicators)} indicators")

    all_summaries = []
    failed = []

    for indicator in indicators:
        print(f"\n[INDICATOR] {indicator}")

        for csa in csas:
            try:
                df_sum = run_one_csa(ROOT, csa, indicator)
                if df_sum is None or df_sum.empty:
                    continue
                all_summaries.append(df_sum)
                print(f"[ok] {csa}")
            except Exception as e:
                failed.append((indicator, csa, str(e)))
                print(f"[fail] {indicator} | {csa}: {e}")

    if not all_summaries:
        print("[error] No results produced.")
        return

    df_all = pd.concat(all_summaries, ignore_index=True)

    out_csv = TABLEDIR / "robustness_allCSAs_allIndicators.csv"
    df_all.to_csv(out_csv, index=False)

    print(f"\n[done] Master robustness table written to:")
    print(out_csv.resolve())

    if failed:
        fail_csv = TABLEDIR / "robustness_allCSAs_allIndicators__FAILED.csv"
        pd.DataFrame(failed, columns=["indicator", "CSA", "error"]).to_csv(fail_csv, index=False)
        print(f"[warn] Some combinations failed. See:")
        print(fail_csv.resolve())

if __name__ == "__main__":
    main()
    
# =============================================================================
# FIGURE 2: Robustness Heatmaps (robustness_index)
# - Separate figures per scenario × window × unit-group
# - Fixed color scales: 0–3 and 0–5
# =============================================================================

IN_CSV = TABLEDIR / "robustness_allCSAs_allIndicators.csv"

HEATDIR = OUTDIR / "fig_robustness_heatmaps"
HEATDIR.mkdir(parents=True, exist_ok=True)

# Fixed scales to test
SCALES = [(0, 3), (0, 5)]

# Load
df = pd.read_csv(IN_CSV)

# Build window label map
WINDOW_LABEL = {name: f"{name} ({y0}–{y1})" for name, y0, y1 in TREND_WINDOWS}

# CSA order
CSA_ORDER = sorted(df["CSA"].unique().tolist())

# Unit grouping (same logic as before)
def unit_group(units):
    u = str(units).strip()
    if u == "days":
        return "Count (days)"
    elif u == "mm":
        return "Precipitation (mm)"
    elif u in ["degC", "degdays"]:
        return "Temperature"
    elif u == "m/s":
        return "Wind"
    elif u in ["score", "values"]:
        return "Index"
    else:
        return "Other"

df["unit_group"] = df["units"].apply(unit_group)

scenarios = sorted(df["scenario"].unique())
windows   = sorted(df["window"].unique())
unit_groups = sorted(df["unit_group"].unique())

for sc in scenarios:
    for w in windows:

        dsw = df[(df["scenario"] == sc) & (df["window"] == w)].copy()
        if dsw.empty:
            continue

        for ug in unit_groups:

            dug = dsw[dsw["unit_group"] == ug].copy()
            if dug.empty:
                continue

            # Pivot to indicator × CSA matrix
            mat = dug.pivot_table(
                index="indicator",
                columns="CSA",
                values="robustness_index"
            )

            # Reorder CSAs
            mat = mat.reindex(columns=[c for c in CSA_ORDER if c in mat.columns])

            # Sort indicators by mean robustness (descending)
            mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]

            for vmin, vmax in SCALES:

                fig, ax = plt.subplots(figsize=(8, 0.5 * len(mat) + 2))

                im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax)

                ax.set_xticks(np.arange(len(mat.columns)))
                ax.set_xticklabels(mat.columns, rotation=45, ha="right")

                ax.set_yticks(np.arange(len(mat.index)))
                ax.set_yticklabels(mat.index)

                ax.set_title(
                    f"Robustness Index — {ug} — {sc} — {WINDOW_LABEL.get(w, w)}\n"
                    f"Fixed scale {vmin}–{vmax}"
                )

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("robustness_index (|median| / IQR)")

                fig.tight_layout()

                fname = (
                    f"robustness_heatmap__{ug.replace(' ','_')}__"
                    f"{sc}__{w}__scale_{vmin}_{vmax}.png"
                )

                fig.savefig(HEATDIR / fname, dpi=220)
                plt.close(fig)

print(f"[done] Robustness heatmaps written to: {HEATDIR.resolve()}")