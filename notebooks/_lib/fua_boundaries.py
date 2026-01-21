# _lib/fua_boundaries.py
from __future__ import annotations

from pathlib import Path
import warnings

import geopandas as gpd


def load_fua_layer(fua_dir: str | Path) -> gpd.GeoDataFrame:
    """
    Load the 'all FUAs' boundary layer once.

    Prefers:
      <fua_dir>/FUA_Boundaries.shp

    Falls back to:
      first *.shp found anywhere under fua_dir.
    """
    fua_dir = Path(fua_dir)

    shp = fua_dir / "FUA_Boundaries.shp"
    if not shp.exists():
        cand = sorted(fua_dir.rglob("*.shp"))
        if not cand:
            raise FileNotFoundError(f"No .shp found under {fua_dir}")
        shp = cand[0]

    gdf = gpd.read_file(shp)
    # stash source path for debugging (optional)
    gdf.attrs["__source_shp__"] = str(shp)
    return gdf


def make_load_fua_boundary(
    fua_all: gpd.GeoDataFrame,
    name_field: str = "FUA_NAME",
    mapping: dict[str, str] | None = None,
    contains_fallback: bool = True,
):
    """
    Create a callable `load_fua_boundary(csa)` that returns the matching FUA geometry
    rows from `fua_all`.

    - `mapping` lets you map CSA names to shapefile names (e.g. "Prague" -> "Praha").
    - Matching is case-insensitive exact match first.
    - If `contains_fallback=True`, uses a substring contains fallback if exact match fails.
    """
    if mapping is None:
        mapping = {}

    if name_field not in fua_all.columns:
        raise KeyError(
            f"'{name_field}' not found in FUA layer columns: {list(fua_all.columns)}"
        )

    s_all = fua_all[name_field].astype(str)

    def load_fua_boundary(csa: str) -> gpd.GeoDataFrame | None:
        target = mapping.get(csa, csa)

        target_norm = str(target).strip().lower()
        s_norm = s_all.str.strip().str.lower()

        # 1) exact match
        m = s_norm == target_norm
        gdf = fua_all.loc[m].copy()

        # 2) contains fallback
        if gdf.empty and contains_fallback:
            m2 = s_norm.str.contains(target_norm, na=False)
            gdf = fua_all.loc[m2].copy()

        if gdf.empty:
            warnings.warn(
                f"No FUA geometry found for CSA='{csa}' (target='{target}') "
                f"using field '{name_field}'."
            )
            return None

        return gdf

    return load_fua_boundary

# -----------------------------------------------------------------------------
# Default CSA list used across CARMINE notebooks
# -----------------------------------------------------------------------------

CSA_LIST = [
    "Athens",
    "Barcelona",
    "Bologna",
    "Birmingham",
    "Funen-Odense",
    "Prague",
    "Rotterdam",
]