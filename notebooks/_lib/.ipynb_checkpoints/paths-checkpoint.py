from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

def find_repo_root(start: Path) -> Path:
    for p in [start.resolve()] + list(start.resolve().parents):
        if (p / ".git").exists():
            return p
    raise RuntimeError("Could not find repo root (.git not found).")

REPO_ROOT = find_repo_root(Path.cwd())


# In this repository, the dataset tree lives directly under the repo root
DATA_ROOT: Path = REPO_ROOT

# Outputs (should NOT be committed)
OUT_ROOT: Path = REPO_ROOT / "outputs"
TABLE_DIR: Path = OUT_ROOT / "tables"
FIG_DIR: Path = OUT_ROOT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def iter_indicators_dirs(root: Path = DATA_ROOT) -> List[Path]:
    """
    Return all INDICATORS directories in the repo, assuming:
        <CSA>/**/INDICATORS/
    but excluding anything under outputs/ or hidden folders.
    """
    ignore_parts = {"outputs", ".git", ".ipynb_checkpoints", ".cache"}
    dirs: List[Path] = []
    for p in root.rglob("INDICATORS"):
        if not p.is_dir():
            continue
        parts = set(p.parts)
        if parts & ignore_parts:
            continue
        dirs.append(p)
    # stable ordering
    return sorted(set(dirs))


def iter_indicator_files(pattern: str = "*.nc", root: Path = DATA_ROOT) -> Iterable[Path]:
    """Yield NetCDF indicator files from all INDICATORS dirs."""
    for d in iter_indicators_dirs(root=root):
        yield from d.rglob(pattern)


def resolve_indicator_by_basename(basename: str, root: Path = DATA_ROOT) -> Path:
    """
    Find a file by basename within any INDICATORS folder.
    Useful when manifests only store filenames.
    Raises FileNotFoundError if not found or ambiguous.
    """
    hits = [p for p in iter_indicator_files(pattern=basename, root=root)]
    if not hits:
        raise FileNotFoundError(f"Could not find {basename} under any INDICATORS folder.")
    if len(hits) > 1:
        msg = "\n".join(str(h) for h in hits[:10])
        raise FileExistsError(
            f"Basename {basename} is ambiguous ({len(hits)} matches). Examples:\n{msg}"
        )
    return hits[0]
