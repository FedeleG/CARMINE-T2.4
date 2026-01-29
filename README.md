# CARMINE-T2.4

This is the GitHub repository set up for the implementation of post-processing tools for calculating climate indicators.

The repository is organized with a folder for each CSA, containing a .txt file to record progress. Within each folder, the python codes for each CSA will be developed.

Please comment with a small message every push you make, so it's easier to understand the modifications made

## Shared Paths Helper

To make the notebooks for post-processing runnable on any local machine without modifying absolute paths, this repository includes a shared paths helper at:

`path-helper-notebooks/_lib/paths.py`

### How it works
- Notebooks are expected to be launched from the `notebooks/` directory.
- The helper automatically detects the repository root by searching up the Directory tree for the `.git` folder.
- Through the helper, notebooks locate data (`INDICATORS/`) and create output folders
  (`outputs/tables/`, `outputs/figures/`) without hard-coded paths.

### Usage in notebooks
In any notebook, import the helper like this:

```python
from _lib.paths import (
    REPO_ROOT,
    DATA_ROOT,
    TABLE_DIR,
    FIG_DIR,
    iter_indicators_dirs,
    iter_indicator_files,
)
```


The helper will create the following directories on first use:

`outputs/tables/`
`outputs/figures/`


These folders are ignored by git (see .gitignore) because outputs
are not versioned.

For more information on the Github codes and data please refer to MS21 and MS4.
