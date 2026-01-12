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

# Table with the indicators for every CSAs

| Prague | Leipzig | Funen-Odense | Athens | Barcelona | Bologna | Brasov | Birmingham |
|--------|---------|--------------|--------|-----------|---------|--------|------------|
| Mean radiant temperature (MRT) | Fire risk occurrence levels for next day | Changes in windstorm intensity and frequency | Number of days with Tmin > 20°C | Maximum Temperatures | Annual cumulative precipitation | Hourly rainfall | Hourly rainfall |
| Thermal comfort indices (e.g. UTCI) | Extreme heat indicators | Changes in maximum daily/subdaily wind speed | Number of hot nights | Heatwave length | Maximum summer temperature | Hourly temperature | Monthly prec. > average |
| Air pollution concentration | Flood indicators |  | Number of hot days | SPI | Average annual temperature | SPI | Indicators on vegetation |
| Air temperature |  |  | Maximum duration of heat waves | Consecutive dry days | Tropical night | Summer Days | Accumulated degree days above threshold for tree species |
| Surface temperature |  |  | CCDs | Fire weather index | CDDs | UTCI | SPI |
| Relative humidity |  |  | Thermal amplitude in degrees |   | Intensity and time duration HWs | WET | Low monthly precipitation |
|  |  |  | Extreme maximum temperature |  | PET and PMV | CDD | Grass phenology indicators |
|  |  |  | Number of dry days |  | Air quality index | Number of days without precipitation | Accumulated degree days above threshold |
|  |  |  | CDD |  | Urban heatwave | Length of drought period | High-resolution hazard maps across the area |
|  |  |  | Number of rainfall days |  | Thermal index | Number of days with snow cover | Tropical nights |
|  |  |  | Maximum number of consecutive wet days |  |  | Number of days with T < -3°C | Hot days |
|  |  |  | Accumulated maximum rainfall in 5 days |  |  | Number of days with SD > 30cm | Summer Days |
|  |  |  | Max, Min wind speed |  |  | Number of days with SD > 5cm | Leaf area index |
