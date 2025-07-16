""" Python script to calculate climate indicators for the CSA Birmingham"""

import xarray as xr
import xclim as xc
from scipy.stats._warnings_errors import FitError
from xclim.core.units import convert_units_to


def wetdays(pr, thresh="10 mm/day", freq="YS"):
    """
    The number of days with daily precipitation at or above a given threshold.

    Parameters
    ----------
    pr : Dataarray
        precipitation
    thresh : str
        Threshold above which wet days are counted
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Number of wetdays (units: days)
    """

    # Assure data is daily:
    pr_daily = pr.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.wetdays(pr_daily, thresh=thresh, freq=freq)
    return da


def cwd(pr, thresh="1 mm/day", freq="YS"):
    """
    The longest number of consecutive days where daily precipitation is at or above a given threshold.

    Parameters
    ----------
    pr : Dataarray
        precipitation
    thresh : str
        Threshold above which wet days are counted
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Maximum number of consecutive wet days (units: days)
    """

    # Assure data is daily:
    pr_daily = pr.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.maximum_consecutive_wet_days(pr_daily, thresh=thresh, freq=freq)
    return da


def cdd(pr, thresh="1 mm/day", freq="YS"):
    """
    The longest number of consecutive days where daily precipitation is below a given threshold.

    Parameters
    ----------
    pr : Dataarray
        precipitation
    thresh : str
        Threshold above which wet days are counted
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Maximum number of consecutive dry days (units: days)
    """

    # Assure data is daily:
    pr_daily = pr.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.maximum_consecutive_dry_days(pr_daily, thresh=thresh, freq=freq)
    return da


def prcptot(pr, freq="YS"):
    """
    Total accumulated precipitation.

    Parameters
    ----------
    pr : Dataarray
        precipitation
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Total precipitation (units: mm)
    """

    # Assure data is daily:

    pr_daily = pr.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    da = xc.indicators.atmos.precip_accumulation(pr_daily, freq=freq)
    return da


def rx1day(pr, freq="YS"):
    """
    Maximum 1-day total precipitation for a given period (freq)

    Parameters
    ----------
    pr : Dataarray
        precipitation
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Maximum daily precipitation (units: mm/day)
    """

    # Assure data is daily:
    pr_daily = pr.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.max_1day_precipitation_amount(pr_daily, freq=freq)
    return da


def spi(pr, freq="MS"):
    """
    Standardized Precipitation Index

    Parameters
    ----------
    pr : Dataarray
        precipitation
    freq: str
        Frequency can be daily ('D'), weekly ('W'), monthly ('MS')
        Weekly frequency will only work if the input array has a “standard” (non-cftime) calendar.

    Returns
    -------
    Dataarray
        Standardized Precipitation Index (units: 1)
        Probability of zero (units: n/a)
    """

    # Assure data is daily:
    pr_daily = pr.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    try:
        da = xc.indicators.atmos.standardized_precipitation_index(pr_daily, freq=freq)
        return da
    except FitError as e:
        # Sometimes calculation of SPI fails, in that case return an empty Dataset
        print(f"Caught FitError: {e}")
        return xr.Dataset()


def txx(tasmax, freq="YS"):
    """
    Maximum of daily maximum temperature over a given period

    Parameters
    ----------
    tasmax : Dataarray
        maximum temperature
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Highest daily temperature over given period (units: degree Celsius)
    """

    # Assure data is daily:
    tasmax_daily = tasmax.resample(time="D").max(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.tx_max(tasmax_daily, freq=freq)
    da = convert_units_to(da, "degC")
    return da


def txn(tasmax, freq="YS"):
    """
    Minimum of daily maximum temperature over a given period

    Parameters
    ----------
    tasmax : Dataarray
        maximum temperature
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Lowest daily temperature over given period (units: degree Celsius)
    """

    # Assure data is daily:
    tasmax_daily = tasmax.resample(time="D").max(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.tx_min(tasmax_daily, freq=freq)
    da = convert_units_to(da, "degC")
    return da


def hot_spell_frequency(tasmax, hot_spell_type, freq="YS"):
    """
    The frequency of hot periods of N days or more,
    during which the temperature over a given time window of days is above a given threshold.

    Parameters
    ----------
    tasmax : Dataarray
        maximum temperature
    hot_spell_type: str
        type of day classification: 'summer' (25 degC, 1 day), 'hot' (30 degC, 1 day), 'heatwave' (27 degC, 3 days)
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Number of days (summer, hot) or spells (heatwave) over a given period (units: 1)
    """

    # Define types:
    hot_spell = {
        "summer": {
            "name": "summer_days",
            "thresh": "25 degC",
            "window": 1,
        },
        "hot": {
            "name": "hot_days",
            "thresh": "30 degC",
            "window": 1,
        },
        "heatwave": {
            "name": "heatwave_spell",
            "thresh": "27 degC",
            "window": 3,
        },
    }

    # Assure data is daily:
    tasmax_daily = tasmax.resample(time="D").max(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.hot_spell_frequency(
        tasmax_daily,
        thresh=hot_spell[hot_spell_type]["thresh"],
        window=hot_spell[hot_spell_type]["window"],
        freq=freq,
    )
    # da.rename(hot_spell[hot_spell_type]["name"])
    # Shorten long_name to avoid segmentation error
    da.attrs["long_name"] = hot_spell[hot_spell_type]["name"]
    return da


def tropical_nights(tasmin, freq="YS"):
    """
    Number of tropical nights

    Parameters
    ----------
    tasmin : Dataarray
        minimum temperature
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Number of tropical nights over given period (units: days)
    """

    # Assure data is daily:
    tasmin_daily = tasmin.resample(time="D").max(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.tropical_nights(tasmin_daily, freq=freq)
    return da


def growing_season_length(tas, freq="YS"):
    """
    Number of days between the first occurrence of a series of days with a daily average temperature above a threshold
    and the first occurrence of a series of days with a daily average temperature below that same threshold,
    occurring after a given calendar date.

    Parameters
    ----------
    tas : Dataarray
        mean temperature
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Length of the growing season over given period, usually years (units: days)
    """

    # Assure data is daily:
    tas_daily = tas.resample(time="D").mean(keep_attrs=True)

    # Calculate indicator
    da = xc.atmos.growing_season_length(tas_daily, freq=freq)
    # Shorten long_name to avoid segmentation error
    da.attrs["long_name"] = "Growing season length"
    return da


def gdd_baker(tasmax, tasmin, tbase=10, freq="YS"):
    """
    Number of degree days calculated according Baker et al. (1980) using maximum and minimum temperature

    Parameters
    ----------
    tasmax : Dataarray
        maximum temperature
    tasmin : Dataarray
        minimum temperature
    tbase : int or float
    freq: str
        Frequency can be monthly ('MS'), seasonal ('QS-DEC'), yearly ('YS') or other

    Returns
    -------
    Dataarray
        Number of growing degree days over given period, usually years (units: 1)
    """

    # Assure data is daily:
    tasmax_daily = tasmax.resample(time="D").max(keep_attrs=True)
    tasmax_daily = convert_units_to(tasmax_daily, "degC")
    tasmin_daily = tasmin.resample(time="D").min(keep_attrs=True)
    tasmin_daily = convert_units_to(tasmin_daily, "degC")

    # Compute average temperature
    tavg = (tasmax_daily + tasmin_daily) / 2

    # Apply the nested conditions
    D = xr.where(
        tasmax_daily <= tbase,
        0,
        xr.where(
            tasmin_daily >= tbase,
            0.5 * (tasmax_daily + tasmin_daily) - tbase,
            xr.where(
                tavg > tbase,
                (tasmax_daily - tbase) / 2 + (tbase - tasmin_daily) / 4,
                (tasmax_daily - tbase) / 4,
            ),
        ),
    )

    gdd = D.resample(time="YS").sum(keep_attrs=True)

    gdd.attrs = {
        "long_name": "Number of growing degree days",
        "description": "Annual Number of growing degree days calculated according to Baker et al (1980)",
        "units": "1",
        "cell_methods": "",
        "standard_name": "growing_degree_days",
    }

    return gdd
