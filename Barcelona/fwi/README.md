# FWI calculation for Barcelona

## Overview

This code calculates the FWI from EMO1 meteorological data (Gomes et al., 2020), adapted from the Global ECMWF Fire Forecast (GEFF) model (Giuseppe et al., 2016).

Gomes, Goncalo; Thiemig, Vera; Skøien, Jon Olav; Ziese, Markus; Rauthe-Schöch, Armin; Rustemeier, Elke; Rehfeldt, Kira; Walawender, Jakub; Kolbe, Christine; Pichon, Damien; Schweim, Christoph; Salamon, Peter (2020): EMO: A high-resolution multi-variable gridded meteorological data set for Europe. European Commission, Joint Research Centre (JRC) [Dataset] doi: 10.2905/0BD84BE4-CEC8-4180-97A6-8B3ADAAC4D26 PID: http://data.europa.eu/89h/0bd84be4-cec8-4180-97a6-8b3adaac4d26

Di Giuseppe, F., Pappenberger, F., Wetterhall, F., Krzeminski, B., Camia, A., Libertá, G. and San Miguel, J., 2016. The potential predictability of fire danger provided by numerical weather prediction. Journal of Applied Meteorology and Climatology, 55(11), pp.2469-2491. https://journals.ametsoc.org/doi/abs/10.1175/JAMC-D-15-0297.1

## Considerations for using EMO1 data

Noon temperature is calculated from daily max and min temperatures using  Sin (14R-1) method from Chow et al., (2007). Chow, D. H. C., & Levermore, G. J. (2007). New algorithm for generating hourly temperature values using daily maximum, minimum and average values from climate models. Building Services Engineering Research and Technology, 28(3), 237-248. 

Wind speed is limited to 15m/s to avoid very high wind speed values in EMO1.

Relative humidity is calculated from vapor pressure and temperature using the August-Roche-Magnus formula.


