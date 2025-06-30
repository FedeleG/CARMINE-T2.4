from dataclasses import dataclass

@dataclass
class latlon_box:
    ur_lat: float
    ur_lon: float
    ll_lat: float
    ll_lon: float
