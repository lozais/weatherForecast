import numpy as np
def wind_speed_dir(u, v):
    speed = np.sqrt(u*u + v*v)
    # meteorological direction: FROM which wind blows, degrees
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, direction
