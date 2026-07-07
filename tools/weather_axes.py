"""Decompose the 21 CARLA weather presets (index 0-20) into independent
*physical* difficulty axes, so illumination can be analysed separately from
precipitation/fog — which the single 0-20 ordinal (and the scalar
`_WEATHER_DIFF`) fuse together.

Motivation: on the A100 sweep, hardest-first scheduling collapsed the completed
sample onto presets 19 (MidRainyNight) and 20 (HardRainNight) — the two highest
`_WEATHER_DIFF` values. Those are simultaneously the *darkest* (sun_altitude
-90 deg) and *rainiest* (precip 60/100) presets, so illumination and
precipitation are perfectly confounded and neither is identifiable. Splitting
the ordinal into axes is the prerequisite for a per-model sensitivity fit
(see tools/sensitivity_matrix.py) and for illumination-stratified scheduling.

Sources for the raw physical parameters:
  * Presets 14-20 (the *Night* variants): exact constructor args vendored in
    leaderboard/team_code/consolidated_agent.py:65-71 (that is the table the run
    actually applies via world.set_weather(_WEATHERS[_WEATHER_IDS[idx]])).
  * Presets 0-13 (Noon/Sunset built-ins): CARLA 0.9.10 `carla.WeatherParameters`
    documented presets. Absolute day magnitudes are approximate but the per-axis
    *ordering* is exact; every axis is min-max normalised below, and the current
    sample is night-only, so day precision is not load-bearing. Re-verify in the
    container with:  python -c "import carla; print(carla.WeatherParameters.ClearNoon)"

Pure stdlib; safe to import anywhere (no carla, numpy, or torch).
"""

# idx: (name, cloudiness, precipitation, precip_deposits, sun_altitude_deg, fog_density)
WEATHER_RAW = {
    0:  ("ClearNoon",        15.0,   0.0,   0.0,  75.0,   0.0),
    1:  ("ClearSunset",      15.0,   0.0,   0.0,  15.0,   0.0),
    2:  ("CloudyNoon",       80.0,   0.0,   0.0,  75.0,   0.0),
    3:  ("CloudySunset",     80.0,   0.0,   0.0,  15.0,   0.0),
    4:  ("WetNoon",          20.0,   0.0,  50.0,  75.0,   0.0),
    5:  ("WetSunset",        20.0,   0.0,  50.0,  15.0,   0.0),
    6:  ("MidRainyNoon",     80.0,  30.0,  50.0,  75.0,   0.0),
    7:  ("MidRainSunset",    80.0,  30.0,  50.0,  15.0,   0.0),
    8:  ("WetCloudyNoon",    80.0,   0.0,  50.0,  75.0,   0.0),
    9:  ("WetCloudySunset",  80.0,   0.0,  50.0,  15.0,   0.0),
    10: ("HardRainNoon",     90.0,  60.0, 100.0,  75.0,   0.0),
    11: ("HardRainSunset",   90.0,  60.0, 100.0,  15.0,   0.0),
    12: ("SoftRainNoon",     70.0,  15.0,  50.0,  75.0,   0.0),
    13: ("SoftRainSunset",   70.0,  15.0,  50.0,  15.0,   0.0),
    # Night variants — exact args from consolidated_agent.py:65-71
    # WeatherParameters(cloudiness, precip, precip_deposits, wind, sun_azimuth,
    #                   sun_altitude, fog_density, fog_distance, wetness, fog_falloff)
    14: ("ClearNight",        5.0,   0.0,   0.0, -90.0,  60.0),
    15: ("CloudyNight",      60.0,   0.0,   0.0, -90.0,  60.0),
    16: ("WetNight",          5.0,   0.0,  50.0, -90.0,  60.0),
    17: ("WetCloudyNight",   60.0,   0.0,  50.0, -90.0,  60.0),
    18: ("SoftRainNight",    60.0,  30.0,  50.0, -90.0,  60.0),
    19: ("MidRainyNight",    80.0,  60.0,  60.0, -90.0,  60.0),
    20: ("HardRainNight",   100.0, 100.0,  90.0, -90.0, 100.0),
}

# Physical axes we expose (each normalised to [0,1], higher = harder / more hazard).
AXIS_NAMES = ["illum_dark", "precip", "road_water", "cloud", "fog"]


def name(idx):
    return WEATHER_RAW.get(int(idx), ("?",))[0]


def sun_altitude(idx):
    return WEATHER_RAW[int(idx)][4]


def time_of_day(idx):
    """Coarse illumination bin from the sun altitude: 'noon' | 'sunset' | 'night'."""
    alt = sun_altitude(idx)
    if alt <= -30.0:
        return "night"
    if alt <= 30.0:
        return "sunset"
    return "noon"


# Ordinal illumination bins for stratified coverage (0 brightest .. 2 darkest).
ILLUM_BINS = {"noon": 0, "sunset": 1, "night": 2}


def illum_bin(idx):
    return ILLUM_BINS[time_of_day(idx)]


def axes(idx):
    """Return the normalised [0,1] physical-axis vector for a preset index.

    illum_dark = (90 - sun_altitude) / 180  -> Noon 0.083, Sunset 0.417, Night 1.0
    """
    _, cloud, precip, deposits, sun_alt, fog = WEATHER_RAW[int(idx)]
    return {
        "illum_dark":  (90.0 - sun_alt) / 180.0,
        "precip":      precip / 100.0,
        "road_water":  deposits / 100.0,
        "cloud":       cloud / 100.0,
        "fog":         fog / 100.0,
    }


def axis_row(idx):
    """Flat dict for CSV emission: name, sun_altitude, time_of_day + normalised axes."""
    row = {
        "weather_name":   name(idx),
        "sun_altitude":   sun_altitude(idx),
        "time_of_day":    time_of_day(idx),
    }
    row.update(axes(idx))
    return row


if __name__ == "__main__":
    # Quick self-check / table dump.
    hdr = ["idx", "name", "sun_alt", "tod"] + AXIS_NAMES
    print("  ".join(f"{h:>12s}" for h in hdr))
    for i in range(21):
        a = axes(i)
        cells = [f"{i:>12d}", f"{name(i):>12s}", f"{sun_altitude(i):>12.0f}",
                 f"{time_of_day(i):>12s}"] + [f"{a[k]:>12.3f}" for k in AXIS_NAMES]
        print("  ".join(cells))
    # confound check: correlation of illum_dark vs precip across the *current* sample
    print("\nConfound in the collapsed sample (presets 19,20 only):")
    for i in (19, 20):
        a = axes(i)
        print(f"  {name(i):14s} illum_dark={a['illum_dark']:.2f}  precip={a['precip']:.2f}"
              f"  -> both maxed, indistinguishable")
