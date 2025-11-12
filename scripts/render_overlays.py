# scripts/render_overlays.py
"""
Render transparent PNG overlays for each step (0..48 h every 3 h).
Robust to lat order and 0..360 vs -180..180 longitudes.

Outputs to out/<var>_+NNN.png and out/manifest.json
"""

import json, glob, pathlib
import numpy as np
import xarray as xr
from PIL import Image

DATA = pathlib.Path("data_raw")
OUT  = pathlib.Path("out"); OUT.mkdir(parents=True, exist_ok=True)

# Fennoscandia bbox [W,S,E,N]  (lon/lat in -180..180)
W, S, E, N = 5.0, 55.0, 35.5, 72.5

# Target render grid (Plate Carrée)
NX, NY = 900, 600

# Steps we aim to publish (hours)
TARGET_STEPS = list(range(0, 49, 3))  # 0,3,...,48

# Variables exposed to UI
VARS = ["msl","t2m","tp_rate","tcwv","ssr_flux","str_flux","gh500","gh850","wspd850"]

def linspace2d():
    lons = np.linspace(W, E, NX, dtype=np.float32)
    lats = np.linspace(N, S, NY, dtype=np.float32)
    return np.meshgrid(lons, lats)
LON2, LAT2 = linspace2d()

def save_png(name, rgba):
    Image.fromarray(rgba, mode="RGBA").save(OUT / f"{name}.png", optimize=True)

def colorize(data, vmin, vmax, base=(0,0,0), top=(0,180,255), alpha=200):
    arr = np.array(data, dtype=float)
    mask = ~np.isfinite(arr)
    rng = max(vmax - vmin, 1e-6)
    t = np.clip((arr - vmin) / rng, 0, 1)
    r = (base[0]*(1-t) + top[0]*t).astype(np.uint8)
    g = (base[1]*(1-t) + top[1]*t).astype(np.uint8)
    b = (base[2]*(1-t) + top[2]*t).astype(np.uint8)
    a = np.full_like(r, alpha, dtype=np.uint8); a[mask] = 0
    return np.dstack([r,g,b,a])

def find_one(pattern):
    files = sorted(glob.glob(str(DATA / pattern)))
    return files[0] if files else None

def open_cf(path):
    try:
        return xr.open_dataset(path, engine="cfgrib")
    except Exception as e:
        print(f"[WARN] open fail {path}: {e}")
        return None

def standardize_lon(lon):
    lon = np.asarray(lon, dtype=float)
    return (lon + 180.0) % 360.0 - 180.0

def subset_bbox(da):
    latn = [n for n in da.dims if "lat" in n][0]
    lonn = [n for n in da.dims if "lon" in n][0]
    lats = da[latn].values
    # latitude slice
    lat_slice = slice(N, S) if lats[0] > lats[-1] else slice(S, N)
    da = da.sel({latn: lat_slice})
    # longitude mask using standardized coord
    da = da.assign_coords({lonn + "_std": (lonn, standardize_lon(da[lonn]))})
    lon_std = da[lonn + "_std"].values
    if lon_std.ndim == 1:
        mask = (lon_std >= W) & (lon_std <= E)
        idx = np.where(mask)[0]
        if idx.size > 0:
            da = da.isel({lonn: idx})
    if (lonn + "_std") in da.coords:
        da = da.drop_vars(lonn + "_std")
    return da

def to_grid(da):
    latn = [n for n in da.dims if "lat" in n][0]
    lonn = [n for n in da.dims if "lon" in n][0]
    da = da.rename({latn:"lat", lonn:"lon"})
    targ_lat = xr.DataArray(LAT2[:,0], dims=("y",))
    targ_lon = xr.DataArray(LON2[0,:], dims=("x",))
    return da.sel(lat=targ_lat, lon=targ_lon, method="nearest").transpose("y","x").values

def get_time_dim(da):
    # prefer 'step' (forecast lead hours), fallback to first time-like
    dims = list(da.dims)
    if "step" in dims: return "step"
    for d in ("time","valid_time","forecast_time"):
        if d in dims: return d
    return None

def hours_from_coord(coord):
    # try to convert 'step' (timedelta-like) to hours; else assume integers
    try:
        # cfgrib step is often in nanoseconds (timedelta64[ns]) → to hours
        vals = coord.values
        if np.issubdtype(vals.dtype, np.timedelta64):
            return (vals / np.timedelta64(1, "h")).astype(int).tolist()
        # plain ints
        return [int(x) for x in vals.tolist()]
    except Exception:
        return list(range(coord.size))

def render_field_series(ds, vname, convert_fn, palette, prefix):
    """Render each TARGET_STEPS frame for a scalar field."""
    tdim = get_time_dim(ds[vname])
    steps_hours = hours_from_coord(ds[tdim]) if tdim else [0]
    hours_to_index = {h:i for i,h in enumerate(steps_hours)}
    produced = []
    for h in TARGET_STEPS:
        idx = hours_to_index.get(h)
        if idx is None: continue
        field = ds[vname].isel({tdim: idx}) if tdim else ds[vname]
        field = subset_bbox(field)
        grid = to_grid(convert_fn(field))
        rgba = colorize(grid, *palette)
        save_png(f"{prefix}_+{h:03d}", rgba)
        produced.append(h)
    return produced

def render_diff_series(ds, vname, convert_fn, palette, prefix, step_hours=3):
    """Render rate/flux from accumulations: frame h uses (h - (h-step_hours))."""
    tdim = get_time_dim(ds[vname])
    steps_hours = hours_from_coord(ds[tdim]) if tdim else []
    hours_to_index = {h:i for i,h in enumerate(steps_hours)}
    produced = []
    for h in TARGET_STEPS:
        if h < step_hours: continue
        i1 = hours_to_index.get(h); i0 = hours_to_index.get(h - step_hours)
        if i1 is None or i0 is None: continue
        a1 = subset_bbox(ds[vname].isel({tdim: i1}))
        a0 = subset_bbox(ds[vname].isel({tdim: i0}))
        g1 = to_grid(a1); g0 = to_grid(a0)
        grid = convert_fn(g1, g0, step_hours)
        rgba = colorize(grid, *palette)
        save_png(f"{prefix}_+{h:03d}", rgba)
        produced.append(h)
    return produced

def main():
    steps_available = set()

    # --- MSLP (Pa -> hPa) ---
    msl = find_one("msl_sfc_*.grib2")
    if msl:
        ds = open_cf(msl)
        try:
            v = [n for n in ds.data_vars if n.startswith("msl")][0]
            made = render_field_series(ds, v, lambda x: x/100.0, (960, 1040), "msl")
            steps_available.update(made)
        except Exception as e:
            print("[WARN] msl:", e)

    # --- 2m T (K -> °C) ---
    t2 = find_one("2t_sfc_*.grib2")
    if t2:
        ds = open_cf(t2)
        try:
            v = [n for n in ds.data_vars if n.startswith("t2") or n=="2t"][0]
            made = render_field_series(ds, v, lambda x: x-273.15, (-25, 25), "t2m")
            steps_available.update(made)
        except Exception as e:
            print("[WARN] t2m:", e)

    # --- Total precipitation rate (mm/h over 3h) from accum (m) ---
    tp = find_one("tp_sfc_*.grib2")
    if tp:
        ds = open_cf(tp)
        try:
            v = [n for n in ds.data_vars if n.startswith("tp")][0]
            def tp_rate(g1, g0, hrs):  # meters -> mm/h over 'hrs'
                return (g1 - g0) * 1000.0 / hrs
            made = render_diff_series(ds, v, tp_rate, (0, 6), "tp_rate", step_hours=3)
            steps_available.update(made)
        except Exception as e:
            print("[WARN] tp_rate:", e)

    # --- Net SW/LW flux (J/m^2 accum) -> W/m^2 over 3h ---
    ssr = find_one("ssr_sfc_*.grib2")
    if ssr:
        ds = open_cf(ssr)
        try:
            v = [n for n in ds.data_vars if n.startswith("ssr")][0]
            def sw_flux(g1, g0, hrs):  # W/m^2
                return (g1 - g0) / (hrs*3600.0)
            made = render_diff_series(ds, v, sw_flux, (0, 500), "ssr_flux", step_hours=3)
            steps_available.update(made)
        except Exception as e:
            print("[WARN] ssr_flux:", e)

    strn = find_one("str_sfc_*.grib2")
    if strn:
        ds = open_cf(strn)
        try:
            v = [n for n in ds.data_vars if n.startswith("str")][0]
            def lw_flux(g1, g0, hrs):
                return (g1 - g0) / (hrs*3600.0)
            made = render_diff_series(ds, v, lw_flux, (-150, 50), "str_flux", step_hours=3)
            steps_available.update(made)
        except Exception as e:
            print("[WARN] str_flux:", e)

    # --- TCWV (kg/m^2) ---
    tcwv = find_one("tcwv_sfc_*.grib2")
    if tcwv:
        ds = open_cf(tcwv)
        try:
            v = [n for n in ds.data_vars if n.startswith("tcwv")][0]
            made = render_field_series(ds, v, lambda x: x, (0, 60), "tcwv")
            steps_available.update(made)
        except Exception as e:
            print("[WARN] tcwv:", e)

    # --- Geopotential (m^2/s^2) -> m at 500/850 hPa ---
    def render_gh(level, vname_guess="gh"):
        f = find_one(f"gh_pl_{level}_*.grib2")
        if not f: return []
        ds = open_cf(f)
        try:
            v = [n for n in ds.data_vars if n.startswith(vname_guess)][0]
            made = render_field_series(ds, v, lambda x: x/9.80665,
                                       (4800, 5800) if level==500 else (1200, 1700),
                                       f"gh{level}")
            return made
        except Exception as e:
            print(f"[WARN] gh{level}:", e); return []

    steps_available.update(render_gh(500))
    steps_available.update(render_gh(850))

    # --- 850-hPa wind speed from u,v (m/s) ---
    u850 = find_one("u_pl_850_*.grib2")
    v850 = find_one("v_pl_850_*.grib2")
    if u850 and v850:
        dsu = open_cf(u850); dsv = open_cf(v850)
        try:
            vu = [n for n in dsu.data_vars if n.startswith("u")][0]
            vv = [n for n in dsv.data_vars if n.startswith("v")][0]
            tdim_u = get_time_dim(dsu[vu]); tdim_v = get_time_dim(dsv[vv])
            hrs_u = hours_from_coord(dsu[tdim_u]) if tdim_u else [0]
            hrs_v = hours_from_coord(dsv[tdim_v]) if tdim_v else [0]
            map_u = {h:i for i,h in enumerate(hrs_u)}
            map_v = {h:i for i,h in enumerate(hrs_v)}
            made = []
            for h in TARGET_STEPS:
                iu = map_u.get(h); iv = map_v.get(h)
                if iu is None or iv is None: continue
                a = subset_bbox(dsu[vu].isel({tdim_u: iu}))
                b = subset_bbox(dsv[vv].isel({tdim_v: iv}))
                uu = to_grid(a); vv = to_grid(b)
                wspd = np.sqrt(uu*uu + vv*vv)
                rgba = colorize(wspd, 0, 40, base=(0,0,0), top=(255,255,255))
                save_png(f"wspd850_+{h:03d}", rgba)
                made.append(h)
            steps_available.update(made)
        except Exception as e:
            print("[WARN] wspd850:", e)

    # --- Manifest: use the intersection of TARGET_STEPS and what we actually rendered ---
    steps = [f"+{h:03d}" for h in sorted(steps_available)]
    if not steps:
        # fallback minimal to keep UI alive
        steps = ["+000","+003","+006","+009"]

    manifest = {
        "run": "latest",           # can be set to an ISO run time later
        "bbox": [W, S, E, N],
        "steps": steps,
        "vars": VARS
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[OK] Overlays written to", OUT, "with", len(steps), "frames")
if __name__ == "__main__":
    main()
