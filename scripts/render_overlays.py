# scripts/render_overlays.py
"""
Render WebP overlays for steps 0..48 h (every 3 h) and write ISO run time into manifest.
"""

import json, glob, pathlib
import numpy as np
import xarray as xr
from PIL import Image

DATA = pathlib.Path("data_raw")
OUT  = pathlib.Path("out"); OUT.mkdir(parents=True, exist_ok=True)

# Fennoscandia bbox [W,S,E,N]
W, S, E, N = 5.0, 55.0, 35.5, 72.5

NX, NY = 900, 600
TARGET_STEPS = list(range(0, 49, 3))  # 0,3,...,48
VARS = ["msl","t2m","tp_rate","tcwv","ssr_flux","str_flux","gh500","gh850","wspd850"]

# ---------- helpers ----------
def linspace2d():
    lons = np.linspace(W, E, NX, dtype=np.float32)
    lats = np.linspace(N, S, NY, dtype=np.float32)
    return np.meshgrid(lons, lats)
LON2, LAT2 = linspace2d()

def save_webp(name, rgba, quality=80):
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(OUT / f"{name}.webp", format="WEBP", quality=quality, method=6)

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
    if not path: return None
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
    lat_slice = slice(N, S) if lats[0] > lats[-1] else slice(S, N)
    da = da.sel({latn: lat_slice})
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
    dims = list(da.dims)
    if "step" in dims: return "step"
    for d in ("time","valid_time","forecast_time"):
        if d in dims: return d
    return None

def hours_from_coord(coord):
    try:
        vals = coord.values
        if np.issubdtype(vals.dtype, np.timedelta64):
            return (vals / np.timedelta64(1, "h")).astype(int).tolist()
        return [int(x) for x in vals.tolist()]
    except Exception:
        return list(range(coord.size))

def first_run_iso(*datasets):
    """Return ISO UTC run time if any dataset has a 'time' coord; else 'latest'."""
    import numpy as np
    for ds in datasets:
        if ds is None:
            continue
        if "time" in getattr(ds, "coords", {}):
            vals = ds["time"].values
            val = vals if np.isscalar(vals) else vals[0]
            try:
                iso = np.datetime_as_string(val, unit="s", timezone="UTC")
                return iso if iso.endswith("Z") else f"{iso}Z"
            except Exception:
                pass
    return "latest"

def render_field_series(ds, vname, convert_fn, palette, prefix):
    tdim = get_time_dim(ds[vname])
    steps_hours = hours_from_coord(ds[tdim]) if tdim else [0]
    hours_to_index = {int(h): int(i) for i,h in enumerate(steps_hours)}
    produced = []
    for h in TARGET_STEPS:
        idx = hours_to_index.get(int(h))
        if idx is None: continue
        field = ds[vname].isel({tdim: int(idx)}) if tdim else ds[vname]
        field = subset_bbox(field)
        grid = to_grid(convert_fn(field))
        rgba = colorize(grid, *palette)
        save_webp(f"{prefix}_+{h:03d}", rgba)
        produced.append(h)
    return produced

def render_diff_series(ds, vname, convert_fn, palette, prefix, step_hours=3):
    tdim = get_time_dim(ds[vname])
    steps_hours = hours_from_coord(ds[tdim]) if tdim else []
    hours_to_index = {int(h): int(i) for i,h in enumerate(steps_hours)}
    produced = []
    for h in TARGET_STEPS:
        if h < step_hours: continue
        i1 = hours_to_index.get(int(h)); i0 = hours_to_index.get(int(h - step_hours))
        if i1 is None or i0 is None: continue
        a1 = subset_bbox(ds[vname].isel({tdim: int(i1)}))
        a0 = subset_bbox(ds[vname].isel({tdim: int(i0)}))
        g1 = to_grid(a1); g0 = to_grid(a0)
        grid = convert_fn(g1, g0, step_hours)
        rgba = colorize(grid, *palette)
        save_webp(f"{prefix}_+{h:03d}", rgba)
        produced.append(h)
    return produced

# ---------- main ----------
def main():
    steps_available = set()

    # Load datasets (some may be None)
    ds_msl   = open_cf(find_one("msl_sfc_*.grib2"))
    ds_t2    = open_cf(find_one("2t_sfc_*.grib2"))
    ds_tp    = open_cf(find_one("tp_sfc_*.grib2"))
    ds_ssr   = open_cf(find_one("ssr_sfc_*.grib2"))
    ds_str   = open_cf(find_one("str_sfc_*.grib2"))
    ds_tcwv  = open_cf(find_one("tcwv_sfc_*.grib2"))
    ds_gh500 = open_cf(find_one("gh_pl_500_*.grib2"))
    ds_gh850 = open_cf(find_one("gh_pl_850_*.grib2"))
    ds_u850  = open_cf(find_one("u_pl_850_*.grib2"))
    ds_v850  = open_cf(find_one("v_pl_850_*.grib2"))

    # MSLP (Pa -> hPa)
    if ds_msl:
        try:
            v = [n for n in ds_msl.data_vars if n.startswith("msl")][0]
            steps_available.update(render_field_series(ds_msl, v, lambda x: x/100.0, (960, 1040), "msl"))
        except Exception as e: print("[WARN] msl:", e)

    # 2 m T (K -> °C)
    if ds_t2:
        try:
            v = [n for n in ds_t2.data_vars if n.startswith("t2") or n=="2t"][0]
            steps_available.update(render_field_series(ds_t2, v, lambda x: x-273.15, (-25, 25), "t2m"))
        except Exception as e: print("[WARN] t2m:", e)

    # Total precipitation rate (mm/h over previous 3h) from accum (m)
    if ds_tp:
        try:
            v = [n for n in ds_tp.data_vars if n.startswith("tp")][0]
            def tp_rate(g1, g0, hrs): return (g1 - g0) * 1000.0 / hrs
            steps_available.update(render_diff_series(ds_tp, v, tp_rate, (0, 6), "tp_rate", step_hours=3))
        except Exception as e: print("[WARN] tp_rate:", e)

    # Net SW flux (J/m^2 accum) -> W/m^2 over 3h
    if ds_ssr:
        try:
            v = [n for n in ds_ssr.data_vars if n.startswith("ssr")][0]
            def sw_flux(g1, g0, hrs): return (g1 - g0) / (hrs*3600.0)
            steps_available.update(render_diff_series(ds_ssr, v, sw_flux, (0, 500), "ssr_flux", step_hours=3))
        except Exception as e: print("[WARN] ssr_flux:", e)

    # Net LW flux (J/m^2 accum) -> W/m^2 over 3h
    if ds_str:
        try:
            v = [n for n in ds_str.data_vars if n.startswith("str")][0]
            def lw_flux(g1, g0, hrs): return (g1 - g0) / (hrs*3600.0)
            steps_available.update(render_diff_series(ds_str, v, lw_flux, (-150, 50), "str_flux", step_hours=3))
        except Exception as e: print("[WARN] str_flux:", e)

    # TCWV (kg/m^2)
    if ds_tcwv:
        try:
            v = [n for n in ds_tcwv.data_vars if n.startswith("tcwv")][0]
            steps_available.update(render_field_series(ds_tcwv, v, lambda x: x, (0, 60), "tcwv"))
        except Exception as e: print("[WARN] tcwv:", e)

    # Geopotential (m^2/s^2) -> m at 500/850 hPa
    def render_gh(ds, level):
        if not ds: return []
        try:
            v = [n for n in ds.data_vars if n.startswith("gh")][0]
            rng = (4800,5800) if level==500 else (1200,1700)
            return render_field_series(ds, v, lambda x: x/9.80665, rng, f"gh{level}")
        except Exception as e: print(f"[WARN] gh{level}:", e); return []
    steps_available.update(render_gh(ds_gh500, 500))
    steps_available.update(render_gh(ds_gh850, 850))

    # 850-hPa wind speed from u,v (m/s)
    if (ds_u850 is not None) and (ds_v850 is not None):
        try:
            vu = [n for n in ds_u850.data_vars if n.startswith("u")][0]
            vv = [n for n in ds_v850.data_vars if n.startswith("v")][0]
            tdu = get_time_dim(ds_u850[vu]); tdv = get_time_dim(ds_v850[vv])
            hrs_u = hours_from_coord(ds_u850[tdu]) if tdu else [0]
            hrs_v = hours_from_coord(ds_v850[tdv]) if tdv else [0]
            map_u = {int(h): int(i) for i,h in enumerate(hrs_u)}
            map_v = {int(h): int(i) for i,h in enumerate(hrs_v)}
            made = []
            for h in TARGET_STEPS:
                iu = map_u.get(int(h)); iv = map_v.get(int(h))
                if iu is None or iv is None: continue
                a = subset_bbox(ds_u850[vu].isel({tdu: int(iu)})) if tdu else subset_bbox(ds_u850[vu])
                b = subset_bbox(ds_v850[vv].isel({tdv: int(iv)})) if tdv else subset_bbox(ds_v850[vv])
                uu = to_grid(a); vv_ = to_grid(b)
                wspd = np.sqrt(uu*uu + vv_*vv_)
                save_webp(f"wspd850_+{h:03d}", colorize(wspd, 0, 40, base=(0,0,0), top=(255,255,255)))
                made.append(h)
            steps_available.update(made)
        except Exception as e:
            print("[WARN] wspd850:", e)

    # Manifest: ISO run time (UTC) if available
    run_iso = first_run_iso(ds_msl, ds_t2, ds_tp, ds_ssr, ds_str, ds_tcwv, ds_gh500, ds_gh850, ds_u850, ds_v850)
    steps = [f"+{h:03d}" for h in sorted(steps_available)] or ["+000","+003","+006","+009"]

    manifest = {
        "run": run_iso,
        "bbox": [W, S, E, N],
        "steps": steps,
        "vars": VARS
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] WebP overlays written to {OUT} with {len(steps)} frames; run={run_iso}")

if __name__ == "__main__":
    main()
