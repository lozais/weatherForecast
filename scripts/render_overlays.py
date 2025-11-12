# scripts/render_overlays.py

"""
Render transparent PNG overlays from ECMWF IFS GRIB2, robust to:
- latitude ascending or descending
- longitude in 0..360 or -180..180
- missing files (falls back to placeholders)

Outputs go to 'out/' then your workflow copies to 'tiles/latest/'.
"""

import json, math, pathlib, glob
import numpy as np
import xarray as xr
from PIL import Image

DATA = pathlib.Path("data_raw")
OUT  = pathlib.Path("out"); OUT.mkdir(parents=True, exist_ok=True)

# Fennoscandia bbox (W,S,E,N) in -180..180 convention
W, S, E, N = 5.0, 55.0, 35.5, 72.5

# Lon/lat grid for rendering (Plate Carrée)
NX, NY = 900, 600  # modest resolution for speed

# Steps we publish and the vars exposed to UI
STEPS = ["+000","+006","+012","+024"]
VARS  = ["msl","t2m","tp_rate","tcwv","ssr_flux","str_flux","gh500","gh850","wspd850"]

def linspace2d():
    lons = np.linspace(W, E, NX, dtype=np.float32)
    lats = np.linspace(N, S, NY, dtype=np.float32)  # top->bottom
    return np.meshgrid(lons, lats)  # lon2, lat2

LON2, LAT2 = linspace2d()

def save_png(name, rgba):
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(OUT / f"{name}.png", optimize=True)

def colorize(data, vmin, vmax, base=(0,0,0), top=(0,180,255), alpha=200):
    arr = np.array(data, dtype=float)
    mask = ~np.isfinite(arr)
    t = (arr - vmin) / max(vmax - vmin, 1e-6)
    t = np.clip(t, 0, 1)
    r = (base[0]*(1-t) + top[0]*t).astype(np.uint8)
    g = (base[1]*(1-t) + top[1]*t).astype(np.uint8)
    b = (base[2]*(1-t) + top[2]*t).astype(np.uint8)
    a = np.full_like(r, alpha, dtype=np.uint8)
    a[mask] = 0
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
    """Return lon in -180..180."""
    lon = np.asarray(lon, dtype=float)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon

def subset_bbox(da):
    """Subset any (lat,lon) field to our bbox, robust to axis naming and order."""
    # find lat/lon names
    latn = [n for n in da.dims if "lat" in n][0]
    lonn = [n for n in da.dims if "lon" in n][0]

    # fetch coord arrays
    lats = da[latn].values
    lons = da[lonn].values

    # normalize longitudes to -180..180 for comparison
    lons_std = standardize_lon(lons)

    # Latitude slice: handle ascending or descending
    if lats[0] > lats[-1]:  # descending (common in ECMWF)
        lat_slice = slice(N, S)
    else:                   # ascending
        lat_slice = slice(S, N)

    da = da.sel({latn: lat_slice})

    # Longitude selection: create mask in standardized space
    # Attach standardized lon as a coord for selection
    da = da.assign_coords({lonn + "_std": (lonn, standardize_lon(da[lonn]))})
    lon_std = da[lonn + "_std"].values
    if lon_std.ndim == 1:
        mask = (lon_std >= W) & (lon_std <= E)
        idx = np.where(mask)[0]
        if idx.size > 0:
            da = da.isel({lonn: idx})
    # drop helper coord
    if (lonn + "_std") in da.coords:
        da = da.drop_vars(lonn + "_std")
    return da

def to_grid(da):
    """Nearest-neighbor sample to our target lon/lat grid."""
    latn = [n for n in da.dims if "lat" in n][0]
    lonn = [n for n in da.dims if "lon" in n][0]
    da = da.rename({latn:"lat", lonn:"lon"})
    targ_lat = xr.DataArray(LAT2[:,0], dims=("y",))
    targ_lon = xr.DataArray(LON2[0,:], dims=("x",))
    res = da.sel(lat=targ_lat, lon=targ_lon, method="nearest").transpose("y","x").values
    return res

def render_placeholders():
    print("[INFO] Rendering placeholder overlays...")
    base = np.full((NY, NX), np.nan, dtype=np.float32)
    entries = {
        "msl_+000": (960, 1040), "msl_+006": (960, 1040),
        "t2m_+000": (-25, 25), "t2m_+006": (-25, 25),
        "tp_rate_+006": (0, 6),
        "gh500_+000": (4800, 5800), "gh850_+000": (1200, 1700),
        "wspd850_+000": (0, 40),
        "ssr_flux_+006": (0, 500), "str_flux_+006": (-150, 50),
    }
    for name, (vmin, vmax) in entries.items():
        rgba = colorize(base, vmin, vmax)
        save_png(name, rgba)

def first_time_index(da):
    """Return the first two indices in the time/step dimension if present, else None."""
    dims = list(da.dims)
    time_like = [d for d in dims if d in ("time","step","valid_time","forecast_time")]
    if not time_like:
        return None, None, None
    tdim = time_like[0]
    n = da.sizes[tdim]
    return tdim, 0 if n>0 else None, 1 if n>1 else None

def main():
    # Discover raw files
    msl = find_one("msl_sfc_*.grib2")
    t2  = find_one("2t_sfc_*.grib2")
    tp  = find_one("tp_sfc_*.grib2")
    ssr = find_one("ssr_sfc_*.grib2")
    strn= find_one("str_sfc_*.grib2")
    tcwv= find_one("tcwv_sfc_*.grib2")

    gh500= find_one("gh_pl_500_*.grib2")
    gh850= find_one("gh_pl_850_*.grib2")
    u850 = find_one("u_pl_850_*.grib2")
    v850 = find_one("v_pl_850_*.grib2")

    have_any = any([msl,t2,tp,ssr,strn,tcwv,gh500,gh850,u850,v850])
    run_id = "latest"

    if not have_any:
        render_placeholders()
    else:
        # --- MSLP (Pa -> hPa), +000 and +006 ---
        if msl:
            ds = open_cf(msl)
            try:
                vname = [n for n in ds.data_vars if n.startswith("msl")][0]
                tdim, i0, i1 = first_time_index(ds[vname])
                for idx, tag in [(i0,"+000"), (i1,"+006")]:
                    if idx is None: continue
                    field = ds[vname].isel({tdim: idx}) if tdim else ds[vname]
                    field = subset_bbox(field)
                    grid = to_grid(field) / 100.0
                    save_png(f"msl_{tag}", colorize(grid, 960, 1040))
            except Exception as e:
                print("[WARN] MSLP render:", e)

        # --- 2m T (K -> °C), +000 and +006 ---
        if t2:
            ds = open_cf(t2)
            try:
                vname = [n for n in ds.data_vars if n.startswith("t2") or n=="2t"][0]
                tdim, i0, i1 = first_time_index(ds[vname])
                for idx, tag in [(i0,"+000"), (i1,"+006")]:
                    if idx is None: continue
                    field = ds[vname].isel({tdim: idx}) if tdim else ds[vname]
                    field = subset_bbox(field) - 273.15
                    grid = to_grid(field)
                    save_png(f"t2m_{tag}", colorize(grid, -25, 25, base=(0,0,0), top=(255,120,90)))
            except Exception as e:
                print("[WARN] T2M render:", e)

        # --- tp rate (mm/h) between +000 and +006 ---
        if tp:
            ds = open_cf(tp)
            try:
                vname = [n for n in ds.data_vars if n.startswith("tp")][0]
                tdim, i0, i1 = first_time_index(ds[vname])
                if i0 is not None and i1 is not None:
                    a0 = subset_bbox(ds[vname].isel({tdim:i0}))
                    a1 = subset_bbox(ds[vname].isel({tdim:i1}))
                    grid = (to_grid(a1) - to_grid(a0)) * 1000.0 / 6.0  # mm/h over 6h window
                    save_png("tp_rate_+006", colorize(grid, 0, 6, base=(0,0,0), top=(0,120,255)))
            except Exception as e:
                print("[WARN] TP render:", e)

        # --- gh500/gh850 (m^2/s^2 -> m) ---
        if gh500:
            ds = open_cf(gh500)
            try:
                vname = [n for n in ds.data_vars if n.startswith("gh")][0]
                tdim, i0, _ = first_time_index(ds[vname])
                if i0 is not None:
                    a = subset_bbox(ds[vname].isel({tdim:i0})) / 9.80665
                    save_png("gh500_+000", colorize(to_grid(a), 4800, 5800, base=(0,0,0), top=(180,255,255)))
            except Exception as e:
                print("[WARN] GH500 render:", e)

        if gh850:
            ds = open_cf(gh850)
            try:
                vname = [n for n in ds.data_vars if n.startswith("gh")][0]
                tdim, i0, _ = first_time_index(ds[vname])
                if i0 is not None:
                    a = subset_bbox(ds[vname].isel({tdim:i0})) / 9.80665
                    save_png("gh850_+000", colorize(to_grid(a), 1200, 1700, base=(0,0,0), top=(180,255,255)))
            except Exception as e:
                print("[WARN] GH850 render:", e)

        # --- 850-hPa wind speed from u,v ---
        if u850 and v850:
            dsu = open_cf(u850); dsv = open_cf(v850)
            try:
                vn_u = [n for n in dsu.data_vars if n.startswith("u")][0]
                vn_v = [n for n in dsv.data_vars if n.startswith("v")][0]
                tu, i0, _ = first_time_index(dsu[vn_u])
                tv, j0, _ = first_time_index(dsv[vn_v])
                if i0 is not None and j0 is not None:
                    a = subset_bbox(dsu[vn_u].isel({tu:i0}))
                    b = subset_bbox(dsv[vn_v].isel({tv:j0}))
                    uu = to_grid(a); vv = to_grid(b)
                    wspd = np.sqrt(uu*uu + vv*vv)
                    save_png("wspd850_+000", colorize(wspd, 0, 40, base=(0,0,0), top=(255,255,255)))
            except Exception as e:
                print("[WARN] WSPD850 render:", e)

        # --- tcwv ---
        if tcwv:
            ds = open_cf(tcwv)
            try:
                vname = [n for n in ds.data_vars if n.startswith("tcwv")][0]
                tdim, i0, _ = first_time_index(ds[vname])
                if i0 is not None:
                    a = subset_bbox(ds[vname].isel({tdim:i0}))
                    save_png("tcwv_+000", colorize(to_grid(a), 0, 60, base=(0,0,0), top=(0,255,180)))
            except Exception as e:
                print("[WARN] TCWV render:", e)

    # Manifest (even for placeholders)
    manifest = {
        "run": run_id,
        "bbox": [W, S, E, N],
        "steps": STEPS,
        "vars": VARS
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[OK] Overlays written to", OUT)

if __name__ == "__main__":
    main()
