# scripts/fetch_open_data.py
import datetime as dt
from pathlib import Path
from ecmwf.opendata import Client

OUTDIR = Path("data_raw"); OUTDIR.mkdir(parents=True, exist_ok=True)

SFC_PARAMS = ["msl", "2t", "tp", "tcwv", "ssr", "str"]
PL_PARAMS  = ["u", "v", "gh"]
PL_LEVELS  = [850, 500]
STEPS_H    = [0, 6]  # extend later

# Let the client infer stream/endpoint for IFS Open Data
c = Client(source="ecmwf")  # model="ifs" and resol="0p25" are defaults

def rename(tmp: Path, final: Path):
    if final.exists():
        final.unlink()
    tmp.replace(final)
    print("[OK]", final.name)

def fetch_sfc(param: str):
    tmp = OUTDIR / f"_tmp_{param}.grib2"
    # Do NOT set stream. The client will infer it for IFS HRES open data.
    result = c.retrieve(
        type="fc",
        step=STEPS_H,          # list is fine (0 and 6h)
        param=param,
        target=str(tmp),
    )
    # result carries the actual run chosen by the server
    run_time = result.get("time")
    run_date = result.get("date")
    run = dt.datetime.strptime(f"{run_date} {run_time:02d}", "%Y-%m-%d %H")
    final = OUTDIR / f"{param}_sfc_{run:%Y%m%d%H}.grib2"
    rename(tmp, final)

def fetch_pl(param: str, level: int):
    tmp = OUTDIR / f"_tmp_{param}_{level}.grib2"
    result = c.retrieve(
        type="fc",
        step=STEPS_H,
        param=param,
        levtype="pl",
        levelist=str(level),
        target=str(tmp),
    )
    run_time = result.get("time")
    run_date = result.get("date")
    run = dt.datetime.strptime(f"{run_date} {run_time:02d}", "%Y-%m-%d %H")
    final = OUTDIR / f"{param}_pl_{level}_{run:%Y%m%d%H}.grib2"
    rename(tmp, final)

def main():
    for p in SFC_PARAMS:
        fetch_sfc(p)
    for p in PL_PARAMS:
        for lv in PL_LEVELS:
            fetch_pl(p, lv)

if __name__ == "__main__":
    main()
