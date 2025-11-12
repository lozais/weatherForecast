# scripts/fetch_open_data.py
import datetime as dt
from pathlib import Path
from ecmwf.opendata import Client

OUTDIR = Path("data_raw"); OUTDIR.mkdir(parents=True, exist_ok=True)

SFC_PARAMS = ["msl","2t","tp","tcwv","ssr","str"]
PL_PARAMS  = ["u","v","gh"]
PL_LEVELS  = [850, 500]
STEPS_H    = [0, 6]  # extend later

now = dt.datetime.utcnow()
cycle = (now.hour // 6) * 6
run = now.replace(hour=cycle, minute=0, second=0, microsecond=0)

# default source is ECMWF Open Data at data.ecmwf.int
c = Client(source="ecmwf")

def fetch_sfc(param: str):
    tmp = OUTDIR / f"_tmp_{param}_{run:%Y%m%d%H}.grib2"
    if not tmp.exists():
        c.retrieve(
            stream="oper",                 # <-- FIXED (was 'hres')
            type="fc",
            date=f"{run:%Y-%m-%d}",
            time=f"{run:%H}",
            step="/".join(str(s) for s in STEPS_H),
            param=param,
            target=str(tmp)
        )
    final = OUTDIR / f"{param}_sfc_{run:%Y%m%d%H}.grib2"
    tmp.replace(final)
    print("[OK] SFC", param, "->", final.name)

def fetch_pl(param: str, level: int):
    tmp = OUTDIR / f"_tmp_{param}_{level}_{run:%Y%m%d%H}.grib2"
    if not tmp.exists():
        c.retrieve(
            stream="oper",                 # <-- FIXED (was 'hres')
            type="fc",
            date=f"{run:%Y-%m-%d}",
            time=f"{run:%H}",
            step="/".join(str(s) for s in STEPS_H),
            param=param,
            levtype="pl",
            levelist=str(level),
            target=str(tmp)
        )
    final = OUTDIR / f"{param}_pl_{level}_{run:%Y%m%d%H}.grib2"
    tmp.replace(final)
    print("[OK] PL", param, level, "->", final.name)

def main():
    for p in SFC_PARAMS:
        fetch_sfc(p)
    for p in PL_PARAMS:
        for lv in PL_LEVELS:
            fetch_pl(p, lv)

if __name__ == "__main__":
    main()
