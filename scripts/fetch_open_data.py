# scripts/fetch_open_data.py
from pathlib import Path
from ecmwf.opendata import Client

OUTDIR = Path("data_raw"); OUTDIR.mkdir(parents=True, exist_ok=True)

# What to fetch
SFC_PARAMS = ["msl", "2t", "tp", "tcwv", "ssr", "str"]
PL_PARAMS  = ["u", "v", "gh"]
PL_LEVELS  = [850, 500]
STEPS_H    = list(range(0, 49, 3))   # 0,3,6,...,48

# Use ECMWF Open Data defaults (IFS 0.25°)
c = Client(source="ecmwf")

def fetch_sfc(param: str):
    tmp = OUTDIR / f"_tmp_{param}.grib2"
    c.retrieve(type="fc", step=STEPS_H, param=param, target=str(tmp))
    final = OUTDIR / f"{param}_sfc_latest.grib2"      # stable name, our renderer globs *_sfc_*.grib2
    if final.exists(): final.unlink()
    tmp.replace(final)
    print("[OK] SFC", param, "->", final.name)

def fetch_pl(param: str, level: int):
    tmp = OUTDIR / f"_tmp_{param}_{level}.grib2"
    c.retrieve(type="fc", step=STEPS_H, param=param, levtype="pl", levelist=str(level), target=str(tmp))
    final = OUTDIR / f"{param}_pl_{level}_latest.grib2"
    if final.exists(): final.unlink()
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

