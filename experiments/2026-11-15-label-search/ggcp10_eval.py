"""Exploration: GGCP10 maize production anomaly (2010-2020), leave one year out."""
import json, sys, urllib.request, re
from pathlib import Path
import numpy as np, pandas as pd, rasterio, lightgbm as lgb
from rasterio.windows import from_bounds
from agrofarm_data.labels.ndvi_job import chirps_monthly, power_season_temperature
from agrofarm_data.labels.ndvi_table import CellGrid
W = Path(sys.argv[1]); CHIRPS = Path(sys.argv[2]); ENSO = Path(sys.argv[3])
BOX = (-92.5, 13.5, -88.0, 18.0); N = 54; RES = 1 / 12
req = urllib.request.Request("https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/G1HBNK", headers={"User-Agent": "agrofarm-data-connector/0.1"})
listing = json.load(urllib.request.urlopen(req, timeout=60))["data"]["latestVersion"]["files"]
files = {}
for f in listing:
    m = re.search(r"Production_(\d{4})_Maize\.tif$", f["dataFile"]["filename"])
    if m: files[int(m.group(1))] = f["dataFile"]["id"]
print([f["dataFile"]["filename"] for f in listing if "Maize" in f["dataFile"]["filename"]])
prod = {}
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_USERAGENT="agrofarm-data-connector/0.1"):
    for y, fid in sorted(files.items()):
        cache = W / f"raw/ggcp10/maize_{y}.npy"
        if not cache.exists():
            with rasterio.open(f"/vsicurl/https://dataverse.harvard.edu/api/access/datafile/{fid}") as s:
                d = s.read(1, window=from_bounds(*BOX, s.transform).round_offsets().round_lengths()).astype(float)
            np.save(cache, d)
        d = np.load(cache); d[d < 0] = np.nan; prod[y] = d
print("years", sorted(prod), flush=True)
grid = CellGrid(BOX[0], BOX[3], RES, N, N)
rain005, months = chirps_monthly(CHIRPS)
# nearest 0.05 degree CHIRPS cell for each 5 arc-minute cell
ri = np.clip(((BOX[3] - grid.latitudes()) / 0.05).astype(int), 0, 89); ci = np.clip(((grid.longitudes() - BOX[0]) / 0.05).astype(int), 0, 89)
rain = rain005[:, ri][:, :, ci]
temp = power_season_temperature(W / "raw/power_t2m_monthly_regional.json", grid)
enso = pd.read_parquet(ENSO); em = enso["month"].dt
def roni(y, a, b): return float(enso[(em.year == y) & em.month.between(a, b)]["roni"].mean())
phase = {y: ("el_nino" if roni(y, 5, 10) >= 0.5 else "la_nina" if roni(y, 5, 10) <= -0.5 else "neutral") for y in prod}
always = np.all([np.nan_to_num(p) > 0 for p in prod.values()], axis=0)
lats, lons = grid.latitudes(), grid.longitudes(); rows = []
for y, p in prod.items():
    def tot(a, b): return rain[np.isin(months, [y * 100 + m for m in range(a, b + 1)])].sum(axis=0)
    s, e, l = tot(5, 10), tot(5, 7), tot(8, 10)
    for i, j in zip(*np.nonzero(always)):
        rows.append((f"{lats[i]:.3f}_{lons[j]:.3f}", y, p[i, j], s[i, j], e[i, j], l[i, j], temp[y][i, j], roni(y, 5, 10), roni(y, 2, 4)))
t = pd.DataFrame(rows, columns=["cell_id", "year", "production", "rain_season_mm", "rain_early_mm", "rain_late_mm", "temp_season_c", "roni_season", "roni_pre_season"]).dropna().reset_index(drop=True)
t.to_parquet(W / "out/ggcp10_cell_year.parquet", index=False)
print("rows", len(t), "cells", t.cell_id.nunique(), "years", t.year.nunique(), "phases", phase)
Q = (0.1, 0.5, 0.9)
PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=4, min_data_in_leaf=20, feature_fraction=1.0, bagging_fraction=1.0, num_threads=2, deterministic=True, force_row_wise=True, verbose=-1, seed=0)
FEATURES = ["rain_anom_pct", "rain_early_anom_pct", "rain_late_anom_pct", "temp_anom_c", "roni_season", "roni_pre_season", "production_normal", "rain_normal_mm", "temp_normal_c"]
def anomalies(table, ref):
    g = ref.groupby("cell_id")
    n = pd.DataFrame({"production_normal": g.production.mean(), "rain_normal_mm": g.rain_season_mm.mean(), "re": g.rain_early_mm.mean(), "rl": g.rain_late_mm.mean(), "temp_normal_c": g.temp_season_c.mean(), "n": g.year.nunique()})
    x = table.join(n, on="cell_id"); x = x[x.n >= 8].copy()
    x["target"] = 100 * (x.production / x.production_normal - 1)
    x["rain_anom_pct"] = 100 * (x.rain_season_mm / x.rain_normal_mm - 1); x["rain_early_anom_pct"] = 100 * (x.rain_early_mm / x.re - 1); x["rain_late_anom_pct"] = 100 * (x.rain_late_mm / x.rl - 1)
    x["temp_anom_c"] = x.temp_season_c - x.temp_normal_c
    return x
def pin(a, p, q): e = a - p; return float(np.mean(np.maximum(q * e, (q - 1) * e)))
per, lm_, lc_ = [], 0.0, 0.0
for y in sorted(t.year.unique()):
    tr_raw, te_raw = t[t.year != y], t[t.year == y]
    tr, te = anomalies(tr_raw, tr_raw), anomalies(te_raw, tr_raw); a = te.target.to_numpy()
    p = np.sort(np.column_stack([lgb.train({**PARAMS, "alpha": q}, lgb.Dataset(tr[FEATURES].to_numpy(float), label=tr.target.to_numpy()), 150).predict(te[FEATURES].to_numpy(float)) for q in Q]), axis=1)
    cq = np.quantile(tr.target.to_numpy(), Q)
    lm = np.mean([pin(a, p[:, i], q) for i, q in enumerate(Q)]); lc = np.mean([pin(a, np.full_like(a, cq[i]), q) for i, q in enumerate(Q)])
    lm_ += lm * len(a); lc_ += lc * len(a)
    per.append({"year": int(y), "phase": phase[int(y)], "rows": len(a), "skill": round(1 - lm / lc, 3), "mean_actual_pct": round(float(a.mean()), 2), "mean_p50_pct": round(float(p[:, 1].mean()), 2), "coverage": round(float(((a >= p[:, 0]) & (a <= p[:, 2])).mean()), 3)})
pos = [r for r in per if r["skill"] > 0]; yr = pd.DataFrame(per)
res = {"pooled_skill": round(1 - lm_ / lc_, 4), "years": len(per), "rows": int(yr.rows.sum()), "share_years_positive": round(len(pos) / len(per), 3), "el_nino_years_positive": sum(r["phase"] == "el_nino" for r in pos), "la_nina_years_positive": sum(r["phase"] == "la_nina" for r in pos), "correlation_of_year_means": round(float(np.corrcoef(yr.mean_actual_pct, yr.mean_p50_pct)[0, 1]), 3), "coverage_p10_p90": round(float(np.average(yr.coverage, weights=yr.rows)), 3), "per_year": per}
res["verdict"] = "beats_climatology" if (res["pooled_skill"] >= 0.05 and res["share_years_positive"] >= 0.6 and res["el_nino_years_positive"] >= 1 and res["la_nina_years_positive"] >= 1) else "no_skill_over_climatology"
print(json.dumps({k: v for k, v in res.items() if k != "per_year"}, indent=1)); print(yr.to_string(index=False))
(W / "out/ggcp10_loyo_result.json").write_text(json.dumps(res, indent=2))
