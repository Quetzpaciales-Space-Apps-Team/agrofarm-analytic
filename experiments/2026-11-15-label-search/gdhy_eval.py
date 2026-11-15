"""Exploration: GDHY maize (major season) yield anomaly against a per-cell trend, leave one year out."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, lightgbm as lgb
from agrofarm_data.labels.ndvi_job import chirps_monthly, power_season_temperature
from agrofarm_data.labels.ndvi_table import CellGrid, SEASON
W = Path(sys.argv[1]); CHIRPS = Path(sys.argv[2]); ENSO = Path(sys.argv[3])
W_, S_, E_, N_ = -92.5, 13.5, -88.0, 18.0
grid = CellGrid(W_, N_, 0.5, 9, 9)
rain005, months = chirps_monthly(CHIRPS)                       # (t, 90, 90)
with np.errstate(all="ignore"):
    rain = np.nanmean(rain005.reshape(len(months), 9, 10, 9, 10), axis=(2, 4))
    share = np.isfinite(rain005).reshape(len(months), 9, 10, 9, 10).mean(axis=(2, 4))
rain[share < 0.5] = np.nan                                     # mostly sea: no value
temp = power_season_temperature(W / "raw/power_t2m_monthly_regional_1995_2016.json", grid)
enso = pd.read_parquet(ENSO); em = enso["month"].dt
def roni(y, a, b): return float(enso[(em.year == y) & em.month.between(a, b)]["roni"].mean())
phase = {y: ("el_nino" if roni(y, 5, 10) >= 0.5 else "la_nina" if roni(y, 5, 10) <= -0.5 else "neutral") for y in range(1995, 2017)}
rows = []
r0, c0 = int((90 - N_) / 0.5), int((360 + W_) / 0.5)
lats, lons = grid.latitudes(), grid.longitudes()
for y in range(1995, 2017):
    with rasterio.open(W / f"gdhy/maize_major/yield_{y}.nc4") as s:
        d = s.read(1).astype(float)[r0:r0 + 9, c0:c0 + 9]; d[d < -1e8] = np.nan
    def tot(a, b):
        sel = np.isin(months, [y * 100 + m for m in range(a, b + 1)]); return rain[sel].sum(axis=0)
    for i in range(9):
        for j in range(9):
            rows.append((f"{lats[i]:.2f}_{lons[j]:.2f}", y, d[i, j], tot(5, 10)[i, j], tot(5, 7)[i, j], tot(8, 10)[i, j], temp[y][i, j], roni(y, 5, 10), roni(y, 2, 4)))
t = pd.DataFrame(rows, columns=["cell_id", "year", "yield_t_ha", "rain_season_mm", "rain_early_mm", "rain_late_mm", "temp_season_c", "roni_season", "roni_pre_season"]).dropna().reset_index(drop=True)
t.to_parquet(W / "out/gdhy_cell_year.parquet", index=False)
print("rows", len(t), "cells", t.cell_id.nunique(), "years", t.year.nunique(), "yield mean/min/max", t.yield_t_ha.mean().round(2), t.yield_t_ha.min().round(2), t.yield_t_ha.max().round(2))

Q = (0.1, 0.5, 0.9)
PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=4, min_data_in_leaf=20, feature_fraction=1.0, bagging_fraction=1.0, num_threads=1, deterministic=True, force_row_wise=True, verbose=-1, seed=0)
FEATURES = ["rain_anom_pct", "rain_early_anom_pct", "rain_late_anom_pct", "temp_anom_c", "roni_season", "roni_pre_season", "yield_normal", "rain_normal_mm", "temp_normal_c"]
def anomalies(table, ref):
    out = []
    for cell, g in table.groupby("cell_id"):
        r = ref[ref.cell_id == cell]
        if r.year.nunique() < 10: continue
        slope, intercept = np.polyfit(r.year, r.yield_t_ha, 1)          # trend without the held-out year
        g = g.copy(); expected = slope * g.year + intercept
        if (expected <= 0).any(): continue
        g["target"] = 100 * (g.yield_t_ha / expected - 1)
        g["yield_normal"] = r.yield_t_ha.mean(); g["rain_normal_mm"] = r.rain_season_mm.mean(); g["temp_normal_c"] = r.temp_season_c.mean()
        g["rain_anom_pct"] = 100 * (g.rain_season_mm / r.rain_season_mm.mean() - 1)
        g["rain_early_anom_pct"] = 100 * (g.rain_early_mm / r.rain_early_mm.mean() - 1)
        g["rain_late_anom_pct"] = 100 * (g.rain_late_mm / r.rain_late_mm.mean() - 1)
        g["temp_anom_c"] = g.temp_season_c - r.temp_season_c.mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)
def pin(a, p, q): e = a - p; return float(np.mean(np.maximum(q * e, (q - 1) * e)))
per, lm_, lc_ = [], 0.0, 0.0
for y in sorted(t.year.unique()):
    tr_raw, te_raw = t[t.year != y], t[t.year == y]
    tr, te = anomalies(tr_raw, tr_raw), anomalies(te_raw, tr_raw)
    a = te.target.to_numpy()
    p = np.sort(np.column_stack([lgb.train({**PARAMS, "alpha": q}, lgb.Dataset(tr[FEATURES].to_numpy(float), label=tr.target.to_numpy()), 150).predict(te[FEATURES].to_numpy(float)) for q in Q]), axis=1)
    cq = np.quantile(tr.target.to_numpy(), Q)
    lm = np.mean([pin(a, p[:, i], q) for i, q in enumerate(Q)]); lc = np.mean([pin(a, np.full_like(a, cq[i]), q) for i, q in enumerate(Q)])
    lm_ += lm * len(a); lc_ += lc * len(a)
    per.append({"year": int(y), "phase": phase[int(y)], "rows": len(a), "skill": round(1 - lm / lc, 3), "mean_actual_pct": round(float(a.mean()), 2), "mean_p50_pct": round(float(p[:, 1].mean()), 2), "coverage": round(float(((a >= p[:, 0]) & (a <= p[:, 2])).mean()), 3)})
pos = [r for r in per if r["skill"] > 0]; yr = pd.DataFrame(per)
res = {"pooled_skill": round(1 - lm_ / lc_, 4), "years": len(per), "rows": int(yr.rows.sum()), "share_years_positive": round(len(pos) / len(per), 3), "el_nino_years_positive": sum(r["phase"] == "el_nino" for r in pos), "la_nina_years_positive": sum(r["phase"] == "la_nina" for r in pos), "correlation_of_year_means": round(float(np.corrcoef(yr.mean_actual_pct, yr.mean_p50_pct)[0, 1]), 3), "coverage_p10_p90": round(float(np.average(yr.coverage, weights=yr.rows)), 3), "per_year": per}
res["verdict"] = "beats_climatology" if (res["pooled_skill"] >= 0.05 and res["share_years_positive"] >= 0.6 and res["el_nino_years_positive"] >= 1 and res["la_nina_years_positive"] >= 1) else "no_skill_over_climatology"
print(json.dumps({k: v for k, v in res.items() if k != "per_year"}, indent=1)); print(yr.to_string(index=False))
(W / "out/gdhy_loyo_result.json").write_text(json.dumps(res, indent=2))
