# Label search for the yield model, 2026-11-15

Question: is there a public label that a model can predict better than "expect a normal year",
from seasonal rainfall, temperature and the ENSO state, in Guatemala and its neighbours?

Answer: no. Three labels were tried and none passed the rule below. `/v1/simulate` stays on its
fixed rules (ADR 0004 in `agrofarm-docs`, decision 1). No trained model is served.

## The rule, written before any model was fit

`rule_written_before_fitting.txt` is the file as written during the work, each block before its
first fit. Every year is predicted by a model that never saw it (leave one whole year out; cell
normals and trends are also computed without that year). A label "works" only if:

1. the pooled pinball loss (p10, p50, p90) is at least 5 % lower than climatology, and
2. the skill is positive in at least 60 % of the years, and
3. it is positive in at least one El Nino year and one La Nina year.

Climatology is the same quantiles of the training anomalies for every row.

## Results

Area: longitude -92.5 to -88.0, latitude 13.5 to 18.0. Season: May to October.
Inputs: CHIRPS v3 monthly rainfall, NASA POWER monthly T2M (about 50 km), NOAA CPC RONI.

| Label | Cells | Years | Skill against climatology | Years with positive skill | Correlation of yearly means | Verdict |
|---|---|---|---|---|---|---|
| Seasonal peak NDVI on cropland (MODIS MOD13Q1) | 969 of 0.05 degrees | 2000 to 2024 (25) | +4.4 % | 84 % | 0.13 | no skill |
| GDHY v1.2/1.3 maize, major season | 50 of 0.5 degrees | 1995 to 2016 (22) | -6.3 % | 55 % | 0.14 | no skill |
| GGCP10 maize production | 1,924 of 5 arc-minutes | 2010 to 2020 (11) | -2.1 % | 36 % | -0.66 | no skill |

Skill is `1 - loss(model) / loss(climatology)`: positive is better than climatology.
"Correlation of yearly means" compares, year by year, the mean observed anomaly of the area with
the mean predicted one. It says whether the model tells a good year from a bad one.

Per-year numbers are in `ndvi_peak_model_card.json`, `gdhy_maize_result.json` and
`ggcp10_maize_result.json`.

## How to read them

- **NDVI** missed the first condition by a little (4.4 % against 5 %) and met the other two. The
  rule is not moved after the fact. More important: the yearly mean of the peak NDVI anomaly moves
  by about 1 % between years, and the model follows it poorly (correlation 0.13). The small skill
  comes from differences between cells inside a year, not from telling years apart, which is what
  a scenario tool needs. Peak NDVI of dense vegetation saturates.
- **GDHY** is worse than climatology. With 50 coarse cells and 22 years there are few independent
  samples, and the product is itself modelled from satellite data and national statistics.
- **GGCP10** has 11 years and a rising production that a per-cell mean does not remove; the model
  predicts the opposite sign of the yearly mean. It is a downscaling of national statistics.

## Limits of this search

- One afternoon of work, one model family (LightGBM quantile), no tuning.
- NDVI was read from the 1 km overview of the 250 m band, so the pixel reliability band could not
  be applied. Clouds are handled only by the seasonal maximum. 2001 has 6 composites and 2025 was
  dropped (4 composites).
- The cropland map is ESA WorldCover 2021 for every year. It marks 2.8 % of the area as cropland,
  which is low for Guatemala: small maize plots are often mapped as grass or shrub.
- Temperature is the nearest point of a 50 km reanalysis grid.
- Rainfall of the same season is an input, so even a model that passed would be a what-if tool,
  not a forecast.
- SPAM 2010 is a single year and cannot train a model over time. It was not used.
- Not tried: seasonal mean or integral of NDVI, EVI, GPP, soil moisture, sub-season timing of dry
  spells, any farm data.

## Reproduce

NDVI (code under test in `services/data` and `services/ml`):

```bash
cd services/data
uv run python -m agrofarm_data.labels.ndvi_job download --out <modis-dir>
uv run python -m agrofarm_data.labels.ndvi_job build --modis <modis-dir> \
  --worldcover <worldcover-dir> --chirps <chirps-monthly-dir> --power <power-json> \
  --enso <enso_monthly.parquet> --out <clean-dir>
cd ../ml
uv run python -m agrofarm_ml.ndvi.training --table <clean-dir>/ndvi_cell_year.parquet \
  --enso <enso_monthly.parquet> --out <model-dir>
```

The NASA POWER file is one call to the monthly regional API for `T2M` over the area.

GDHY and GGCP10 were run with the two scripts in this folder. They are exploratory scripts: no
tests, no lint, kept only so the numbers can be checked.

```bash
cd services/data
uv run --with lightgbm python ../../experiments/2026-11-15-label-search/gdhy_eval.py <work-dir> <chirps-monthly-dir> <enso_monthly.parquet>
uv run --with lightgbm python ../../experiments/2026-11-15-label-search/ggcp10_eval.py <work-dir> <chirps-monthly-dir> <enso_monthly.parquet>
```

`<work-dir>` needs `gdhy/maize_major/yield_YYYY.nc4` (GDHY zip from PANGAEA,
doi:10.1594/PANGAEA.909132, sha256 `731d03171ff9a5afb4837f51d6941fb58085ee69c62e767810e9395756648c76`),
`raw/power_t2m_monthly_regional*.json` and an `out/` folder. GGCP10 (Harvard Dataverse,
doi:10.7910/DVN/G1HBNK, CC0 1.0) is read over the network, only the window of the area.

## Sources

MODIS MOD13Q1 v061 (NASA LP DAAC, read from the Microsoft Planetary Computer; license wording of
the collection not verified). ESA WorldCover 2021 v200 (CC BY 4.0). CHIRPS v3 (Climate Hazards
Center, UC Santa Barbara). NASA POWER. NOAA CPC RONI. GDHY (Iizumi and Sakai; license on the
PANGAEA page not checked here). GGCP10 (CC0 1.0).
