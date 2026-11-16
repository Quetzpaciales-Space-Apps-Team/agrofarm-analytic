# Regional vegetation outlook

`POST /v1/vegetation/simulate` serves the previously trained LightGBM quantile models, separately from the fixed-rule yield endpoint. `NDVI_MODEL_DIR` loads an artifact at startup, verifies its files and the owner's limited-use acceptance, and fails startup when a configured artifact is invalid. Without a configured model the route returns 503.

The target is the seasonal peak NDVI anomaly relative to the reference cell's own 2000–2024 mean. The nearest trained cropland cell must be within 15 km. These are mixed regional cells at approximately 0.05 degrees, not farm boundaries or crop-specific yield. Rainfall changes are applied uniformly to early, late and full seasons; ENSO phase maps to RONI +1/0/-1 for both seasonal and preseason inputs. These are conditional inputs, not predicted weather or ENSO.

Leave-one-year-out evaluation improved pooled quantile loss by 4.41% against climatology, below the predeclared 5% goal. Annual mean correlation was 0.127 and p10–p90 coverage was 75.6% rather than the nominal 80%. The historical evaluation and its `no_skill_over_climatology` verdict remain unchanged. ADR 0005 records the owner's choice to use it with these limitations.

Attributions are scenario-minus-baseline TreeSHAP contributions for the raw median booster. They explain its change, not causation. Quantile sorting can move the raw median; `explained_median_delta` names precisely what is explained. Out-of-range input features are returned explicitly.

The pinned GitHub release is verified before it is baked into the Cloud Run image. This does not implement the separate Cloud Storage registry issue.
