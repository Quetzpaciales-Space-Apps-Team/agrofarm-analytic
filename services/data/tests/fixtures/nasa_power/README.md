NASA POWER Daily API responses recorded on 2026-11-15 for 2024-01-01.

- `point.json`: 14.59 latitude, -90.53 longitude, seven AG parameters.
- `regional_*.json`: 14 to 16 latitude, -91 to -89 longitude, one AG parameter per file.

Source: https://power.larc.nasa.gov/api/temporal/daily/point and
https://power.larc.nasa.gov/api/temporal/daily/regional.
Attribution: NASA Langley Research Center POWER Project. The values are source
responses, not synthetic observations. Tests never call the live API.

Regional endpoint limits, confirmed live on 2026-11-15 with two small requests that failed fast (HTTP 422): a box must span 2 to 10 degrees in latitude and in longitude, one variable per request.
Parameter names (T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN, WS2M) and the fill value -999 were re-confirmed on the point endpoint with one request. User-Agent: agrofarm-data-connector/0.1.
