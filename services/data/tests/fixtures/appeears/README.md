# Recorded AppEEARS answers

Recorded on 2026-11-15 from the real API with one `area` task for the box `-90.65 14.45 -90.40 14.75`
and the dates 2025-06-01 to 2025-06-30 (task `4d960e29-8ead-4879-be97-62992af7551f`).

- `products.json`, `layers_*.json`: the public catalog (`GET /product`, `GET /product/<id>`), cut to the
  three products and the layers the connector uses.
- `task_submit_response.json`: the answer of `POST /task`.
- `*-Statistics.csv`, `*-pixel-reliability-Statistics-QA.csv`, `*-pixel-reliability-lookup.csv`: files of the
  bundle, byte for byte (their SHA-256 matches the bundle listing).
- `bundle_listing.json`: `GET /bundle/<task id>`, **cut** to the files the tests need plus three GeoTIFF
  entries, and **without the signed `s3_url` of each file**.

Not recorded on purpose: the account name that AppEEARS echoes in the task status and in the bundle's
`*-request.json`. No token and no password appear here.
