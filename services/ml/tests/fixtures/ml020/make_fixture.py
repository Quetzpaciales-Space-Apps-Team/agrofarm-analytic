"""Regenerate the synthetic fixture: uv run python tests/fixtures/ml020/make_fixture.py

Same generator as tests/conftest.py (seed 7), 1990 to 2010, rainfall rounded to 0.01 mm.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from conftest import make_climate, make_enso  # type: ignore[import-not-found]  # noqa: E402

enso = make_enso("1989-01-01", "2011-12-01")
climate = make_climate(enso, "1990-01-01", "2010-12-31", seed=7)
climate["rainfall_mm"] = climate["rainfall_mm"].round(2)
climate.to_csv(HERE / "climate.csv", index=False)
enso.assign(oni=enso["oni"].round(3), roni=enso["roni"].round(3)).to_csv(
    HERE / "enso.csv", index=False
)
