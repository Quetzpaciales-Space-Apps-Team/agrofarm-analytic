Synthetic fixture for the ML-020 CI gate. NOT real data: a made-up station with a strong ENSO
effect on rainfall (see tests/conftest.py). `climate.csv` is daily rainfall, `enso.csv` monthly
ONI/RONI. `baseline.json` stores the pinball loss of the baseline model on the held-out years.
Regenerate the data with `make_fixture.py` and the baseline with `agrofarm_ml.gate --update`;
a changed baseline needs a reason in the pull request (rule G2: never loosen it to pass).
