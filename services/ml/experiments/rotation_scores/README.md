# Rotation score sensitivity evaluation

Generated with OpenAI Codex (gpt-6.1-sol). Human review pending.

## Purpose and selection rule

Compare the existing provisional constants with smaller and larger adjustments.
Choose before inspecting results: reject any variant producing out-of-range scores
or intervals; prefer retaining the deployed constants when all pass and no observed
field outcome distinguishes their accuracy. Do not maximize a hand-picked strategy's
score or treat a desired winner as a measured label. This is software verification
and sensitivity analysis, not calibration or proof of agronomic quality.

## Reproduce

From `services/ml`, with the locked environment:

```bash
uv run python experiments/rotation_scores/evaluate.py --out experiments/rotation_scores/results.json
make check
```

The JSON records input file hashes, constants, cases, winner sets including ties,
overlap of the leading score ranges and the neutral example. It has no timestamp
and is deterministic for the same inputs. `test_rotation_evaluation.py` checks
repeatability and restoration of serving constants after the experiment.

## Inputs and iterations

Forty-five synthetic scenarios combine rainfall changes of -80, -30, 0, 50 and
100 percent; temperature changes of -5, 0 and 8 degrees C; and three ENSO phases.
Ten profiles cover equal weights, each of the seven priorities alone, water-limited
conditions and avoidance of beans. Every run compares maize/fallow against the
three alternatives. The current baseline is an example, not verified CEDA practice.

The four iterations multiply six constants together: the two legume bonuses, the
bare-fallow penalty and the dry, wet and warming slopes. Other constants and all
crop traits remain fixed. Winner changes use the rounded API scores; ties count
as sets, so winner counts can sum to more than the number of cases.

| Multiplier | Interpretation | Cases | Bounded scores and intervals | Changed winner sets against current |
|---|---|---|---|---|
| 1.0 | Current constants, reference iteration | 450 | Yes | 0 |
| 0.0 | Disable these six adjustments | 450 | Yes | 45 |
| 0.5 | Reduce these six adjustments by half | 450 | Yes | 45 |
| 1.5 | Increase these six adjustments by half | 450 | Yes | 45 |

In each variant the leading two heuristic ranges overlap in 449 of 450 cases.
That overlap does not establish statistical equivalence or a significance level.
It shows why a small point-score lead should not be presented as a confident choice.
Joint scaling does not isolate each parameter and is not a search for an optimum.

## Selected result

Retain multiplier 1.0 and the existing provisional constants. All variants pass
the numeric bounds check, but the available evidence cannot select an empirically
better magnitude. Do not describe the retained setting as an agronomic optimum.
The result depends strongly on the selected priorities and on provisional traits.

For equal priorities, neutral ENSO and zero climate changes:

| Sequence | Fit points out of 100 | Heuristic range, points out of 100 |
|---|---|---|
| Maize then fallow | 27.5 | 17.5 to 37.5 |
| Maize then black beans | 48.6 | 38.6 to 58.6 |
| Maize then cover crop | 45.0 | 35.0 to 55.0 |
| Maize then black beans then cover crop | 51.4 | 41.4 to 61.4 |

These are relative fit scores, not yield, income, probability or a farm-specific
prescription. A cover crop has no selected species; its nitrogen score assumes a
legume mixture. Periods have equal weight regardless of duration. There are no
local crop calendars, prices, costs or crop-specific field labels in this evaluation.

## Changes supported by the tests

Five regression cases first failed on unsupported equipment claims, missing
constraint/ENSO evidence and missing cover-species and interval qualifications.
The serving fix corrects explanations and increments the rule-engine patch version
to `rules-0.1.1`. Trait values, score constants and formulas remain identical.
An explicit comparison against the pre-change engine confirmed identical scores,
intervals and candidate identities for all 450 reference cases.
Forty-five boundary cases verify finite bounded scores and ordered intervals.
An additional check verifies invariance to proportional scaling of all weights.

Future calibration requires observed outcomes or independently reviewed strategy
ratings, the actual current rotation, management constraints and a predeclared
evaluation method. Interval calibration needs observations too. Passing these tests
does not resolve those missing inputs or substitute for local expert review.
