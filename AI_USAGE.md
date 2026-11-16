# AI usage in `agrofarm-analytic`

This file is the public statement required by the NASA Space Apps Challenge 2026 submission rules: where and how AI tools were used. The rule that keeps it true is G11 in [`AGENTS.md`](AGENTS.md).

## Summary

AI coding assistants were used as tools to write and review code, tests, CI workflows, contracts and documentation. They did not choose the idea of the solution. A team member reviews every pull request before it is merged.

## Tools

| Tool | Model | Used by | Used for |
|---|---|---|---|
| OpenAI Codex | `gpt-6.1-sol` | `JorgeVilledaS` | ML-040 explanation corrections, contract-boundary regression tests, reproducible synthetic score sensitivity experiment and evaluation documentation. No field calibration performed; human review pending. |
| OpenAI Codex | `gpt-6.1-sol` | `JorgeVilledaS` | ML-038 crop-trait numeric validation, regression tests and local checks. Human review pending. |
| Claude Code | `claude-fable-5-2` | `eldmark` | Repository rules, CI, scaffolds, connectors, tests, documentation, data analysis |
| Claude Code | `claude-sonnet-5-5` | `eldmark` | Rotation engine constraint fix (ML-040), bare-field detection (DS-056), model registry (ML-050), AppEEARS connector (DS-030), reconciling the rotation PRs, issue triage, documentation |
| Codex | `gpt-6` | `jdivass` | Connector code, recorded-fixture tests, documentation, review |
| OpenAI Codex | `GPT-6` | `eldmark` | Maize-leaf dataset audit, CPU classifier, grouped evaluation, private image API, deployment checks and documentation; recording owner-confirmed crop-traits review while preserving source-verification limits |
| Antigravity | `Gemini 3.8` | `Angel Chavez (angcoder-c)` | Repository synchronization, AppEEARS/Earthdata credentials and ML verification |

Team members who used another tool add a row here in their next pull request (rule G11.3).

## How to trace each use

- **Commits.** From 2026-11-14, every AI-assisted commit ends with an `Assisted-by: <tool> (<model>)` trailer:
  `git log --grep="^Assisted-by:" --format="%h %an %s"`
- **Pull requests.** Every pull request has an "AI use" section that says what the tool did and who reviewed it.
- **Before the rule.** Commits authored by `eldmark` (Marco Diaz) on 2026-11-14 before this file existed were written with Claude Code and reviewed by the author. They do not carry the trailer because history is never rewritten (rule G8). Other authors: not recorded, each author states it here.

## Not generated with AI

- The project idea and the choice of challenge.
- Source datasets (INSIVUMEH, NOAA, NASA, CHIRPS, SoilGrids and the others in the dataset catalog).

OpenAI Codex (GPT-6), 2026-11-15: connected the owner-accepted regional NDVI outlook to serving, preserved the failed original acceptance rule, added quantiles and median SHAP differences, and pinned the release artifact for staging. Human owner approved limited use; no field validation is implied.
