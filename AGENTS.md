# seasonal-analysis — Agent Conventions

`seasonal-analysis` is a pure Python package that implements the seasonal
pattern / seasonal-window engine (curve modes, cost-aware scoring, data-snooping
significance, promotion evidence report). It is consumed by `tradesurge`
(the `tradesurge_artifacts_api` worker imports `seasonal_analysis.*`; on the
`matrillosub1` dev host the repo is bind-mounted read-only into the worker
container).

## Git workflow

This repo follows the standard branch model used across the fleet:

- **`development` is the integration branch.** Cut feature branches from
  `development` and open PRs **into `development`**. Never target `main` with a
  normal feature PR.
- **`main` is release-only.** It advances solely through a deliberate release
  merge from `development`, on explicit instruction. Never commit or push
  directly to `main`.
- Keep the working checkout on `development` for routine work.
- Branch names: `feature/...`, `fix/...`, or a board-item-derived name.

## Environment

- Python: see `.python-version` / `pyproject.toml`.
- Install for development: `pip install -e .`
- Run tests: `python -m pytest` from the repo root.

## Consumers

- `tradesurge` (`~/apps/tradesurge`) is the primary consumer; its SEAS board
  cards live in the companion notes repo `tss-notes` (`~/apps/tradesurge-notes`),
  not here.
