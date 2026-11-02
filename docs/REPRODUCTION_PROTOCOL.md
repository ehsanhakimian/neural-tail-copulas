# Reproduction protocol

1. Make a clean Python environment (3.11 or 3.12 recommended; packages were also used under Colab). Install `requirements.txt`.
2. Run mathematical checks: `python scripts/test_copula_math.py`, `python scripts/test_walkforward.py`, `python scripts/test_four_rotation.py`. The prospective four-rotation implementation has *not* been evaluated on a holdout.
3. Run synthetic replication: `python scripts/run_multiseed.py --config code/config.json --out outputs/multiseed --seeds 50 --start-seed 20261001`. Long computations can be resumed.
4. Retrieve external data: `python scripts/fetch_independent_universe.py --out data/external --start 2008-01-01 --end 2026-11-01`. This also creates `data/external/observations.csv` (ignored by Git).
5. Execute density evaluation: `python scripts/run_walkforward.py --input data/external/observations.csv --out outputs/external_scores --epochs 18 --cadence 26 --rolling-window 260 --features price`.
6. Execute matched risk evaluation: `python scripts/matched_rolling_var_es.py --input data/external/observations.csv --out outputs/external_risk --epochs 18 --cadence 26 --window 260 --draws 3000 --features price`.
7. Audit results: `python scripts/audit_external.py --panel data/external/observations.csv --scores outputs/external_scores/walkforward_test_scores.csv --risk outputs/external_risk/matched_risk_forecasts.csv --out outputs/audit`.

This external evaluation was conducted after earlier related tests; it is an external *asset-universe* validation, not a prospective unseen-time experiment. The latest model extension in `scripts/four_rotation_clayton_PROSPECTIVE.py` is untrained and must not be presented as a verified improvement. The code may take hours and need considerably more RAM/CPU than basic examples.
