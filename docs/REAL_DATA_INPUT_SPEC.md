# Data needed before any real-finance claims

Send `returns.csv` with `date` plus adjusted daily arithmetic return columns for a market root and at least three other risky assets; indicate symbols, source, adjustment convention, and timezone. Send `macro.csv` containing `observation_date`, `release_timestamp`, and macro indicators in separate columns. The exact date that each indicator became publicly available matters more than the label of its reference period. Provide a data dictionary for each variable and record data provenance and licensing.

Run `python scripts/validate_real_data.py --returns returns.csv --macro macro.csv` after placing both files inside this publisher folder. This validator does not perform estimation, and success does not imply forecasting performance. The real-data estimation pipeline remains to be developed and independently verified.
