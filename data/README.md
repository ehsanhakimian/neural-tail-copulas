# Data provenance and redistribution

The GitHub repository intentionally does **not** contain raw adjusted-price time series, third-party licensed price snapshots, confidential material, or FRED API keys.

To reconstruct the external six-ETF input, execute `python scripts/fetch_independent_universe.py --out data/external --start 2008-01-01 --end 2026-11-01` while online. The script obtains available data from Yahoo Finance using `yfinance`; retrospectively adjusted prices, corporate actions and endpoint behavior may differ across acquisition dates. Use the archived dataset checksum in the private research package when checking exact historical reproduction.

For original-sector ETF macro inputs, execute `python scripts/fetch_modern_market.py --out data/us_sector --start 2005-01-01 --end 2026-11-01` (requires internet). FRED/ALFRED values vary by series/vintage and source policies. Historical-vintage collection requires a personal `FRED_API_KEY` environment variable. Do not place it in files or commits.

Synthetic summary outputs and author-generated evaluation statistics in `results/` can be inspected without retrieving the vendor price data. They do not replace the historical raw inputs for complete re-estimation.
