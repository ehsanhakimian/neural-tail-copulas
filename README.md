#! Macro-Conditioned Neural Copulas for Asymmetric Tail Risk

Author:Ehsan Hakimian · University of Applied Sciences for Business & Society BFI Vienna · ORCID [0009-0001-4721-1361](https://orcid.org/0009-0001-4721-1361) · Academic contact: fhb241835@hwg.at.

### What this repository contains

Research code, checked copula mathematics, reproducibility configurations, synthetic replication summaries, author-generated financial model-performance statistics, and figure exports associated with the paper *Macro-Conditioned Neural Copulas for Asymmetric Tail Risk*. The work evaluates recurrent Clayton/survival-Clayton C-vines, a restricted Gaussian second tree, and established benchmark copulas.

Main empirical finding: A two-tree extension improves the neural first-tree model in the external asset-universe test, but dynamic Student-t and EWMA Gaussian outperform the neural specification in held-out financial density forecasts. No general neural forecasting superiority is claimed. A four-rotation alternative is prospective and unvalidated, not part of the reported performance comparison.

#### Layout

- `code/`: shared simulation and empirical engines, fixed configuration.
- `scripts/`: data acquisition, model training, walk-forward evaluation, risk backtests, statistical audits, and tests.
- `results/`: selected author-generated aggregate and evaluation outputs (no raw vendor data).
- `figures/`: selected manuscript figures.
- `data/`: source acquisition and licensing notes; local source data are git-ignored.
- `docs/REPRODUCTION_PROTOCOL.md`: commands for verification and full reruns.
- `docs/AI_DISCLOSURE_AND_RESEARCH_INTEGRITY.md`: research disclosure and review notes.

#### Quick start (Ubuntu/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/test_copula_math.py
python scripts/test_walkforward.py
python scripts/test_four_rotation.py
```

For an exact archived historical rerun, the original date-stamped data snapshots and the same package versions are needed. Public price endpoints and ex-post adjusted histories can change. See `data/README.md` for the restricted third-party data policy.

### Transparency and data policy

Outputs are from simulation, observed financial assets and FRED/ALFRED historical-information checks. Individual tests on overlapping dates are exploratory. The public repository excludes private API keys, raw third-party price downloads and internal journal correspondence. See `docs/REPRODUCTION_PROTOCOL.md` for scripts.

The author received no external funding and reports no competing interests. OpenAI ChatGPT assisted with language, organization, programming and editorial working drafts; the author is responsible for independent verification of scientific claims and the final submission.

### Citation

Hakimian, E. (2026). Macro-Conditioned Neural Copulas for Asymmetric Tail Risk. Manuscript in preparation. Do not cite a DOI until one is assigned.

### Licensing

This initial upload does not grant an open-source license automatically. The author should choose an explicit code license (for example MIT or BSD-3-Clause) only after checking reuse rights for all included material. Third-party data are not licensed by this repository.
