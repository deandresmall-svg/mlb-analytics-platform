# MLB GitHub Streamlit Analytics

A GitHub-ready Streamlit repository for MLB schedule and box-score ingestion, historical backfills, point-in-time rolling features, weather and park context, chronological model validation, and separate models for game winners, batter hits, batter home runs, and starting-pitcher strikeouts.

## Included now

- Live and historical MLB schedule sync
- Completed-game box-score ingestion
- Team game logs and rolling 14/30-game offense
- Point-in-time team win percentage and rest
- Starting-pitcher rolling ERA and K/BB
- Three-game bullpen pitch workload
- Park-factor lookup and Open-Meteo weather joins
- Chronological 80/20 holdout splits
- Calibrated logistic models for home win, 1+ hit, and HR
- Poisson strikeout-count model
- Streamlit pages for slate, backfill, models, data quality, and calibration
- Docker, GitHub Actions, tests, environment template, and Streamlit config
- No slip optimizer

## Important data warning

The feature builders use only rows dated before the game being predicted. That avoids the most obvious future leakage. Park factors in `venues.py` are starter approximations and should be replaced with your chosen yearly source. Weather is historical reanalysis for past dates and forecast data for upcoming dates. Player models currently use box-score rolling form; Statcast pitch-level enhancements can be layered in later without changing the target-specific architecture.

## Repository setup

```bash
git clone YOUR_REPOSITORY_URL
cd MLB_GitHub_Streamlit_Analytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
export PYTHONPATH=src
```

## Backfill data

Start with one season or a smaller test window:

```bash
PYTHONPATH=src python scripts/backfill.py --start 2025-03-18 --end 2025-09-28
```

This can make many MLB and weather requests. Run it from a persistent environment, not from a short Streamlit web session.

## Train all models

```bash
PYTHONPATH=src python scripts/train_models.py
```

## Run Streamlit

```bash
PYTHONPATH=src streamlit run app/Home.py
```

## Deploy from GitHub to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose the repository and branch.
3. Use `app/Home.py` as the entrypoint.
4. Select Python 3.12.
5. Add secrets only when you introduce private providers. The included MLB and Open-Meteo sources do not require keys.

Streamlit Community Cloud installs dependencies from `requirements.txt`. The app should not perform a multi-season backfill during page startup; build the database separately and use a persistent database for production.

## Persistence warning

SQLite works for local development and demos. Streamlit Community Cloud storage is not a durable production database. For a continuously updated deployed app, set `MLB_DATABASE_URL` to a persistent PostgreSQL database and adapt the SQLite-specific upsert syntax if needed, or generate and commit a read-only SQLite snapshot through GitHub Actions.

## GitHub Actions

- `tests.yml` runs Ruff and pytest on pushes and pull requests.
- `daily-sync.yml` can sync yesterday/today and commit a database snapshot. Review repository size and MLB data-use requirements before enabling automatic database commits.

## Model targets

| Model | Target | Current feature source |
|---|---|---|
| Home win | Home team wins | Team offense, starter form, bullpen, rest, park, weather |
| 1+ hit | Batter records at least one hit | Batter rolling box-score form |
| Home run | Batter records a HR | Batter rolling HR and power form |
| Strikeouts | Starter strikeout count | Starter rolling K/IP, pitches, innings |

These are analytical baselines, not guaranteed betting edges. Add lineups, handedness, opponent quality, umpire, pitch mix, and Statcast quality-of-contact before treating player probabilities as production-grade.
