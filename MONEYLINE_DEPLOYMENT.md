# Moneyline deployment

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create or edit the Streamlit app.
3. Select `streamlit_app.py` as the main file.
4. Select Python 3.12.
5. Add `ODDS_API_KEY` to Streamlit Secrets only if sportsbook odds are required.

The site can sync the current MLB slate. Historical backfills and model training should run outside Streamlit.

## Colab/offline training

```bash
git clone https://github.com/deandresmall-svg/mlb-analytics-platform.git
cd mlb-analytics-platform
pip install -r requirements.txt
PYTHONPATH=src python scripts/train_moneyline.py --start 2025-03-18 --end 2025-09-28
```

The trainer creates:

- `data/mlb_analytics.db`
- `models/home_win.joblib`

The challenger is accepted only when it:

- improves Brier score by at least 0.002 across rolling validation;
- wins at least three of four chronological folds; and
- improves Brier score by at least 0.002 on the untouched final 20%.

If rejected, an existing production model is not overwritten.

## Daily use

1. Open the Streamlit site.
2. Upload `mlb_analytics.db` and `home_win.joblib` in the sidebar if they are not committed to the repository.
3. Select the slate date.
4. Press **Sync slate**.
5. Review the Moneyline tab for model probability and fair American odds.
6. Compare model probability with no-vig market probability before recording a play.
7. Download the current database/model after any in-session update because Streamlit storage is temporary.

Only upload joblib model files that you created or trust.
