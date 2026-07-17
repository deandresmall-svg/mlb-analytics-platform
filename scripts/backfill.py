import argparse
from datetime import date
from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService
p=argparse.ArgumentParser();p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--no-boxscores',action='store_true');a=p.parse_args()
print(AnalyticsService(settings).backfill(date.fromisoformat(a.start),date.fromisoformat(a.end),not a.no_boxscores))
