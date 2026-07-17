from pprint import pprint
from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService
pprint(AnalyticsService(settings).train_all())
