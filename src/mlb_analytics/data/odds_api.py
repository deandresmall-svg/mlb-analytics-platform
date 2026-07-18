from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from difflib import SequenceMatcher
import re
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd
import requests

MLB_PROP_MARKETS = ("batter_hits", "batter_home_runs", "pitcher_strikeouts")
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TEAM_ALIASES = {
    "arizona diamondbacks": "diamondbacks",
    "atlanta braves": "braves",
    "baltimore orioles": "orioles",
    "boston red sox": "red sox",
    "chicago cubs": "cubs",
    "chicago white sox": "white sox",
    "cincinnati reds": "reds",
    "cleveland guardians": "guardians",
    "colorado rockies": "rockies",
    "detroit tigers": "tigers",
    "houston astros": "astros",
    "kansas city royals": "royals",
    "los angeles angels": "angels",
    "los angeles dodgers": "dodgers",
    "miami marlins": "marlins",
    "milwaukee brewers": "brewers",
    "minnesota twins": "twins",
    "new york mets": "mets",
    "new york yankees": "yankees",
    "athletics": "athletics",
    "philadelphia phillies": "phillies",
    "pittsburgh pirates": "pirates",
    "san diego padres": "padres",
    "san francisco giants": "giants",
    "seattle mariners": "mariners",
    "st louis cardinals": "cardinals",
    "tampa bay rays": "rays",
    "texas rangers": "rangers",
    "toronto blue jays": "blue jays",
    "washington nationals": "nationals",
}


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(text.split())


def normalize_team(value: object) -> str:
    normalized = normalize_name(value)
    return TEAM_ALIASES.get(normalized, normalized)


def american_implied_probability(price: object) -> float | None:
    try:
        odds = float(price)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def no_vig_probability(over_price: object, under_price: object) -> float | None:
    over = american_implied_probability(over_price)
    under = american_implied_probability(under_price)
    if over is None or under is None or over + under <= 0:
        return None
    return over / (over + under)


@dataclass
class OddsUsage:
    remaining: int | None = None
    used: int | None = None
    last: int | None = None


class OddsAPIError(RuntimeError):
    pass


class OddsAPIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        timeout: float = 25.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.usage = OddsUsage()

    def _get(self, path: str, params: dict | None = None):
        if not self.api_key:
            raise OddsAPIError("ODDS_API_KEY is not configured")
        query = {"apiKey": self.api_key, **(params or {})}
        try:
            response = requests.get(
                f"{self.base_url}/{path.lstrip('/')}",
                params=query,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OddsAPIError(f"Odds API request failed: {exc}") from exc

        self.usage = OddsUsage(
            remaining=self._header_int(response, "x-requests-remaining"),
            used=self._header_int(response, "x-requests-used"),
            last=self._header_int(response, "x-requests-last"),
        )
        if not response.ok:
            try:
                detail = response.json().get("message") or response.json().get("error")
            except Exception:
                detail = response.text[:300]
            raise OddsAPIError(
                f"Odds API returned HTTP {response.status_code}: {detail or 'unknown error'}"
            )
        return response.json()

    @staticmethod
    def _header_int(response, name: str) -> int | None:
        try:
            return int(response.headers.get(name))
        except (TypeError, ValueError):
            return None

    def events_for_date(
        self,
        slate_date: date,
        sport: str = "baseball_mlb",
    ) -> list[dict]:
        # Treat the selected date as the MLB slate date in US Eastern time,
        # then convert the bounds to UTC for the API.
        start_local = datetime.combine(slate_date, time.min, tzinfo=EASTERN)
        next_day_local = datetime.combine(
            slate_date + timedelta(days=1),
            time.min,
            tzinfo=EASTERN,
        )
        end_local = next_day_local - timedelta(seconds=1)

        # The Odds API requires whole-second UTC timestamps in the exact
        # YYYY-MM-DDTHH:MM:SSZ format. datetime.isoformat() on time.max would
        # include microseconds, which causes HTTP 422 validation errors.
        start = start_local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = end_local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self._get(
            f"sports/{sport}/events",
            {
                "dateFormat": "iso",
                "commenceTimeFrom": start,
                "commenceTimeTo": end,
            },
        )

    def event_odds(
        self,
        event_id: str,
        markets: tuple[str, ...] = MLB_PROP_MARKETS,
        regions: str = "us",
        bookmakers: str | None = None,
        sport: str = "baseball_mlb",
    ) -> dict:
        params = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"sports/{sport}/events/{event_id}/odds", params)

    @staticmethod
    def match_event(game_row: pd.Series, events: list[dict]) -> dict | None:
        home = normalize_team(game_row.get("home_team"))
        away = normalize_team(game_row.get("away_team"))
        best: dict | None = None
        best_score = 0.0
        for event in events:
            event_home = normalize_team(event.get("home_team"))
            event_away = normalize_team(event.get("away_team"))
            direct = (
                SequenceMatcher(None, home, event_home).ratio()
                + SequenceMatcher(None, away, event_away).ratio()
            ) / 2
            if direct > best_score:
                best, best_score = event, direct
        return best if best_score >= 0.82 else None

    @staticmethod
    def flatten_props(event: dict) -> pd.DataFrame:
        rows: list[dict] = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    description = outcome.get("description")
                    player = description or outcome.get("name")
                    side = outcome.get("name") if description else "Yes"
                    rows.append(
                        {
                            "event_id": event.get("id"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "commence_time": event.get("commence_time"),
                            "bookmaker_key": bookmaker.get("key"),
                            "bookmaker": bookmaker.get("title"),
                            "market": market.get("key"),
                            "last_update": market.get("last_update"),
                            "player": player,
                            "player_normalized": normalize_name(player),
                            "side": side,
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                            "implied_probability": american_implied_probability(
                                outcome.get("price")
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["point"] = pd.to_numeric(frame["point"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        return frame


def paired_no_vig_probability(odds: pd.DataFrame, quote: pd.Series) -> float | None:
    """Return the no-vig probability for a quote using the paired side at the same book/line."""
    if odds.empty:
        return None
    side = str(quote.get("side", "")).lower()
    opposite = "under" if side == "over" else "over" if side == "under" else None
    if opposite is None:
        return None
    mask = (
        (odds["event_id"] == quote.get("event_id"))
        & (odds["bookmaker_key"] == quote.get("bookmaker_key"))
        & (odds["market"] == quote.get("market"))
        & (odds["player_normalized"] == quote.get("player_normalized"))
        & (odds["side"].astype(str).str.lower() == opposite)
    )
    point = quote.get("point")
    if pd.notna(point):
        mask &= pd.to_numeric(odds["point"], errors="coerce").eq(float(point))
    pair = odds.loc[mask]
    if pair.empty:
        return None
    opposite_price = pair.iloc[0].get("price")
    if side == "over":
        return no_vig_probability(quote.get("price"), opposite_price)
    under_fair = no_vig_probability(opposite_price, quote.get("price"))
    return None if under_fair is None else 1.0 - under_fair
