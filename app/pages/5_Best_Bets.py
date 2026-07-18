from __future__ import annotations

from datetime import date
from difflib import SequenceMatcher
from math import exp, factorial, isfinite

import pandas as pd
import streamlit as st

from mlb_analytics.config import settings
from mlb_analytics.data.odds_api import normalize_name, paired_no_vig_probability
from mlb_analytics.services.pipeline import AnalyticsService


def pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.1%}" if isfinite(number) else "N/A"


def fair_american(probability: object) -> str:
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return "N/A"
    if not isfinite(p) or p <= 0 or p >= 1:
        return "N/A"
    odds = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return f"{odds:+.0f}"


def decimal_return(price: float) -> float:
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def expected_value(probability: float, price: float) -> float:
    return probability * decimal_return(price) - 1


def poisson_over(mean: float, line: float) -> float:
    threshold = int(line // 1 + 1)
    cdf = sum(exp(-mean) * mean**k / factorial(k) for k in range(threshold))
    return max(0.0, min(1.0, 1 - cdf))


def matched_quotes(
    odds: pd.DataFrame,
    player: str,
    market: str,
    game_pk: object = None,
    side: str = "Over",
    point: float | None = None,
) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    required = {"market", "side", "player_normalized", "price"}
    if not required.issubset(odds.columns):
        return pd.DataFrame()
    frame = odds[
        (odds["market"] == market)
        & (odds["side"].astype(str).str.lower() == side.lower())
    ].copy()
    if game_pk is not None and "game_pk" in frame.columns:
        frame = frame[frame["game_pk"] == game_pk]
    if point is not None:
        frame = frame[pd.to_numeric(frame["point"], errors="coerce").eq(point)]
    if frame.empty:
        return frame
    target = normalize_name(player)
    frame["name_score"] = frame["player_normalized"].map(
        lambda value: SequenceMatcher(None, target, value).ratio()
    )
    frame = frame[frame["name_score"] >= 0.84]
    frame["price_num"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame.dropna(subset=["price_num"])


def best_player_price(*args, **kwargs) -> pd.Series | None:
    frame = matched_quotes(*args, **kwargs)
    if frame.empty:
        return None
    return frame.sort_values(
        ["price_num", "name_score"], ascending=[False, False]
    ).iloc[0]


def hit_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("batting_order", 9) <= 4:
        reasons.append("top-4 lineup spot")
    if row.get("projected_pa", 0) >= 4.4:
        reasons.append("strong PA projection")
    if row.get("hit_rate_10", 0) >= 0.7:
        reasons.append("70%+ recent hit rate")
    return ", ".join(reasons[:3]) or "model feature blend"


def hr_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("batting_order", 9) <= 4:
        reasons.append("top-4 lineup spot")
    if row.get("iso_30", 0) >= 0.2:
        reasons.append("strong 30-game ISO")
    if row.get("opponent_sp_hr_per_9_5", 0) >= 1.3:
        reasons.append("starter allows power")
    return ", ".join(reasons[:3]) or "model feature blend"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_slate_odds(slate_date: date) -> dict:
    """Cache sportsbook requests for five minutes to protect API credits."""
    return AnalyticsService(settings).odds_for_slate(slate_date)


st.set_page_config(page_title="Today's Best Bets", page_icon="⭐", layout="wide")
st.title("⭐ Today's Model + Market Board")
st.caption(
    "Model probabilities are compared with the best available sportsbook price. "
    "Odds are cached for five minutes; always confirm a line before betting."
)

svc = AnalyticsService(settings)
slate_date = st.date_input("Slate date", date.today())

controls = st.columns([1, 1, 1, 2])
with controls[0]:
    refresh_odds = st.button(
        "Refresh sportsbook odds", type="primary", width="stretch"
    )
with controls[1]:
    st.metric("Odds API", "Connected" if settings.odds_api_key else "Key missing")
with controls[2]:
    st.metric("Regions", settings.odds_regions or "us")

if refresh_odds:
    fetch_slate_odds.clear()

with st.spinner("Building full-slate predictions..."):
    outputs = svc.slate_prop_predictions(slate_date)
batters, pitchers = outputs["batters"], outputs["pitchers"]

if settings.odds_api_key:
    with st.spinner("Loading MLB player props..."):
        payload = fetch_slate_odds(slate_date)
else:
    payload = {
        "rows": pd.DataFrame(),
        "usage": {},
        "error": "ODDS_API_KEY is not configured.",
        "event_errors": [],
        "matched_games": 0,
        "requested_games": 0,
    }

odds = payload.get("rows", pd.DataFrame())
usage = payload.get("usage", {})

if payload.get("error"):
    st.warning(payload["error"])
else:
    status = st.columns(4)
    status[0].metric("Matched games", f"{payload.get('matched_games', 0)}/{payload.get('requested_games', 0)}")
    status[1].metric("Odds rows", f"{len(odds):,}")
    status[2].metric("Credits remaining", usage.get("remaining", "N/A"))
    status[3].metric("Refresh cost", usage.get("refresh_cost", usage.get("last", "N/A")))

errors = payload.get("event_errors", [])
if errors:
    with st.expander(f"Odds warnings ({len(errors)})"):
        for message in errors:
            st.write(f"• {message}")

if batters.empty and pitchers.empty:
    st.info(
        "No prediction rows are available. Sync the selected slate and make sure "
        "the models are trained."
    )
    st.stop()

hit_tab, hr_tab, k_tab = st.tabs(
    ["Top 1+ Hit", "Top Home Run", "Pitcher Strikeouts"]
)

with hit_tab:
    minimum = st.slider("Minimum hit probability", 0.40, 0.85, 0.55, 0.01)
    hit = batters[batters.get("hit_probability", 0) >= minimum].copy()
    rows: list[dict] = []
    for _, row in hit.iterrows():
        quote = best_player_price(
            odds,
            row["player_name"],
            "batter_hits",
            game_pk=row.get("game_pk"),
            point=0.5,
        )
        price = float(quote["price"]) if quote is not None else None
        raw_market = (
            float(quote["implied_probability"]) if quote is not None else None
        )
        no_vig = paired_no_vig_probability(odds, quote) if quote is not None else None
        model_p = float(row["hit_probability"])
        rows.append(
            {
                "Player": row["player_name"],
                "Team": row["team"],
                "Matchup": row["matchup"],
                "Model probability": model_p,
                "Model fair odds": fair_american(model_p),
                "Best price": f"{price:+.0f}" if price is not None else "N/A",
                "Book": quote["bookmaker"] if quote is not None else "N/A",
                "Raw implied": raw_market,
                "No-vig market": no_vig,
                "Model edge": model_p - no_vig if no_vig is not None else None,
                "EV per $1": expected_value(model_p, price) if price is not None else None,
                "Projected order": row.get("batting_order"),
                "Projected PA": row.get("projected_pa"),
                "Opposing starter": row.get("opponent_pitcher"),
                "Why": hit_reasons(row),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No hitters meet the selected probability filter.")
    else:
        table = table.sort_values(
            ["EV per $1", "Model probability"], ascending=False, na_position="last"
        )
        for column in ["Model probability", "Raw implied", "No-vig market", "Model edge"]:
            table[column] = table[column].map(pct)
        table["EV per $1"] = pd.to_numeric(
            table["EV per $1"], errors="coerce"
        ).map(lambda value: f"{value:+.1%}" if pd.notna(value) else "N/A")
        st.dataframe(table.head(40), width="stretch", hide_index=True)

with hr_tab:
    minimum = st.slider("Minimum HR probability", 0.03, 0.40, 0.08, 0.01)
    hr = batters[batters.get("home_run_probability", 0) >= minimum].copy()
    rows = []
    for _, row in hr.iterrows():
        quote = best_player_price(
            odds,
            row["player_name"],
            "batter_home_runs",
            game_pk=row.get("game_pk"),
            point=0.5,
        )
        price = float(quote["price"]) if quote is not None else None
        raw_market = (
            float(quote["implied_probability"]) if quote is not None else None
        )
        no_vig = paired_no_vig_probability(odds, quote) if quote is not None else None
        model_p = float(row["home_run_probability"])
        rows.append(
            {
                "Player": row["player_name"],
                "Team": row["team"],
                "Matchup": row["matchup"],
                "Model probability": model_p,
                "Model fair odds": fair_american(model_p),
                "Best price": f"{price:+.0f}" if price is not None else "N/A",
                "Book": quote["bookmaker"] if quote is not None else "N/A",
                "Raw implied": raw_market,
                "No-vig market": no_vig,
                "Model edge": model_p - no_vig if no_vig is not None else None,
                "EV per $1": expected_value(model_p, price) if price is not None else None,
                "Projected order": row.get("batting_order"),
                "Projected PA": row.get("projected_pa"),
                "Opposing starter": row.get("opponent_pitcher"),
                "Why": hr_reasons(row),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No hitters meet the selected probability filter.")
    else:
        table = table.sort_values(
            ["EV per $1", "Model probability"], ascending=False, na_position="last"
        )
        for column in ["Model probability", "Raw implied", "No-vig market", "Model edge"]:
            table[column] = table[column].map(pct)
        table["EV per $1"] = pd.to_numeric(
            table["EV per $1"], errors="coerce"
        ).map(lambda value: f"{value:+.1%}" if pd.notna(value) else "N/A")
        st.dataframe(table.head(40), width="stretch", hide_index=True)

with k_tab:
    rows = []
    for _, row in pitchers.iterrows():
        player_quotes = matched_quotes(
            odds,
            row["player_name"],
            "pitcher_strikeouts",
            game_pk=row.get("game_pk"),
            side="Over",
        )
        best = None
        best_ev = None
        best_probability = None
        for _, quote in player_quotes.iterrows():
            if pd.isna(quote.get("point")):
                continue
            line = float(quote["point"])
            price = float(quote["price"])
            probability = poisson_over(float(row["projected_strikeouts"]), line)
            ev = expected_value(probability, price)
            if best_ev is None or ev > best_ev:
                best, best_ev, best_probability = quote, ev, probability
        no_vig = paired_no_vig_probability(odds, best) if best is not None else None
        rows.append(
            {
                "Pitcher": row["player_name"],
                "Team": row["team"],
                "Opponent": row["opponent"],
                "Matchup": row["matchup"],
                "Projected K": round(float(row["projected_strikeouts"]), 2),
                "Best over line": best["point"] if best is not None else None,
                "Best price": f"{float(best['price']):+.0f}" if best is not None else "N/A",
                "Book": best["bookmaker"] if best is not None else "N/A",
                "Model over probability": best_probability,
                "No-vig market": no_vig,
                "Model edge": best_probability - no_vig if best_probability is not None and no_vig is not None else None,
                "EV per $1": best_ev,
                "Recent K/start": round(float(row.get("pitcher_k_avg_5", 0)), 2),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No pitcher projections are available.")
    else:
        table = table.sort_values(
            ["EV per $1", "Projected K"], ascending=False, na_position="last"
        )
        for column in ["Model over probability", "No-vig market", "Model edge"]:
            table[column] = table[column].map(pct)
        table["EV per $1"] = pd.to_numeric(
            table["EV per $1"], errors="coerce"
        ).map(lambda value: f"{value:+.1%}" if pd.notna(value) else "N/A")
        st.dataframe(table.head(40), width="stretch", hide_index=True)

st.caption(
    "Raw implied probability includes sportsbook vig. No-vig market probability "
    "uses the Over and Under prices from the same sportsbook at the same line."
)
