from __future__ import annotations

from datetime import date
from math import isfinite

import pandas as pd
import streamlit as st

from mlb_analytics.config import settings
from mlb_analytics.features.game_features import build_game_features
from mlb_analytics.services.pipeline import AnalyticsService


def probability_to_american(probability: float) -> str:
    """Convert a win probability to fair American odds."""
    if not isfinite(probability) or probability <= 0 or probability >= 1:
        return "N/A"
    if probability >= 0.5:
        odds = -100 * probability / (1 - probability)
    else:
        odds = 100 * (1 - probability) / probability
    return f"{odds:+.0f}"


def format_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not isfinite(number):
        return "N/A"
    return f"{number:.1%}"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def load_table(service: AnalyticsService, table_name: str) -> pd.DataFrame:
    """Load a database table without crashing the Streamlit page."""
    try:
        return service.repo.query(f"SELECT * FROM {table_name}")
    except Exception:
        return pd.DataFrame()


def batter_summary(
    batting: pd.DataFrame,
    team_id: int,
    cutoff_date: date,
) -> pd.DataFrame:
    """Create point-in-time rolling batter indicators for one team."""
    if batting.empty:
        return pd.DataFrame()

    frame = batting.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    cutoff = pd.Timestamp(cutoff_date)

    frame = frame[
        (frame["team_id"] == int(team_id))
        & (frame["game_date"] < cutoff)
    ].sort_values(["player_id", "game_date"])

    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for (player_id, player_name), group in frame.groupby(
        ["player_id", "player_name"],
        dropna=False,
    ):
        group = group.sort_values("game_date")
        last_30 = group.tail(30)
        last_10 = group.tail(10)

        if last_10.empty:
            continue

        last_10_pa = float(last_10["plate_appearances"].sum())
        last_30_pa = float(last_30["plate_appearances"].sum())
        last_10_ab = float(last_10["at_bats"].sum())
        last_30_ab = float(last_30["at_bats"].sum())

        rows.append(
            {
                "player_id": player_id,
                "player": player_name,
                "games_10": len(last_10),
                "games_30": len(last_30),
                "last_game": last_30["game_date"].max(),
                "pa_10": int(last_10_pa),
                "hit_game_rate_10": float((last_10["hits"] > 0).mean()),
                "hit_game_rate_30": float((last_30["hits"] > 0).mean()),
                "batting_avg_10": safe_divide(
                    last_10["hits"].sum(),
                    last_10_ab,
                ),
                "batting_avg_30": safe_divide(
                    last_30["hits"].sum(),
                    last_30_ab,
                ),
                "hits_10": int(last_10["hits"].sum()),
                "hits_30": int(last_30["hits"].sum()),
                "hr_10": int(last_10["home_runs"].sum()),
                "hr_30": int(last_30["home_runs"].sum()),
                "hr_game_rate_10": float(
                    (last_10["home_runs"] > 0).mean()
                ),
                "hr_pa_rate_30": safe_divide(
                    last_30["home_runs"].sum(),
                    last_30_pa,
                ),
                "tb_pa_30": safe_divide(
                    last_30["total_bases"].sum(),
                    last_30_pa,
                ),
                "strikeout_rate_30": safe_divide(
                    last_30["strikeouts"].sum(),
                    last_30_pa,
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Recent plate appearances are used only as a practical activity filter.
    return result.sort_values(
        ["pa_10", "hit_game_rate_10"],
        ascending=[False, False],
    ).reset_index(drop=True)


def pitcher_summary(
    pitching: pd.DataFrame,
    player_id: object,
    cutoff_date: date,
) -> dict[str, object] | None:
    """Summarize a probable pitcher's five most recent starts."""
    if pitching.empty or pd.isna(player_id):
        return None

    frame = pitching.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    cutoff = pd.Timestamp(cutoff_date)

    frame = frame[
        (frame["player_id"] == int(player_id))
        & (frame["game_date"] < cutoff)
        & (frame["games_started"] > 0)
    ].sort_values("game_date")

    recent = frame.tail(5)
    if recent.empty:
        return None

    innings = float(recent["innings_pitched"].sum())
    strikeouts = float(recent["strikeouts"].sum())
    walks = float(recent["walks"].sum())
    hits = float(recent["hits"].sum())
    earned_runs = float(recent["earned_runs"].sum())

    return {
        "starts": len(recent),
        "last_start": recent["game_date"].max(),
        "avg_innings": float(recent["innings_pitched"].mean()),
        "avg_pitches": float(recent["pitches_thrown"].mean()),
        "avg_strikeouts": float(recent["strikeouts"].mean()),
        "k_per_inning": safe_divide(strikeouts, innings),
        "era": safe_divide(earned_runs * 9, innings),
        "whip": safe_divide(walks + hits, innings),
        "avg_hits_allowed": float(recent["hits"].mean()),
        "avg_walks": float(recent["walks"].mean()),
        "avg_home_runs_allowed": float(recent["home_runs"].mean()),
    }


def display_batter_table(
    summary: pd.DataFrame,
    mode: str,
) -> None:
    if summary.empty:
        st.info(
            "No historical batter rows are available for this team before "
            "the selected game."
        )
        return

    if mode == "hits":
        output = summary[
            [
                "player",
                "games_10",
                "pa_10",
                "hit_game_rate_10",
                "hit_game_rate_30",
                "batting_avg_10",
                "batting_avg_30",
                "hits_10",
                "hits_30",
                "strikeout_rate_30",
            ]
        ].copy()

        output.columns = [
            "Player",
            "G (10)",
            "PA (10)",
            "1+ Hit Rate (10)",
            "1+ Hit Rate (30)",
            "AVG (10)",
            "AVG (30)",
            "Hits (10)",
            "Hits (30)",
            "K Rate (30)",
        ]

        for column in [
            "1+ Hit Rate (10)",
            "1+ Hit Rate (30)",
            "K Rate (30)",
        ]:
            output[column] = output[column].map(format_percent)

        for column in ["AVG (10)", "AVG (30)"]:
            output[column] = output[column].map(
                lambda value: format_number(value, 3)
            )

    else:
        output = summary[
            [
                "player",
                "games_10",
                "pa_10",
                "hr_10",
                "hr_30",
                "hr_game_rate_10",
                "hr_pa_rate_30",
                "tb_pa_30",
                "strikeout_rate_30",
            ]
        ].copy()

        output.columns = [
            "Player",
            "G (10)",
            "PA (10)",
            "HR (10)",
            "HR (30)",
            "HR Game Rate (10)",
            "HR/PA (30)",
            "TB/PA (30)",
            "K Rate (30)",
        ]

        for column in [
            "HR Game Rate (10)",
            "HR/PA (30)",
            "K Rate (30)",
        ]:
            output[column] = output[column].map(format_percent)

        output["TB/PA (30)"] = output["TB/PA (30)"].map(
            lambda value: format_number(value, 3)
        )

    st.dataframe(
        output,
        use_container_width=True,
        hide_index=True,
    )


def display_pitcher_metrics(
    pitcher_name: object,
    summary: dict[str, object] | None,
) -> None:
    st.markdown(f"#### {pitcher_name or 'TBD'}")

    if summary is None:
        st.info("No prior-start data is available for this probable pitcher.")
        return

    first_row = st.columns(4)
    first_row[0].metric("Recent starts", int(summary["starts"]))
    first_row[1].metric(
        "Average innings",
        format_number(summary["avg_innings"]),
    )
    first_row[2].metric(
        "Average pitches",
        format_number(summary["avg_pitches"], 0),
    )
    first_row[3].metric(
        "Average strikeouts",
        format_number(summary["avg_strikeouts"]),
    )

    second_row = st.columns(4)
    second_row[0].metric(
        "K per inning",
        format_number(summary["k_per_inning"], 3),
    )
    second_row[1].metric(
        "Recent ERA",
        format_number(summary["era"]),
    )
    second_row[2].metric(
        "Recent WHIP",
        format_number(summary["whip"]),
    )
    second_row[3].metric(
        "Hits allowed/start",
        format_number(summary["avg_hits_allowed"]),
    )

    third_row = st.columns(2)
    third_row[0].metric(
        "Walks/start",
        format_number(summary["avg_walks"]),
    )
    third_row[1].metric(
        "HR allowed/start",
        format_number(summary["avg_home_runs_allowed"]),
    )


st.set_page_config(
    page_title="MLB Analytics Platform",
    page_icon="⚾",
    layout="wide",
)

svc = AnalyticsService(settings)

st.title("⚾ MLB Analytics Platform")
st.caption(
    "Daily MLB slate, matchup analysis, model outputs, and data-quality tools."
)

slate_date = st.date_input("Slate date", date.today())

sync_col, train_col, count_col = st.columns([1.2, 1.2, 0.8])

with sync_col:
    if st.button(
        "Sync slate",
        type="primary",
        use_container_width=True,
    ):
        try:
            svc.sync_schedule(slate_date)
            st.success("Slate updated")
            st.rerun()
        except Exception as exc:
            st.error(f"Slate sync failed: {exc}")

with train_col:
    if st.button(
        "Train all models",
        use_container_width=True,
    ):
        try:
            st.json(svc.train_all())
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

coverage = svc.repo.coverage()
database_games = 0
if not coverage.empty and "games" in coverage.columns:
    database_games = int(coverage["games"].iloc[0])

with count_col:
    st.metric("Database games", database_games)

games = svc.game_predictions(slate_date)

if games.empty:
    st.info("No games stored for this date. Sync the slate first.")
    st.stop()

games = games.reset_index(drop=True).copy()

team_stats = load_table(svc, "team_game_stats")
pitcher_stats = load_table(svc, "pitcher_game_stats")
batting_stats = load_table(svc, "player_game_batting")

feature_error = None
features = pd.DataFrame()
try:
    features = build_game_features(games, team_stats, pitcher_stats)
    features = features.reset_index(drop=True)
except Exception as exc:
    feature_error = str(exc)

display_columns = [
    "game_time",
    "away_team",
    "home_team",
    "away_probable_pitcher",
    "home_probable_pitcher",
    "venue",
    "status",
]

if "home_win_probability" in games.columns:
    games["pick"] = games.apply(
        lambda row: (
            row["home_team"]
            if row["home_win_probability"] >= 0.5
            else row["away_team"]
        ),
        axis=1,
    )
    games["confidence"] = games[
        ["home_win_probability", "away_win_probability"]
    ].max(axis=1)
    display_columns += ["pick", "confidence"]

available_display_columns = [
    column for column in display_columns if column in games.columns
]

table = games[available_display_columns].copy()
if "confidence" in table.columns:
    table["confidence"] = table["confidence"].map(format_percent)

st.subheader("Today's slate")
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)

games["matchup_label"] = games.apply(
    lambda row: (
        f"{row['away_team']} at {row['home_team']} — "
        f"{row.get('game_time', '')}"
    ),
    axis=1,
)

selected_index = st.selectbox(
    "Select a matchup",
    options=games.index.tolist(),
    format_func=lambda index: games.loc[index, "matchup_label"],
)

selected = games.loc[selected_index]
selected_features = (
    features.loc[selected_index]
    if not features.empty and selected_index in features.index
    else pd.Series(dtype="object")
)

away_batters = batter_summary(
    batting_stats,
    int(selected["away_team_id"]),
    slate_date,
)
home_batters = batter_summary(
    batting_stats,
    int(selected["home_team_id"]),
    slate_date,
)

away_pitcher = pitcher_summary(
    pitcher_stats,
    selected.get("away_probable_pitcher_id"),
    slate_date,
)
home_pitcher = pitcher_summary(
    pitcher_stats,
    selected.get("home_probable_pitcher_id"),
    slate_date,
)

st.divider()
st.header(f"{selected['away_team']} at {selected['home_team']}")

overview_tab, moneyline_tab, hits_tab, hr_tab, pitchers_tab = st.tabs(
    [
        "Overview",
        "Moneyline",
        "Hits",
        "Home Runs",
        "Pitchers",
    ]
)

with overview_tab:
    left, middle, right = st.columns(3)

    with left:
        st.subheader("Away")
        st.write(f"**Team:** {selected.get('away_team', 'Unknown')}")
        st.write(
            "**Probable pitcher:** "
            f"{selected.get('away_probable_pitcher', 'TBD')}"
        )

    with middle:
        st.subheader("Game")
        st.write(f"**Time:** {selected.get('game_time', 'Unknown')}")
        st.write(f"**Venue:** {selected.get('venue', 'Unknown')}")
        st.write(f"**Status:** {selected.get('status', 'Unknown')}")

    with right:
        st.subheader("Home")
        st.write(f"**Team:** {selected.get('home_team', 'Unknown')}")
        st.write(
            "**Probable pitcher:** "
            f"{selected.get('home_probable_pitcher', 'TBD')}"
        )

    if not selected_features.empty:
        weather_cols = st.columns(4)
        weather_cols[0].metric(
            "Temperature",
            f"{format_number(selected_features.get('temperature'), 0)}°F",
        )
        weather_cols[1].metric(
            "Humidity",
            f"{format_number(selected_features.get('humidity'), 0)}%",
        )
        weather_cols[2].metric(
            "Wind",
            f"{format_number(selected_features.get('wind_speed'), 0)} mph",
        )
        weather_cols[3].metric(
            "Park factor",
            format_number(selected_features.get("park_factor"), 3),
        )

with moneyline_tab:
    if "home_win_probability" not in games.columns:
        st.warning(
            "Train the home-win model after completing a historical backfill."
        )
    else:
        away_probability = float(selected["away_win_probability"])
        home_probability = float(selected["home_win_probability"])

        away_col, home_col = st.columns(2)
        away_col.metric(
            selected["away_team"],
            format_percent(away_probability),
            help=f"Fair moneyline: {probability_to_american(away_probability)}",
        )
        home_col.metric(
            selected["home_team"],
            format_percent(home_probability),
            help=f"Fair moneyline: {probability_to_american(home_probability)}",
        )

        odds_col1, odds_col2 = st.columns(2)
        odds_col1.metric(
            f"{selected['away_team']} fair odds",
            probability_to_american(away_probability),
        )
        odds_col2.metric(
            f"{selected['home_team']} fair odds",
            probability_to_american(home_probability),
        )

        probability_chart = pd.DataFrame(
            {
                "Team": [
                    selected["away_team"],
                    selected["home_team"],
                ],
                "Win probability": [
                    away_probability,
                    home_probability,
                ],
            }
        ).set_index("Team")
        st.bar_chart(probability_chart)

        pick = (
            selected["home_team"]
            if home_probability >= away_probability
            else selected["away_team"]
        )
        confidence = max(home_probability, away_probability)
        st.success(f"Model pick: {pick} ({confidence:.1%})")

        st.subheader("Model inputs")

        if feature_error:
            st.error(f"Feature builder error: {feature_error}")
        elif selected_features.empty:
            st.warning("No feature row was produced for this matchup.")
        else:
            away_col, home_col = st.columns(2)

            with away_col:
                st.markdown(f"#### {selected['away_team']}")
                st.metric(
                    "30-game win rate",
                    format_percent(selected_features.get("away_win_pct_30")),
                )
                st.metric(
                    "14-game runs/game",
                    format_number(selected_features.get("away_runs_pg_14")),
                )
                st.metric(
                    "14-game OPS",
                    format_number(selected_features.get("away_ops_14"), 3),
                )
                st.metric(
                    "Starter ERA",
                    format_number(selected_features.get("away_sp_era_5")),
                )
                st.metric(
                    "Starter K/BB",
                    format_number(selected_features.get("away_sp_kbb_5")),
                )
                st.metric(
                    "Bullpen pitches, last 3",
                    format_number(
                        selected_features.get("away_bullpen_pitches_3"),
                        0,
                    ),
                )
                st.metric(
                    "Rest days",
                    format_number(selected_features.get("away_rest"), 0),
                )

            with home_col:
                st.markdown(f"#### {selected['home_team']}")
                st.metric(
                    "30-game win rate",
                    format_percent(selected_features.get("home_win_pct_30")),
                )
                st.metric(
                    "14-game runs/game",
                    format_number(selected_features.get("home_runs_pg_14")),
                )
                st.metric(
                    "14-game OPS",
                    format_number(selected_features.get("home_ops_14"), 3),
                )
                st.metric(
                    "Starter ERA",
                    format_number(selected_features.get("home_sp_era_5")),
                )
                st.metric(
                    "Starter K/BB",
                    format_number(selected_features.get("home_sp_kbb_5")),
                )
                st.metric(
                    "Bullpen pitches, last 3",
                    format_number(
                        selected_features.get("home_bullpen_pitches_3"),
                        0,
                    ),
                )
                st.metric(
                    "Rest days",
                    format_number(selected_features.get("home_rest"), 0),
                )

            with st.expander("View complete feature row"):
                st.dataframe(
                    selected_features.to_frame("Value"),
                    use_container_width=True,
                )

        st.caption(
            "Probabilities are meaningful only after a historical backfill, "
            "chronological validation, and calibration review."
        )

with hits_tab:
    st.subheader("Batter hit indicators")
    st.warning(
        "These are historical rolling indicators, not final 1+ hit "
        "probabilities. Confirmed lineups and a trained live prediction "
        "pipeline still need to be connected."
    )

    away_hit_tab, home_hit_tab = st.tabs(
        [selected["away_team"], selected["home_team"]]
    )

    with away_hit_tab:
        display_batter_table(away_batters, "hits")

    with home_hit_tab:
        display_batter_table(home_batters, "hits")

with hr_tab:
    st.subheader("Home-run indicators")
    st.warning(
        "These are rolling power indicators, not sportsbook-ready home-run "
        "probabilities. Statcast quality, pitch-type matchups, confirmed "
        "lineups, weather, and calibrated HR modeling will be added later."
    )

    away_hr_tab, home_hr_tab = st.tabs(
        [selected["away_team"], selected["home_team"]]
    )

    with away_hr_tab:
        display_batter_table(away_batters, "hr")

    with home_hr_tab:
        display_batter_table(home_batters, "hr")

with pitchers_tab:
    st.subheader("Probable starter indicators")
    st.caption(
        "Metrics use up to the five most recent starts before the selected "
        "game date."
    )

    away_pitcher_tab, home_pitcher_tab = st.tabs(
        [selected["away_team"], selected["home_team"]]
    )

    with away_pitcher_tab:
        display_pitcher_metrics(
            selected.get("away_probable_pitcher", "TBD"),
            away_pitcher,
        )

    with home_pitcher_tab:
        display_pitcher_metrics(
            selected.get("home_probable_pitcher", "TBD"),
            home_pitcher,
        )

st.warning(
    "Models require a historical backfill before their probabilities are "
    "meaningful. No slip optimizer is included."
)
