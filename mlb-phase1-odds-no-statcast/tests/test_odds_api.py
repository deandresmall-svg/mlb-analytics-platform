import pandas as pd

from mlb_analytics.data.odds_api import (
    OddsAPIClient,
    american_implied_probability,
    no_vig_probability,
    paired_no_vig_probability,
)


def sample_payload():
    return {
        "id": "evt1",
        "home_team": "Miami Marlins",
        "away_team": "New York Mets",
        "commence_time": "2026-07-18T23:10:00Z",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "batter_hits",
                        "last_update": "2026-07-18T20:00:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Juan Soto", "price": -170, "point": 0.5},
                            {"name": "Under", "description": "Juan Soto", "price": 135, "point": 0.5},
                        ],
                    }
                ],
            }
        ],
    }


def test_flatten_props_and_no_vig():
    frame = OddsAPIClient.flatten_props(sample_payload())
    assert len(frame) == 2
    over = frame[frame["side"] == "Over"].iloc[0]
    assert over["player_normalized"] == "juan soto"
    fair = paired_no_vig_probability(frame, over)
    expected = no_vig_probability(-170, 135)
    assert fair is not None
    assert abs(fair - expected) < 1e-12


def test_american_implied_probability():
    assert round(american_implied_probability(-200), 6) == round(2 / 3, 6)
    assert round(american_implied_probability(200), 6) == round(1 / 3, 6)


def test_match_event_uses_team_names():
    game = pd.Series({"home_team": "Miami Marlins", "away_team": "New York Mets"})
    events = [
        {"id": "wrong", "home_team": "Boston Red Sox", "away_team": "New York Yankees"},
        {"id": "right", "home_team": "Miami Marlins", "away_team": "New York Mets"},
    ]
    assert OddsAPIClient.match_event(game, events)["id"] == "right"
