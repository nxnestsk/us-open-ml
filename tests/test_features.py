"""Testes das features: o futuro nunca pode influenciar o passado."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elo import compute_elo  # noqa: E402
from src.features import FEATURES, build_features, flip_features  # noqa: E402
from src.simulate import make_hypo_rows, simulate  # noqa: E402


def _fake_matches(n=600, players=14, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 900, n)), unit="D")
    a = rng.integers(1, players + 1, n)
    b = (a + rng.integers(1, players, n) - 1) % players + 1
    a_wins = rng.random(n) < 0.55
    df = pd.DataFrame({
        "winner_id": np.where(a_wins, a, b), "loser_id": np.where(a_wins, b, a),
        "winner_name": "w", "loser_name": "l",
        "date": dates, "tourney_start": dates, "year": dates.year,
        "tourney_id": [f"T{d.toordinal() // 10}" for d in dates], "tourney_name": "T",
        "match_num": np.arange(n), "surface": rng.choice(["Hard", "Clay"], n),
        "tourney_level": "A", "best_of": 3, "source": "sackmann",
        "is_walkover": False, "is_retirement": rng.random(n) < 0.06,
        "winner_rank": rng.integers(1, 200, n).astype(float), "loser_rank": rng.integers(1, 200, n).astype(float),
        "winner_rank_points": 500.0, "loser_rank_points": 400.0,
        "winner_hand": "R", "loser_hand": "L", "winner_ht": 185.0, "loser_ht": 190.0,
        "winner_age": 25.0, "loser_age": 27.0,
    })
    return df


def test_future_matches_do_not_change_past_features():
    """Calcular só com os primeiros k jogos tem de dar as mesmas features que com todos."""
    df = _fake_matches()
    k = 350
    full = build_features(df, compute_elo(df))
    part = build_features(df.iloc[:k].copy(), compute_elo(df.iloc[:k].copy()))
    assert_frame_equal(full.iloc[:k][FEATURES], part[FEATURES])


def test_hypothetical_rows_do_not_change_real_features():
    """Perguntar 'e se A jogasse com B?' não pode alterar ratings nem histórico."""
    df = _fake_matches()
    ent = pd.DataFrame({"id": [1, 2, 3, 4], "name": list("abcd"), "rank": [1.0, 2, 3, 4], "pts": [9.0, 8, 7, 6]})
    hypo = make_hypo_rows(ent, pd.Timestamp("2021-06-01"), "t", 3)
    both = pd.concat([df, hypo], ignore_index=True)
    both = both.sort_values(["date", "tourney_id", "match_num"], kind="stable").reset_index(drop=True)
    both["is_walkover"] = both["is_walkover"].astype(bool)
    Xb = build_features(both, compute_elo(both))
    Xr = build_features(df, compute_elo(df))
    real = Xb[~Xb.is_hypo].reset_index(drop=True)
    assert_frame_equal(real[FEATURES], Xr[FEATURES])
    assert Xb.is_hypo.sum() == 6  # 4 jogadores -> 6 pares


def test_flip_is_involution():
    df = _fake_matches(200)
    X = build_features(df, compute_elo(df))[FEATURES]
    assert_frame_equal(flip_features(flip_features(X)), X)


def test_simulation_probabilities_are_consistent():
    P = np.full((8, 8), 0.5)
    M = simulate(P, n=40000, seed=1)
    assert abs(M[:, 3].sum() - 1.0) < 1e-9          # há exatamente um campeão por simulação
    assert np.allclose(M[:, 3], 1 / 8, atol=0.01)   # jogadores iguais: 1/8 cada
    # um jogador dominante ganha quase sempre
    P2 = P.copy()
    P2[0, :], P2[:, 0] = 0.99, 0.01
    P2[0, 0] = 0.5
    assert simulate(P2, n=20000, seed=1)[0, 3] > 0.9
