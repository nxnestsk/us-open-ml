"""Testes do Elo: garantem que não há fuga de dados e que a matemática está certa."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elo import INITIAL, compute_elo, expected, k_factor  # noqa: E402


def _matches(rows):
    return pd.DataFrame(
        rows, columns=["winner_id", "loser_id", "surface", "tourney_level", "is_walkover"]
    )


def test_expected_is_symmetric_and_half_when_equal():
    assert expected(1500, 1500) == 0.5
    assert abs(expected(1700, 1500) + expected(1500, 1700) - 1.0) < 1e-12


def test_first_match_uses_initial_rating():
    # pré-jogo: ninguém tem histórico, logo ambos valem 1500
    df = _matches([(1, 2, "Hard", "A", False)])
    out = compute_elo(df)
    assert out.loc[0, "w_elo"] == INITIAL and out.loc[0, "l_elo"] == INITIAL


def test_result_only_affects_later_matches():
    # o rating pré-jogo do jogo 0 não pode depender do resultado do jogo 0
    a = compute_elo(_matches([(1, 2, "Hard", "A", False), (1, 3, "Hard", "A", False)]))
    b = compute_elo(_matches([(2, 1, "Hard", "A", False), (1, 3, "Hard", "A", False)]))
    assert a.loc[0, "w_elo"] == b.loc[0, "w_elo"] == INITIAL
    assert a.loc[1, "w_elo"] > INITIAL > b.loc[1, "w_elo"]  # jogador 1 ganhou / perdeu antes


def test_walkover_does_not_update():
    out = compute_elo(_matches([(1, 2, "Hard", "A", True), (1, 2, "Hard", "A", False)]))
    assert out.loc[1, "w_elo"] == INITIAL


def test_k_decreases_with_experience():
    assert k_factor(0) > k_factor(50) > k_factor(500)
