"""Ratings Elo (geral e por superfície) calculados sem fuga de dados.

Princípio central: para cada jogo, primeiro LEMOS os ratings (o que se sabia
antes de o jogo acontecer) e só depois os ATUALIZAMOS com o resultado.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

INITIAL = 1500.0

# Parâmetros do Elo. Os valores por defeito são os da FiveThirtyEight (v1);
# `scripts/04_tune_elo.py` procura valores melhores sem olhar para 2023 em diante.
DEFAULT_PARAMS = dict(offset=5.0, shape=0.4, scale=250.0, slam_bonus=1.1, surf_k=1.0, blend_w=0.5)


def expected(r_a: float, r_b: float) -> float:
    """Probabilidade de A ganhar a B, dada a diferença de rating."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def k_factor(n_matches: int, offset: float = 5.0, shape: float = 0.4, scale: float = 250.0) -> float:
    """K decresce com a experiência: jogadores novos mexem mais no rating."""
    return scale / (n_matches + offset) ** shape


def compute_elo(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Devolve, por jogo, os ratings PRÉ-JOGO de vencedor e perdedor.

    `df` tem de estar ordenado cronologicamente (ver src.data.load_matches).
    """
    P = {**DEFAULT_PARAMS, **(params or {})}
    overall = defaultdict(lambda: INITIAL)
    surface = defaultdict(lambda: INITIAL)
    n_all = defaultdict(int)
    n_surf = defaultdict(int)

    cols = ["w_elo", "l_elo", "w_selo", "l_selo", "w_n", "l_n"]
    out = np.zeros((len(df), len(cols)))

    w_ids = df["winner_id"].to_numpy()
    l_ids = df["loser_id"].to_numpy()
    surfs = df["surface"].fillna("Unknown").to_numpy()
    levels = df["tourney_level"].fillna("").to_numpy()
    skip = df["is_walkover"].to_numpy()

    for i in range(len(df)):
        w, l, s = w_ids[i], l_ids[i], surfs[i]
        # 1) LER (pré-jogo)
        out[i] = (overall[w], overall[l], surface[w, s], surface[l, s], n_all[w], n_all[l])
        if skip[i]:  # walkover: não houve jogo, não atualiza
            continue
        # 2) ATUALIZAR (pós-jogo)
        bonus = P["slam_bonus"] if levels[i] == "G" else 1.0
        for store, key_w, key_l, cnt, mult in (
            (overall, w, l, n_all, 1.0),
            (surface, (w, s), (l, s), n_surf, P["surf_k"]),
        ):
            p_w = expected(store[key_w], store[key_l])
            kw = k_factor(cnt[key_w], P["offset"], P["shape"], P["scale"]) * bonus * mult
            kl = k_factor(cnt[key_l], P["offset"], P["shape"], P["scale"]) * bonus * mult
            store[key_w] += kw * (1.0 - p_w)
            store[key_l] -= kl * (1.0 - p_w)
            cnt[key_w] += 1
            cnt[key_l] += 1
    return pd.DataFrame(out, columns=cols, index=df.index)


def win_prob(elo_w: np.ndarray, elo_l: np.ndarray) -> np.ndarray:
    """Probabilidade que o Elo atribuía ao jogador que acabou por ganhar."""
    return 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
