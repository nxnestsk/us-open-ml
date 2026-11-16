"""Simulação Monte Carlo de um quadro de eliminação direta.

Ideia: o modelo dá P(A vence B) para qualquer par. Sorteamos cada jogo com essa
probabilidade, repetimos o torneio milhares de vezes e contamos quantas vezes
cada jogador chega a cada ronda e ganha o título.
"""
import itertools

import numpy as np
import pandas as pd

from .features import FEATURES, flip_features


# ---------------------------------------------------------------- quadro
def bracket_order(t: pd.DataFrame) -> list[int]:
    """Reconstrói a ordem do quadro (128 posições) a partir dos jogos efetivamente disputados.

    Cada jogo da ronda r tem como "filhos" os dois jogos da ronda r-1 ganhos pelos seus
    dois jogadores. A travessia em ordem dá uma lista onde posições vizinhas se defrontam.
    (Backtest: usa o sorteio real, inferido dos resultados. Não usa quem ganhou
    para calcular probabilidades, só para saber quem joga com quem.)
    """
    t = t.dropna(subset=["round_idx"])
    won = {(int(r.round_idx), int(r.winner_id)): r for r in t.itertuples()}

    def expand(m) -> list[int]:
        r = int(m.round_idx)
        if r == 1:
            return [int(m.winner_id), int(m.loser_id)]
        out: list[int] = []
        for p in (int(m.winner_id), int(m.loser_id)):
            out += expand(won[(r - 1, p)])
        return out

    final = t[t.round_idx == t.round_idx.max()].iloc[0]
    final = won[(int(final.round_idx), int(final.winner_id))]
    return expand(final)


def actual_reach(t: pd.DataFrame, players: list[int]) -> dict[int, int]:
    """Número de rondas ganhas por jogador (0 = perdeu na 1.ª; 7 = campeão)."""
    t = t.dropna(subset=["round_idx"])
    wins = t.groupby("winner_id").size()
    return {p: int(wins.get(p, 0)) for p in players}


# ------------------------------------------------- linhas hipotéticas
def make_hypo_rows(entrants: pd.DataFrame, start: pd.Timestamp, group: str, best_of: int) -> pd.DataFrame:
    """Todos os pares possíveis de um quadro, como jogos 'a acontecer' na véspera do torneio.

    Entram na mesma máquina de features que os jogos reais, mas com is_walkover=True para
    NÃO atualizarem ratings nem histórico (são perguntas, não resultados).
    `entrants` tem colunas: id, name, rank, pts.
    """
    rows = []
    for a, b in itertools.combinations(entrants.itertuples(), 2):
        rows.append({
            "winner_id": a.id, "loser_id": b.id, "winner_name": a.name, "loser_name": b.name,
            "winner_rank": a.rank, "loser_rank": b.rank,
            "winner_rank_points": a.pts, "loser_rank_points": b.pts,
        })
    h = pd.DataFrame(rows)
    h["surface"] = "Hard"
    h["tourney_level"] = "G"
    h["best_of"] = best_of
    h["date"] = start - pd.Timedelta(days=1)  # processado depois de tudo o que é anterior ao torneio
    h["tourney_start"] = start
    h["tourney_id"] = f"HYPO-{group}"
    h["tourney_name"] = f"HYPO {group}"
    h["match_num"] = 0
    h["round_idx"] = np.nan
    h["is_walkover"] = True
    h["is_retirement"] = False
    h["source"] = "hypo"
    h["is_hypo"] = True
    h["group"] = group
    h["year"] = start.year
    return h


# --------------------------------------------------------- probabilidades
def ensemble_p1(models: dict, X: pd.DataFrame) -> np.ndarray:
    """P(jogador 1 vence), média logística + LightGBM, simetrizada."""
    feats = models["features"]

    def raw(Z: pd.DataFrame) -> np.ndarray:
        return 0.5 * models["lr"].predict_proba(Z[feats])[:, 1] + 0.5 * models["gbm"].predict_proba(Z[feats])[:, 1]

    return 0.5 * (raw(X) + 1.0 - raw(flip_features(X)))


def prob_matrix(rows: pd.DataFrame, ids: list[int], col: str) -> np.ndarray:
    """Matriz P[i, j] = P(ids[i] vence ids[j]) a partir das linhas de pares (p1 vs p2)."""
    idx = {p: k for k, p in enumerate(ids)}
    P = np.full((len(ids), len(ids)), 0.5)
    i = rows["p1_id"].map(idx).to_numpy()
    j = rows["p2_id"].map(idx).to_numpy()
    p = rows[col].to_numpy()
    P[i, j] = p
    P[j, i] = 1 - p
    return P


# ---------------------------------------------------------- Monte Carlo
def simulate(P: np.ndarray, n: int = 20000, seed: int = 11, random_draw: bool = False) -> np.ndarray:
    """Devolve M[jogador, k] = P(ganhar pelo menos k rondas), k = 0..log2(m).

    Jogadores já estão por ordem do quadro (vizinhos jogam entre si). Com
    `random_draw=True` o quadro é sorteado de novo em cada simulação.
    """
    rng = np.random.default_rng(seed)
    m = P.shape[0]
    rounds = int(np.log2(m))
    if random_draw:
        cur = np.argsort(rng.random((n, m)), axis=1)
    else:
        cur = np.tile(np.arange(m), (n, 1))
    M = np.zeros((m, rounds + 1))
    M[:, 0] = 1.0
    for r in range(rounds):
        a, b = cur[:, 0::2], cur[:, 1::2]
        win = rng.random(a.shape) < P[a, b]
        cur = np.where(win, a, b)
        M[:, r + 1] = np.bincount(cur.ravel(), minlength=m) / n
    return M
