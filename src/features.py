"""Features pré-jogo, sem fuga de dados.

Regra: para o jogo i, só usamos informação de jogos ANTERIORES (estado dos
jogadores lido antes de o atualizar com o resultado de i).
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

DIFF_COLS = [
    "elo_diff", "selo_diff", "blend_diff", "exp_diff", "rank_log_diff", "pts_log_diff",
    "age_diff", "ht_diff", "form10_diff", "m14_diff", "m30_diff", "rest_diff", "cur_diff",
    "spw_diff", "rpw_diff", "h2h_diff", "blend_x_bo5_diff", "blend_x_slam_diff",
    # proxies de lesão (ver _injury_snapshot): não há registo de lesões em dados abertos
    "inj_rate_diff", "inj365_diff", "layoff_diff", "lastinj_diff",
]
INJURY_COLS = ["inj_rate_diff", "inj365_diff", "layoff_diff", "lastinj_diff"]
CONTEXT_COLS = ["is_slam", "bo5", "hard", "clay", "grass"]
SWAP_COLS = [("lefty_1", "lefty_2")]
FEATURES = DIFF_COLS + CONTEXT_COLS + ["lefty_1", "lefty_2"]
FEATURES_NO_INJURY = [c for c in FEATURES if c not in INJURY_COLS]


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _snapshot(hist, today: int, last_day):
    """Forma e fadiga de um jogador, a partir do histórico ANTERIOR ao jogo."""
    h = list(hist)
    last10 = h[-10:]
    form10 = (sum(x[1] for x in last10) + 2.5) / (len(last10) + 5)  # encolhida para 0,5
    prior = [x for x in h if x[0] < today]
    m14 = sum(1 for x in prior if today - x[0] <= 14)
    m30 = sum(1 for x in prior if today - x[0] <= 30)
    cur = sum(1 for x in h if x[0] == today)  # jogos já feitos neste torneio
    rest = min(today - prior[-1][0], 60) if prior else 60
    sp = [x[2] for x in h[-20:] if x[2] == x[2]]
    rp = [x[3] for x in h[-20:] if x[3] == x[3]]
    return (form10, m14, m30, rest, cur,
            float(np.mean(sp)) if sp else np.nan,
            float(np.mean(rp)) if rp else np.nan)


INJ_PRIOR, INJ_PRIOR_N = 0.03, 20  # taxa típica de desistência por jogo, com peso de 20 jogos


def _injury_snapshot(st: dict, today: int):
    """Proxies de lesão de um jogador, a partir de eventos ANTERIORES ao jogo.

    Evento = desistência a meio do jogo (RET) ou antes de entrar em campo (W/O) do próprio jogador.
    - inj_rate: taxa de eventos na carreira, encolhida para a média (1 evento em 5 jogos não é 20%)
    - inj365: eventos nos últimos 365 dias
    - layoff: dias desde o último jogo ou evento (até 365): regressos de paragens longas
    - lastinj: 1 se o último evento registado foi uma desistência
    """
    ev = st["events"]
    rate = (st["n_events"] + INJ_PRIOR * INJ_PRIOR_N) / (st["n"] + INJ_PRIOR_N)
    last365 = sum(1 for x in ev if 0 < today - x <= 365)
    layoff = min(today - st["last_day"], 365) if st["last_day"] is not None else 365
    return rate, last365, layoff, 1.0 if st["last_was_event"] else 0.0


def build_features(df: pd.DataFrame, elo: pd.DataFrame, seed: int = 7, blend_w: float = 0.5) -> pd.DataFrame:
    """Uma linha por jogo jogado, orientada aleatoriamente (jogador 1 vs 2).

    A orientação aleatória impede que o modelo aprenda "o vencedor é sempre a
    coluna A". `y = 1` se o jogador 1 ganhou.
    """
    d = df.join(elo).reset_index(drop=True)
    n = len(d)
    day = d["tourney_start"].map(pd.Timestamp.toordinal).to_numpy()
    wid, lid = d["winner_id"].to_numpy(), d["loser_id"].to_numpy()
    skip = d["is_walkover"].to_numpy()

    w_sv, w_1, w_2 = _num(d, "w_svpt"), _num(d, "w_1stWon"), _num(d, "w_2ndWon")
    l_sv, l_1, l_2 = _num(d, "l_svpt"), _num(d, "l_1stWon"), _num(d, "l_2ndWon")
    with np.errstate(divide="ignore", invalid="ignore"):
        w_spw = np.where(w_sv > 0, (w_1 + w_2) / w_sv, np.nan)
        l_spw = np.where(l_sv > 0, (l_1 + l_2) / l_sv, np.nan)
    w_rpw, l_rpw = 1 - l_spw, 1 - w_spw  # pontos de devolução ganhos = pontos de serviço do rival perdidos

    # atributos do jogador: usa o valor do jogo se existir, senão o último conhecido
    cols = {
        "ht": (_num(d, "winner_ht"), _num(d, "loser_ht")),
        "age": (_num(d, "winner_age"), _num(d, "loser_age")),
    }
    hand_w = d["winner_hand"].to_numpy() if "winner_hand" in d else np.full(n, None)
    hand_l = d["loser_hand"].to_numpy() if "loser_hand" in d else np.full(n, None)

    hist = defaultdict(lambda: deque(maxlen=60))
    last_attr: dict = {}  # id -> [ht, age, age_day, hand]
    h2h = defaultdict(lambda: [0, 0])  # (id_menor, id_maior) -> vitórias de cada um

    is_hypo = d["is_hypo"].fillna(False).astype(bool).to_numpy() if "is_hypo" in d else np.zeros(n, bool)
    is_ret = d["is_retirement"].fillna(False).astype(bool).to_numpy() if "is_retirement" in d else np.zeros(n, bool)
    inj = defaultdict(lambda: {"events": deque(maxlen=40), "n_events": 0, "n": 0, "last_day": None, "last_was_event": False})
    IW = np.zeros((n, 4))
    IL = np.zeros((n, 4))

    W = np.full((n, 7), np.nan)  # snapshot do vencedor
    L = np.full((n, 7), np.nan)
    attrs_w = np.full((n, 3), np.nan)  # ht, age, lefty
    attrs_l = np.full((n, 3), np.nan)
    hw = np.zeros(n)
    hl = np.zeros(n)

    def attrs(pid, i, ht, age, hand):
        st = last_attr.get(pid, [np.nan, np.nan, day[i], None])
        if ht == ht:
            st[0] = ht
        if age == age:
            st[1], st[2] = age, day[i]
        if isinstance(hand, str) and hand in ("L", "R"):
            st[3] = hand
        last_attr[pid] = st
        age_now = st[1] + (day[i] - st[2]) / 365.25 if st[1] == st[1] else np.nan
        return st[0], age_now, (1.0 if st[3] == "L" else 0.0 if st[3] == "R" else np.nan)

    for i in range(n):
        w, l = wid[i], lid[i]
        W[i] = _snapshot(hist[w], day[i], None)
        L[i] = _snapshot(hist[l], day[i], None)
        attrs_w[i] = attrs(w, i, cols["ht"][0][i], cols["age"][0][i], hand_w[i])
        attrs_l[i] = attrs(l, i, cols["ht"][1][i], cols["age"][1][i], hand_l[i])
        IW[i] = _injury_snapshot(inj[w], day[i])
        IL[i] = _injury_snapshot(inj[l], day[i])
        key = (min(w, l), max(w, l))
        rec = h2h[key]
        hw[i], hl[i] = (rec[0], rec[1]) if w < l else (rec[1], rec[0])
        if skip[i]:
            if not is_hypo[i]:  # walkover real: o perdedor desistiu antes de jogar
                st = inj[l]
                st["events"].append(day[i]); st["n_events"] += 1; st["n"] += 1
                st["last_day"], st["last_was_event"] = day[i], True
            continue
        # eventos de lesão: só DEPOIS de ler tudo
        sw, sl = inj[w], inj[l]
        sw["n"] += 1; sw["last_day"], sw["last_was_event"] = day[i], False
        sl["n"] += 1; sl["last_day"] = day[i]
        if is_ret[i]:
            sl["events"].append(day[i]); sl["n_events"] += 1; sl["last_was_event"] = True
        else:
            sl["last_was_event"] = False
        # só DEPOIS de ler tudo é que se atualiza o estado
        hist[w].append((day[i], 1.0, w_spw[i], w_rpw[i]))
        hist[l].append((day[i], 0.0, l_spw[i], l_rpw[i]))
        rec[0 if w < l else 1] += 1

    # orientação aleatória
    # A orientação depende só da identidade do jogo (hash), não da posição na tabela: assim,
    # acrescentar ou remover linhas não muda a orientação dos restantes jogos.
    key = pd.DataFrame({"w": wid, "l": lid, "d": day, "m": d["match_num"].to_numpy(), "s": seed})
    flip = (pd.util.hash_pandas_object(key, index=False).to_numpy() >> np.uint64(7)) & np.uint64(1) == 1  # True: jogador 1 = perdedor
    pick = lambda a, b: np.where(flip[:, None] if a.ndim > 1 else flip, b, a)  # noqa: E731
    S1, S2 = pick(W, L), pick(L, W)
    A1, A2 = pick(attrs_w, attrs_l), pick(attrs_l, attrs_w)
    J1, J2 = pick(IW, IL), pick(IL, IW)

    def col(a, b):
        return np.where(flip, b, a)

    elo1, elo2 = col(d.w_elo.to_numpy(), d.l_elo.to_numpy()), col(d.l_elo.to_numpy(), d.w_elo.to_numpy())
    se1, se2 = col(d.w_selo.to_numpy(), d.l_selo.to_numpy()), col(d.l_selo.to_numpy(), d.w_selo.to_numpy())
    n1, n2 = col(d.w_n.to_numpy(), d.l_n.to_numpy()), col(d.l_n.to_numpy(), d.w_n.to_numpy())
    rk1 = col(_num(d, "winner_rank"), _num(d, "loser_rank"))
    rk2 = col(_num(d, "loser_rank"), _num(d, "winner_rank"))
    pt1 = col(_num(d, "winner_rank_points"), _num(d, "loser_rank_points"))
    pt2 = col(_num(d, "loser_rank_points"), _num(d, "winner_rank_points"))
    h1, h2 = col(hw, hl), col(hl, hw)
    n_h = h1 + h2

    best_of = pd.to_numeric(d.get("best_of", pd.Series(3, index=d.index)), errors="coerce").fillna(3).to_numpy()
    is_slam = (d["tourney_level"] == "G").to_numpy().astype(float)
    bo5 = (best_of >= 5).astype(float)
    # mistura do Elo geral com o de superfície; blend_w=0,5 era a escolha da v1
    blend1, blend2 = blend_w * elo1 + (1 - blend_w) * se1, blend_w * elo2 + (1 - blend_w) * se2
    blend = blend1 - blend2

    surf = d["surface"].fillna("")
    out = pd.DataFrame({
        "elo_diff": elo1 - elo2,
        "selo_diff": se1 - se2,
        "blend_diff": blend,
        "exp_diff": np.log1p(n1) - np.log1p(n2),
        # ranking não registado -> trata como 1500.º (pior que quase todos)
        "rank_log_diff": np.log(np.nan_to_num(rk1, nan=1500.0)) - np.log(np.nan_to_num(rk2, nan=1500.0)),
        "pts_log_diff": np.log1p(np.nan_to_num(pt1, nan=0.0)) - np.log1p(np.nan_to_num(pt2, nan=0.0)),
        "age_diff": A1[:, 1] - A2[:, 1],
        "ht_diff": A1[:, 0] - A2[:, 0],
        "form10_diff": S1[:, 0] - S2[:, 0],
        "m14_diff": S1[:, 1] - S2[:, 1],
        "m30_diff": S1[:, 2] - S2[:, 2],
        "rest_diff": S1[:, 3] - S2[:, 3],
        "cur_diff": S1[:, 4] - S2[:, 4],
        "spw_diff": S1[:, 5] - S2[:, 5],
        "rpw_diff": S1[:, 6] - S2[:, 6],
        "h2h_diff": (h1 + 1) / (n_h + 2) - (h2 + 1) / (n_h + 2),  # suavizado: sem jogos anteriores = 0
        "inj_rate_diff": J1[:, 0] - J2[:, 0],
        "inj365_diff": J1[:, 1] - J2[:, 1],
        "layoff_diff": J1[:, 2] - J2[:, 2],
        "lastinj_diff": J1[:, 3] - J2[:, 3],
        "blend_x_bo5_diff": blend * bo5,
        "blend_x_slam_diff": blend * is_slam,
        "is_slam": is_slam,
        "bo5": bo5,
        "hard": (surf == "Hard").to_numpy().astype(float),
        "clay": (surf == "Clay").to_numpy().astype(float),
        "grass": (surf == "Grass").to_numpy().astype(float),
        "lefty_1": A1[:, 2],
        "lefty_2": A2[:, 2],
    })
    out["y"] = (~flip).astype(int)  # 1 se o jogador 1 ganhou
    out["p1_is_winner"] = ~flip
    out["year"] = d["year"].to_numpy()
    out["date"] = d["date"].to_numpy()
    out["tourney_name"] = d["tourney_name"].to_numpy()
    out["source"] = d["source"].to_numpy()
    # linhas hipotéticas (confrontos que ainda não aconteceram, usados na simulação)
    out["is_hypo"] = d["is_hypo"].fillna(False).astype(bool).to_numpy() if "is_hypo" in d else False
    out["group"] = d["group"].to_numpy() if "group" in d else ""
    out["p1_id"] = np.where(flip, lid, wid)
    out["p2_id"] = np.where(flip, wid, lid)
    for c in ["AvgW", "AvgL"]:
        out[c] = _num(d, c)
    out["elo_p1_prob"] = 1 / (1 + 10 ** (-blend / 400))
    keep = (~skip) | out["is_hypo"].to_numpy()
    return out.loc[keep].reset_index(drop=True)


def flip_features(X: pd.DataFrame) -> pd.DataFrame:
    """Troca jogador 1 <-> 2: nega as diferenças e troca as colunas por jogador.

    Um bom modelo tem de dar P(1 vence) = 1 - P(2 vence). Usamos isto para
    forçar essa simetria nas previsões.
    """
    F = X.copy()
    for c in DIFF_COLS:
        if c in F:  # funciona também com subconjuntos de colunas (ablação)
            F[c] = -F[c]
    for a, b in SWAP_COLS:
        if a in F and b in F:
            F[a], F[b] = X[b], X[a]
    return F
