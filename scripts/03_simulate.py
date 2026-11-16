"""Fase 3: simulação do US Open (backtest 2023-2026) e perspetiva futura.

Backtest: o modelo só vê informação ANTERIOR ao início do torneio. O sorteio do quadro
é o real (inferido dos resultados); quem ganhou cada jogo nunca entra nas probabilidades.
"""
import sys
from pathlib import Path

import json

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consolas Windows (cp1252) não imprimem "−" nem "→"
sys.path.insert(0, str(ROOT))
from src.data import load_matches  # noqa: E402
from src.elo import compute_elo  # noqa: E402
from src.features import build_features  # noqa: E402
from src.simulate import (actual_reach, bracket_order, ensemble_p1, make_hypo_rows,  # noqa: E402
                          prob_matrix, simulate)

YEARS = [2023, 2024, 2025, 2026]
N_SIM = 20000


def event_logloss(M: np.ndarray, wins: np.ndarray) -> float:
    """Log loss binário médio de 'ganhar >= k rondas', k=1..7, para os 128 jogadores."""
    ll = []
    for k in range(1, 8):
        p = np.clip(M[:, k], 1e-4, 1 - 1e-4)
        y = (wins >= k).astype(float)
        ll.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(ll))


def run(tour: str):
    df = load_matches(tour)
    best_of = 5 if tour == "atp" else 3
    groups, hypos = {}, []
    # nomes completos (do Sackmann) para mostrar de forma consistente; o tennis-data usa 'Apelido I.'
    sk = df[df.source == "sackmann"]
    nm = pd.concat([sk[["winner_id", "winner_name"]].set_axis(["id", "name"], axis=1),
                    sk[["loser_id", "loser_name"]].set_axis(["id", "name"], axis=1)]
                   ).drop_duplicates("id", keep="last").set_index("id")["name"].to_dict()

    # --- backtests: um US Open por ano
    for y in YEARS:
        t = df[df.tourney_name.str.contains("us open", case=False, na=False) & (df.year == y)]
        t = t[t.source == t.source.iloc[0]] if len(t) else t
        order = bracket_order(t)
        r1 = t[t.round_idx == 1]
        info = {}
        for r in r1.itertuples():
            info[int(r.winner_id)] = (r.winner_name, r.winner_rank, r.winner_rank_points)
            info[int(r.loser_id)] = (r.loser_name, r.loser_rank, r.loser_rank_points)
        ent = pd.DataFrame({"id": order, "name": [info[p][0] for p in order],
                            "rank": [info[p][1] for p in order], "pts": [info[p][2] for p in order]})
        ent[["rank", "pts"]] = ent[["rank", "pts"]].astype(float).fillna({"rank": 1500.0, "pts": 0.0})
        ent["name"] = [nm.get(i, n) for i, n in zip(ent.id, ent.name)]
        start = t.tourney_start.min()
        g = f"{tour}-{y}"
        groups[g] = dict(ent=ent, wins=actual_reach(t, order), random_draw=False, year=y)
        hypos.append(make_hypo_rows(ent, start, g, best_of))

    # --- perspetiva: próximo Slam em piso duro, quadro sorteado ao acaso (sem cabeças de série)
    last = df.date.max()
    rec = df[df.date >= last - pd.Timedelta(days=120)]
    both = pd.concat([
        rec[["date", "winner_id", "winner_name", "winner_rank", "winner_rank_points"]].set_axis(["date", "id", "name", "rank", "pts"], axis=1),
        rec[["date", "loser_id", "loser_name", "loser_rank", "loser_rank_points"]].set_axis(["date", "id", "name", "rank", "pts"], axis=1),
    ]).dropna(subset=["rank"]).sort_values("date").groupby("id").last().reset_index()
    ent = both.sort_values("rank").head(128)[["id", "name", "rank", "pts"]].reset_index(drop=True)
    ent["pts"] = ent["pts"].fillna(0.0)
    ent["name"] = [nm.get(i, n) for i, n in zip(ent.id, ent.name)]
    g = f"{tour}-outlook"
    groups[g] = dict(ent=ent, wins=None, random_draw=True, year=int(last.year) + 1)
    hypos.append(make_hypo_rows(ent, last + pd.Timedelta(days=1), g, best_of))

    # --- features (mesma máquina dos jogos reais) com os pares hipotéticos inseridos
    full = pd.concat([df] + hypos, ignore_index=True)
    full["is_walkover"] = full["is_walkover"].astype(bool)
    full = full.sort_values(["date", "tourney_id", "match_num"], kind="stable").reset_index(drop=True)
    # Elo com os parâmetros otimizados (04_tune_elo.py) e modelos walk-forward (05_walkforward.py):
    # o US Open do ano Y usa o modelo treinado só com dados até Y-3 (validação Y-2..Y-1).
    params = json.loads((ROOT / "models" / f"elo_params_{tour}.json").read_text())
    X = build_features(full, compute_elo(full, params), blend_w=params["blend_w"])
    H = X[X.is_hypo].copy()
    H["p_model"] = np.nan
    for g in H.group.unique():
        key = "outlook" if g.endswith("outlook") else g.split("-")[1]
        mdl = joblib.load(ROOT / "models" / f"{tour}_wf_{key}.joblib")
        m = (H.group == g).to_numpy()
        H.loc[m, "p_model"] = ensemble_p1(mdl, H[m])
    H["p_elo"] = H["elo_p1_prob"]

    sims, pairs, summ = [], [], []
    names = {}
    for g, info in groups.items():
        ent, rows = info["ent"], H[H.group == g]
        ids = ent.id.astype(int).tolist()
        Pm, Pe = prob_matrix(rows, ids, "p_model"), prob_matrix(rows, ids, "p_elo")
        Mm = simulate(Pm, N_SIM, random_draw=info["random_draw"])
        Me = simulate(Pe, N_SIM, random_draw=info["random_draw"])
        d = ent.assign(group=g, year=info["year"], tour=tour, slot=range(len(ent)))
        for k in range(1, 8):
            d[f"pm_{k}"], d[f"pe_{k}"] = Mm[:, k], Me[:, k]
        if info["wins"] is not None:
            wins = np.array([info["wins"][i] for i in ids])
            d["actual_wins"] = wins
            champ = int(np.argmax(wins))
            summ.append(dict(
                tour=tour, year=info["year"], champion=ent.name[champ],
                p_title_model=Mm[champ, 7], rank_model=int((Mm[:, 7] > Mm[champ, 7]).sum()) + 1,
                p_title_elo=Me[champ, 7], rank_elo=int((Me[:, 7] > Me[champ, 7]).sum()) + 1,
                favourite_model=ent.name[int(np.argmax(Mm[:, 7]))], p_fav_model=Mm[:, 7].max(),
                ll_model=event_logloss(Mm, wins), ll_elo=event_logloss(Me, wins)))
        sims.append(d)
        r = rows[["group", "p1_id", "p2_id", "p_model", "p_elo"]].copy()
        r["p1_name"] = r.p1_id.map(dict(zip(ent.id, ent.name)))
        r["p2_name"] = r.p2_id.map(dict(zip(ent.id, ent.name)))
        pairs.append(r)

    out = ROOT / "data" / "processed"
    pd.concat(sims).to_csv(out / f"sim_{tour}.csv", index=False)
    pd.concat(pairs).to_csv(out / f"pairs_{tour}.csv", index=False)
    return pd.DataFrame(summ), pd.concat(sims)


if __name__ == "__main__":
    lines = ["# Fase 3 — Simulação do quadro e backtest", "", "Gerado por `scripts/03_simulate.py`.",
             f"{N_SIM:,} simulações por torneio. O modelo só usa informação anterior ao início de cada torneio.", ""]
    allsum = []
    for tour in ["atp", "wta"]:
        s, sims = run(tour)
        allsum.append(s)
        lines += [f"## {tour.upper()} — campeão real vs probabilidade do modelo", "",
                  "| Ano | Campeão | P(título) modelo | Posição | P(título) Elo | Posição Elo | Favorito do modelo (P) | LogLoss rondas modelo | LogLoss rondas Elo |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in s.itertuples():
            lines.append(f"| {r.year} | {r.champion} | {r.p_title_model:.1%} | {r.rank_model} | {r.p_title_elo:.1%} | {r.rank_elo} | "
                         f"{r.favourite_model} ({r.p_fav_model:.1%}) | {r.ll_model:.4f} | {r.ll_elo:.4f} |")
        top = sims[sims.group == f"{tour}-outlook"].sort_values("pm_7", ascending=False).head(10)
        lines += ["", f"### {tour.upper()} — perspetiva (quadro sorteado ao acaso, top 128 do ranking atual, piso duro)", "",
                  "| Jogador | Ranking | P(título) |", "|---|---|---|"]
        lines += [f"| {r.name} | {int(r.rank)} | {r.pm_7:.1%} |" for r in top.itertuples()]
        lines.append("")
    pd.concat(allsum).to_csv(ROOT / "data" / "processed" / "backtest_summary.csv", index=False)
    (ROOT / "reports" / "03_simulacao.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
