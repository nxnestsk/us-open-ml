"""Fase 1: exploração dos dados + baseline Elo. Gera reports/01_eda_elo.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_matches  # noqa: E402
from src.elo import compute_elo, win_prob  # noqa: E402

TEST_FROM = 2023  # treina (calcula ratings) no passado, avalia de 2023 em diante


def metrics(p_winner: np.ndarray) -> dict:
    """p_winner = prob. atribuída a quem ganhou. Simetrizamos para AUC/Brier."""
    y = np.r_[np.ones(len(p_winner)), np.zeros(len(p_winner))]
    p = np.r_[p_winner, 1 - p_winner]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(p_winner),
        "acc": float((p_winner > 0.5).mean()),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def fmt(name: str, m: dict) -> str:
    return f"| {name} | {m['n']} | {m['acc']:.3f} | {m['logloss']:.4f} | {m['brier']:.4f} | {m['auc']:.3f} |"


def run(tour: str) -> list[str]:
    df = load_matches(tour)
    elo = compute_elo(df)
    df = df.join(elo)
    df.to_csv(ROOT / "data" / "processed" / f"{tour}_elo.csv.gz", index=False)

    played = df[~df["is_walkover"]].copy()
    played["p_overall"] = win_prob(played.w_elo, played.l_elo)
    played["p_surface"] = win_prob(played.w_selo, played.l_selo)
    played["p_blend"] = win_prob((played.w_elo + played.w_selo) / 2, (played.l_elo + played.l_selo) / 2)
    test = played[played.year >= TEST_FROM]
    slam = test[test.tourney_level == "G"]
    uso = test[test.tourney_name.str.contains("US Open", case=False, na=False)]

    lines = [f"## {tour.upper()}", ""]
    lines += [
        f"- Jogos totais carregados: **{len(df):,}** ({df.date.min().date()} a {df.date.max().date()})",
        f"- Jogos de teste (>= {TEST_FROM}, sem walkovers): **{len(test):,}**",
        "",
        "### Superfícies (todos os anos desde 1985)",
        "",
        (df.surface.value_counts(normalize=True).round(3) * 100).to_frame("%").T.to_markdown(),
        "",
        "### Baseline: quanto acerta cada rating (período de teste)",
        "",
        "| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for label, subset in [("todos", test), ("Grand Slams", slam), ("US Open", uso)]:
        for name in ["p_overall", "p_surface", "p_blend"]:
            lines.append(fmt(f"Elo {name[2:]} — {label}", metrics(subset[name].to_numpy())))
    # baseline ingénuo: o melhor ranking ganha
    rk = test.dropna(subset=["winner_rank", "loser_rank"])
    rk_acc = float((rk.winner_rank < rk.loser_rank).mean())
    lines += ["", f"- Baseline ingénuo (melhor ranking ganha): accuracy **{rk_acc:.3f}** em {len(rk):,} jogos.", ""]

    # comparação com o mercado (odds médias) onde existirem
    if "AvgW" in test:
        mk = test.dropna(subset=["AvgW", "AvgL"]).copy()
        mk[["AvgW", "AvgL"]] = mk[["AvgW", "AvgL"]].astype(float)
        if len(mk):
            pm = (1 / mk.AvgW) / (1 / mk.AvgW + 1 / mk.AvgL)  # remove a margem da casa
            lines += [
                f"### Elo vs mercado (jogos pós-Roland Garros 2026 com odds, n={len(mk)})",
                "",
                "| Modelo | Jogos | Accuracy | LogLoss | Brier | AUC |",
                "|---|---|---|---|---|---|",
                fmt("Mercado (odds médias)", metrics(pm.to_numpy())),
                fmt("Elo blend", metrics(mk.p_blend.to_numpy())),
                "",
            ]

    return lines


if __name__ == "__main__":
    out = ["# Fase 1 — Exploração e baseline Elo", "", "Gerado por `scripts/01_eda_elo.py`.", ""]
    for tour in ["atp", "wta"]:
        out += run(tour)
    (ROOT / "reports" / "01_eda_elo.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
