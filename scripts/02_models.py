"""Fase 2: features, modelos, calibração e comparação com Elo e mercado.

Divisão temporal (nunca aleatória):
  treino 1995-2020 | validação 2021-2022 (early stopping + calibração) | teste 2023+
"""
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_matches  # noqa: E402
from src.elo import compute_elo  # noqa: E402
from src.features import FEATURES, build_features, flip_features  # noqa: E402
from src.metrics import HEADER, metrics, reliability, row  # noqa: E402

TRAIN = (1995, 2020)
VAL = (2021, 2022)
TEST_FROM = 2023

GROUPS = [  # ablação: acrescentamos grupos de features um a um
    ("só Elo mistura", ["blend_diff"]),
    ("+ Elo geral/superfície/experiência", ["elo_diff", "selo_diff", "exp_diff"]),
    ("+ ranking e pontos", ["rank_log_diff", "pts_log_diff"]),
    ("+ forma e fadiga", ["form10_diff", "m14_diff", "m30_diff", "rest_diff", "cur_diff"]),
    ("+ serviço e devolução", ["spw_diff", "rpw_diff"]),
    ("+ idade, altura, mão, H2H", ["age_diff", "ht_diff", "h2h_diff", "lefty_1", "lefty_2"]),
    ("+ contexto (Slam, 5 sets, piso)", ["is_slam", "bo5", "hard", "clay", "grass",
                                          "blend_x_bo5_diff", "blend_x_slam_diff"]),
]


def sym(predict, X: pd.DataFrame) -> np.ndarray:
    """Força P(1 vence) = 1 - P(2 vence): média da previsão normal e da trocada."""
    return 0.5 * (predict(X) + 1.0 - predict(flip_features(X)))


def logit_model(cols):
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))
    return pipe


def p_win(p1, p1_is_winner):
    return np.where(p1_is_winner, p1, 1 - p1)


def boot_diff(ll_a, ll_b, n=1000, seed=1):
    """Diferença média de log loss (a - b) com IC 95% por bootstrap pareado."""
    rng = np.random.default_rng(seed)
    d = np.asarray(ll_a) - np.asarray(ll_b)
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def run(tour: str) -> list[str]:
    df = load_matches(tour)
    X = build_features(df, compute_elo(df))
    X.to_csv(ROOT / "data" / "processed" / f"{tour}_features.csv.gz", index=False)

    tr = X[X.year.between(*TRAIN)]
    va = X[X.year.between(*VAL)]
    te = X[X.year >= TEST_FROM].reset_index(drop=True)
    L = [f"## {tour.upper()}", "",
         f"Treino {TRAIN[0]}–{TRAIN[1]}: {len(tr):,} jogos · validação {VAL[0]}–{VAL[1]}: {len(va):,} · "
         f"teste {TEST_FROM}+: {len(te):,}", ""]

    # --- ablação com regressão logística ---
    L += ["### Ablação: o que cada grupo de features acrescenta (regressão logística, teste)", "", *HEADER]
    cols: list[str] = []
    for name, add in GROUPS:
        cols = cols + add
        m = logit_model(cols).fit(tr[cols], tr.y)
        p = sym(lambda Z: m.predict_proba(Z)[:, 1], te[cols])
        L.append(row(name, metrics(p_win(p, te.p1_is_winner))))
    L.append("")

    # --- modelos finais com todas as features ---
    lr = logit_model(FEATURES).fit(tr[FEATURES], tr.y)
    p_lr = lambda Z: lr.predict_proba(Z[FEATURES])[:, 1]  # noqa: E731

    gbm = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10,
                             random_state=7, verbose=-1)
    gbm.fit(tr[FEATURES], tr.y, eval_set=[(va[FEATURES], va.y)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(150, verbose=False)])
    p_gb = lambda Z: gbm.predict_proba(Z[FEATURES])[:, 1]  # noqa: E731

    def p_ens(Z):
        return 0.5 * p_lr(Z) + 0.5 * p_gb(Z)

    # calibração isotónica ajustada na VALIDAÇÃO, mantendo a simetria
    pv = sym(p_ens, va)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(
        np.r_[pv, 1 - pv], np.r_[va.y.to_numpy(), 1 - va.y.to_numpy()])

    def cal(p):
        return 0.5 * (iso.predict(p) + 1 - iso.predict(1 - p))

    preds = {
        "Elo mistura (baseline)": te.elo_p1_prob.to_numpy(),
        "Regressão logística": sym(p_lr, te),
        "LightGBM": sym(p_gb, te),
        "Ensemble (média)": sym(p_ens, te),
    }
    preds["Ensemble calibrado"] = cal(preds["Ensemble (média)"])

    subsets = {
        "todos": np.ones(len(te), bool),
        "Grand Slams": (te.is_slam == 1).to_numpy(),
        "US Open": te.tourney_name.str.contains("US Open", case=False, na=False).to_numpy(),
    }
    L += [f"### Modelos finais (LightGBM parou em {gbm.best_iteration_} árvores)", "", *HEADER]
    for sname, mask in subsets.items():
        for name, p in preds.items():
            L.append(row(f"{name} — {sname}", metrics(p_win(p[mask], te.p1_is_winner[mask]))))
    L.append("")

    # --- IC da melhoria sobre o Elo ---
    ll = {k: -np.log(np.clip(p_win(p, te.p1_is_winner), 1e-6, 1)) for k, p in preds.items()}
    d, lo, hi = boot_diff(ll["Ensemble (média)"], ll["Elo mistura (baseline)"])
    L += [f"**Ensemble − Elo (log loss, todos os jogos de teste): {d:+.4f} "
          f"(IC 95%: {lo:+.4f} a {hi:+.4f})**. Negativo = melhor.", ""]

    # --- mercado ---
    mk = te.AvgW.notna() & te.AvgL.notna() & (te.source == "tennis-data")
    if mk.sum() > 100:
        pm =(1 / te.AvgW[mk]) / (1 / te.AvgW[mk] + 1 / te.AvgL[mk])  # P(vencedor) sem margem
        L += [f"### Contra o mercado (jogos depois de Roland Garros 2026 com odds, n={int(mk.sum())})", "", *HEADER,
              row("Mercado (odds médias)", metrics(pm.to_numpy()))]
        for name in ["Elo mistura (baseline)", "Regressão logística", "LightGBM", "Ensemble (média)", "Ensemble calibrado"]:
            L.append(row(name, metrics(p_win(preds[name][mk.to_numpy()], te.p1_is_winner[mk]))))
        llm = -np.log(pm.to_numpy())
        d, lo, hi = boot_diff(ll["Ensemble (média)"][mk.to_numpy()], llm)
        L += ["", f"**Ensemble − mercado (log loss): {d:+.4f} (IC 95%: {lo:+.4f} a {hi:+.4f})**. "
              "Positivo = o mercado continua melhor.", ""]

    # --- calibração ---
    L += ["### Calibração do ensemble (média, sem isotónica) no teste", "",
          "| Prob. prevista | Jogos | Média prevista | Frequência observada |", "|---|---|---|---|"]
    pw = p_win(preds["Ensemble (média)"], te.p1_is_winner)
    for lo_, hi_, cnt, mp, fo in reliability(pw):
        L.append(f"| {lo_:.1f}–{hi_:.1f} | {cnt} | {mp:.3f} | {fo:.3f} |")
    L.append("")

    # --- importância ---
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(10)
    L += ["### Importância das features no LightGBM (ganho, top 10)", "", "| Feature | Importância |", "|---|---|"]
    L += [f"| `{k}` | {v:.1%} |" for k, v in imp.items()]
    L.append("")

    joblib.dump({"lr": lr, "gbm": gbm, "iso": iso, "features": FEATURES}, ROOT / "models" / f"{tour}_models.joblib")
    return L


if __name__ == "__main__":
    out = ["# Fase 2 — Features e modelos", "", "Gerado por `scripts/02_models.py`.", ""]
    for t in ["atp", "wta"]:
        out += run(t)
    (ROOT / "reports" / "02_modelos.md").write_text("\n".join(out), encoding="utf-8")
    print("done")
