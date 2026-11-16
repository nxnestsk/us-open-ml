"""Fase 4b: Elo otimizado + treino walk-forward com pesos temporais.

Para cada ano de teste Y (2023..2026) treina-se só com o passado:
  treino 1995..Y-3 | validação Y-2..Y-1 (paragem antecipada) | teste Y
O período de meia-vida dos pesos é escolhido na validação de 2021-2022, antes de qualquer ano de teste.
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consolas Windows (cp1252) não imprimem "−" nem "→"
sys.path.insert(0, str(ROOT))
from src.data import load_matches  # noqa: E402
from src.elo import compute_elo  # noqa: E402
from src.features import FEATURES, FEATURES_NO_INJURY, build_features, flip_features  # noqa: E402
from src.metrics import HEADER, boot_diff, metrics, row  # noqa: E402
from src.simulate import ensemble_p1  # noqa: E402

TEST_YEARS = [2023, 2024, 2025, 2026]
HALF_LIVES = [None, 4, 8, 12]


def weights(years: np.ndarray, half_life):
    if half_life is None:
        return np.ones(len(years))
    return 0.5 ** ((years.max() - years) / half_life)


def make_logit():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))


def sym(pred, X):
    return 0.5 * (pred(X) + 1.0 - pred(flip_features(X)))


def p_win(p1, is_w):
    return np.where(is_w, p1, 1 - p1)


def fit_fold(X, last_train_year: int, val_years: tuple, half_life, feats=None):
    """Treina logística + LightGBM com o passado e devolve o dicionário de modelos."""
    feats = feats or FEATURES
    tr = X[X.year.between(1995, last_train_year)]
    va = X[X.year.between(*val_years)]
    w = weights(tr.year.to_numpy(), half_life)
    lr = make_logit().fit(tr[feats], tr.y, logisticregression__sample_weight=w)
    gbm = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10,
                             random_state=7, verbose=-1)
    gbm.fit(tr[feats], tr.y, sample_weight=w, eval_set=[(va[feats], va.y)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(150, verbose=False)])
    return {"lr": lr, "gbm": gbm, "features": feats}


def run(tour: str) -> list[str]:
    params = json.loads((ROOT / "models" / f"elo_params_{tour}.json").read_text())
    df = load_matches(tour)
    X = build_features(df, compute_elo(df, params), blend_w=params["blend_w"])
    X.to_csv(ROOT / "data" / "processed" / f"{tour}_features_v2.csv.gz", index=False)

    # --- 1) meia-vida escolhida na validação 2021-22, com treino até 2020 (nenhum ano de teste)
    tr, va = X[X.year.between(1995, 2020)], X[X.year.between(2021, 2022)]
    sel = {}
    for hl in HALF_LIVES:
        m = make_logit().fit(tr[FEATURES], tr.y, logisticregression__sample_weight=weights(tr.year.to_numpy(), hl))
        p = sym(lambda Z: m.predict_proba(Z[FEATURES])[:, 1], va)
        sel[hl] = log_loss(np.r_[va.y, 1 - va.y], np.r_[p, 1 - p])
    best_hl = min(sel, key=sel.get)

    # --- 2) walk-forward
    te_all = X[X.year >= TEST_YEARS[0]].reset_index(drop=True)
    p_wf = np.zeros(len(te_all))
    p_ni = np.zeros(len(te_all))  # walk-forward idêntico mas SEM as features de lesão
    iters = {}
    for Y in TEST_YEARS:
        mdl = fit_fold(X, Y - 3, (Y - 2, Y - 1), best_hl)
        joblib.dump(mdl, ROOT / "models" / f"{tour}_wf_{Y}.joblib")
        m = (te_all.year == Y).to_numpy()
        p_wf[m] = ensemble_p1(mdl, te_all[m])
        iters[Y] = mdl["gbm"].best_iteration_
        mni = fit_fold(X, Y - 3, (Y - 2, Y - 1), best_hl, FEATURES_NO_INJURY)
        p_ni[m] = ensemble_p1(mni, te_all[m])
    # modelo para a perspetiva futura: treino até 2023, validação 2024-2026
    joblib.dump(fit_fold(X, 2023, (2024, 2026), best_hl), ROOT / "models" / f"{tour}_wf_outlook.joblib")

    # --- 3) comparações no mesmo conjunto de teste
    # v1: modelos treinados até 2020 + Elo FiveThirtyEight (ficheiro de features da v1)
    X1 = pd.read_csv(ROOT / "data" / "processed" / f"{tour}_features.csv.gz")
    te1 = X1[X1.year >= TEST_YEARS[0]].reset_index(drop=True)
    assert (te1.p1_id.to_numpy() == te_all.p1_id.to_numpy()).all() and len(te1) == len(te_all)
    p_v1 = ensemble_p1(joblib.load(ROOT / "models" / f"{tour}_models.joblib"), te1)
    # v2 estático: Elo otimizado, mas treino fixo até 2020 (isola o efeito do walk-forward)
    st = fit_fold(X, 2020, (2021, 2022), None)
    p_st = ensemble_p1(st, te_all)

    is_w = te_all.p1_is_winner.to_numpy()
    preds = {"Elo v1 (baseline)": te1.elo_p1_prob.to_numpy(), "Ensemble v1 (treino até 2020)": p_v1,
             "Ensemble com Elo otimizado, treino até 2020": p_st, "Ensemble walk-forward sem lesões": p_ni,
             "Ensemble walk-forward (final)": p_wf}
    ll = {k: -np.log(np.clip(p_win(p, is_w), 1e-6, 1)) for k, p in preds.items()}

    L = [f"## {tour.upper()}", "",
         f"Meia-vida dos pesos escolhida na validação 2021–22: **{best_hl if best_hl else 'sem pesos'}** "
         f"(log loss de validação: " + ", ".join(f"{'sem' if k is None else k}: {v:.4f}" for k, v in sel.items()) + ").",
         f"Árvores do LightGBM por ano de teste: " + ", ".join(f"{y}: {n}" for y, n in iters.items()) + ".", "",
         "### Teste 2023+ (mesmos jogos para todos)", "", *HEADER]
    for k, p in preds.items():
        L.append(row(k, metrics(p_win(p, is_w))))
    L += ["", "### Diferença em log loss (IC 95%, bootstrap pareado; negativo = o primeiro é melhor)", "",
          "| Comparação | Diferença | IC 95% |", "|---|---|---|"]
    for a, b in [("Ensemble com Elo otimizado, treino até 2020", "Ensemble v1 (treino até 2020)"),
                 ("Ensemble walk-forward (final)", "Ensemble com Elo otimizado, treino até 2020"),
                 ("Ensemble walk-forward (final)", "Ensemble walk-forward sem lesões"),
                 ("Ensemble walk-forward (final)", "Ensemble v1 (treino até 2020)")]:
        d, lo, hi = boot_diff(ll[a], ll[b])
        L.append(f"| {a} − {b} | {d:+.4f} | {lo:+.4f} a {hi:+.4f} |")
    L += ["", "### Por ano (log loss)", "", "| Ano | Jogos | Ensemble v1 | Walk-forward |", "|---|---|---|---|"]
    for Y in TEST_YEARS:
        m = (te_all.year == Y).to_numpy()
        L.append(f"| {Y} | {int(m.sum())} | {ll['Ensemble v1 (treino até 2020)'][m].mean():.4f} | "
                 f"{ll['Ensemble walk-forward (final)'][m].mean():.4f} |")
    sub = {"Grand Slams": (te_all.is_slam == 1).to_numpy(),
           "US Open": te_all.tourney_name.str.contains("US Open", case=False, na=False).to_numpy()}
    L += ["", "### Grand Slams e US Open", "", *HEADER]
    for sname, mask in sub.items():
        for k in ["Ensemble v1 (treino até 2020)", "Ensemble walk-forward (final)"]:
            L.append(row(f"{k} — {sname}", metrics(p_win(preds[k][mask], is_w[mask]))))

    mk = (te_all.AvgW.notna() & te_all.AvgL.notna() & (te_all.source == "tennis-data")).to_numpy()
    if mk.sum() > 100:
        pm = ((1 / te_all.AvgW[mk]) / (1 / te_all.AvgW[mk] + 1 / te_all.AvgL[mk])).to_numpy()
        L += ["", f"### Contra o mercado (depois de Roland Garros 2026, n={int(mk.sum())})", "", *HEADER,
              row("Mercado (odds médias)", metrics(pm))]
        for k in ["Ensemble v1 (treino até 2020)", "Ensemble walk-forward (final)"]:
            L.append(row(k, metrics(p_win(preds[k][mk], is_w[mk]))))
        llm = -np.log(pm)
        L.append("")
        for k in ["Ensemble v1 (treino até 2020)", "Ensemble walk-forward (final)"]:
            d, lo, hi = boot_diff(ll[k][mk], llm)
            L.append(f"- {k} − mercado: **{d:+.4f}** (IC 95%: {lo:+.4f} a {hi:+.4f}). Positivo = o mercado é melhor.")
    L.append("")
    return L


if __name__ == "__main__":
    out = ["# Fase 4b — Elo otimizado e treino walk-forward", "", "Gerado por `scripts/05_walkforward.py`.", ""]
    for t in ["atp", "wta"]:
        out += run(t)
    (ROOT / "reports" / "05_walkforward.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
