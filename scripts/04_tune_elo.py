"""Fase 4a: otimizar os parâmetros do Elo SEM olhar para 2023 em diante.

Procura aleatória + refinamento local. O critério é o log loss do Elo (mistura geral/superfície, com
uma escala ajustada) em 2012-2022. Só depois se mede o resultado em 2023+ (nunca usado na procura).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consolas Windows (cp1252) não imprimem "−" nem "→"
sys.path.insert(0, str(ROOT))
from src.data import load_matches  # noqa: E402
from src.elo import DEFAULT_PARAMS, compute_elo  # noqa: E402

TUNE_YEARS = (2012, 2022)  # janela de avaliação da procura
SPACE = dict(scale=(120, 420), shape=(0.25, 0.65), offset=(2, 12), slam_bonus=(1.0, 1.4),
             surf_k=(0.5, 1.6), blend_w=(0.2, 0.8))

_df = None
_arr = None


def _init(tour: str):
    global _df, _arr
    df = load_matches(tour)
    _df = df[df.year <= TUNE_YEARS[1]].reset_index(drop=True)  # a procura nunca vê 2023+
    _arr = dict(year=_df.year.to_numpy(), skip=_df.is_walkover.to_numpy())


def fit_slope(d: np.ndarray) -> tuple[float, float]:
    """Escala `a` que minimiza o log loss de P(vencedor) = sigmoid(a * d / 100)."""
    f = lambda a: float(np.mean(np.log1p(np.exp(-a * d / 100.0))))  # noqa: E731
    r = minimize_scalar(f, bounds=(0.05, 3.0), method="bounded")
    return float(r.x), float(r.fun)


def _blend_diff(elo, bw):
    return bw * (elo.w_elo.to_numpy() - elo.l_elo.to_numpy()) + (1 - bw) * (elo.w_selo.to_numpy() - elo.l_selo.to_numpy())


def evaluate(params: dict) -> float:
    elo = compute_elo(_df, params)
    d = _blend_diff(elo, params["blend_w"])
    m = (_arr["year"] >= TUNE_YEARS[0]) & ~_arr["skip"]
    return fit_slope(d[m])[1]


def sample(rng) -> dict:
    return {k: float(rng.uniform(*v)) for k, v in SPACE.items()}


def perturb(base: dict, rng, scale=0.12) -> dict:
    out = {}
    for k, (lo, hi) in SPACE.items():
        out[k] = float(np.clip(base[k] * (1 + rng.normal(0, scale)), lo, hi))
    return out


def run(tour: str):
    rng = np.random.default_rng(5)
    baseline = {k: DEFAULT_PARAMS[k] for k in SPACE}
    trials = [baseline] + [sample(rng) for _ in range(31)]
    with ProcessPoolExecutor(max_workers=7, initializer=_init, initargs=(tour,)) as ex:
        scores = list(ex.map(evaluate, trials))
        for _ in range(3):  # refinamento local à volta do melhor
            best = trials[int(np.argmin(scores))]
            new = [perturb(best, rng) for _ in range(14)]
            trials += new
            scores += list(ex.map(evaluate, new))
    order = np.argsort(scores)
    best = trials[int(order[0])]

    # avaliação final: 2023+ nunca foi usado na procura
    df = load_matches(tour)
    played = ~df.is_walkover.to_numpy()
    res = {}
    for name, par in (("v1 (FiveThirtyEight)", baseline), ("otimizado", best)):
        elo = compute_elo(df, par)
        d = _blend_diff(elo, par["blend_w"])
        yr = df.year.to_numpy()
        a, ll_tune = fit_slope(d[(yr >= TUNE_YEARS[0]) & (yr <= TUNE_YEARS[1]) & played])
        te = (yr >= 2023) & played
        ll_test = float(np.mean(np.log1p(np.exp(-a * d[te] / 100.0))))  # escala fixada na janela de procura
        res[name] = dict(params=par, a=a, ll_tune=ll_tune, ll_test=ll_test, n_test=int(te.sum()))
    (ROOT / "models" / f"elo_params_{tour}.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return res, [(scores[i], trials[i]) for i in order[:5]], len(trials)


if __name__ == "__main__":
    out = ["# Fase 4a — Otimização do Elo", "",
           f"Procura em 2012–2022 (nunca vê 2023+). Critério: log loss do Elo com a escala ajustada.", ""]
    for tour in ["atp", "wta"]:
        res, top, n = run(tour)
        out += [f"## {tour.upper()} ({n} combinações testadas)", "",
                "| Elo | Log loss 2012–2022 (procura) | Log loss 2023+ (teste, nunca usado) |", "|---|---|---|"]
        for k, v in res.items():
            out.append(f"| {k} | {v['ll_tune']:.4f} | {v['ll_test']:.4f} |")
        out += ["", "| Parâmetro | v1 | otimizado |", "|---|---|---|"]
        for k in SPACE:
            out.append(f"| `{k}` | {res['v1 (FiveThirtyEight)']['params'][k]:.3f} | {res['otimizado']['params'][k]:.3f} |")
        out += ["", "Melhores 5 combinações (log loss na janela de procura):", ""]
        out += [f"- {s:.4f}: " + ", ".join(f"{k}={v:.2f}" for k, v in p.items()) for s, p in top]
        out.append("")
    (ROOT / "reports" / "04_elo_tuning.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
