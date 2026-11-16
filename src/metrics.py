"""Métricas probabilísticas, partilhadas por todos os scripts."""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def metrics(p_winner: np.ndarray) -> dict:
    """`p_winner` = probabilidade atribuída a quem ACABOU por ganhar.

    Simetrizamos (cada jogo conta como vencedor com p e perdedor com 1-p) para
    que AUC e Brier não dependam de quem calhou ser o "jogador 1".
    """
    p_winner = np.asarray(p_winner, dtype=float)
    y = np.r_[np.ones(len(p_winner)), np.zeros(len(p_winner))]
    p = np.clip(np.r_[p_winner, 1 - p_winner], 1e-6, 1 - 1e-6)
    return {
        "n": len(p_winner),
        "acc": float((p_winner > 0.5).mean()),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def row(name: str, m: dict) -> str:
    return f"| {name} | {m['n']} | {m['acc']:.3f} | {m['logloss']:.4f} | {m['brier']:.4f} | {m['auc']:.3f} |"


HEADER = ["| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |", "|---|---|---|---|---|---|"]


def reliability(p_winner: np.ndarray, bins: int = 10) -> list[tuple]:
    """Tabela de calibração: previsto vs observado, por intervalo de probabilidade.

    Usa a visão simetrizada: cada jogo dá dois pontos (p, ganhou=1) e (1-p, ganhou=0).
    """
    p = np.r_[p_winner, 1 - p_winner]
    y = np.r_[np.ones(len(p_winner)), np.zeros(len(p_winner))]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((edges[b], edges[b + 1], int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out


def boot_diff(ll_a, ll_b, n: int = 1000, seed: int = 1):
    """Diferença média de log loss (a - b) com IC 95% por bootstrap pareado. Negativo = a é melhor."""
    rng = np.random.default_rng(seed)
    d = np.asarray(ll_a, dtype=float) - np.asarray(ll_b, dtype=float)
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
