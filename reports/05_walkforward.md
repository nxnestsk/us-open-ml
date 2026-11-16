# Fase 4b — Elo otimizado e treino walk-forward

Gerado por `scripts/05_walkforward.py`.

## ATP

Meia-vida dos pesos escolhida na validação 2021–22: **sem pesos** (log loss de validação: sem: 0.5956, 4: 0.5960, 8: 0.5957, 12: 0.5956).
Árvores do LightGBM por ano de teste: 2023: 503, 2024: 646, 2025: 554, 2026: 379.

### Teste 2023+ (mesmos jogos para todos)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Elo v1 (baseline) | 11638 | 0.637 | 0.6398 | 0.2232 | 0.699 |
| Ensemble v1 (treino até 2020) | 11638 | 0.654 | 0.6122 | 0.2129 | 0.720 |
| Ensemble com Elo otimizado, treino até 2020 | 11638 | 0.656 | 0.6112 | 0.2125 | 0.721 |
| Ensemble walk-forward sem lesões | 11638 | 0.656 | 0.6113 | 0.2125 | 0.721 |
| Ensemble walk-forward (final) | 11638 | 0.656 | 0.6110 | 0.2124 | 0.721 |

### Diferença em log loss (IC 95%, bootstrap pareado; negativo = o primeiro é melhor)

| Comparação | Diferença | IC 95% |
|---|---|---|
| Ensemble com Elo otimizado, treino até 2020 − Ensemble v1 (treino até 2020) | -0.0010 | -0.0017 a -0.0003 |
| Ensemble walk-forward (final) − Ensemble com Elo otimizado, treino até 2020 | -0.0002 | -0.0004 a +0.0001 |
| Ensemble walk-forward (final) − Ensemble walk-forward sem lesões | -0.0003 | -0.0007 a +0.0001 |
| Ensemble walk-forward (final) − Ensemble v1 (treino até 2020) | -0.0012 | -0.0019 a -0.0005 |

### Por ano (log loss)

| Ano | Jogos | Ensemble v1 | Walk-forward |
|---|---|---|---|
| 2023 | 2966 | 0.6152 | 0.6139 |
| 2024 | 3138 | 0.6067 | 0.6059 |
| 2025 | 2840 | 0.6149 | 0.6126 |
| 2026 | 2694 | 0.6124 | 0.6120 |

### Grand Slams e US Open

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Ensemble v1 (treino até 2020) — Grand Slams | 2021 | 0.721 | 0.5436 | 0.1833 | 0.798 |
| Ensemble walk-forward (final) — Grand Slams | 2021 | 0.724 | 0.5419 | 0.1829 | 0.799 |
| Ensemble v1 (treino até 2020) — US Open | 507 | 0.688 | 0.5662 | 0.1936 | 0.776 |
| Ensemble walk-forward (final) — US Open | 507 | 0.698 | 0.5646 | 0.1930 | 0.778 |

### Contra o mercado (depois de Roland Garros 2026, n=1256)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Mercado (odds médias) | 1256 | 0.652 | 0.6166 | 0.2146 | 0.715 |
| Ensemble v1 (treino até 2020) | 1256 | 0.639 | 0.6228 | 0.2178 | 0.704 |
| Ensemble walk-forward (final) | 1256 | 0.639 | 0.6231 | 0.2179 | 0.704 |

- Ensemble v1 (treino até 2020) − mercado: **+0.0063** (IC 95%: -0.0054 a +0.0176). Positivo = o mercado é melhor.
- Ensemble walk-forward (final) − mercado: **+0.0066** (IC 95%: -0.0051 a +0.0177). Positivo = o mercado é melhor.

## WTA

Meia-vida dos pesos escolhida na validação 2021–22: **4** (log loss de validação: sem: 0.6005, 4: 0.6001, 8: 0.6001, 12: 0.6002).
Árvores do LightGBM por ano de teste: 2023: 503, 2024: 292, 2025: 392, 2026: 392.

### Teste 2023+ (mesmos jogos para todos)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Elo v1 (baseline) | 10780 | 0.647 | 0.6253 | 0.2177 | 0.711 |
| Ensemble v1 (treino até 2020) | 10780 | 0.664 | 0.6060 | 0.2099 | 0.729 |
| Ensemble com Elo otimizado, treino até 2020 | 10780 | 0.664 | 0.6063 | 0.2101 | 0.729 |
| Ensemble walk-forward sem lesões | 10780 | 0.666 | 0.6062 | 0.2100 | 0.729 |
| Ensemble walk-forward (final) | 10780 | 0.667 | 0.6058 | 0.2097 | 0.730 |

### Diferença em log loss (IC 95%, bootstrap pareado; negativo = o primeiro é melhor)

| Comparação | Diferença | IC 95% |
|---|---|---|
| Ensemble com Elo otimizado, treino até 2020 − Ensemble v1 (treino até 2020) | +0.0004 | -0.0003 a +0.0010 |
| Ensemble walk-forward (final) − Ensemble com Elo otimizado, treino até 2020 | -0.0006 | -0.0015 a +0.0004 |
| Ensemble walk-forward (final) − Ensemble walk-forward sem lesões | -0.0005 | -0.0009 a +0.0000 |
| Ensemble walk-forward (final) − Ensemble v1 (treino até 2020) | -0.0002 | -0.0013 a +0.0009 |

### Por ano (log loss)

| Ano | Jogos | Ensemble v1 | Walk-forward |
|---|---|---|---|
| 2023 | 2788 | 0.5997 | 0.6007 |
| 2024 | 2774 | 0.6027 | 0.6019 |
| 2025 | 2674 | 0.6154 | 0.6148 |
| 2026 | 2544 | 0.6065 | 0.6060 |

### Grand Slams e US Open

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Ensemble v1 (treino até 2020) — Grand Slams | 2027 | 0.709 | 0.5575 | 0.1894 | 0.784 |
| Ensemble walk-forward (final) — Grand Slams | 2027 | 0.709 | 0.5591 | 0.1897 | 0.783 |
| Ensemble v1 (treino até 2020) — US Open | 505 | 0.745 | 0.5146 | 0.1716 | 0.823 |
| Ensemble walk-forward (final) — US Open | 505 | 0.747 | 0.5193 | 0.1729 | 0.822 |

### Contra o mercado (depois de Roland Garros 2026, n=1258)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Mercado (odds médias) | 1258 | 0.684 | 0.5795 | 0.1989 | 0.759 |
| Ensemble v1 (treino até 2020) | 1258 | 0.656 | 0.6106 | 0.2122 | 0.723 |
| Ensemble walk-forward (final) | 1258 | 0.657 | 0.6088 | 0.2112 | 0.725 |

- Ensemble v1 (treino até 2020) − mercado: **+0.0311** (IC 95%: +0.0190 a +0.0435). Positivo = o mercado é melhor.
- Ensemble walk-forward (final) − mercado: **+0.0293** (IC 95%: +0.0178 a +0.0409). Positivo = o mercado é melhor.
