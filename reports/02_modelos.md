# Fase 2 — Features e modelos

Gerado por `scripts/02_models.py`.

## ATP

Treino 1995–2020: 80,930 jogos · validação 2021–2022: 5,613 · teste 2023+: 11,638

### Ablação: o que cada grupo de features acrescenta (regressão logística, teste)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| só Elo mistura | 11638 | 0.637 | 0.6302 | 0.2203 | 0.699 |
| + Elo geral/superfície/experiência | 11638 | 0.637 | 0.6298 | 0.2201 | 0.699 |
| + ranking e pontos | 11638 | 0.647 | 0.6198 | 0.2160 | 0.711 |
| + forma e fadiga | 11638 | 0.648 | 0.6187 | 0.2156 | 0.712 |
| + serviço e devolução | 11638 | 0.650 | 0.6173 | 0.2150 | 0.714 |
| + idade, altura, mão, H2H | 11638 | 0.653 | 0.6151 | 0.2141 | 0.717 |
| + contexto (Slam, 5 sets, piso) | 11638 | 0.652 | 0.6142 | 0.2137 | 0.718 |

### Modelos finais (LightGBM parou em 641 árvores)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Elo mistura (baseline) — todos | 11638 | 0.637 | 0.6398 | 0.2232 | 0.699 |
| Regressão logística — todos | 11638 | 0.652 | 0.6142 | 0.2137 | 0.718 |
| LightGBM — todos | 11638 | 0.655 | 0.6123 | 0.2128 | 0.721 |
| Ensemble (média) — todos | 11638 | 0.654 | 0.6122 | 0.2129 | 0.720 |
| Ensemble calibrado — todos | 11638 | 0.646 | 0.6149 | 0.2132 | 0.719 |
| Elo mistura (baseline) — Grand Slams | 2021 | 0.698 | 0.5648 | 0.1917 | 0.780 |
| Regressão logística — Grand Slams | 2021 | 0.721 | 0.5455 | 0.1842 | 0.796 |
| LightGBM — Grand Slams | 2021 | 0.724 | 0.5437 | 0.1830 | 0.799 |
| Ensemble (média) — Grand Slams | 2021 | 0.721 | 0.5436 | 0.1833 | 0.798 |
| Ensemble calibrado — Grand Slams | 2021 | 0.714 | 0.5526 | 0.1841 | 0.798 |
| Elo mistura (baseline) — US Open | 507 | 0.673 | 0.5870 | 0.2018 | 0.758 |
| Regressão logística — US Open | 507 | 0.686 | 0.5693 | 0.1948 | 0.774 |
| LightGBM — US Open | 507 | 0.696 | 0.5672 | 0.1931 | 0.778 |
| Ensemble (média) — US Open | 507 | 0.688 | 0.5662 | 0.1936 | 0.776 |
| Ensemble calibrado — US Open | 507 | 0.680 | 0.5724 | 0.1938 | 0.776 |

**Ensemble − Elo (log loss, todos os jogos de teste): -0.0276 (IC 95%: -0.0321 a -0.0237)**. Negativo = melhor.

### Contra o mercado (jogos depois de Roland Garros 2026 com odds, n=1256)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Mercado (odds médias) | 1256 | 0.652 | 0.6166 | 0.2146 | 0.715 |
| Elo mistura (baseline) | 1256 | 0.633 | 0.6437 | 0.2247 | 0.694 |
| Regressão logística | 1256 | 0.646 | 0.6268 | 0.2195 | 0.697 |
| LightGBM | 1256 | 0.645 | 0.6245 | 0.2181 | 0.705 |
| Ensemble (média) | 1256 | 0.639 | 0.6228 | 0.2178 | 0.704 |
| Ensemble calibrado | 1256 | 0.634 | 0.6257 | 0.2190 | 0.701 |

**Ensemble − mercado (log loss): +0.0063 (IC 95%: -0.0054 a +0.0176)**. Positivo = o mercado continua melhor.

### Calibração do ensemble (média, sem isotónica) no teste

| Prob. prevista | Jogos | Média prevista | Frequência observada |
|---|---|---|---|
| 0.0–0.1 | 478 | 0.067 | 0.065 |
| 0.1–0.2 | 1381 | 0.157 | 0.159 |
| 0.2–0.3 | 2697 | 0.254 | 0.282 |
| 0.3–0.4 | 3356 | 0.351 | 0.384 |
| 0.4–0.5 | 3725 | 0.451 | 0.464 |
| 0.5–0.6 | 3727 | 0.549 | 0.536 |
| 0.6–0.7 | 3356 | 0.649 | 0.616 |
| 0.7–0.8 | 2697 | 0.746 | 0.718 |
| 0.8–0.9 | 1381 | 0.843 | 0.841 |
| 0.9–1.0 | 478 | 0.933 | 0.935 |

### Importância das features no LightGBM (ganho, top 10)

| Feature | Importância |
|---|---|
| `blend_diff` | 55.6% |
| `pts_log_diff` | 9.6% |
| `selo_diff` | 9.3% |
| `elo_diff` | 3.8% |
| `rank_log_diff` | 3.7% |
| `spw_diff` | 3.4% |
| `age_diff` | 3.2% |
| `rpw_diff` | 1.8% |
| `blend_x_bo5_diff` | 1.6% |
| `ht_diff` | 1.3% |

## WTA

Treino 1995–2020: 71,598 jogos · validação 2021–2022: 5,137 · teste 2023+: 10,780

### Ablação: o que cada grupo de features acrescenta (regressão logística, teste)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| só Elo mistura | 10780 | 0.647 | 0.6201 | 0.2160 | 0.711 |
| + Elo geral/superfície/experiência | 10780 | 0.649 | 0.6193 | 0.2156 | 0.712 |
| + ranking e pontos | 10780 | 0.658 | 0.6113 | 0.2122 | 0.722 |
| + forma e fadiga | 10780 | 0.660 | 0.6092 | 0.2113 | 0.725 |
| + serviço e devolução | 10780 | 0.664 | 0.6082 | 0.2109 | 0.726 |
| + idade, altura, mão, H2H | 10780 | 0.666 | 0.6070 | 0.2102 | 0.728 |
| + contexto (Slam, 5 sets, piso) | 10780 | 0.666 | 0.6069 | 0.2102 | 0.728 |

### Modelos finais (LightGBM parou em 861 árvores)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Elo mistura (baseline) — todos | 10780 | 0.647 | 0.6253 | 0.2177 | 0.711 |
| Regressão logística — todos | 10780 | 0.666 | 0.6069 | 0.2102 | 0.728 |
| LightGBM — todos | 10780 | 0.663 | 0.6066 | 0.2103 | 0.729 |
| Ensemble (média) — todos | 10780 | 0.664 | 0.6060 | 0.2099 | 0.729 |
| Ensemble calibrado — todos | 10780 | 0.619 | 0.6090 | 0.2106 | 0.729 |
| Elo mistura (baseline) — Grand Slams | 2027 | 0.689 | 0.5750 | 0.1960 | 0.770 |
| Regressão logística — Grand Slams | 2027 | 0.710 | 0.5587 | 0.1898 | 0.783 |
| LightGBM — Grand Slams | 2027 | 0.705 | 0.5577 | 0.1894 | 0.784 |
| Ensemble (média) — Grand Slams | 2027 | 0.709 | 0.5575 | 0.1894 | 0.784 |
| Ensemble calibrado — Grand Slams | 2027 | 0.676 | 0.5583 | 0.1896 | 0.784 |
| Elo mistura (baseline) — US Open | 505 | 0.739 | 0.5360 | 0.1796 | 0.807 |
| Regressão logística — US Open | 505 | 0.749 | 0.5170 | 0.1723 | 0.822 |
| LightGBM — US Open | 505 | 0.741 | 0.5135 | 0.1715 | 0.824 |
| Ensemble (média) — US Open | 505 | 0.745 | 0.5146 | 0.1716 | 0.823 |
| Ensemble calibrado — US Open | 505 | 0.719 | 0.5187 | 0.1730 | 0.824 |

**Ensemble − Elo (log loss, todos os jogos de teste): -0.0194 (IC 95%: -0.0233 a -0.0157)**. Negativo = melhor.

### Contra o mercado (jogos depois de Roland Garros 2026 com odds, n=1258)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Mercado (odds médias) | 1258 | 0.684 | 0.5795 | 0.1989 | 0.759 |
| Elo mistura (baseline) | 1258 | 0.658 | 0.6234 | 0.2164 | 0.715 |
| Regressão logística | 1258 | 0.660 | 0.6098 | 0.2117 | 0.724 |
| LightGBM | 1258 | 0.655 | 0.6134 | 0.2133 | 0.722 |
| Ensemble (média) | 1258 | 0.656 | 0.6106 | 0.2122 | 0.723 |
| Ensemble calibrado | 1258 | 0.616 | 0.6131 | 0.2130 | 0.721 |

**Ensemble − mercado (log loss): +0.0311 (IC 95%: +0.0190 a +0.0435)**. Positivo = o mercado continua melhor.

### Calibração do ensemble (média, sem isotónica) no teste

| Prob. prevista | Jogos | Média prevista | Frequência observada |
|---|---|---|---|
| 0.0–0.1 | 435 | 0.072 | 0.055 |
| 0.1–0.2 | 1478 | 0.155 | 0.175 |
| 0.2–0.3 | 2378 | 0.253 | 0.267 |
| 0.3–0.4 | 3067 | 0.352 | 0.376 |
| 0.4–0.5 | 3421 | 0.451 | 0.454 |
| 0.5–0.6 | 3423 | 0.549 | 0.546 |
| 0.6–0.7 | 3067 | 0.648 | 0.624 |
| 0.7–0.8 | 2378 | 0.747 | 0.733 |
| 0.8–0.9 | 1478 | 0.845 | 0.825 |
| 0.9–1.0 | 435 | 0.928 | 0.945 |

### Importância das features no LightGBM (ganho, top 10)

| Feature | Importância |
|---|---|
| `blend_diff` | 43.0% |
| `elo_diff` | 19.5% |
| `pts_log_diff` | 12.0% |
| `rank_log_diff` | 5.0% |
| `age_diff` | 2.6% |
| `selo_diff` | 2.5% |
| `exp_diff` | 2.1% |
| `ht_diff` | 2.0% |
| `spw_diff` | 2.0% |
| `m30_diff` | 1.5% |
