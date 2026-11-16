# Fase 1 — Exploração e baseline Elo

Gerado por `scripts/01_eda_elo.py`.

## ATP

- Jogos totais carregados: **135,226** (1985-01-07 a 2026-11-08)
- Jogos de teste (>= 2023, sem walkovers): **11,638**

### Superfícies (todos os anos desde 1985)

|    |   Hard |   Clay |   Grass |   Carpet |
|:---|-------:|-------:|--------:|---------:|
| %  |     49 |   33.5 |     9.8 |      7.7 |

### Baseline: quanto acerta cada rating (período de teste)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Elo overall — todos | 11638 | 0.640 | 0.6456 | 0.2250 | 0.697 |
| Elo surface — todos | 11638 | 0.630 | 0.6552 | 0.2291 | 0.687 |
| Elo blend — todos | 11638 | 0.637 | 0.6398 | 0.2232 | 0.699 |
| Elo overall — Grand Slams | 2021 | 0.699 | 0.5744 | 0.1945 | 0.775 |
| Elo surface — Grand Slams | 2021 | 0.690 | 0.5790 | 0.1977 | 0.767 |
| Elo blend — Grand Slams | 2021 | 0.698 | 0.5648 | 0.1917 | 0.780 |
| Elo overall — US Open | 507 | 0.675 | 0.5831 | 0.1993 | 0.764 |
| Elo surface — US Open | 507 | 0.675 | 0.6075 | 0.2099 | 0.742 |
| Elo blend — US Open | 507 | 0.673 | 0.5870 | 0.2018 | 0.758 |

- Baseline ingénuo (melhor ranking ganha): accuracy **0.638** em 11,447 jogos.

### Elo vs mercado (jogos pós-Roland Garros 2026 com odds, n=1256)

| Modelo | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Mercado (odds médias) | 1256 | 0.652 | 0.6166 | 0.2146 | 0.715 |
| Elo blend | 1256 | 0.633 | 0.6437 | 0.2247 | 0.694 |

## WTA

- Jogos totais carregados: **115,296** (1985-01-02 a 2026-11-08)
- Jogos de teste (>= 2023, sem walkovers): **10,780**

### Superfícies (todos os anos desde 1985)

|    |   Hard |   Clay |   Grass |   Carpet |
|:---|-------:|-------:|--------:|---------:|
| %  |   53.5 |     30 |    10.7 |      5.8 |

### Baseline: quanto acerta cada rating (período de teste)

| Modelo / conjunto | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Elo overall — todos | 10780 | 0.647 | 0.6302 | 0.2193 | 0.709 |
| Elo surface — todos | 10780 | 0.632 | 0.6396 | 0.2234 | 0.697 |
| Elo blend — todos | 10780 | 0.647 | 0.6253 | 0.2177 | 0.711 |
| Elo overall — Grand Slams | 2027 | 0.694 | 0.5796 | 0.1971 | 0.769 |
| Elo surface — Grand Slams | 2027 | 0.680 | 0.5916 | 0.2030 | 0.754 |
| Elo blend — Grand Slams | 2027 | 0.689 | 0.5750 | 0.1960 | 0.770 |
| Elo overall — US Open | 505 | 0.737 | 0.5360 | 0.1800 | 0.806 |
| Elo surface — US Open | 505 | 0.737 | 0.5464 | 0.1831 | 0.799 |
| Elo blend — US Open | 505 | 0.739 | 0.5360 | 0.1796 | 0.807 |

- Baseline ingénuo (melhor ranking ganha): accuracy **0.634** em 10,662 jogos.

### Elo vs mercado (jogos pós-Roland Garros 2026 com odds, n=1258)

| Modelo | Jogos | Accuracy | LogLoss | Brier | AUC |
|---|---|---|---|---|---|
| Mercado (odds médias) | 1258 | 0.684 | 0.5795 | 0.1989 | 0.759 |
| Elo blend | 1258 | 0.658 | 0.6234 | 0.2164 | 0.715 |
