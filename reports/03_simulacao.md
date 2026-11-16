# Fase 3 — Simulação do quadro e backtest

Gerado por `scripts/03_simulate.py`.
20,000 simulações por torneio. O modelo só usa informação anterior ao início de cada torneio.

## ATP — campeão real vs probabilidade do modelo

| Ano | Campeão | P(título) modelo | Posição | P(título) Elo | Posição Elo | Favorito do modelo (P) | LogLoss rondas modelo | LogLoss rondas Elo |
|---|---|---|---|---|---|---|---|---|
| 2023 | Novak Djokovic | 36.8% | 1 | 53.8% | 1 | Novak Djokovic (36.8%) | 0.2257 | 0.2277 |
| 2024 | Jannik Sinner | 37.9% | 1 | 32.2% | 2 | Jannik Sinner (37.9%) | 0.2113 | 0.2211 |
| 2025 | Carlos Alcaraz | 27.1% | 2 | 23.6% | 2 | Jannik Sinner (47.1%) | 0.2312 | 0.2302 |
| 2026 | Alexander Zverev | 15.5% | 2 | 10.8% | 3 | Carlos Alcaraz (35.3%) | 0.2328 | 0.2452 |

### ATP — perspetiva (quadro sorteado ao acaso, top 128 do ranking atual, piso duro)

| Jogador | Ranking | P(título) |
|---|---|---|
| Jannik Sinner | 1 | 41.0% |
| Carlos Alcaraz | 3 | 28.8% |
| Alexander Zverev | 1 | 6.2% |
| Ben Shelton | 4 | 4.3% |
| Novak Djokovic | 11 | 3.6% |
| Daniil Medvedev | 5 | 2.1% |
| Taylor Fritz | 12 | 1.8% |
| Alex De Minaur | 9 | 1.6% |
| Arthur Fils | 7 | 1.6% |
| Tommy Paul | 14 | 1.2% |

## WTA — campeão real vs probabilidade do modelo

| Ano | Campeão | P(título) modelo | Posição | P(título) Elo | Posição Elo | Favorito do modelo (P) | LogLoss rondas modelo | LogLoss rondas Elo |
|---|---|---|---|---|---|---|---|---|
| 2023 | Coco Gauff | 10.5% | 3 | 8.2% | 3 | Iga Swiatek (27.5%) | 0.2023 | 0.2084 |
| 2024 | Aryna Sabalenka | 18.1% | 2 | 19.5% | 2 | Iga Swiatek (33.7%) | 0.2140 | 0.2253 |
| 2025 | Aryna Sabalenka | 17.6% | 2 | 26.3% | 2 | Iga Swiatek (26.7%) | 0.2218 | 0.2262 |
| 2026 | Elena Rybakina | 16.8% | 2 | 16.0% | 2 | Aryna Sabalenka (21.1%) | 0.1740 | 0.1745 |

### WTA — perspetiva (quadro sorteado ao acaso, top 128 do ranking atual, piso duro)

| Jogador | Ranking | P(título) |
|---|---|---|
| Elena Rybakina | 1 | 17.8% |
| Aryna Sabalenka | 2 | 13.2% |
| Coco Gauff | 4 | 11.1% |
| Iga Swiatek | 9 | 7.0% |
| Jessica Pegula | 4 | 5.7% |
| Mirra Andreeva | 5 | 5.1% |
| Diana Shnaider | 13 | 5.0% |
| Karolina Muchova | 7 | 4.0% |
| Linda Noskova | 6 | 3.7% |
| Elina Svitolina | 6 | 3.3% |
