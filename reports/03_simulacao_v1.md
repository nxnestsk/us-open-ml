# Fase 3 — Simulação do quadro e backtest

Gerado por `scripts/03_simulate.py`.
20,000 simulações por torneio. O modelo só usa informação anterior ao início de cada torneio.

## ATP — campeão real vs probabilidade do modelo

| Ano | Campeão | P(título) modelo | Posição | P(título) Elo | Posição Elo | Favorito do modelo (P) | LogLoss rondas modelo | LogLoss rondas Elo |
|---|---|---|---|---|---|---|---|---|
| 2023 | Novak Djokovic | 31.0% | 1 | 50.0% | 1 | Novak Djokovic (31.0%) | 0.2280 | 0.2334 |
| 2024 | Jannik Sinner | 37.3% | 1 | 32.6% | 2 | Jannik Sinner (37.3%) | 0.2108 | 0.2226 |
| 2025 | Carlos Alcaraz | 23.2% | 2 | 19.7% | 2 | Jannik Sinner (50.4%) | 0.2349 | 0.2372 |
| 2026 | Alexander Zverev | 14.1% | 2 | 9.1% | 3 | Carlos Alcaraz (40.4%) | 0.2306 | 0.2472 |

### ATP — perspetiva (quadro sorteado ao acaso, top 128 do ranking atual, piso duro)

| Jogador | Ranking | P(título) |
|---|---|---|
| Jannik Sinner | 1 | 40.0% |
| Carlos Alcaraz | 3 | 27.0% |
| Alexander Zverev | 1 | 7.0% |
| Ben Shelton | 4 | 4.3% |
| Novak Djokovic | 11 | 3.5% |
| Daniil Medvedev | 5 | 2.2% |
| Taylor Fritz | 12 | 2.0% |
| Arthur Fils | 7 | 1.8% |
| Alex De Minaur | 9 | 1.7% |
| Tommy Paul | 14 | 1.2% |

## WTA — campeão real vs probabilidade do modelo

| Ano | Campeão | P(título) modelo | Posição | P(título) Elo | Posição Elo | Favorito do modelo (P) | LogLoss rondas modelo | LogLoss rondas Elo |
|---|---|---|---|---|---|---|---|---|
| 2023 | Coco Gauff | 9.9% | 3 | 7.3% | 4 | Iga Swiatek (32.5%) | 0.2041 | 0.2143 |
| 2024 | Aryna Sabalenka | 19.9% | 2 | 17.7% | 2 | Iga Swiatek (40.0%) | 0.2136 | 0.2263 |
| 2025 | Aryna Sabalenka | 19.2% | 2 | 24.2% | 2 | Iga Swiatek (31.1%) | 0.2195 | 0.2229 |
| 2026 | Elena Rybakina | 21.3% | 2 | 17.3% | 2 | Aryna Sabalenka (24.2%) | 0.1693 | 0.1747 |

### WTA — perspetiva (quadro sorteado ao acaso, top 128 do ranking atual, piso duro)

| Jogador | Ranking | P(título) |
|---|---|---|
| Elena Rybakina | 1 | 22.0% |
| Aryna Sabalenka | 2 | 14.9% |
| Coco Gauff | 4 | 10.1% |
| Iga Swiatek | 9 | 8.2% |
| Jessica Pegula | 4 | 5.3% |
| Mirra Andreeva | 5 | 5.0% |
| Diana Shnaider | 13 | 4.9% |
| Karolina Muchova | 7 | 3.5% |
| Elina Svitolina | 6 | 3.3% |
| Linda Noskova | 6 | 2.8% |
