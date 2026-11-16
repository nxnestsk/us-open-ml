# Fase 4a — Otimização do Elo

Procura em 2012–2022 (nunca vê 2023+). Critério: log loss do Elo com a escala ajustada.

## ATP (74 combinações testadas)

| Elo | Log loss 2012–2022 (procura) | Log loss 2023+ (teste, nunca usado) |
|---|---|---|
| v1 (FiveThirtyEight) | 0.5991 | 0.6296 |
| otimizado | 0.5981 | 0.6280 |

| Parâmetro | v1 | otimizado |
|---|---|---|
| `scale` | 250.000 | 228.460 |
| `shape` | 0.400 | 0.303 |
| `offset` | 5.000 | 3.776 |
| `slam_bonus` | 1.100 | 1.146 |
| `surf_k` | 1.000 | 0.500 |
| `blend_w` | 0.500 | 0.498 |

Melhores 5 combinações (log loss na janela de procura):

- 0.5981: scale=228.46, shape=0.30, offset=3.78, slam_bonus=1.15, surf_k=0.50, blend_w=0.50
- 0.5982: scale=257.88, shape=0.32, offset=3.96, slam_bonus=1.20, surf_k=0.50, blend_w=0.53
- 0.5982: scale=269.85, shape=0.34, offset=4.13, slam_bonus=1.11, surf_k=0.50, blend_w=0.55
- 0.5982: scale=296.48, shape=0.35, offset=4.29, slam_bonus=1.01, surf_k=0.50, blend_w=0.49
- 0.5982: scale=243.44, shape=0.32, offset=4.49, slam_bonus=1.40, surf_k=0.50, blend_w=0.50

## WTA (74 combinações testadas)

| Elo | Log loss 2012–2022 (procura) | Log loss 2023+ (teste, nunca usado) |
|---|---|---|
| v1 (FiveThirtyEight) | 0.6181 | 0.6199 |
| otimizado | 0.6166 | 0.6191 |

| Parâmetro | v1 | otimizado |
|---|---|---|
| `scale` | 250.000 | 297.127 |
| `shape` | 0.400 | 0.375 |
| `offset` | 5.000 | 10.839 |
| `slam_bonus` | 1.100 | 1.214 |
| `surf_k` | 1.000 | 0.922 |
| `blend_w` | 0.500 | 0.662 |

Melhores 5 combinações (log loss na janela de procura):

- 0.6166: scale=297.13, shape=0.38, offset=10.84, slam_bonus=1.21, surf_k=0.92, blend_w=0.66
- 0.6167: scale=314.74, shape=0.38, offset=9.57, slam_bonus=1.04, surf_k=0.96, blend_w=0.70
- 0.6167: scale=278.84, shape=0.36, offset=9.11, slam_bonus=1.00, surf_k=0.96, blend_w=0.67
- 0.6167: scale=362.44, shape=0.38, offset=9.65, slam_bonus=1.05, surf_k=0.69, blend_w=0.75
- 0.6167: scale=307.30, shape=0.38, offset=6.70, slam_bonus=1.00, surf_k=1.00, blend_w=0.73
