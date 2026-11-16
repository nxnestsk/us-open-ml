> **Documento histórico.** Este é o plano do início do projeto. Algumas ideias não foram feitas (Glicko-2, SHAP, minutos jogados) ou deram resultados diferentes do esperado. O estado final e os resultados estão no [README](../README.md) e nos relatórios `01` a `05`.

# US Open ML — Plano inicial

## Objetivo
Estimar a probabilidade de cada jogador ganhar o US Open (quadro masculino e feminino), com um processo reprodutível e estudável passo a passo.

## Inspiração e onde divergimos do vídeo
O vídeo usa features de Elo (geral, por superfície, por recinto), ranking/pontos em log, estatísticas de serviço e uma regressão logística avaliada com accuracy, Brier, log loss e AUC. Mantemos essa base e acrescentamos:

| Vídeo | Este projeto | Porquê |
|---|---|---|
| Só prever o jogo | Prever o jogo **e simular o quadro** (Monte Carlo) | O que interessa é o campeão, não o jogo isolado |
| Split aleatório (provável) | **Validação temporal** (treinar no passado, testar no futuro) | Evita fuga de dados; o ténis é uma série temporal |
| Só accuracy/AUC | **Brier + log loss + calibração** | Para simular, as probabilidades têm de estar bem calibradas |
| Um modelo | Baseline Elo → logística → gradient boosting | Mostra o ganho de cada passo |
| — | Comparação com odds de mercado (tennis-data.co.uk) | Benchmark difícil e honesto |

## Ideias planeadas (o que foi feito e o que não)
- **Feito:** Elo com K variável e por superfície com mistura; features só com informação anterior ao jogo, com testes automáticos anti-fuga; ensemble de logística e LightGBM; intervalos de confiança por bootstrap; comparação com o mercado.
- **Feito, com ganho pequeno ou não provado:** afinação do Elo, treino walk-forward com pesos temporais, indicadores aproximados de risco de lesão (desistências).
- **Não feito:** Glicko-2, SHAP, minutos jogados e viagens, previsão com intervalos de credibilidade, sensibilidade a lesões.
- **Honestidade estatística:** o mercado de apostas é muito eficiente; o objetivo realista é aproximar as odds, não "ganhar-lhes" de forma garantida.

## Fontes de dados
- Jeff Sackmann `tennis_atp` e `tennis_wta` (GitHub, licença CC BY-NC-SA: uso não comercial, citar autor).
- Odds e resultados recentes: tennis-data.co.uk.

## Fases
1. Estrutura e dados
2. Exploração (EDA)
3. Elo (geral, por superfície)
4. Features sem fuga de dados (só informação anterior ao jogo)
5. Modelos e validação temporal
6. Simulação do quadro
7. Backtest em US Opens passados
8. Dashboard

Cada fase gera um relatório em `reports/`.
