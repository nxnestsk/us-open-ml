# US Open ML — Plano e decisões

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

## Melhorias adicionais (para além do vídeo)
- **Ratings mais ricos:** Elo com K variável (mais volátil para novos jogadores), Glicko-2 (incerteza do rating) e Elo por superfície com mistura do Elo geral.
- **Fadiga e calendário:** minutos jogados nos últimos 7/14 dias, dias de descanso, sets longos, viagem entre torneios.
- **Matchups:** mão dominante, altura, estilo (serviço vs. devolução), head-to-head com suavização bayesiana.
- **Features só com informação anterior ao jogo**, com testes automáticos que garantem que não há fuga de dados.
- **Ensemble:** logística + LightGBM/XGBoost, com calibração (isotónica/Platt) e média das probabilidades.
- **Incerteza no campeão:** intervalos de confiança na simulação e análise de sensibilidade (e se o jogador X estiver lesionado?).
- **Interpretabilidade:** importância das features e SHAP para estudar o que o modelo aprendeu.
- **Honestidade estatística:** o mercado de apostas é muito eficiente; o objetivo realista é igualar ou aproximar as odds, não "ganhar-lhes" de forma garantida.

## Fontes de dados
- Jeff Sackmann `tennis_atp` e `tennis_wta` (GitHub, licença CC BY-NC-SA: uso não comercial, citar autor).
- Odds históricas: tennis-data.co.uk (fase posterior).

## Fases
1. Estrutura e dados
2. Exploração (EDA)
3. Elo (geral, por superfície)
4. Features sem fuga de dados (só informação anterior ao jogo)
5. Modelos e validação temporal
6. Simulação do quadro
7. Backtest em US Opens passados
8. Dashboard

Cada fase gera uma secção em `reports/`.
