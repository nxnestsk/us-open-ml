"""Front end do projeto US Open ML. Correr com: streamlit run app/streamlit_app.py"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
APP_DIR = Path(__file__).resolve().parent

ROUND_LABELS = ["2.ª ronda", "3.ª ronda", "Oitavos", "Quartos", "Meias", "Final", "Campeão"]
RESULT_LABELS = ["Perdeu na 1.ª ronda", "Perdeu na 2.ª ronda", "Perdeu na 3.ª ronda", "Perdeu nos oitavos",
                 "Perdeu nos quartos", "Perdeu nas meias", "Perdeu a final", "Campeão"]

st.set_page_config(page_title="US Open ML", layout="wide")


@st.cache_data
def load_sim(tour: str) -> pd.DataFrame:
    return pd.read_csv(PROC / f"sim_{tour}.csv")


@st.cache_data
def load_pairs(tour: str) -> pd.DataFrame:
    return pd.read_csv(PROC / f"pairs_{tour}.csv")


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(PROC / "backtest_summary.csv")


@st.cache_data
def load_elo(tour: str) -> pd.DataFrame:
    cols = ["date", "year", "winner_id", "loser_id", "surface", "tourney_level", "is_walkover",
            "w_elo", "l_elo", "w_selo", "l_selo"]
    return pd.read_csv(PROC / f"{tour}_elo.csv.gz", usecols=cols, parse_dates=["date"])


def read_report(name: str) -> str:
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else "_Relatório ainda não gerado._"


def group_label(g: str) -> str:
    year = g.split("-", 1)[1]
    return "Perspetiva (quadro sorteado ao acaso)" if year == "outlook" else f"US Open {year} (backtest)"


st.sidebar.title("US Open ML")
page = st.sidebar.radio("Página", ["Resumo", "Simulação", "Confronto direto", "Dados e Elo", "Modelos"])
st.sidebar.caption("Probabilidades estimadas por um ensemble (regressão logística + LightGBM) "
                   "treinado ano a ano só com dados anteriores a cada torneio, com features sem fuga de dados.")

# ============================================================== Resumo
if page == "Resumo":
    s = load_summary()
    # Dados para a página animada (app/resumo.html). Tudo vem do backtest já gerado.
    rows = []
    for r in s.itertuples():
        rows.append(dict(tour=r.tour, year=int(r.year), champion=r.champion, pm=float(r.p_title_model),
                         pe=float(r.p_title_elo), rank_model=int(r.rank_model), favourite=r.favourite_model,
                         p_fav=float(r.p_fav_model), fav_is_champion=bool(r.favourite_model == r.champion)))
    kpi = dict(n=len(s), top2=int((s.rank_model <= 2).sum()), mean=float(s.p_title_model.mean()),
               better=int((s.ll_model < s.ll_elo).sum()))
    html = (APP_DIR / "resumo.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(dict(rows=rows, kpi=kpi), ensure_ascii=False))
    # Só nesta página: o azul-noite da página animada estende-se à área principal (a barra lateral não muda).
    st.markdown(
        """<style>
        [data-testid="stApp"], [data-testid="stMain"], [data-testid="stHeader"] { background: #081630 !important; }
        [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.iframe(html)  # altura automática (height="content")
    with st.expander("Tabela completa do backtest"):
        t = s.assign(
            tour=s.tour.str.upper(),
            p_title_model=(s.p_title_model * 100).round(1), p_title_elo=(s.p_title_elo * 100).round(1),
            p_fav_model=(s.p_fav_model * 100).round(1),
        )[["tour", "year", "champion", "p_title_model", "rank_model", "p_title_elo", "rank_elo",
           "favourite_model", "p_fav_model"]]
        t.columns = ["Circuito", "Ano", "Campeão real", "P(título) modelo %", "Posição modelo",
                     "P(título) Elo %", "Posição Elo", "Favorito do modelo", "P(favorito) %"]
        st.dataframe(t, hide_index=True, width="stretch")

# ============================================================ Simulação
elif page == "Simulação":
    st.title("Simulação do quadro")
    c1, c2, c3 = st.columns([1, 2, 1])
    tour = c1.selectbox("Circuito", ["atp", "wta"], format_func=str.upper)
    sim = load_sim(tour)
    g = c2.selectbox("Torneio", sorted(sim.group.unique(), reverse=True), format_func=group_label, key=f"sim_group_{tour}")
    topn = c3.slider("Jogadores a mostrar", 5, 30, 12)
    d = sim[sim.group == g].copy()
    d["p_title_model"], d["p_title_elo"] = d.pm_7, d.pe_7
    show = d.sort_values("p_title_model", ascending=False).head(topn)

    fav = show.iloc[0]
    m = st.columns(3)
    m[0].metric("Favorito do modelo", fav["name"], f"{fav.p_title_model:.1%}")
    if "actual_wins" in d and d.actual_wins.notna().any():
        champ = d[d.actual_wins == 7].iloc[0]
        pos = int((d.p_title_model > champ.p_title_model).sum()) + 1
        m[1].metric("Campeão real", champ["name"], f"{champ.p_title_model:.1%} (posição {pos})")
        m[2].metric("Probabilidade do Elo para o campeão", f"{champ.p_title_elo:.1%}")
    else:
        m[1].metric("Tipo", "Perspetiva", help="Top 128 do ranking, sem cabeças de série, piso duro")

    long = show.melt(id_vars="name", value_vars=["p_title_model", "p_title_elo"], var_name="modelo", value_name="p")
    long["modelo"] = long.modelo.map({"p_title_model": "Ensemble", "p_title_elo": "Só Elo"})
    chart = alt.Chart(long).mark_bar().encode(
        y=alt.Y("name:N", sort=show["name"].tolist(), title=None),
        x=alt.X("p:Q", title="Probabilidade de ser campeão", axis=alt.Axis(format="%")),
        yOffset="modelo:N", color=alt.Color("modelo:N", title=None),
        tooltip=["name", "modelo", alt.Tooltip("p:Q", format=".1%")],
    ).properties(height=max(300, topn * 34))
    st.altair_chart(chart, width="stretch")

    st.subheader("Probabilidade de chegar a cada ronda (ensemble)")
    tab = show[["name", "rank"] + [f"pm_{k}" for k in range(1, 8)]].copy()
    tab.columns = ["Jogador", "Ranking"] + ROUND_LABELS
    if "actual_wins" in show and show.actual_wins.notna().any():
        tab["Resultado real"] = show.actual_wins.astype(int).map(dict(enumerate(RESULT_LABELS))).values
    fmt = {c: "{:.1%}" for c in ROUND_LABELS}
    st.dataframe(tab.style.format(fmt).background_gradient(subset=ROUND_LABELS, cmap="Greens", vmin=0, vmax=1),
                 hide_index=True, width="stretch")
    st.caption("20 000 simulações. Probabilidades de chegar à ronda indicada, dado o quadro do torneio.")

# ======================================================= Confronto direto
elif page == "Confronto direto":
    st.title("Confronto direto")
    c1, c2 = st.columns(2)
    tour = c1.selectbox("Circuito", ["atp", "wta"], format_func=str.upper)
    pairs = load_pairs(tour)
    g = c2.selectbox("Contexto (estado dos ratings na véspera de…)", sorted(pairs.group.unique(), reverse=True),
                     format_func=group_label, key=f"pair_group_{tour}")
    pg = pairs[pairs.group == g]
    players = sorted(set(pg.p1_name) | set(pg.p2_name))
    a, b = st.columns(2)
    pa = a.selectbox("Jogador A", players, index=0, key=f"pa_{g}")
    pb = b.selectbox("Jogador B", players, index=1, key=f"pb_{g}")
    if pa == pb:
        st.warning("Escolhe dois jogadores diferentes.")
    else:
        r = pg[((pg.p1_name == pa) & (pg.p2_name == pb)) | ((pg.p1_name == pb) & (pg.p2_name == pa))].iloc[0]
        flip = r.p1_name != pa
        pm = 1 - r.p_model if flip else r.p_model
        pe = 1 - r.p_elo if flip else r.p_elo
        c = st.columns(2)
        c[0].metric(f"P({pa} vence) — ensemble", f"{pm:.1%}")
        c[1].metric(f"P({pa} vence) — só Elo", f"{pe:.1%}")
        st.progress(float(pm), text=f"{pa} {pm:.0%}  ·  {pb} {1 - pm:.0%}")
        st.caption("Piso duro, melhor de 5 sets (ATP) ou 3 (WTA), Grand Slam, com o estado dos jogadores "
                   "na véspera do torneio escolhido.")

# ============================================================ Dados e Elo
elif page == "Dados e Elo":
    st.title("Dados e evolução do Elo")
    tour = st.selectbox("Circuito", ["atp", "wta"], format_func=str.upper)
    if not (PROC / f"{tour}_elo.csv.gz").exists():
        st.info("Esta página usa ficheiros calculados que não estão no repositório. "
                "Corre `scripts/00_download_data.py` e depois `scripts/01_eda_elo.py` (ver o README) e recarrega a página.")
        st.stop()
    elo = load_elo(tour)
    st.write(f"**{len(elo):,}** jogos, de {elo.date.min().date()} a {elo.date.max().date()}. "
             "Fontes: Sackmann (espelho) até Roland Garros 2026; tennis-data.co.uk depois.")
    per_year = elo.groupby("year").size().rename("Jogos").reset_index()
    st.altair_chart(alt.Chart(per_year).mark_bar().encode(x="year:O", y="Jogos:Q",
                                                         tooltip=["year", "Jogos"]), width="stretch")

    sim = load_sim(tour)
    names = (sim[["id", "name", "pm_7"]].sort_values("pm_7", ascending=False)
             .drop_duplicates("id").set_index("name")["id"])
    chosen = st.multiselect("Jogadores", names.index.tolist(), default=names.index.tolist()[:3])
    kind = st.radio("Rating", ["Mistura (geral + superfície)", "Geral"], horizontal=True)
    frames = []
    for n in chosen:
        pid = int(names[n])
        w = elo[elo.winner_id == pid][["date", "w_elo", "w_selo"]].set_axis(["date", "elo", "selo"], axis=1)
        l = elo[elo.loser_id == pid][["date", "l_elo", "l_selo"]].set_axis(["date", "elo", "selo"], axis=1)
        x = pd.concat([w, l]).sort_values("date")
        x["Elo"] = x.elo if kind == "Geral" else (x.elo + x.selo) / 2
        x["Jogador"] = n
        frames.append(x[["date", "Elo", "Jogador"]])
    if frames:
        data = pd.concat(frames)
        data = data[data.date >= "2015-01-01"]
        st.altair_chart(alt.Chart(data).mark_line().encode(
            x="date:T", y=alt.Y("Elo:Q", scale=alt.Scale(zero=False)), color="Jogador:N",
            tooltip=["Jogador", "date", alt.Tooltip("Elo:Q", format=".0f")]), width="stretch")
        st.caption("Rating **antes** de cada jogo (desde 2015). Os ids sem correspondência entre fontes "
                   "aparecem só no período em que a fonte os identifica.")

# =============================================================== Modelos
else:
    st.title("Modelos e avaliação")
    st.write("Relatórios gerados pelos scripts do projeto. O modelo final é treinado ano a ano (walk-forward), "
             "sempre só com o passado; o teste é 2023 em diante. A versão 1 (treino fixo até 2020) fica como comparação.")
    t1, t2, t3, t4, t5 = st.tabs(["Modelo final", "Elo: baseline", "Elo: afinação", "Versão 1 (treino até 2020)",
                                  "Simulação e backtest"])
    t1.markdown(read_report("05_walkforward.md"))
    t2.markdown(read_report("01_eda_elo.md"))
    t3.markdown(read_report("04_elo_tuning.md"))
    t4.markdown(read_report("02_modelos.md"))
    t5.markdown(read_report("03_simulacao.md"))
