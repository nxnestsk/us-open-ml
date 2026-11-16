"""Carregamento e junção dos dados de jogos ATP/WTA.

Fontes abertas:
- Sackmann (espelho): resultados detalhados até Roland Garros 2026 (inclusive).
- tennis-data.co.uk: resultados + odds de 2026, usados para o período
  seguinte a Roland Garros (Wimbledon, US Open, Masters, ...).
  Esta fonte NÃO tem estatísticas de serviço nem minutos.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sackmann_mirror"
TDUK = ROOT / "data" / "raw" / "tennis_data_uk"

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 4, "QF": 5, "SF": 6, "F": 7}
TD_ROUNDS = {"1st Round": None, "2nd Round": None, "3rd Round": None, "4th Round": None,
             "Quarterfinals": "QF", "Semifinals": "SF", "The Final": "F", "Round Robin": "RR"}
TD_LEVEL = {"Grand Slam": "G", "Masters 1000": "M", "ATP500": "A", "ATP250": "A",
            "Masters Cup": "F", "WTA1000": "PM", "WTA500": "P", "WTA250": "I",
            "Premier": "P", "Premier 5": "PM", "Premier Mandatory": "PM", "International": "I",
            "WTA Finals": "F"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def _load_sackmann(tour: str, min_year: int) -> pd.DataFrame:
    files = sorted((RAW / tour).glob(f"{tour}_matches_[12]*.csv"))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["date"].dt.year >= min_year].dropna(subset=["winner_id", "loser_id", "date"])
    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce").fillna(0)
    df["source"] = "sackmann"
    df["tourney_start"] = df["date"]  # no Sackmann 'date' já é o início do torneio
    df["round_idx"] = df["round"].map({"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7})
    return df


def _player_index(df: pd.DataFrame) -> dict:
    """Mapa 'apelido+inicial' -> id, a partir de jogadores recentes do Sackmann."""
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=900)]
    names = pd.concat([
        recent[["winner_id", "winner_name", "date"]].set_axis(["id", "name", "date"], axis=1),
        recent[["loser_id", "loser_name", "date"]].set_axis(["id", "name", "date"], axis=1),
    ]).dropna().sort_values("date")
    last_seen = names.groupby(["id", "name"], as_index=False)["date"].max()
    counts = names.groupby("id").size()
    index: dict = {}
    for _, r in last_seen.iterrows():
        tokens = str(r["name"]).split()
        if len(tokens) < 2:
            continue
        initial = _norm(tokens[0])[:1]
        for k in range(1, len(tokens)):
            key = _norm("".join(tokens[k:])) + initial
            pid = int(r["id"])
            # em colisão, ganha quem tem mais jogos recentes
            if key not in index or counts[pid] > counts[index[key]]:
                index[key] = pid
    return index


def _td_key(name: str) -> str:
    """'Auger-Aliassime F.' -> 'augeraliassimef'."""
    tokens = str(name).strip().split()
    # as iniciais vêm no fim e terminam em '.', ex.: 'J.M.' ou 'Dar.'
    k = len(tokens)
    while k > 1 and tokens[k - 1].endswith("."):
        k -= 1
    last = "".join(tokens[:k])
    init = tokens[k][0] if k < len(tokens) else ""
    return _norm(last) + init.lower()


def _extension(tour: str, after: pd.Timestamp, index: dict) -> pd.DataFrame:
    f = TDUK / f"{tour}_2026.xlsx"
    if not f.exists():
        return pd.DataFrame()
    td = pd.read_excel(f)
    td = td[td["Date"] > after].copy()
    if td.empty:
        return td
    # ids: correspondência por nome; sem correspondência -> id sintético negativo
    synthetic: dict = {}

    def pid(name: str) -> int:
        key = _td_key(name)
        if key in index:
            return index[key]
        return synthetic.setdefault(key, -(len(synthetic) + 1))

    out = pd.DataFrame({
        "tourney_id": "2026-TD-" + td["Tournament"].astype(str),
        "tourney_name": td["Tournament"],
        "surface": td["Surface"],
        # ATP usa a coluna 'Series', WTA usa 'Tier'
        "tourney_level": td["Series" if "Series" in td else "Tier"].map(TD_LEVEL).fillna("A"),
        "date": td["Date"],
        # início do torneio = 1.ª data do torneio, para alinhar com o Sackmann
        "tourney_start": td.groupby("Tournament")["Date"].transform("min"),
        "best_of": td["Best of"],
        "round_raw": td["Round"],
        "winner_id": td["Winner"].map(pid),
        "loser_id": td["Loser"].map(pid),
        "winner_name": td["Winner"],
        "loser_name": td["Loser"],
        "winner_rank": td["WRank"],
        "loser_rank": td["LRank"],
        "winner_rank_points": td["WPts"],
        "loser_rank_points": td["LPts"],
        "is_walkover": td["Comment"].isin(["Walkover"]),
        "is_retirement": td["Comment"].isin(["Retired"]),
        "source": "tennis-data",
    })
    for c in ["B365W", "B365L", "PSW", "PSL", "AvgW", "AvgL", "MaxW", "MaxL"]:
        if c in td:
            out[c] = td[c]
    # Rondas: rondas iniciais dependem do tamanho do quadro do torneio
    order = {"1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4, "Round Robin": 4,
             "Quarterfinals": 5, "Semifinals": 6, "The Final": 7}
    out["match_num"] = td["Round"].map(order).fillna(0).values
    out["round_idx"] = td["Round"].map({k: v for k, v in order.items() if k != "Round Robin"}).values
    out["unmatched_names"] = False
    out.attrs["synthetic_ids"] = len(synthetic)
    return out


def load_matches(tour: str = "atp", min_year: int = 1985, extend: bool = True) -> pd.DataFrame:
    """Jogos ordenados cronologicamente (obrigatório para evitar fuga de dados)."""
    df = _load_sackmann(tour, min_year)
    index = _player_index(df)
    last_slam = df.loc[(df["tourney_name"] == "Roland Garros") & (df["date"].dt.year == 2026)]
    if extend and not last_slam.empty:
        td_rg = pd.read_excel(TDUK / f"{tour}_2026.xlsx")
        td_rg = td_rg[td_rg["Tournament"].str.contains("French|Roland", na=False)]
        cutoff = td_rg["Date"].max()
        ext = _extension(tour, cutoff, index)
        if not ext.empty:
            for c in ext.columns:
                if c not in df.columns:
                    df[c] = pd.NA
            df = pd.concat([df, ext[df.columns.intersection(ext.columns)]], ignore_index=True)
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    df["year"] = df["date"].dt.year
    # qualidade de dados: o WTA tem 'clay' em minúsculas em alguns jogos
    df["surface"] = df["surface"].astype("string").str.strip().str.title()
    df = df.sort_values(["date", "tourney_id", "match_num"], kind="stable").reset_index(drop=True)
    score = df["score"].fillna("") if "score" in df else pd.Series("", index=df.index)
    df["is_walkover"] = df["is_walkover"].fillna(False).astype(bool) | score.str.contains("W/O", regex=False)
    df["is_retirement"] = df["is_retirement"].fillna(False).astype(bool) | score.str.contains("RET|DEF", regex=True)
    return df
