# -*- coding: utf-8 -*-
"""
Painel interativo para previsão da Copa do Mundo 2026 - versão 3.

Como rodar no Anaconda Prompt:

    cd %USERPROFILE%\\Downloads
    pip install -r requirements_copa2026_v3.txt
    streamlit run copa2026_modelo_interativo_v3.py

O que esta versão adiciona em relação ao protótipo:
- Aba para montar base histórica via resultados internacionais;
- Reconstrói Elo pré-jogo automaticamente;
- Aba de calibração por máxima verossimilhança;
- Estima alpha, beta e gamma a partir dessa base histórica;
- Permite usar parâmetros estimados ou parâmetros manuais;
- Mantém confronto direto, matriz de todos os jogos, simulação Monte Carlo e árvore/bracket aproximada;
- Permite editar Elos, grupos e sedes direto no painel.

Observação importante:
- Os Elos iniciais e grupos abaixo são uma base editável de trabalho, não uma coleta automática.
- Para previsão rigorosa, use Elo pré-jogo em sua base histórica. Não use Elo atual para jogos antigos.
- O bracket do mata-mata é aproximado por seeding de campanha, não a tabela oficial exata da FIFA.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from scipy.special import gammaln


# ============================================================
# 1. Dados iniciais editáveis
# ============================================================

DEFAULT_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Ratings iniciais aproximados em escala Elo. Edite no painel para usar sua fonte preferida.
DEFAULT_ELO = {
    "Spain": 2155,
    "Argentina": 2114,
    "France": 2063,
    "England": 2021,
    "Brazil": 1991,
    "Portugal": 1985,
    "Netherlands": 1940,
    "Germany": 1910,
    "Uruguay": 1890,
    "Mexico": 1875,
    "Colombia": 1870,
    "Belgium": 1845,
    "Morocco": 1840,
    "Croatia": 1810,
    "Japan": 1800,
    "Senegal": 1790,
    "Norway": 1780,
    "Switzerland": 1765,
    "South Korea": 1758,
    "Czechia": 1740,
    "United States": 1725,
    "Austria": 1720,
    "Türkiye": 1715,
    "Sweden": 1710,
    "Ecuador": 1705,
    "Paraguay": 1690,
    "Australia": 1675,
    "Algeria": 1670,
    "Egypt": 1665,
    "Côte d'Ivoire": 1655,
    "Scotland": 1650,
    "Canada": 1645,
    "Tunisia": 1630,
    "Ghana": 1615,
    "IR Iran": 1610,
    "Qatar": 1540,
    "Saudi Arabia": 1535,
    "Bosnia and Herzegovina": 1530,
    "Panama": 1520,
    "Iraq": 1515,
    "South Africa": 1517,
    "Uzbekistan": 1505,
    "Jordan": 1490,
    "Congo DR": 1485,
    "New Zealand": 1475,
    "Haiti": 1450,
    "Curaçao": 1440,
    "Cabo Verde": 1435,
}

HOSTS = {"Mexico", "United States", "Canada"}

ROUND_COLUMNS = ["R32", "R16", "QF", "SF", "Final", "Champion"]


# ============================================================
# 2. Utilitários de dados
# ============================================================


def default_team_table() -> pd.DataFrame:
    rows = []
    for group, teams in DEFAULT_GROUPS.items():
        for team in teams:
            rows.append(
                {
                    "team": team,
                    "group": group,
                    "elo": int(DEFAULT_ELO.get(team, 1600)),
                    "host": bool(team in HOSTS),
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "team"]).reset_index(drop=True)



def clean_team_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["team", "group", "elo", "host"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tabela de seleções sem colunas obrigatórias: {missing}")

    out = df[required].copy()
    out["team"] = out["team"].astype(str).str.strip()
    out["group"] = out["group"].astype(str).str.strip().str.upper()
    out["elo"] = pd.to_numeric(out["elo"], errors="coerce").fillna(1600).astype(float)

    if out["host"].dtype != bool:
        out["host"] = out["host"].astype(str).str.lower().isin(["true", "1", "yes", "sim", "s"])

    out = out[out["team"] != ""].drop_duplicates(subset=["team"], keep="last")
    return out.reset_index(drop=True)



def team_maps(team_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, bool], Dict[str, List[str]]]:
    clean = clean_team_table(team_df)
    elo = dict(zip(clean["team"], clean["elo"]))
    host = dict(zip(clean["team"], clean["host"]))
    groups = {
        g: sorted(sub["team"].tolist())
        for g, sub in clean.groupby("group", sort=True)
    }
    return elo, host, groups



def validate_groups(groups: Dict[str, List[str]]) -> List[str]:
    warnings = []
    if len(groups) != 12:
        warnings.append(f"Foram encontrados {len(groups)} grupos. O formato 2026 completo tem 12 grupos.")
    for group, teams in groups.items():
        if len(teams) != 4:
            warnings.append(f"Grupo {group} tem {len(teams)} seleções. O esperado é 4.")
    return warnings



def csv_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# 3. Modelo Poisson e probabilidades
# ============================================================


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-12)
    return math.exp(-lam) * (lam**k) / math.factorial(k)



def poisson_logpmf_vec(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(lam, 1e-12)
    return k * np.log(lam) - lam - gammaln(k + 1)



def get_active_params() -> Dict[str, float | str]:
    """Retorna parâmetros ativos guardados na sessão ou defaults manuais."""
    if "active_params" not in st.session_state:
        st.session_state.active_params = {
            "mode": "manual_ratio",
            "mu": 2.40,
            "beta": 1.00,
            "host_bonus_elo": 80.0,
            "alpha": math.log(1.20),
            "gamma": 0.15,
            "estimated": False,
        }
    return st.session_state.active_params



def set_active_params(params: Dict[str, float | str]) -> None:
    st.session_state.active_params = params



def match_lambdas_manual_ratio(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    mu: float = 2.40,
    beta: float = 1.00,
    host_bonus_elo: float = 80.0,
) -> Tuple[float, float]:
    """
    Modelo do protótipo:

    ratio = exp(beta * diff_elo / 400)
    lambda_a + lambda_b = mu
    lambda_a / lambda_b = ratio
    """
    elo_a = float(elo.get(team_a, 1600.0)) + (host_bonus_elo if host.get(team_a, False) else 0.0)
    elo_b = float(elo.get(team_b, 1600.0)) + (host_bonus_elo if host.get(team_b, False) else 0.0)

    diff = elo_a - elo_b
    ratio = math.exp(beta * diff / 400.0)

    lam_a = mu * ratio / (1.0 + ratio)
    lam_b = mu / (1.0 + ratio)
    return float(lam_a), float(lam_b)



def match_lambdas_estimated_loglinear(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[float, float]:
    """
    Modelo estimável:

    log(lambda_a) = alpha + beta * (Elo_a - Elo_b)/400 + gamma * Home_a
    log(lambda_b) = alpha + beta * (Elo_b - Elo_a)/400 + gamma * Home_b

    Aqui Home vale 1 para seleções anfitriãs no torneio e 0 caso contrário.
    """
    elo_a = float(elo.get(team_a, 1600.0))
    elo_b = float(elo.get(team_b, 1600.0))
    d = (elo_a - elo_b) / 400.0

    home_a = 1.0 if host.get(team_a, False) else 0.0
    home_b = 1.0 if host.get(team_b, False) else 0.0

    lam_a = math.exp(alpha + beta * d + gamma * home_a)
    lam_b = math.exp(alpha - beta * d + gamma * home_b)
    return float(lam_a), float(lam_b)



def match_lambdas(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
) -> Tuple[float, float]:
    mode = str(params.get("mode", "manual_ratio"))
    if mode == "estimated_loglinear":
        return match_lambdas_estimated_loglinear(
            team_a,
            team_b,
            elo,
            host,
            alpha=float(params.get("alpha", math.log(1.20))),
            beta=float(params.get("beta", 1.0)),
            gamma=float(params.get("gamma", 0.15)),
        )

    return match_lambdas_manual_ratio(
        team_a,
        team_b,
        elo,
        host,
        mu=float(params.get("mu", 2.40)),
        beta=float(params.get("beta", 1.0)),
        host_bonus_elo=float(params.get("host_bonus_elo", 80.0)),
    )



def score_probability_table(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    max_goals: int = 10,
) -> pd.DataFrame:
    lam_a, lam_b = match_lambdas(team_a, team_b, elo, host, params)
    rows = []
    for ga in range(max_goals + 1):
        p_ga = poisson_pmf(ga, lam_a)
        for gb in range(max_goals + 1):
            rows.append(
                {
                    "team_a": team_a,
                    "team_b": team_b,
                    "gols_a": ga,
                    "gols_b": gb,
                    "placar": f"{ga}-{gb}",
                    "prob": p_ga * poisson_pmf(gb, lam_b),
                    "lambda_a": lam_a,
                    "lambda_b": lam_b,
                }
            )
    out = pd.DataFrame(rows)
    out["prob"] = out["prob"] / out["prob"].sum()
    return out



def match_probabilities(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    max_goals: int = 10,
) -> Dict[str, float]:
    df = score_probability_table(team_a, team_b, elo, host, params, max_goals=max_goals)
    p_a = float(df.loc[df["gols_a"] > df["gols_b"], "prob"].sum())
    p_draw = float(df.loc[df["gols_a"] == df["gols_b"], "prob"].sum())
    p_b = float(df.loc[df["gols_a"] < df["gols_b"], "prob"].sum())
    p_over25 = float(df.loc[df["gols_a"] + df["gols_b"] >= 3, "prob"].sum())
    p_btts = float(df.loc[(df["gols_a"] > 0) & (df["gols_b"] > 0), "prob"].sum())
    p_a_clean_win = float(df.loc[(df["gols_a"] > df["gols_b"]) & (df["gols_b"] == 0), "prob"].sum())
    p_b_clean_win = float(df.loc[(df["gols_b"] > df["gols_a"]) & (df["gols_a"] == 0), "prob"].sum())
    p_a_margin2 = float(df.loc[df["gols_a"] - df["gols_b"] >= 2, "prob"].sum())
    p_b_margin2 = float(df.loc[df["gols_b"] - df["gols_a"] >= 2, "prob"].sum())
    lam_a = float(df["lambda_a"].iloc[0])
    lam_b = float(df["lambda_b"].iloc[0])
    return {
        "p_a": p_a,
        "p_draw": p_draw,
        "p_b": p_b,
        "p_over25": p_over25,
        "p_under25": 1.0 - p_over25,
        "p_btts": p_btts,
        "p_no_btts": 1.0 - p_btts,
        "p_a_clean_win": p_a_clean_win,
        "p_b_clean_win": p_b_clean_win,
        "p_a_margin2": p_a_margin2,
        "p_b_margin2": p_b_margin2,
        "lambda_a": lam_a,
        "lambda_b": lam_b,
    }


# ============================================================
# 4. Estimação por máxima verossimilhança
# ============================================================


REQUIRED_HIST_COLS = ["team_a", "team_b", "goals_a", "goals_b", "elo_a", "elo_b", "home_a", "home_b"]



def historical_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-07-14", "2024-07-10", "2024-06-20", "2023-11-21"],
            "team_a": ["Spain", "Argentina", "Brazil", "Mexico"],
            "team_b": ["England", "Colombia", "Colombia", "Honduras"],
            "goals_a": [2, 1, 1, 2],
            "goals_b": [1, 0, 2, 0],
            "elo_a": [2100, 2090, 2020, 1840],
            "elo_b": [2020, 1870, 1870, 1500],
            "home_a": [0, 0, 0, 1],
            "home_b": [0, 0, 0, 0],
            "weight": [1.0, 1.0, 1.0, 0.8],
        }
    )





# ============================================================
# 4A. Construção automática de base histórica e Elo pré-jogo
# ============================================================

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

APP_TO_RESULTS_NAME = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "United States": "United States",
    "South Korea": "South Korea",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
}


def results_name_for_app_team(team: str) -> str:
    return APP_TO_RESULTS_NAME.get(team, team)


@st.cache_data(show_spinner=False)
def load_international_results_from_url(url: str = RESULTS_URL) -> pd.DataFrame:
    """Baixa a base pública martj42/international_results diretamente do GitHub."""
    return pd.read_csv(url)


def clean_results_source(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "country",
        "neutral",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Base de resultados sem colunas obrigatórias: {missing}")

    out = raw.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out = out.dropna(subset=["home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)

    # neutral pode vir como bool ou string.
    if out["neutral"].dtype != bool:
        out["neutral"] = out["neutral"].astype(str).str.lower().isin(["true", "1", "yes", "sim"])

    # Algumas versões podem não ter city; criamos para evitar erro de seleção de colunas.
    if "city" not in out.columns:
        out["city"] = ""

    return out.sort_values("date").reset_index(drop=True)


def expected_score_elo(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))


def actual_score_from_goals(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a == goals_b:
        return 0.5
    return 0.0


def goal_difference_multiplier(goal_diff: int) -> float:
    """Multiplicador simples para placares elásticos na atualização Elo."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def tournament_k_multiplier(tournament: str) -> float:
    """Peso do jogo na atualização do Elo reconstruído."""
    t = str(tournament).lower()
    if "friendly" in t:
        return 0.60
    if "fifa world cup" in t and "qualification" not in t:
        return 2.20
    if any(x in t for x in ["uefa euro", "copa américa", "copa america", "african cup", "afc asian cup", "concacaf gold cup", "uefa nations league"]):
        return 1.60
    if "qualification" in t or "qualifier" in t:
        return 1.20
    return 1.00


def tournament_estimation_weight(tournament: str) -> float:
    """Peso usado na verossimilhança do modelo de gols."""
    t = str(tournament).lower()
    if "friendly" in t:
        return 0.50
    if "fifa world cup" in t and "qualification" not in t:
        return 1.75
    if any(x in t for x in ["uefa euro", "copa américa", "copa america", "african cup", "afc asian cup", "concacaf gold cup", "uefa nations league"]):
        return 1.35
    if "qualification" in t or "qualifier" in t:
        return 1.10
    return 1.00


def build_pregame_elo_dataset(
    results_raw: pd.DataFrame,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    initial_elo: float = 1500.0,
    k_base: float = 30.0,
    home_advantage_elo: float = 75.0,
    use_goal_diff_multiplier: bool = True,
    use_tournament_k: bool = True,
    use_competition_weight: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstrói um Elo interno jogo a jogo e devolve uma base com Elo pré-jogo.

    Importante: o Elo salvo em elo_a e elo_b é sempre o rating ANTES da partida.
    """
    results = clean_results_source(results_raw)
    start_ts = pd.to_datetime(start_date)
    results = results[results["date"] >= start_ts].copy()
    if end_date:
        end_ts = pd.to_datetime(end_date)
        results = results[results["date"] <= end_ts].copy()
    results = results.sort_values("date").reset_index(drop=True)

    elo_ratings: Dict[str, float] = {}
    rows = []

    for _, row in results.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        goals_a = int(row["home_score"])
        goals_b = int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = row.get("tournament", "")

        if team_a not in elo_ratings:
            elo_ratings[team_a] = float(initial_elo)
        if team_b not in elo_ratings:
            elo_ratings[team_b] = float(initial_elo)

        elo_a_pre = float(elo_ratings[team_a])
        elo_b_pre = float(elo_ratings[team_b])

        home_a = 0.0 if neutral else 1.0
        home_b = 0.0

        # Na expectativa do Elo, aplicamos vantagem de casa apenas quando o jogo não é neutro.
        exp_a = expected_score_elo(
            elo_a_pre + home_advantage_elo * home_a,
            elo_b_pre + home_advantage_elo * home_b,
        )
        score_a = actual_score_from_goals(goals_a, goals_b)

        k_match = float(k_base)
        if use_tournament_k:
            k_match *= tournament_k_multiplier(tournament)
        if use_goal_diff_multiplier:
            k_match *= goal_difference_multiplier(goals_a - goals_b)

        delta = k_match * (score_a - exp_a)
        elo_ratings[team_a] = elo_a_pre + delta
        elo_ratings[team_b] = elo_b_pre - delta

        rows.append(
            {
                "date": row["date"],
                "team_a": team_a,
                "team_b": team_b,
                "goals_a": goals_a,
                "goals_b": goals_b,
                "elo_a": elo_a_pre,
                "elo_b": elo_b_pre,
                "home_a": home_a,
                "home_b": home_b,
                "tournament": tournament,
                "country": row.get("country", ""),
                "city": row.get("city", ""),
                "neutral": neutral,
                "weight": tournament_estimation_weight(tournament) if use_competition_weight else 1.0,
            }
        )

    hist = pd.DataFrame(rows)
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"])

    latest = pd.DataFrame(
        [{"source_team": k, "elo": v} for k, v in elo_ratings.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True)

    return hist, latest


def update_team_table_with_rebuilt_elos(team_df: pd.DataFrame, latest_elo_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Atualiza os Elos da tabela da Copa usando o último Elo reconstruído disponível."""
    latest_map = dict(zip(latest_elo_df["source_team"], latest_elo_df["elo"]))
    out = clean_team_table(team_df).copy()
    not_found = []
    new_elos = []

    for _, row in out.iterrows():
        app_team = row["team"]
        source_name = results_name_for_app_team(app_team)
        if source_name in latest_map:
            new_elos.append(round(float(latest_map[source_name])))
        else:
            new_elos.append(row["elo"])
            not_found.append(app_team)

    out["elo"] = new_elos
    return out, not_found

def clean_historical_games(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_HIST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Base histórica sem colunas obrigatórias: {missing}")

    out = df.copy()
    for col in ["team_a", "team_b"]:
        out[col] = out[col].astype(str).str.strip()

    for col in ["goals_a", "goals_b", "elo_a", "elo_b", "home_a", "home_b"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["goals_a", "goals_b", "elo_a", "elo_b", "home_a", "home_b"])
    out["goals_a"] = out["goals_a"].astype(int)
    out["goals_b"] = out["goals_b"].astype(int)
    out["home_a"] = out["home_a"].astype(float)
    out["home_b"] = out["home_b"].astype(float)

    if "weight" not in out.columns:
        out["weight"] = 1.0
    else:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return out.reset_index(drop=True)



def add_time_decay_weights(df: pd.DataFrame, half_life_years: float) -> pd.DataFrame:
    """Multiplica a coluna weight por peso temporal, se existir coluna date."""
    out = df.copy()
    if "date" not in out.columns or out["date"].isna().all():
        return out

    max_date = out["date"].max()
    age_years = (max_date - out["date"]).dt.days.fillna(0) / 365.25
    decay = np.exp(-math.log(2) * age_years / max(half_life_years, 1e-6))
    out["weight"] = out["weight"] * decay
    return out



def nll_loglinear(params_arr: np.ndarray, data: pd.DataFrame) -> float:
    alpha, beta, gamma = params_arr
    d = (data["elo_a"].values - data["elo_b"].values) / 400.0
    home_a = data["home_a"].values
    home_b = data["home_b"].values
    goals_a = data["goals_a"].values
    goals_b = data["goals_b"].values
    weights = data["weight"].values

    lambda_a = np.exp(alpha + beta * d + gamma * home_a)
    lambda_b = np.exp(alpha - beta * d + gamma * home_b)

    ll = poisson_logpmf_vec(goals_a, lambda_a) + poisson_logpmf_vec(goals_b, lambda_b)
    return float(-np.sum(weights * ll))



def fit_loglinear_model(data: pd.DataFrame) -> Dict[str, float | bool | str]:
    if len(data) < 20:
        # Não impede estimação, mas avisa no retorno.
        sample_warning = True
    else:
        sample_warning = False

    # Chute inicial: média por time e coeficientes moderados.
    mean_goals_per_team = max((data["goals_a"].mean() + data["goals_b"].mean()) / 2.0, 0.2)
    initial = np.array([math.log(mean_goals_per_team), 1.0, 0.15])

    bounds = [
        (math.log(0.15), math.log(5.0)),  # alpha
        (-5.0, 5.0),                      # beta
        (-1.0, 1.0),                      # gamma em log-gols
    ]

    res = minimize(
        nll_loglinear,
        initial,
        args=(data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000},
    )

    alpha, beta, gamma = res.x
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "n_obs": int(len(data)),
        "nll": float(res.fun),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "base_goals_per_team": float(math.exp(alpha)),
        "base_total_goals": float(2.0 * math.exp(alpha)),
        "home_multiplier": float(math.exp(gamma)),
        "sample_warning": bool(sample_warning),
    }



def evaluate_log_score(data: pd.DataFrame, alpha: float, beta: float, gamma: float) -> float:
    nll = nll_loglinear(np.array([alpha, beta, gamma]), data)
    total_weight = max(float(data["weight"].sum()), 1e-12)
    return -nll / total_weight


# ============================================================
# 5. Simulação do torneio
# ============================================================



def simulate_match_score(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    rng: np.random.Generator,
) -> Tuple[int, int]:
    lam_a, lam_b = match_lambdas(team_a, team_b, elo, host, params)
    return int(rng.poisson(lam_a)), int(rng.poisson(lam_b))



def simulate_group(
    group_name: str,
    teams: List[str],
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.DataFrame(
        {
            "team": teams,
            "group": group_name,
            "pts": 0,
            "w": 0,
            "d": 0,
            "l": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "elo": [float(elo.get(t, 1600.0)) for t in teams],
            "tie_noise": rng.random(len(teams)) / 100000.0,
        }
    ).set_index("team")

    games = []
    for team_a, team_b in combinations(teams, 2):
        ga, gb = simulate_match_score(team_a, team_b, elo, host, params, rng)
        games.append(
            {
                "group": group_name,
                "team_a": team_a,
                "team_b": team_b,
                "goals_a": ga,
                "goals_b": gb,
                "score": f"{team_a} {ga} x {gb} {team_b}",
            }
        )

        table.loc[team_a, "gf"] += ga
        table.loc[team_a, "ga"] += gb
        table.loc[team_b, "gf"] += gb
        table.loc[team_b, "ga"] += ga

        if ga > gb:
            table.loc[team_a, "pts"] += 3
            table.loc[team_a, "w"] += 1
            table.loc[team_b, "l"] += 1
        elif gb > ga:
            table.loc[team_b, "pts"] += 3
            table.loc[team_b, "w"] += 1
            table.loc[team_a, "l"] += 1
        else:
            table.loc[team_a, "pts"] += 1
            table.loc[team_b, "pts"] += 1
            table.loc[team_a, "d"] += 1
            table.loc[team_b, "d"] += 1

    table["gd"] = table["gf"] - table["ga"]
    table = table.reset_index()
    table = table.sort_values(
        ["pts", "gd", "gf", "w", "elo", "tie_noise"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    table["group_rank"] = np.arange(1, len(table) + 1)
    return table, pd.DataFrame(games)



def simulate_group_stage(
    groups: Dict[str, List[str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    tables = []
    games = []
    for group_name, teams in sorted(groups.items()):
        table, group_games = simulate_group(group_name, teams, elo, host, params, rng)
        tables.append(table)
        games.append(group_games)

    table_all = pd.concat(tables, ignore_index=True)
    games_all = pd.concat(games, ignore_index=True) if games else pd.DataFrame()

    first_second = table_all[table_all["group_rank"].isin([1, 2])].copy()
    thirds = table_all[table_all["group_rank"] == 3].copy()
    best_thirds = thirds.sort_values(
        ["pts", "gd", "gf", "w", "elo", "tie_noise"],
        ascending=[False, False, False, False, False, False],
    ).head(8)

    qualified_table = pd.concat([first_second, best_thirds], ignore_index=True)
    qualified_table = qualified_table.sort_values(
        ["group_rank", "pts", "gd", "gf", "w", "elo", "tie_noise"],
        ascending=[True, False, False, False, False, False, False],
    ).reset_index(drop=True)

    qualified = qualified_table["team"].tolist()
    return table_all, games_all, qualified, qualified_table



def build_round32_bracket(qualified_table: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Árvore aproximada por seeding.

    A FIFA usa cruzamentos específicos e combinações de melhores terceiros. Aqui usamos:
    seed 1 x seed 32, seed 16 x seed 17, seed 8 x seed 25 etc., preservando uma árvore balanceada.
    """
    seeds = qualified_table.copy()
    seeds = seeds.sort_values(
        ["group_rank", "pts", "gd", "gf", "w", "elo", "tie_noise"],
        ascending=[True, False, False, False, False, False, False],
    ).reset_index(drop=True)
    teams = seeds["team"].tolist()

    if len(teams) != 32:
        raise ValueError(f"A fase de 32 precisa de 32 seleções, mas recebeu {len(teams)}.")

    # Ordem de bracket aproximada para evitar que seed 1 e 2 se encontrem antes da final.
    seed_order = [
        (0, 31), (15, 16), (7, 24), (8, 23),
        (3, 28), (12, 19), (4, 27), (11, 20),
        (1, 30), (14, 17), (6, 25), (9, 22),
        (2, 29), (13, 18), (5, 26), (10, 21),
    ]
    return [(teams[i], teams[j]) for i, j in seed_order]



def knockout_winner(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    rng: np.random.Generator,
    penalty_sensitivity: float = 0.08,
) -> Tuple[str, Dict[str, object]]:
    ga, gb = simulate_match_score(team_a, team_b, elo, host, params, rng)
    method = "90min"
    if ga > gb:
        winner = team_a
    elif gb > ga:
        winner = team_b
    else:
        diff = (float(elo.get(team_a, 1600.0)) - float(elo.get(team_b, 1600.0))) / 400.0
        p_a_pen = 0.50 + penalty_sensitivity * diff
        p_a_pen = min(max(p_a_pen, 0.40), 0.60)
        winner = team_a if rng.random() < p_a_pen else team_b
        method = "penalties"

    return winner, {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": ga,
        "goals_b": gb,
        "winner": winner,
        "method": method,
    }



def simulate_knockout(
    bracket: List[Tuple[str, str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    rng: np.random.Generator,
    penalty_sensitivity: float,
) -> Tuple[str, Dict[str, List[Dict[str, object]]]]:
    history: Dict[str, List[Dict[str, object]]] = {"R32": [], "R16": [], "QF": [], "SF": [], "Final": []}

    current_pairs = bracket
    for round_name in ["R32", "R16", "QF", "SF", "Final"]:
        winners = []
        matches = []
        for team_a, team_b in current_pairs:
            winner, match_info = knockout_winner(
                team_a, team_b, elo, host, params, rng, penalty_sensitivity=penalty_sensitivity
            )
            winners.append(winner)
            matches.append(match_info)
        history[round_name] = matches
        if round_name != "Final":
            current_pairs = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    champion = history["Final"][0]["winner"]
    return champion, history



def simulate_tournament_once(
    groups: Dict[str, List[str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    rng: np.random.Generator,
    penalty_sensitivity: float = 0.08,
) -> Dict[str, object]:
    table_all, games_all, qualified, qualified_table = simulate_group_stage(groups, elo, host, params, rng)
    bracket = build_round32_bracket(qualified_table)
    champion, ko_history = simulate_knockout(bracket, elo, host, params, rng, penalty_sensitivity)
    return {
        "group_table": table_all,
        "group_games": games_all,
        "qualified": qualified,
        "qualified_table": qualified_table,
        "bracket": bracket,
        "ko_history": ko_history,
        "champion": champion,
    }



def simulate_many_tournaments(
    groups: Dict[str, List[str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    params: Dict[str, float | str],
    n_sims: int,
    seed: int,
    penalty_sensitivity: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [team for group_teams in groups.values() for team in group_teams]
    counts = {team: {col: 0 for col in ROUND_COLUMNS} for team in teams}

    for _ in range(n_sims):
        sim = simulate_tournament_once(groups, elo, host, params, rng, penalty_sensitivity)

        qualified = set(sim["qualified"])
        for team in qualified:
            counts[team]["R32"] += 1

        for round_name in ["R32", "R16", "QF", "SF", "Final"]:
            matches = sim["ko_history"][round_name]
            winners = [m["winner"] for m in matches]
            next_col = {
                "R32": "R16",
                "R16": "QF",
                "QF": "SF",
                "SF": "Final",
                "Final": "Champion",
            }[round_name]
            for winner in winners:
                counts[winner][next_col] += 1

    rows = []
    for team in teams:
        row = {"team": team, "elo": elo.get(team, 1600.0)}
        row.update({col: counts[team][col] / n_sims for col in ROUND_COLUMNS})
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)


# ============================================================
# 6. UI
# ============================================================


st.set_page_config(page_title="Modelo Copa 2026 - Poisson Elo", layout="wide")

st.title("Modelo interativo para previsão da Copa 2026")
st.caption(
    "Versão 3: baixa/usa base histórica de jogos, reconstrói Elo pré-jogo, estima parâmetros por máxima verossimilhança, simula confrontos e a Copa."
)

if "team_df" not in st.session_state:
    st.session_state.team_df = default_team_table()

params = get_active_params()

with st.sidebar:
    st.header("Parâmetros ativos")

    mode_label = "Estimado: Poisson log-linear" if params.get("mode") == "estimated_loglinear" else "Manual: razão de gols"
    st.info(f"Modelo ativo: {mode_label}")

    if params.get("mode") == "estimated_loglinear":
        st.metric("alpha", f"{float(params.get('alpha', 0)):.3f}")
        st.metric("beta estimado", f"{float(params.get('beta', 0)):.3f}")
        st.metric("gamma mando", f"{float(params.get('gamma', 0)):.3f}")
        st.caption(f"Média base por time: {math.exp(float(params.get('alpha', math.log(1.2)))):.2f}")
        st.caption(f"Multiplicador de mando: {math.exp(float(params.get('gamma', 0.15))):.2f}x")
    else:
        manual_mu = st.number_input("Média total de gols", min_value=0.5, max_value=6.0, value=float(params.get("mu", 2.40)), step=0.05)
        manual_beta = st.number_input("Beta manual", min_value=-5.0, max_value=5.0, value=float(params.get("beta", 1.00)), step=0.05)
        manual_host_bonus = st.number_input("Bônus de sede em pontos Elo", min_value=0.0, max_value=250.0, value=float(params.get("host_bonus_elo", 80.0)), step=5.0)
        if st.button("Aplicar parâmetros manuais"):
            set_active_params(
                {
                    "mode": "manual_ratio",
                    "mu": manual_mu,
                    "beta": manual_beta,
                    "host_bonus_elo": manual_host_bonus,
                    "alpha": math.log(max(manual_mu / 2.0, 1e-6)),
                    "gamma": 0.15,
                    "estimated": False,
                }
            )
            st.rerun()

    st.divider()
    if st.button("Voltar para modelo manual padrão"):
        set_active_params(
            {
                "mode": "manual_ratio",
                "mu": 2.40,
                "beta": 1.00,
                "host_bonus_elo": 80.0,
                "alpha": math.log(1.20),
                "gamma": 0.15,
                "estimated": False,
            }
        )
        st.rerun()


tab_dados, tab_basehist, tab_calib, tab_match, tab_pairs, tab_sim, tab_tree, tab_help = st.tabs(
    [
        "1. Dados da Copa",
        "2. Base histórica + Elo",
        "3. Calibrar parâmetros",
        "4. Confronto direto",
        "5. Todos os jogos possíveis",
        "6. Simular Copa",
        "7. Árvore/bracket",
        "8. Como o modelo funciona",
    ]
)


with tab_dados:
    st.subheader("Base de seleções")
    st.write(
        "Edite grupos, Elos e marcação de sede. O app usa esta tabela em todas as previsões."
    )

    uploaded_teams = st.file_uploader(
        "Opcional: carregar CSV de seleções com colunas team, group, elo, host",
        type=["csv"],
        key="teams_uploader",
    )
    if uploaded_teams is not None:
        try:
            loaded = pd.read_csv(uploaded_teams)
            st.session_state.team_df = clean_team_table(loaded)
            st.success("Tabela carregada com sucesso.")
        except Exception as exc:
            st.error(f"Erro ao carregar tabela: {exc}")

    edited = st.data_editor(
        st.session_state.team_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "team": st.column_config.TextColumn("team", required=True),
            "group": st.column_config.TextColumn("group", required=True),
            "elo": st.column_config.NumberColumn("elo", min_value=0, max_value=3000, step=1),
            "host": st.column_config.CheckboxColumn("host"),
        },
        key="team_editor",
    )

    try:
        st.session_state.team_df = clean_team_table(edited)
        elo, host, groups = team_maps(st.session_state.team_df)
        warnings = validate_groups(groups)
        for w in warnings:
            st.warning(w)
        st.download_button(
            "Baixar tabela de seleções editada",
            data=csv_download_bytes(st.session_state.team_df),
            file_name="selecoes_copa2026_editavel.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"A tabela atual tem problema: {exc}")


with tab_basehist:
    st.subheader("Montar base histórica e reconstruir Elo pré-jogo")
    st.write(
        "Esta aba monta automaticamente a base necessária para estimar o modelo. "
        "Ela usa resultados internacionais, calcula um Elo interno jogo a jogo e salva, para cada partida, o Elo antes do jogo."
    )

    st.markdown(
        f"""
        **Fonte padrão dos resultados:** `martj42/international_results`  
        URL usada pelo app: `{RESULTS_URL}`

        A base tem placares históricos de seleções masculinas. O app não usa Elo pronto da internet: ele reconstrói um Elo interno a partir desses placares.
        """
    )

    source_mode = st.radio(
        "Como carregar a base de resultados?",
        ["Baixar automaticamente do GitHub", "Subir CSV manualmente"],
        horizontal=True,
    )

    raw_results = None
    if source_mode == "Baixar automaticamente do GitHub":
        if st.button("Baixar base de resultados", type="primary"):
            try:
                with st.spinner("Baixando results.csv..."):
                    raw_results = load_international_results_from_url(RESULTS_URL)
                st.session_state.raw_results = raw_results
                st.success(f"Base baixada: {len(raw_results):,} linhas.")
            except Exception as exc:
                st.error(
                    "Não consegui baixar a base automaticamente. "
                    "Verifique a conexão ou use a opção de subir CSV manualmente."
                )
                st.exception(exc)
    else:
        uploaded_results = st.file_uploader(
            "Subir results.csv com colunas date, home_team, away_team, home_score, away_score, tournament, country, neutral",
            type=["csv"],
            key="results_source_uploader",
        )
        if uploaded_results is not None:
            try:
                raw_results = pd.read_csv(uploaded_results)
                st.session_state.raw_results = raw_results
                st.success(f"Base carregada: {len(raw_results):,} linhas.")
            except Exception as exc:
                st.error(f"Erro ao ler CSV: {exc}")

    if "raw_results" in st.session_state:
        try:
            preview = clean_results_source(st.session_state.raw_results)
            st.write("Amostra da base de resultados:")
            st.dataframe(preview.tail(10), use_container_width=True)

            min_date_available = preview["date"].min().date()
            max_date_available = preview["date"].max().date()

            st.divider()
            st.markdown("### Parâmetros para reconstrução do Elo")
            c1, c2, c3 = st.columns(3)
            with c1:
                start_date = st.date_input("Usar jogos a partir de", value=pd.to_datetime("2000-01-01").date(), min_value=min_date_available, max_value=max_date_available)
            with c2:
                end_date = st.date_input("Usar jogos até", value=max_date_available, min_value=min_date_available, max_value=max_date_available)
            with c3:
                initial_elo = st.number_input("Elo inicial para novas seleções", min_value=1000.0, max_value=2000.0, value=1500.0, step=25.0)

            c4, c5, c6 = st.columns(3)
            with c4:
                k_base = st.number_input("K base do Elo", min_value=5.0, max_value=80.0, value=30.0, step=1.0)
            with c5:
                home_adv_elo = st.number_input("Vantagem de casa no Elo", min_value=0.0, max_value=200.0, value=75.0, step=5.0)
            with c6:
                use_goal_diff = st.checkbox("Usar multiplicador por saldo do placar", value=True)

            c7, c8 = st.columns(2)
            with c7:
                use_tournament_k = st.checkbox("K maior para competições importantes", value=True)
            with c8:
                use_competition_weight = st.checkbox("Criar pesos de estimação por tipo de competição", value=True)

            if st.button("Construir base com Elo pré-jogo", type="primary"):
                with st.spinner("Reconstruindo Elo e montando base histórica..."):
                    hist_model, latest_elo = build_pregame_elo_dataset(
                        st.session_state.raw_results,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        initial_elo=float(initial_elo),
                        k_base=float(k_base),
                        home_advantage_elo=float(home_adv_elo),
                        use_goal_diff_multiplier=bool(use_goal_diff),
                        use_tournament_k=bool(use_tournament_k),
                        use_competition_weight=bool(use_competition_weight),
                    )
                st.session_state.hist_model_df = hist_model
                st.session_state.latest_elo_df = latest_elo
                st.success(f"Base construída: {len(hist_model):,} jogos; {len(latest_elo):,} seleções com Elo.")

            if "hist_model_df" in st.session_state:
                hist_model = st.session_state.hist_model_df
                latest_elo = st.session_state.latest_elo_df

                st.divider()
                st.markdown("### Base gerada")
                m1, m2, m3 = st.columns(3)
                m1.metric("Jogos", f"{len(hist_model):,}")
                m2.metric("Seleções com Elo", f"{len(latest_elo):,}")
                m3.metric("Última data", str(hist_model["date"].max().date()) if len(hist_model) else "-")

                st.write("Amostra da base para estimação:")
                st.dataframe(hist_model.tail(20), use_container_width=True)

                st.write("Top 20 Elo reconstruído:")
                st.dataframe(latest_elo.head(20), use_container_width=True, hide_index=True)

                col_down1, col_down2, col_update = st.columns(3)
                with col_down1:
                    st.download_button(
                        "Baixar base histórica com Elo pré-jogo",
                        data=csv_download_bytes(hist_model),
                        file_name="jogos_historicos_com_elo_pre_jogo.csv",
                        mime="text/csv",
                    )
                with col_down2:
                    st.download_button(
                        "Baixar último Elo reconstruído",
                        data=csv_download_bytes(latest_elo),
                        file_name="elos_reconstruidos_atuais.csv",
                        mime="text/csv",
                    )
                with col_update:
                    if st.button("Atualizar Elos da Copa com esta base"):
                        updated_team_df, not_found = update_team_table_with_rebuilt_elos(st.session_state.team_df, latest_elo)
                        st.session_state.team_df = updated_team_df
                        st.success("Elos da tabela da Copa atualizados.")
                        if not_found:
                            st.warning("Não encontrei correspondência para: " + ", ".join(not_found))
                        st.rerun()
        except Exception as exc:
            st.error(f"Erro ao processar a base de resultados: {exc}")
    else:
        st.info("Baixe a base automaticamente ou suba o CSV para começar.")


with tab_calib:
    st.subheader("Calibrar alpha, beta e gamma com dados históricos")
    st.write(
        "Aqui o beta deixa de ser escolhido na mão. O app estima os parâmetros por máxima verossimilhança "
        "usando a base histórica com Elo pré-jogo. Você pode usar a base criada na aba anterior ou subir um CSV próprio."
    )

    st.markdown(
        """
        **Colunas obrigatórias do CSV histórico**

        - `team_a`, `team_b`: seleções;
        - `goals_a`, `goals_b`: placar observado;
        - `elo_a`, `elo_b`: Elo de cada seleção **antes do jogo**;
        - `home_a`, `home_b`: 1 se a seleção jogou em casa, 0 caso contrário.

        Colunas opcionais:
        - `date`: data do jogo, para peso temporal;
        - `weight`: peso manual do jogo. Exemplo: amistoso 0.5, Copa 1.75.
        """
    )

    st.download_button(
        "Baixar modelo de CSV histórico",
        data=csv_download_bytes(historical_template()),
        file_name="template_jogos_historicos.csv",
        mime="text/csv",
    )

    hist_source = st.radio(
        "Fonte da base para estimação",
        ["Usar base gerada na aba 2", "Subir CSV próprio"],
        horizontal=True,
    )

    hist = None
    if hist_source == "Usar base gerada na aba 2":
        if "hist_model_df" in st.session_state:
            try:
                hist = clean_historical_games(st.session_state.hist_model_df)
                st.success(f"Usando base gerada na aba 2: {len(hist):,} jogos válidos.")
            except Exception as exc:
                st.error(f"A base gerada não pôde ser usada: {exc}")
        else:
            st.info("Ainda não há base gerada. Vá para a aba 2 e clique em 'Construir base com Elo pré-jogo'.")
    else:
        uploaded_hist = st.file_uploader("Carregar jogos históricos em CSV", type=["csv"], key="hist_uploader_v3")
        if uploaded_hist is not None:
            try:
                hist_raw = pd.read_csv(uploaded_hist)
                hist = clean_historical_games(hist_raw)
                st.success(f"Base carregada: {len(hist):,} jogos válidos.")
            except Exception as exc:
                st.error(f"Erro na base histórica: {exc}")

    if hist is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            use_decay = st.checkbox("Usar peso temporal", value=True)
        with col2:
            half_life = st.number_input("Meia-vida dos pesos, em anos", min_value=0.25, max_value=30.0, value=6.0, step=0.25)
        with col3:
            min_year = st.number_input("Filtrar a partir do ano", min_value=1872, max_value=2026, value=2000, step=1)

        hist_fit = hist.copy()
        if "date" in hist_fit.columns:
            hist_fit = hist_fit[hist_fit["date"].dt.year >= int(min_year)].copy()
        if use_decay:
            hist_fit = add_time_decay_weights(hist_fit, half_life)

        st.write("Amostra da base usada na estimação:")
        st.dataframe(hist_fit.tail(20), use_container_width=True)

        cA, cB, cC = st.columns(3)
        cA.metric("Jogos usados", f"{len(hist_fit):,}")
        cB.metric("Peso médio", f"{hist_fit['weight'].mean():.2f}")
        cC.metric("Gols médios/jogo", f"{(hist_fit['goals_a'].mean() + hist_fit['goals_b'].mean()):.2f}")

        if st.button("Estimar parâmetros", type="primary"):
            fit = fit_loglinear_model(hist_fit)

            st.session_state.last_fit = fit
            st.session_state.last_hist_n = len(hist_fit)

            if fit["sample_warning"]:
                st.warning("A base tem menos de 20 jogos. O resultado pode ser muito instável.")

            if fit["success"]:
                st.success("Estimação concluída.")
            else:
                st.warning(f"O otimizador terminou com aviso: {fit['message']}")

            set_active_params(
                {
                    "mode": "estimated_loglinear",
                    "alpha": fit["alpha"],
                    "beta": fit["beta"],
                    "gamma": fit["gamma"],
                    "mu": fit["base_total_goals"],
                    "host_bonus_elo": 80.0,
                    "estimated": True,
                }
            )
            st.success("Parâmetros estimados aplicados ao app.")
            st.rerun()

    if "last_fit" in st.session_state:
        fit = st.session_state.last_fit
        st.divider()
        st.subheader("Última estimação da sessão")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("alpha", f"{fit['alpha']:.3f}")
        c2.metric("beta", f"{fit['beta']:.3f}")
        c3.metric("gamma", f"{fit['gamma']:.3f}")
        c4.metric("gols base/jogo", f"{fit['base_total_goals']:.2f}")
        st.write(
            f"Interpretação: mando multiplica os gols esperados por aproximadamente **{fit['home_multiplier']:.2f}x**. "
            f"O beta estimado substitui o beta exógeno do protótipo."
        )
        st.json({k: v for k, v in fit.items() if k not in ["sample_warning"]})


with tab_match:
    st.subheader("Previsão de confronto direto")

    try:
        elo, host, groups = team_maps(st.session_state.team_df)
        teams = sorted(elo.keys())

        col_a, col_b = st.columns(2)
        with col_a:
            team_a = st.selectbox("Seleção A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
        with col_b:
            default_b = "Morocco" if "Morocco" in teams else teams[min(1, len(teams) - 1)]
            team_b = st.selectbox("Seleção B", teams, index=teams.index(default_b))

        max_goals = st.slider("Máximo de gols considerado na matriz de placares", 6, 15, 10)

        if team_a == team_b:
            st.warning("Escolha duas seleções diferentes.")
        else:
            probs = match_probabilities(team_a, team_b, elo, host, get_active_params(), max_goals=max_goals)
            lam_a, lam_b = probs["lambda_a"], probs["lambda_b"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"λ {team_a}", f"{lam_a:.2f}")
            c2.metric(f"λ {team_b}", f"{lam_b:.2f}")
            c3.metric(f"P({team_a})", f"{probs['p_a']:.1%}")
            c4.metric(f"P({team_b})", f"{probs['p_b']:.1%}")

            st.metric("Empate", f"{probs['p_draw']:.1%}")

            event_df = pd.DataFrame(
                {
                    "evento": [
                        f"{team_a} vence",
                        "Empate",
                        f"{team_b} vence",
                        "Mais de 2.5 gols",
                        "Menos de 2.5 gols",
                        "Ambos marcam",
                        f"{team_a} vence por 2+ gols",
                        f"{team_b} vence por 2+ gols",
                    ],
                    "probabilidade": [
                        probs["p_a"],
                        probs["p_draw"],
                        probs["p_b"],
                        probs["p_over25"],
                        probs["p_under25"],
                        probs["p_btts"],
                        probs["p_a_margin2"],
                        probs["p_b_margin2"],
                    ],
                }
            )
            event_df["probabilidade"] = event_df["probabilidade"].map(lambda x: f"{x:.1%}")
            st.dataframe(event_df, use_container_width=True, hide_index=True)

            scores = score_probability_table(team_a, team_b, elo, host, get_active_params(), max_goals=max_goals)
            top_scores = scores.sort_values("prob", ascending=False).head(12).copy()
            top_scores["prob"] = top_scores["prob"].map(lambda x: f"{x:.1%}")
            st.write("Placares mais prováveis")
            st.dataframe(top_scores[["placar", "prob", "lambda_a", "lambda_b"]], use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Erro ao calcular confronto: {exc}")


with tab_pairs:
    st.subheader("Todos os confrontos possíveis entre as seleções")

    try:
        elo, host, groups = team_maps(st.session_state.team_df)
        teams = sorted(elo.keys())
        if st.button("Calcular matriz de confrontos"):
            rows = []
            params_now = get_active_params()
            for team_a, team_b in combinations(teams, 2):
                p = match_probabilities(team_a, team_b, elo, host, params_now, max_goals=10)
                rows.append(
                    {
                        "team_a": team_a,
                        "team_b": team_b,
                        "lambda_a": p["lambda_a"],
                        "lambda_b": p["lambda_b"],
                        "p_a_win": p["p_a"],
                        "p_draw": p["p_draw"],
                        "p_b_win": p["p_b"],
                        "p_over25": p["p_over25"],
                        "p_btts": p["p_btts"],
                    }
                )
            pair_df = pd.DataFrame(rows)
            st.session_state.pair_df = pair_df

        if "pair_df" in st.session_state:
            df_show = st.session_state.pair_df.copy()
            st.dataframe(df_show, use_container_width=True)
            st.download_button(
                "Baixar matriz de confrontos em CSV",
                data=csv_download_bytes(df_show),
                file_name="matriz_confrontos_copa2026.csv",
                mime="text/csv",
            )
        else:
            st.info("Clique no botão para calcular os 1.128 confrontos possíveis se houver 48 seleções.")
    except Exception as exc:
        st.error(f"Erro ao calcular matriz: {exc}")


with tab_sim:
    st.subheader("Simulação Monte Carlo da Copa")

    try:
        elo, host, groups = team_maps(st.session_state.team_df)
        warnings = validate_groups(groups)
        for w in warnings:
            st.warning(w)

        c1, c2, c3 = st.columns(3)
        with c1:
            n_sims = st.number_input("Número de simulações", min_value=100, max_value=100000, value=5000, step=100)
        with c2:
            seed = st.number_input("Seed aleatória", min_value=1, max_value=999999, value=42, step=1)
        with c3:
            penalty_sens = st.number_input("Sensibilidade nos pênaltis", min_value=0.0, max_value=0.30, value=0.08, step=0.01)

        if st.button("Rodar simulação", type="primary"):
            with st.spinner("Simulando torneios..."):
                results = simulate_many_tournaments(
                    groups,
                    elo,
                    host,
                    get_active_params(),
                    n_sims=int(n_sims),
                    seed=int(seed),
                    penalty_sensitivity=float(penalty_sens),
                )
            st.session_state.sim_results = results
            st.success("Simulação concluída.")

        if "sim_results" in st.session_state:
            out = st.session_state.sim_results.copy()
            pct_cols = ROUND_COLUMNS
            display_df = out.copy()
            for col in pct_cols:
                display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True)
            st.download_button(
                "Baixar resultados da simulação em CSV",
                data=csv_download_bytes(out),
                file_name="probabilidades_copa2026.csv",
                mime="text/csv",
            )
        else:
            st.info("Rode a simulação para obter probabilidades por fase.")

    except Exception as exc:
        st.error(f"Erro na simulação: {exc}")


with tab_tree:
    st.subheader("Uma Copa simulada: grupos e árvore aproximada")

    try:
        elo, host, groups = team_maps(st.session_state.team_df)
        c1, c2 = st.columns(2)
        with c1:
            tree_seed = st.number_input("Seed da árvore", min_value=1, max_value=999999, value=123, step=1)
        with c2:
            tree_pen = st.number_input("Sensibilidade pênaltis da árvore", min_value=0.0, max_value=0.30, value=0.08, step=0.01, key="tree_pen")

        if st.button("Gerar uma árvore simulada"):
            rng = np.random.default_rng(int(tree_seed))
            sim = simulate_tournament_once(groups, elo, host, get_active_params(), rng, penalty_sensitivity=float(tree_pen))
            st.session_state.one_sim = sim

        if "one_sim" in st.session_state:
            sim = st.session_state.one_sim
            st.write(f"Campeão simulado: **{sim['champion']}**")

            with st.expander("Tabela da fase de grupos"):
                st.dataframe(sim["group_table"], use_container_width=True)

            with st.expander("Jogos da fase de grupos"):
                st.dataframe(sim["group_games"], use_container_width=True)

            st.write("Mata-mata")
            for round_name in ["R32", "R16", "QF", "SF", "Final"]:
                st.markdown(f"### {round_name}")
                rows = []
                for m in sim["ko_history"][round_name]:
                    suffix = " (pênaltis)" if m["method"] == "penalties" else ""
                    rows.append(
                        {
                            "jogo": f"{m['team_a']} {m['goals_a']} x {m['goals_b']} {m['team_b']}{suffix}",
                            "classificado": m["winner"],
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Gere uma árvore para visualizar uma trajetória possível do torneio.")
    except Exception as exc:
        st.error(f"Erro ao gerar árvore: {exc}")


with tab_help:
    st.subheader("Como o modelo funciona")

    st.markdown(
        r"""
        ## 1. Dados da Copa e base histórica

        O app parte de uma tabela de seleções com grupo, Elo e indicação de sede. Esses dados são editáveis.

        A versão 3 também pode baixar uma base pública de resultados internacionais, reconstruir um Elo interno jogo a jogo e salvar o Elo de cada seleção antes da partida. Essa base é usada para estimar os parâmetros do modelo de gols.

        ## 2. Modelo manual do protótipo

        No modo manual, o app usa:

        $$
        ratio = \exp\left(\beta \frac{Elo_A - Elo_B}{400}\right)
        $$

        $$
        \lambda_A = \mu \frac{ratio}{1+ratio}
        $$

        $$
        \lambda_B = \mu \frac{1}{1+ratio}
        $$

        onde $\mu$ é a média total de gols do jogo.

        ## 3. Modelo estimado

        Na aba de calibração, o app estima:

        $$
        \log(\lambda_A) = \alpha + \beta \frac{Elo_A - Elo_B}{400} + \gamma Home_A
        $$

        $$
        \log(\lambda_B) = \alpha + \beta \frac{Elo_B - Elo_A}{400} + \gamma Home_B
        $$

        A estimação escolhe $\alpha$, $\beta$ e $\gamma$ para maximizar a verossimilhança dos placares observados.

        ## 4. Placares

        Para cada partida:

        $$
        G_A \sim Poisson(\lambda_A), \quad G_B \sim Poisson(\lambda_B)
        $$

        Isso gera probabilidades de placares, vitória, empate, over/under, ambos marcam etc.

        ## 5. Torneio

        O app simula os grupos, classifica 12 primeiros, 12 segundos e 8 melhores terceiros, e depois joga um mata-mata aproximado por seeding.
        As probabilidades finais são frequências em Monte Carlo.

        ## 6. Cuidado central

        Para calibrar direito, a base histórica precisa ter o Elo **antes de cada jogo**. Usar Elo atual para jogos antigos gera vazamento de informação.

        Na aba 2, o app evita esse problema porque reconstrói o Elo em ordem cronológica e grava `elo_a` e `elo_b` antes de atualizar o rating depois de cada jogo.
        """
    )
