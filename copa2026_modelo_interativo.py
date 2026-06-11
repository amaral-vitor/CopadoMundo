# -*- coding: utf-8 -*-
"""
Painel interativo para previsão de jogos da Copa do Mundo 2026.

Como rodar:
1) No Anaconda Prompt, instale dependências:
   pip install streamlit pandas numpy

2) Rode o app:
   streamlit run copa2026_modelo_interativo.py

O modelo combina:
- ratings Elo editáveis;
- vantagem de sede;
- modelo Poisson para gols;
- simulação Monte Carlo da fase de grupos e mata-mata;
- matriz de todos os confrontos possíveis entre as 48 seleções.

Observação: o bracket de mata-mata é uma aproximação por seeding de campanha.
Para reprodução exata oficial da FIFA, substitua a função build_round32_bracket
por uma tabela oficial de cruzamentos, incluindo as combinações de melhores terceiros.
"""

import math
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


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

# Ratings iniciais aproximados. Edite no painel para usar sua fonte preferida.
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
    "Colombia": 1870,
    "Mexico": 1875,
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
    "South Africa": 1517,
    "Qatar": 1540,
    "Saudi Arabia": 1535,
    "Bosnia and Herzegovina": 1530,
    "Panama": 1520,
    "Iraq": 1515,
    "Uzbekistan": 1505,
    "Jordan": 1490,
    "Congo DR": 1485,
    "New Zealand": 1475,
    "Haiti": 1450,
    "Curaçao": 1440,
    "Cabo Verde": 1435,
}

HOSTS = {"Mexico", "United States", "Canada"}


def default_team_table() -> pd.DataFrame:
    rows = []
    for group, teams in DEFAULT_GROUPS.items():
        for team in teams:
            rows.append(
                {
                    "team": team,
                    "group": group,
                    "elo": int(DEFAULT_ELO.get(team, 1600)),
                    "host": team in HOSTS,
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "team"]).reset_index(drop=True)


# ============================================================
# 2. Núcleo matemático
# ============================================================


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def match_lambdas(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float = 2.40,
    beta: float = 1.00,
    host_bonus_elo: float = 80.0,
) -> Tuple[float, float]:
    """
    Converte diferença de Elo em gols esperados.

    ratio = exp(beta * diff_elo / 400)
    lambda_a + lambda_b = base_total_goals
    lambda_a / lambda_b = ratio
    """
    elo_a = float(elo.get(team_a, 1600)) + (host_bonus_elo if host.get(team_a, False) else 0.0)
    elo_b = float(elo.get(team_b, 1600)) + (host_bonus_elo if host.get(team_b, False) else 0.0)

    diff = elo_a - elo_b
    ratio = math.exp(beta * diff / 400.0)

    lam_a = base_total_goals * ratio / (1.0 + ratio)
    lam_b = base_total_goals / (1.0 + ratio)

    return lam_a, lam_b


def score_probability_table(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    max_goals: int = 10,
) -> pd.DataFrame:
    lam_a, lam_b = match_lambdas(
        team_a,
        team_b,
        elo,
        host,
        base_total_goals=base_total_goals,
        beta=beta,
        host_bonus_elo=host_bonus_elo,
    )

    rows = []
    for ga in range(max_goals + 1):
        for gb in range(max_goals + 1):
            p = poisson_pmf(ga, lam_a) * poisson_pmf(gb, lam_b)
            rows.append(
                {
                    "team_a": team_a,
                    "team_b": team_b,
                    "gols_a": ga,
                    "gols_b": gb,
                    "placar": f"{ga}-{gb}",
                    "prob": p,
                    "lambda_a": lam_a,
                    "lambda_b": lam_b,
                }
            )

    df = pd.DataFrame(rows)
    # Normaliza para retirar o pequeno erro de truncamento no max_goals.
    df["prob"] = df["prob"] / df["prob"].sum()
    return df


def match_probabilities(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    max_goals: int = 10,
) -> Dict[str, float]:
    df = score_probability_table(
        team_a,
        team_b,
        elo,
        host,
        base_total_goals,
        beta,
        host_bonus_elo,
        max_goals=max_goals,
    )

    p_a = df.loc[df["gols_a"] > df["gols_b"], "prob"].sum()
    p_d = df.loc[df["gols_a"] == df["gols_b"], "prob"].sum()
    p_b = df.loc[df["gols_a"] < df["gols_b"], "prob"].sum()
    p_over_25 = df.loc[df["gols_a"] + df["gols_b"] >= 3, "prob"].sum()
    p_btts = df.loc[(df["gols_a"] > 0) & (df["gols_b"] > 0), "prob"].sum()

    return {
        "team_a": team_a,
        "team_b": team_b,
        "lambda_a": float(df["lambda_a"].iloc[0]),
        "lambda_b": float(df["lambda_b"].iloc[0]),
        "p_a": float(p_a),
        "p_draw": float(p_d),
        "p_b": float(p_b),
        "p_over_25": float(p_over_25),
        "p_under_25": float(1 - p_over_25),
        "p_btts": float(p_btts),
    }


def simulate_score(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    lam_a, lam_b = match_lambdas(
        team_a,
        team_b,
        elo,
        host,
        base_total_goals=base_total_goals,
        beta=beta,
        host_bonus_elo=host_bonus_elo,
    )
    return int(rng.poisson(lam_a)), int(rng.poisson(lam_b))


def simulate_knockout_winner(
    team_a: str,
    team_b: str,
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    penalty_sensitivity: float,
    rng: np.random.Generator,
) -> Tuple[str, int, int, str]:
    ga, gb = simulate_score(
        team_a,
        team_b,
        elo,
        host,
        base_total_goals,
        beta,
        host_bonus_elo,
        rng,
    )

    if ga > gb:
        return team_a, ga, gb, "90min"
    if gb > ga:
        return team_b, ga, gb, "90min"

    effective_a = float(elo.get(team_a, 1600)) + (host_bonus_elo if host.get(team_a, False) else 0.0)
    effective_b = float(elo.get(team_b, 1600)) + (host_bonus_elo if host.get(team_b, False) else 0.0)
    diff = effective_a - effective_b

    p_pen_a = 0.50 + penalty_sensitivity * diff / 400.0
    p_pen_a = min(max(p_pen_a, 0.40), 0.60)

    winner = team_a if rng.random() < p_pen_a else team_b
    return winner, ga, gb, "pens"


# ============================================================
# 3. Fase de grupos e árvore/bracket
# ============================================================


def simulate_group(
    group_name: str,
    teams: List[str],
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.DataFrame(
        {
            "team": teams,
            "group": group_name,
            "pts": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "elo": [elo.get(t, 1600) for t in teams],
            "tie_noise": rng.random(len(teams)),
        }
    ).set_index("team")

    match_rows = []

    for team_a, team_b in combinations(teams, 2):
        ga, gb = simulate_score(
            team_a,
            team_b,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            rng,
        )

        table.loc[team_a, "gf"] += ga
        table.loc[team_a, "ga"] += gb
        table.loc[team_b, "gf"] += gb
        table.loc[team_b, "ga"] += ga

        if ga > gb:
            table.loc[team_a, "pts"] += 3
            table.loc[team_a, "wins"] += 1
            table.loc[team_b, "losses"] += 1
        elif gb > ga:
            table.loc[team_b, "pts"] += 3
            table.loc[team_b, "wins"] += 1
            table.loc[team_a, "losses"] += 1
        else:
            table.loc[team_a, "pts"] += 1
            table.loc[team_b, "pts"] += 1
            table.loc[team_a, "draws"] += 1
            table.loc[team_b, "draws"] += 1

        match_rows.append(
            {
                "group": group_name,
                "team_a": team_a,
                "team_b": team_b,
                "score": f"{ga}-{gb}",
            }
        )

    table["gd"] = table["gf"] - table["ga"]
    table = table.reset_index()

    table = table.sort_values(
        ["pts", "gd", "gf", "wins", "elo", "tie_noise"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    table["place"] = np.arange(1, len(table) + 1)

    return table.drop(columns=["tie_noise"]), pd.DataFrame(match_rows)


def simulate_group_stage(
    groups: Dict[str, List[str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_tables = []
    all_matches = []

    for group_name, teams in groups.items():
        table, matches = simulate_group(
            group_name,
            teams,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            rng,
        )
        all_tables.append(table)
        all_matches.append(matches)

    standings = pd.concat(all_tables, ignore_index=True)
    matches_df = pd.concat(all_matches, ignore_index=True)

    direct = standings.loc[standings["place"] <= 2].copy()
    thirds = standings.loc[standings["place"] == 3].copy()
    best_thirds = thirds.sort_values(
        ["pts", "gd", "gf", "wins", "elo"],
        ascending=[False, False, False, False, False],
    ).head(8)

    qualified = pd.concat([direct, best_thirds], ignore_index=True)
    qualified["qualified_as"] = np.where(
        qualified["place"] <= 2,
        qualified["group"].astype(str) + qualified["place"].astype(str),
        qualified["group"].astype(str) + "3",
    )

    return standings, qualified, matches_df


def build_round32_bracket(qualified: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Bracket simplificado por seeding de campanha.

    Ordena os 32 classificados por:
    1) colocação no grupo;
    2) pontos;
    3) saldo;
    4) gols pró;
    5) Elo.

    Depois monta uma árvore balanceada. Isso NÃO é a chave oficial da FIFA.
    É uma aproximação útil para simulação de cenários e probabilidades gerais.
    """
    q = qualified.sort_values(
        ["place", "pts", "gd", "gf", "elo"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    seeds = q["team"].tolist()

    pair_indices = [
        (0, 31),
        (15, 16),
        (7, 24),
        (8, 23),
        (3, 28),
        (12, 19),
        (4, 27),
        (11, 20),
        (1, 30),
        (14, 17),
        (6, 25),
        (9, 22),
        (2, 29),
        (13, 18),
        (5, 26),
        (10, 21),
    ]

    return [(seeds[i], seeds[j]) for i, j in pair_indices]


def simulate_knockout_tree(
    round32_pairs: List[Tuple[str, str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    penalty_sensitivity: float,
    rng: np.random.Generator,
) -> Tuple[str, Dict[str, List[str]], pd.DataFrame]:
    rounds = {
        "R32": [],
        "R16": [],
        "QF": [],
        "SF": [],
        "Final": [],
        "Champion": [],
    }
    log_rows = []

    current_pairs = round32_pairs
    round_order = ["R32", "R16", "QF", "SF", "Final"]
    next_round_name = {
        "R32": "R16",
        "R16": "QF",
        "QF": "SF",
        "SF": "Final",
        "Final": "Champion",
    }

    for round_name in round_order:
        winners = []
        for team_a, team_b in current_pairs:
            winner, ga, gb, method = simulate_knockout_winner(
                team_a,
                team_b,
                elo,
                host,
                base_total_goals,
                beta,
                host_bonus_elo,
                penalty_sensitivity,
                rng,
            )
            winners.append(winner)
            log_rows.append(
                {
                    "round": round_name,
                    "team_a": team_a,
                    "team_b": team_b,
                    "score_90min": f"{ga}-{gb}",
                    "winner": winner,
                    "method": method,
                }
            )

        rounds[next_round_name[round_name]] = winners

        if len(winners) == 1:
            break

        current_pairs = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    champion = rounds["Champion"][0]
    return champion, rounds, pd.DataFrame(log_rows)


# ============================================================
# 4. Simulações agregadas
# ============================================================


def run_tournament_simulations(
    groups: Dict[str, List[str]],
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
    penalty_sensitivity: float,
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted(elo.keys())

    counts = pd.DataFrame(
        0,
        index=teams,
        columns=["R32", "R16", "QF", "SF", "Final", "Champion"],
        dtype=float,
    )

    progress = st.progress(0)

    for s in range(n_sims):
        standings, qualified, _ = simulate_group_stage(
            groups,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            rng,
        )
        round32_pairs = build_round32_bracket(qualified)
        champion, rounds, _ = simulate_knockout_tree(
            round32_pairs,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            penalty_sensitivity,
            rng,
        )

        for t in qualified["team"].tolist():
            counts.loc[t, "R32"] += 1
        for stage in ["R16", "QF", "SF", "Final", "Champion"]:
            for t in rounds[stage]:
                counts.loc[t, stage] += 1

        if (s + 1) % max(1, n_sims // 100) == 0:
            progress.progress((s + 1) / n_sims)

    progress.empty()

    probs = counts / n_sims
    probs = probs.reset_index().rename(columns={"index": "team"})
    probs["elo"] = probs["team"].map(elo)
    probs["group"] = probs["team"].map({t: g for g, ts in groups.items() for t in ts})
    probs = probs.sort_values("Champion", ascending=False).reset_index(drop=True)
    return probs


def all_pairwise_probabilities(
    teams: List[str],
    elo: Dict[str, float],
    host: Dict[str, bool],
    base_total_goals: float,
    beta: float,
    host_bonus_elo: float,
) -> pd.DataFrame:
    rows = []
    for team_a, team_b in combinations(teams, 2):
        pr = match_probabilities(
            team_a,
            team_b,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
        )
        rows.append(pr)
    return pd.DataFrame(rows)


# ============================================================
# 5. Interface Streamlit
# ============================================================

st.set_page_config(
    page_title="Modelo Copa 2026",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Modelo interativo de previsão — Copa do Mundo 2026")
st.caption(
    "Poisson + Elo + Monte Carlo. Use como laboratório: edite ratings, bônus de sede e número de simulações."
)

with st.sidebar:
    st.header("Parâmetros")
    base_total_goals = st.slider("Média esperada de gols por jogo", 1.50, 3.50, 2.40, 0.05)
    beta = st.slider("Sensibilidade do Elo", 0.20, 2.00, 1.00, 0.05)
    host_bonus_elo = st.slider("Bônus de sede, em pontos Elo", 0, 180, 80, 5)
    penalty_sensitivity = st.slider("Sensibilidade em pênaltis", 0.00, 0.20, 0.05, 0.01)
    max_goals = st.slider("Máximo de gols na matriz de placares", 5, 14, 10, 1)
    n_sims = st.slider("Simulações Monte Carlo", 100, 20000, 2000, 100)
    seed = st.number_input("Semente aleatória", value=42, step=1)

st.subheader("Base de seleções")

uploaded = st.file_uploader(
    "Opcional: envie CSV com colunas team, group, elo, host",
    type=["csv"],
)

if uploaded is not None:
    teams_df = pd.read_csv(uploaded)
    required_cols = {"team", "group", "elo", "host"}
    if not required_cols.issubset(set(teams_df.columns)):
        st.error("O CSV precisa ter as colunas: team, group, elo, host.")
        st.stop()
else:
    teams_df = default_team_table()

edited = st.data_editor(
    teams_df,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "team": st.column_config.TextColumn("Seleção"),
        "group": st.column_config.TextColumn("Grupo"),
        "elo": st.column_config.NumberColumn("Elo", min_value=1000, max_value=2400, step=1),
        "host": st.column_config.CheckboxColumn("Sede?"),
    },
)

edited["elo"] = pd.to_numeric(edited["elo"], errors="coerce").fillna(1600)
edited["host"] = edited["host"].astype(bool)

elo = dict(zip(edited["team"], edited["elo"]))
host = dict(zip(edited["team"], edited["host"]))
groups = {
    g: edited.loc[edited["group"] == g, "team"].tolist()
    for g in sorted(edited["group"].unique())
}
teams = edited["team"].tolist()

if len(teams) != 48:
    st.warning(f"A base atual tem {len(teams)} seleções. O formato completo da Copa 2026 usa 48.")

bad_groups = {g: ts for g, ts in groups.items() if len(ts) != 4}
if bad_groups:
    st.warning("Há grupos com número diferente de 4 seleções. Revise antes de simular a Copa completa.")

# ------------------------------------------------------------
# Abas
# ------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Confronto direto",
        "Todos os pares possíveis",
        "Fase de grupos",
        "Simular Copa",
        "Árvore de uma simulação",
    ]
)

with tab1:
    st.subheader("Estimador de confronto direto")
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Seleção A", teams, index=0)
    with c2:
        default_b = 1 if len(teams) > 1 else 0
        team_b = st.selectbox("Seleção B", teams, index=default_b)

    if team_a == team_b:
        st.info("Escolha duas seleções diferentes.")
    else:
        probs = match_probabilities(
            team_a,
            team_b,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            max_goals=max_goals,
        )
        score_df = score_probability_table(
            team_a,
            team_b,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            max_goals=max_goals,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{team_a} vence", f"{probs['p_a']:.1%}")
        m2.metric("Empate", f"{probs['p_draw']:.1%}")
        m3.metric(f"{team_b} vence", f"{probs['p_b']:.1%}")
        m4.metric("Gols esperados", f"{probs['lambda_a']:.2f} x {probs['lambda_b']:.2f}")

        e1, e2, e3 = st.columns(3)
        e1.metric("Over 2.5", f"{probs['p_over_25']:.1%}")
        e2.metric("Under 2.5", f"{probs['p_under_25']:.1%}")
        e3.metric("Ambos marcam", f"{probs['p_btts']:.1%}")

        top_scores = (
            score_df.sort_values("prob", ascending=False)
            .head(12)[["placar", "prob"]]
            .assign(prob=lambda x: x["prob"].map(lambda v: f"{v:.1%}"))
        )
        st.write("Placares mais prováveis")
        st.dataframe(top_scores, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Todos os confrontos possíveis entre as seleções")
    st.write(
        "Aqui o painel calcula os 1.128 pares possíveis entre 48 seleções. "
        "Isso é útil para estudar confrontos que ainda podem aparecer no mata-mata."
    )

    pair_df = all_pairwise_probabilities(
        teams,
        elo,
        host,
        base_total_goals,
        beta,
        host_bonus_elo,
    )

    focus = st.multiselect("Filtrar seleções", teams, default=[])
    view_df = pair_df.copy()
    if focus:
        view_df = view_df.loc[
            view_df["team_a"].isin(focus) | view_df["team_b"].isin(focus)
        ]

    display_df = view_df.copy()
    for col in ["p_a", "p_draw", "p_b", "p_over_25", "p_under_25", "p_btts"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.1%}")
    display_df["lambda_a"] = display_df["lambda_a"].map(lambda v: f"{v:.2f}")
    display_df["lambda_b"] = display_df["lambda_b"].map(lambda v: f"{v:.2f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar confrontos possíveis em CSV",
        pair_df.to_csv(index=False).encode("utf-8"),
        file_name="confrontos_possiveis_copa2026.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Uma simulação da fase de grupos")
    if st.button("Simular fase de grupos"):
        rng = np.random.default_rng(int(seed))
        standings, qualified, group_matches = simulate_group_stage(
            groups,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            rng,
        )
        st.write("Jogos simulados")
        st.dataframe(group_matches, use_container_width=True, hide_index=True)
        st.write("Classificação por grupo")
        st.dataframe(standings, use_container_width=True, hide_index=True)
        st.write("Classificados para a fase de 32")
        st.dataframe(
            qualified.sort_values(["group", "place"]),
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.subheader("Monte Carlo da Copa completa")
    st.write(
        "Roda várias Copas simuladas e estima probabilidade de alcançar cada fase. "
        "O mata-mata usa uma árvore aproximada por seeding de campanha."
    )

    if st.button("Rodar Monte Carlo"):
        result = run_tournament_simulations(
            groups,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            penalty_sensitivity,
            int(n_sims),
            int(seed),
        )

        st.write("Probabilidades por seleção")
        show = result.copy()
        for col in ["R32", "R16", "QF", "SF", "Final", "Champion"]:
            show[col] = show[col].map(lambda v: f"{v:.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.write("Top 15 candidatos ao título")
        chart_df = result.head(15).set_index("team")[["Champion"]]
        st.bar_chart(chart_df)

        st.download_button(
            "Baixar resultados do Monte Carlo em CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="probabilidades_copa2026_monte_carlo.csv",
            mime="text/csv",
        )

with tab5:
    st.subheader("Árvore de uma Copa simulada")
    st.write(
        "Mostra uma realização completa do torneio. Boa para entender caminhos possíveis, "
        "mas não deve ser lida como previsão pontual."
    )

    if st.button("Gerar árvore de uma simulação"):
        rng = np.random.default_rng(int(seed))
        standings, qualified, _ = simulate_group_stage(
            groups,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            rng,
        )
        round32_pairs = build_round32_bracket(qualified)
        champion, rounds, log_df = simulate_knockout_tree(
            round32_pairs,
            elo,
            host,
            base_total_goals,
            beta,
            host_bonus_elo,
            penalty_sensitivity,
            rng,
        )

        st.success(f"Campeão simulado: {champion}")
        st.write("Classificados e seed aproximado")
        q_seeded = qualified.sort_values(
            ["place", "pts", "gd", "gf", "elo"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
        q_seeded.insert(0, "seed", np.arange(1, len(q_seeded) + 1))
        st.dataframe(q_seeded, use_container_width=True, hide_index=True)

        st.write("Mata-mata simulado")
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Baixar árvore simulada em CSV",
            log_df.to_csv(index=False).encode("utf-8"),
            file_name="arvore_simulada_copa2026.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Modelo educacional: substitua os ratings por Elo/FIFA atualizados e, se necessário, implemente a chave oficial da FIFA para maior precisão."
)
