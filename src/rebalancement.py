"""
rebalancement.py — Plan de rebalancement vers le portefeuille optimal
======================================================================
Compare le portefeuille actuel (depuis IBKR) avec le portefeuille cible
et calcule exactement les ordres ACHAT / VENTE a passer.
"""

import numpy as np
import pandas as pd
from recuperer_positions import recuperer_portefeuille
from donnees import telecharger_prix, nettoyer_tickers, CORRECTIONS
from stats import calculer_rendements, statistiques_annuelles
from optimiseur import portefeuille_max_sharpe, portefeuille_min_variance
from paths import data

NAV_TOTALE       = 1_472_484
TAUX_SANS_RISQUE = 0.04
POIDS_MAX        = 0.15    # plafond par position (15% max) — evite surconcentration


def optimiser_avec_plafond(mu, cov, taux_rf, poids_max):
    """Max Sharpe avec contrainte de poids maximum par action."""
    from scipy.optimize import minimize

    n = len(mu)

    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - taux_rf) / v

    contraintes = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bornes = [(0, poids_max)] * n

    res = minimize(neg_sharpe, np.ones(n)/n, method="SLSQP",
                   bounds=bornes, constraints=contraintes,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def plan_rebalancement(portefeuille_actuel, poids_cibles, tickers_cibles,
                       nav, titre="Max Sharpe (plafonné 15%)"):
    """
    Compare positions actuelles vs cibles et genere les ordres.

    portefeuille_actuel : dict {ticker_yf: valeur_actuelle_en_$}
    poids_cibles        : array de poids optimaux
    tickers_cibles      : liste des tickers correspondants
    """
    # Valeur actuelle par ticker (0 si absent du portefeuille actuel)
    val_actuelle = {t: portefeuille_actuel.get(t, 0.0) for t in tickers_cibles}

    # Valeur cible par ticker
    val_cible = {t: poids_cibles[i] * nav
                 for i, t in enumerate(tickers_cibles)
                 if poids_cibles[i] > 0.0005}

    # Tous les tickers concernés (cibles + actuels hors cible)
    tous_tickers = set(val_cible.keys()) | set(portefeuille_actuel.keys())

    ordres = []
    for t in tous_tickers:
        actuel = portefeuille_actuel.get(t, 0.0)
        cible  = val_cible.get(t, 0.0)
        delta  = cible - actuel

        if abs(delta) < 100:   # ignore les ajustements < 100$
            continue

        ordres.append({
            "Ticker":        t,
            "Valeur actuelle ($)": round(actuel),
            "Valeur cible ($)":    round(cible),
            "Delta ($)":           round(delta),
            "Action":              "ACHAT" if delta > 0 else "VENTE",
        })

    df = pd.DataFrame(ordres).sort_values("Delta ($)", ascending=True)

    ventes  = df[df["Action"] == "VENTE"]
    achats  = df[df["Action"] == "ACHAT"]
    total_v = abs(ventes["Delta ($)"].sum())
    total_a = achats["Delta ($)"].sum()

    print(f"\n{'='*62}")
    print(f"  PLAN DE REBALANCEMENT — {titre}")
    print(f"{'='*62}")
    print(f"  NAV totale       : {nav:>12,.0f} $")
    print(f"  Total a vendre   : {total_v:>12,.0f} $")
    print(f"  Total a acheter  : {total_a:>12,.0f} $")
    print(f"  Nb d'ordres      : {len(df)} ({len(ventes)} ventes + {len(achats)} achats)")

    print(f"\n--- VENTES ---")
    print(ventes[["Ticker","Valeur actuelle ($)","Valeur cible ($)","Delta ($)"]].to_string(index=False))

    print(f"\n--- ACHATS ---")
    print(achats[["Ticker","Valeur actuelle ($)","Valeur cible ($)","Delta ($)"]].to_string(index=False))

    nom = titre.replace(" ", "_").replace("(","").replace(")","").replace("%","").lower()
    df.to_csv(data(f"rebalancement_{nom}.csv"), index=False)
    print(f"\n  Sauvegarde : {data(f'rebalancement_{nom}.csv')}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

print("Recuperation du portefeuille actuel...")
positions = recuperer_portefeuille()

# Valeur actuelle par ticker yfinance
portefeuille_actuel = {}
for p in positions:
    t = CORRECTIONS.get(p["ticker_ibkr"], p["ticker_yf"])
    portefeuille_actuel[t] = p["valeur"]

# Donnees et stats
tickers = nettoyer_tickers(positions)
prix    = telecharger_prix(tickers)
rend_df = calculer_rendements(prix)
mu, cov, sigma = statistiques_annuelles(rend_df)

# Portefeuille cible : Max Sharpe avec plafond 15%
print("Optimisation avec plafond 15% par position...")
w_ms_plafond = optimiser_avec_plafond(mu.values, cov.values,
                                       TAUX_SANS_RISQUE, POIDS_MAX)

vol  = np.sqrt(w_ms_plafond @ cov.values @ w_ms_plafond)
rend = w_ms_plafond @ mu.values
print(f"Portefeuille cible : rend={rend:.2%}  vol={vol:.2%}  Sharpe={(rend-TAUX_SANS_RISQUE)/vol:.2f}")

# Plan de rebalancement
plan_rebalancement(
    portefeuille_actuel,
    w_ms_plafond,
    mu.index.tolist(),
    NAV_TOTALE,
    titre="Max Sharpe plafonne 15%"
)
