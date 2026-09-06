"""
allocation.py — Allocation complete du portefeuille optimise
============================================================
Genere le tableau complet des montants a investir en dollars
pour les deux portefeuilles optimaux (Max Sharpe et Min Variance).
"""

import numpy as np
import pandas as pd
from donnees import telecharger_prix
from stats import calculer_rendements, statistiques_annuelles
from optimiseur import portefeuille_min_variance, portefeuille_max_sharpe
from paths import data

NAV_TOTALE = 1_472_484   # dollars — NAV compte demo IBKR

TAUX_SANS_RISQUE = 0.04


def generer_allocation(w, tickers, nav, titre):
    poids = pd.Series(w, index=tickers)
    poids = poids[poids > 0.0005].sort_values(ascending=False)  # ignore < 0.05%

    df = pd.DataFrame({
        "Poids (%)":        (poids * 100).round(2),
        "Montant ($)":      (poids * nav).round(0).astype(int),
    })
    df.index.name = "Ticker"

    vol   = np.sqrt(w @ cov.values @ w)
    rend  = w @ mu.values
    sharpe = (rend - TAUX_SANS_RISQUE) / vol

    print(f"\n{'='*58}")
    print(f"  {titre}")
    print(f"{'='*58}")
    print(f"  Rendement annuel attendu : {rend:.2%}")
    print(f"  Volatilite annuelle      : {vol:.2%}")
    print(f"  Ratio de Sharpe          : {sharpe:.2f}")
    print(f"  NAV investie             : {nav:,.0f} $")
    print(f"  Nombre de positions      : {len(df)}")
    print(f"\n{df.to_string()}")

    nom_csv = data(titre.replace(" ", "_").lower() + "_allocation.csv")
    df.to_csv(nom_csv)
    print(f"\n  Sauvegarde : {nom_csv}")
    return df


# --- Chargement des donnees ---
prix = telecharger_prix([])   # depuis le cache
rend_df = calculer_rendements(prix)
mu, cov, sigma = statistiques_annuelles(rend_df)

# --- Optimisation ---
w_ms = portefeuille_max_sharpe(mu.values, cov.values, TAUX_SANS_RISQUE)
w_mv = portefeuille_min_variance(mu.values, cov.values)

# --- Allocations ---
df_ms = generer_allocation(w_ms, mu.index.tolist(), NAV_TOTALE, "Portefeuille Max Sharpe")
df_mv = generer_allocation(w_mv, mu.index.tolist(), NAV_TOTALE, "Portefeuille Min Variance")

# --- Comparaison cote a cote ---
print(f"\n{'='*58}")
print("  COMPARAISON MAX SHARPE vs MIN VARIANCE")
print(f"{'='*58}")
compare = df_ms.join(df_mv, how="outer", lsuffix="_MS", rsuffix="_MV").fillna(0)
compare["Poids (%)_MS"]  = compare["Poids (%)_MS"].round(2)
compare["Poids (%)_MV"]  = compare["Poids (%)_MV"].round(2)
compare["Montant ($)_MS"] = compare["Montant ($)_MS"].astype(int)
compare["Montant ($)_MV"] = compare["Montant ($)_MV"].astype(int)
compare.columns = ["Poids MS(%)", "Montant MS($)", "Poids MV(%)", "Montant MV($)"]
compare = compare.sort_values("Poids MS(%)", ascending=False)
print(compare.to_string())
compare.to_csv(data("comparaison_allocations.csv"))
print(f"\n  Sauvegarde : {data('comparaison_allocations.csv')}")
