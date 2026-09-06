"""
risque.py — Mesures de risque avancees : VaR, CVaR, Drawdown, Stress Tests
===========================================================================

VaR (Value at Risk) :
    "Je perds au maximum X$ dans 95% des cas sur 1 jour"
    → quantile des pertes historiques

CVaR (Conditional VaR / Expected Shortfall) :
    "Quand ca depasse la VaR, je perds en moyenne X$"
    → moyenne des pires pertes au-dela de la VaR
    → plus prudent que la VaR car capture les queues de distribution

Drawdown :
    "La pire chute depuis un sommet sur toute la periode"
    → mesure la douleur reelle d'un investisseur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from donnees import telecharger_prix
from stats import calculer_rendements, statistiques_annuelles
from optimiseur import portefeuille_max_sharpe, portefeuille_min_variance
from paths import graph

TAUX_SANS_RISQUE = 0.04
NAV              = 1_472_484


# ---------------------------------------------------------------------------
# 1. VAR ET CVAR
# ---------------------------------------------------------------------------

def var_historique(rendements_port, nav, confiance=0.95):
    """
    VaR historique : quantile empirique des pertes.
    Pas d'hypothese sur la distribution — utilise les vrais rendements passes.
    """
    seuil = np.percentile(rendements_port, (1 - confiance) * 100)
    var   = -seuil * nav
    return var


def cvar_historique(rendements_port, nav, confiance=0.95):
    """
    CVaR : moyenne des pertes qui depassent la VaR.
    Appele aussi Expected Shortfall (ES).
    """
    seuil           = np.percentile(rendements_port, (1 - confiance) * 100)
    queues_extremes = rendements_port[rendements_port <= seuil]
    cvar            = -queues_extremes.mean() * nav
    return cvar


def var_parametrique(rendements_port, nav, confiance=0.95):
    """
    VaR parametrique : suppose une distribution normale.
    Rapide mais sous-estime les queues epaisses (fat tails).
    """
    mu_j    = rendements_port.mean()
    sigma_j = rendements_port.std()
    z       = stats.norm.ppf(1 - confiance)
    var     = -(mu_j + z * sigma_j) * nav
    return var


def rapport_risque(rendements_port, nav, nom="Portefeuille", confiance=0.95):
    """Rapport complet VaR/CVaR pour un portefeuille."""
    var_hist  = var_historique(rendements_port, nav, confiance)
    cvar_hist = cvar_historique(rendements_port, nav, confiance)
    var_param = var_parametrique(rendements_port, nav, confiance)

    rend_ann  = rendements_port.mean() * 252
    vol_ann   = rendements_port.std()  * np.sqrt(252)
    sharpe    = (rend_ann - TAUX_SANS_RISQUE) / vol_ann

    print(f"\n  {nom}")
    print(f"  {'─'*50}")
    print(f"  Rendement annuel     : {rend_ann:>8.2%}")
    print(f"  Volatilite annuelle  : {vol_ann:>8.2%}")
    print(f"  Ratio de Sharpe      : {sharpe:>8.2f}")
    print(f"  VaR {confiance:.0%} historique  : {var_hist:>10,.0f} $ / jour")
    print(f"  VaR {confiance:.0%} parametrique: {var_param:>10,.0f} $ / jour")
    print(f"  CVaR {confiance:.0%} (ES)        : {cvar_hist:>10,.0f} $ / jour")
    print(f"  → En cas de mauvaise journee (top 5%), perte moyenne : {cvar_hist:,.0f} $")

    return {"var_hist": var_hist, "cvar_hist": cvar_hist,
            "var_param": var_param, "rend_ann": rend_ann,
            "vol_ann": vol_ann, "sharpe": sharpe}


# ---------------------------------------------------------------------------
# 2. DRAWDOWN
# ---------------------------------------------------------------------------

def calculer_drawdown(valeur_portefeuille):
    """
    Calcule la serie de drawdown : chute en % depuis le dernier sommet.

    drawdown[t] = (valeur[t] - max(valeur[0..t])) / max(valeur[0..t])
    """
    sommet    = valeur_portefeuille.cummax()
    drawdown  = (valeur_portefeuille - sommet) / sommet
    return drawdown


def max_drawdown(valeur_portefeuille):
    """Retourne le drawdown maximum (pire chute depuis un sommet)."""
    dd = calculer_drawdown(valeur_portefeuille)
    return dd.min()


def reconstruire_valeur(rendements_port, nav_initiale=100):
    """Reconstruit la courbe de valeur a partir des rendements journaliers."""
    return nav_initiale * (1 + rendements_port).cumprod()


# ---------------------------------------------------------------------------
# 3. STRESS TESTS
# ---------------------------------------------------------------------------

CRISES = {
    "Covid (fev-mars 2020)":     ("2020-02-19", "2020-03-23"),
    "Hausse taux Fed (2022)":    ("2022-01-03", "2022-10-12"),
    "Mini-crash oct 2023":       ("2023-07-31", "2023-10-27"),
    "Krach tarifaire 2025":      ("2025-02-19", "2025-04-08"),
}

def stress_test(prix, poids, tickers, nav):
    """
    Simule les pertes du portefeuille pendant les grandes crises historiques.
    """
    print(f"\n  {'Crise':<30} {'Duree':>8} {'Perte (%)':>10} {'Perte ($)':>12}")
    print(f"  {'─'*64}")

    resultats = []
    for nom_crise, (debut, fin) in CRISES.items():
        try:
            prix_crise = prix.loc[debut:fin, tickers]
            if len(prix_crise) < 2:
                continue
            # Rendement du portefeuille sur la periode
            rend_periode = (prix_crise.iloc[-1] / prix_crise.iloc[0] - 1)
            rend_port    = float(rend_periode @ poids)
            perte_dollar = rend_port * nav
            n_jours      = len(prix_crise)

            print(f"  {nom_crise:<30} {n_jours:>6}j  {rend_port:>9.2%}  {perte_dollar:>12,.0f} $")
            resultats.append((nom_crise, rend_port, perte_dollar, n_jours))
        except Exception:
            continue

    return resultats


# ---------------------------------------------------------------------------
# 4. VISUALISATION
# ---------------------------------------------------------------------------

def visualiser_risque(prix, poids_actuel, poids_optimal, tickers,
                      rendements, nav, sauvegarder=None):
    if sauvegarder is None:
        sauvegarder = graph("analyse_risque.png")

    r_actuel  = rendements[tickers] @ poids_actuel
    r_optimal = rendements[tickers] @ poids_optimal

    val_actuel  = reconstruire_valeur(r_actuel)
    val_optimal = reconstruire_valeur(r_optimal)
    dd_actuel   = calculer_drawdown(val_actuel)
    dd_optimal  = calculer_drawdown(val_optimal)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Analyse du Risque — Portefeuille IBKR Demo",
                 color="white", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    couleurs = {"actuel": "#ff6b6b", "optimal": "#00e5ff"}

    # --- 1. Courbe de valeur normalisee ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0f0f1a")
    ax1.plot(val_actuel.index,  val_actuel.values,  color=couleurs["actuel"],
             linewidth=1.8, label="Portefeuille actuel")
    ax1.plot(val_optimal.index, val_optimal.values, color=couleurs["optimal"],
             linewidth=1.8, label="Portefeuille optimal (Max Sharpe)")
    ax1.axhline(100, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
    ax1.set_title("Evolution de la valeur (base 100)", color="white")
    ax1.set_ylabel("Valeur", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333355")

    # --- 2. Drawdown ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0f0f1a")
    ax2.fill_between(dd_actuel.index,  dd_actuel.values  * 100, 0,
                     color=couleurs["actuel"],  alpha=0.5, label="Actuel")
    ax2.fill_between(dd_optimal.index, dd_optimal.values * 100, 0,
                     color=couleurs["optimal"], alpha=0.5, label="Optimal")
    ax2.set_title("Drawdown (%)", color="white")
    ax2.set_ylabel("Chute depuis le sommet (%)", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    mdd_a = max_drawdown(val_actuel)  * 100
    mdd_o = max_drawdown(val_optimal) * 100
    ax2.set_title(f"Drawdown  |  Max actuel: {mdd_a:.1f}%  |  Max optimal: {mdd_o:.1f}%",
                  color="white", fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

    # --- 3. Distribution des rendements journaliers ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0f0f1a")
    ax3.hist(r_actuel  * 100, bins=60, color=couleurs["actuel"],
             alpha=0.6, density=True, label="Actuel")
    ax3.hist(r_optimal * 100, bins=60, color=couleurs["optimal"],
             alpha=0.6, density=True, label="Optimal")

    var_a = var_historique(r_actuel,  1, 0.95) * 100
    var_o = var_historique(r_optimal, 1, 0.95) * 100
    ax3.axvline(-var_a, color=couleurs["actuel"],  linestyle="--",
                linewidth=1.5, label=f"VaR 95% actuel  = {var_a:.2f}%")
    ax3.axvline(-var_o, color=couleurs["optimal"], linestyle="--",
                linewidth=1.5, label=f"VaR 95% optimal = {var_o:.2f}%")
    ax3.set_title("Distribution des rendements journaliers", color="white")
    ax3.set_xlabel("Rendement (%)", color="white")
    ax3.set_ylabel("Densite", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#333355")

    # --- 4. Tableau recapitulatif ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0f0f1a")
    ax4.axis("off")

    var_a_usd  = var_historique(r_actuel,  nav, 0.95)
    cvar_a_usd = cvar_historique(r_actuel,  nav, 0.95)
    var_o_usd  = var_historique(r_optimal, nav, 0.95)
    cvar_o_usd = cvar_historique(r_optimal, nav, 0.95)
    rend_a   = r_actuel.mean()  * 252
    rend_o   = r_optimal.mean() * 252
    vol_a    = r_actuel.std()   * np.sqrt(252)
    vol_o    = r_optimal.std()  * np.sqrt(252)

    data = [
        ["Metrique",            "Actuel",                    "Optimal"],
        ["Rendement annuel",    f"{rend_a:.2%}",             f"{rend_o:.2%}"],
        ["Volatilite annuelle", f"{vol_a:.2%}",              f"{vol_o:.2%}"],
        ["Sharpe",              f"{(rend_a-0.04)/vol_a:.2f}",f"{(rend_o-0.04)/vol_o:.2f}"],
        ["VaR 95% ($/jour)",    f"{var_a_usd:,.0f}",           f"{var_o_usd:,.0f}"],
        ["CVaR 95% ($/jour)",   f"{cvar_a_usd:,.0f}",          f"{cvar_o_usd:,.0f}"],
        ["Max Drawdown",        f"{mdd_a:.1f}%",             f"{mdd_o:.1f}%"],
    ]

    table = ax4.table(cellText=data[1:], colLabels=data[0],
                      cellLoc="center", loc="center",
                      bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1a1a2e" if r == 0 else "#0f0f1a")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333355")
        if r > 0 and c == 2:
            cell.set_facecolor("#0d2b0d")
    ax4.set_title("Tableau comparatif", color="white", fontsize=10)

    plt.savefig(sauvegarder, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Graphique sauvegarde : {sauvegarder}")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ANALYSE DU RISQUE — VaR / CVaR / DRAWDOWN")
    print("=" * 60)

    prix       = telecharger_prix([])
    rendements = calculer_rendements(prix)
    mu, cov, sigma = statistiques_annuelles(rendements)

    # Portefeuille actuel : equi-pondere (approximation)
    n = len(mu)
    w_actuel  = np.ones(n) / n
    w_optimal = portefeuille_max_sharpe(mu.values, cov.values, TAUX_SANS_RISQUE)

    r_actuel  = (rendements @ w_actuel)
    r_optimal = (rendements @ w_optimal)

    # --- Rapports VaR / CVaR ---
    print("\n[ VaR et CVaR — Comparaison ]")
    stats_a = rapport_risque(r_actuel,  NAV, "Portefeuille actuel (equi-pondere)")
    stats_o = rapport_risque(r_optimal, NAV, "Portefeuille optimal (Max Sharpe)")

    # --- Drawdown ---
    print("\n[ Drawdown Maximum ]")
    val_a = reconstruire_valeur(r_actuel)
    val_o = reconstruire_valeur(r_optimal)
    mdd_a = max_drawdown(val_a)
    mdd_o = max_drawdown(val_o)
    print(f"  Portefeuille actuel  : Max Drawdown = {mdd_a:.2%}")
    print(f"  Portefeuille optimal : Max Drawdown = {mdd_o:.2%}")

    # --- Stress Tests ---
    print("\n[ Stress Tests — Portefeuille optimal pendant les crises ]")
    stress_test(prix, w_optimal, mu.index.tolist(), NAV)

    print("\n[ Stress Tests — Portefeuille actuel pendant les crises ]")
    stress_test(prix, w_actuel, mu.index.tolist(), NAV)

    # --- Visualisation ---
    print("\n[ Generation des graphiques... ]")
    visualiser_risque(prix, w_actuel, w_optimal,
                      mu.index.tolist(), rendements, NAV)

    print("\nTermine.")
