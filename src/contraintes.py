"""
contraintes.py — Optimisation avec contraintes réelles
=======================================================
Un vrai hedge fund ne peut pas juste minimiser la variance.
Il doit respecter des regles pratiques :

1. LIQUIDITE    : ne pas investir plus de X% du volume journalier moyen
2. SECTEURS     : pas plus de 30% dans un meme secteur (diversification)
3. TURNOVER     : limiter les achats/ventes pour reduire les frais
4. POIDS MIN/MAX: entre 1% et 15% par position (pas de micro-positions)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

from donnees import telecharger_prix
from stats import calculer_rendements, statistiques_annuelles
from optimiseur import portefeuille_max_sharpe
from paths import graph, data

TAUX_SANS_RISQUE = 0.04
NAV              = 1_472_484

# ---------------------------------------------------------------------------
# 1. CLASSIFICATION SECTORIELLE
# ---------------------------------------------------------------------------

SECTEURS = {
    # Technologie
    "AAPL": "Technologie", "MSFT": "Technologie", "GOOGL": "Technologie",
    "META": "Technologie", "NVDA": "Technologie", "ORCL": "Technologie",
    "IBM": "Technologie",  "KLAC": "Technologie", "VGT": "Technologie",
    "PLTR": "Technologie", "TEM": "Technologie",

    # Consommation discretionnaire
    "AMZN": "Conso. discretionnaire", "TSLA": "Conso. discretionnaire",
    "HD": "Conso. discretionnaire",   "NVR": "Conso. discretionnaire",
    "DPZ": "Conso. discretionnaire",  "POOL": "Conso. discretionnaire",
    "CHTR": "Conso. discretionnaire", "NFLX": "Conso. discretionnaire",
    "FWONK": "Conso. discretionnaire","LLYVK": "Conso. discretionnaire",
    "LILAK": "Conso. discretionnaire","LLYVA": "Conso. discretionnaire",

    # Finance
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "BLK": "Finance",
    "AXP": "Finance", "COF": "Finance", "JEF": "Finance", "MCO": "Finance",
    "CB": "Finance",  "AON": "Finance", "IBKR": "Finance", "ALLY": "Finance",
    "GLE.PA": "Finance", "BN.PA": "Finance", "AGN.AS": "Finance",

    # Sante
    "JNJ": "Sante", "UNH": "Sante", "MCK": "Sante", "DVA": "Sante",
    "VRSN": "Sante",

    # Energie
    "XOM": "Energie", "OXY": "Energie", "VST": "Energie",

    # Consommation de base
    "KO": "Conso. de base", "KR": "Conso. de base", "KHC": "Conso. de base",
    "DEO": "Conso. de base", "STZ": "Conso. de base",
    "NESN.SW": "Conso. de base",

    # Industrie
    "GE": "Industrie", "GD": "Industrie", "FDX": "Industrie",
    "GWW": "Industrie", "LPX": "Industrie", "ROP": "Industrie",
    "AIR.PA": "Industrie", "STLAP.PA": "Industrie",

    # Materiaux / Or
    "GLD": "Matieres premieres", "NEM": "Matieres premieres",

    # Telecom
    "TMUS": "Telecom", "SIRI": "Telecom", "ENGI.PA": "Telecom",

    # Europe diversifie
    "ASML.AS": "Technologie", "AI.PA": "Technologie",
    "CAC.PA": "Finance",      "CAP.PA": "Technologie",
    "DSY.PA": "Technologie",  "KER.PA": "Conso. discretionnaire",
    "OR.PA": "Conso. de base","PHIA.AS": "Sante",
    "PGHN.SW": "Finance",     "MA": "Finance", "V": "Finance",

    # ETFs
    "QQQ": "ETF", "SPY": "ETF", "VGT": "ETF",
    "EQIX": "Immobilier",

    # Divers
    "ASST": "Divers", "NUAI": "Technologie", "FXC": "Devises",
    "FXE": "Devises", "FXF": "Devises", "FXY": "Devises",
}


def get_secteur(ticker):
    return SECTEURS.get(ticker, "Autre")


# ---------------------------------------------------------------------------
# 2. OPTIMISATION AVEC CONTRAINTES REELLES
# ---------------------------------------------------------------------------

def optimiser_contraint(mu, cov, tickers,
                        poids_max=0.15,
                        poids_min=0.01,
                        max_secteur=0.30,
                        poids_actuels=None,
                        max_turnover=0.40,
                        taux_rf=TAUX_SANS_RISQUE):
    """
    Maximise le Sharpe avec contraintes réelles.

    Contraintes :
    - Poids entre poids_min et poids_max par action
    - Pas plus de max_secteur dans un meme secteur
    - Turnover max par rapport aux poids actuels
    """
    n = len(mu)

    # Matrice sectorielle : secteurs[i] = nom du secteur de l'action i
    secteurs = [get_secteur(t) for t in tickers]
    noms_secteurs = list(set(secteurs))

    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        if v < 1e-10:
            return 0
        return -(r - taux_rf) / v

    contraintes = [
        # Budget : somme des poids = 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]

    # Contrainte sectorielle : chaque secteur <= max_secteur
    for secteur in noms_secteurs:
        idx = [i for i, s in enumerate(secteurs) if s == secteur]
        if len(idx) > 1:
            contraintes.append({
                "type": "ineq",
                "fun": lambda w, ix=idx: max_secteur - np.sum(w[ix])
            })

    # Contrainte de turnover
    if poids_actuels is not None:
        contraintes.append({
            "type": "ineq",
            "fun": lambda w: max_turnover - np.sum(np.abs(w - poids_actuels)) / 2
        })

    bornes = [(poids_min, poids_max)] * n
    w0     = np.ones(n) / n

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bornes, constraints=contraintes,
                   options={"ftol": 1e-10, "maxiter": 2000})

    return res.x, res.success


# ---------------------------------------------------------------------------
# 3. ANALYSE DE LIQUIDITE
# ---------------------------------------------------------------------------

def analyser_liquidite(prix, poids, tickers, nav):
    """
    Verifie si les positions sont liquides.
    Estime le volume moyen journalier et compare a la position cible.
    Retourne les actions potentiellement illiquides.
    """
    print(f"\n  {'Ticker':<14} {'Poids':>7} {'Montant ($)':>12} {'Liquidite':>12}")
    print(f"  {'─'*50}")

    alertes = []
    for i, t in enumerate(tickers):
        if poids[i] < 0.001:
            continue
        montant = poids[i] * nav
        # Heuristique : actions < 50M$ de capitalisation journaliere = illiquide
        # Pour simplifier on marque comme "A verifier" si poids > 10%
        liquidite = "OK" if poids[i] <= 0.12 else "A verifier"
        if liquidite != "OK":
            alertes.append(t)
        print(f"  {t:<14} {poids[i]:>6.1%}  {montant:>12,.0f}   {liquidite}")

    return alertes


# ---------------------------------------------------------------------------
# 4. ANALYSE DE TURNOVER
# ---------------------------------------------------------------------------

def analyser_turnover(poids_actuel, poids_cible, tickers, nav):
    """
    Calcule le turnover (% du portefeuille qui change de mains).
    Turnover = somme des achats = somme des ventes (en % de la NAV).
    """
    delta   = np.abs(poids_cible - poids_actuel)
    turnover = np.sum(delta) / 2   # divise par 2 car achat = vente

    cout_estime = turnover * nav * 0.001   # hypothese : 0.1% de frais

    print(f"\n  Turnover du rebalancement : {turnover:.1%} de la NAV")
    print(f"  Montant total a echanger  : {turnover * nav:,.0f} $")
    print(f"  Frais estimes (0.1%)      : {cout_estime:,.0f} $")

    return turnover


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------

def visualiser_contraintes(poids_libre, poids_contraint, tickers, mu, cov):
    """Compare portefeuille sans contraintes vs avec contraintes."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Impact des contraintes réelles sur l'allocation",
                 color="white", fontsize=13, fontweight="bold")

    import matplotlib.cm as cm

    for ax, poids, titre in zip(
        axes,
        [poids_libre, poids_contraint],
        ["Sans contraintes (Max Sharpe pur)", "Avec contraintes reelles"]
    ):
        ax.set_facecolor("#0f0f1a")
        series = pd.Series(poids, index=tickers)
        series = series[series > 0.005].sort_values(ascending=False).head(15)

        # Colorer par secteur
        couleurs_secteur = {
            "Technologie": "#00e5ff", "Finance": "#ffd700",
            "Sante": "#00ff88",       "Energie": "#ff6b35",
            "Conso. de base": "#c77dff", "Industrie": "#ff6b6b",
            "Matieres premieres": "#ffd700", "Telecom": "#74b9ff",
            "Conso. discretionnaire": "#fd79a8", "Autre": "#b2bec3",
            "ETF": "#a29bfe", "Immobilier": "#55efc4",
        }
        couleurs = [couleurs_secteur.get(get_secteur(t), "#b2bec3")
                    for t in series.index]

        bars = ax.barh(series.index, series.values * 100,
                       color=couleurs, edgecolor="#333355", linewidth=0.5)

        # Valeur sur chaque barre
        for bar, val in zip(bars, series.values):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va="center", color="white", fontsize=8)

        vol  = np.sqrt(poids @ cov @ poids)
        rend = poids @ mu
        sharpe = (rend - TAUX_SANS_RISQUE) / vol
        n_pos = (poids > 0.005).sum()

        ax.set_title(f"{titre}\nRend={rend:.1%}  Vol={vol:.1%}  Sharpe={sharpe:.2f}  "
                     f"Positions={n_pos}",
                     color="white", fontsize=9)
        ax.set_xlabel("Poids (%)", color="white")
        ax.tick_params(colors="white")
        ax.invert_yaxis()
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")

    plt.tight_layout()
    plt.savefig(graph("contraintes_reelles.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Graphique sauvegarde : {graph('contraintes_reelles.png')}")
    plt.show()


def visualiser_secteurs(poids_libre, poids_contraint, tickers):
    """Compare l'exposition sectorielle avant/apres contraintes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Exposition sectorielle — Sans vs Avec contraintes",
                 color="white", fontsize=13, fontweight="bold")

    import matplotlib.cm as cm

    for ax, poids, titre in zip(
        axes,
        [poids_libre, poids_contraint],
        ["Sans contraintes", "Avec contraintes (max 30%/secteur)"]
    ):
        ax.set_facecolor("#0f0f1a")
        expo = {}
        for i, t in enumerate(tickers):
            s = get_secteur(t)
            expo[s] = expo.get(s, 0) + poids[i]

        expo = {k: v for k, v in expo.items() if v > 0.005}
        expo = dict(sorted(expo.items(), key=lambda x: x[1], reverse=True))

        couleurs = cm.plasma(np.linspace(0.1, 0.9, len(expo)))
        wedges, texts, autotexts = ax.pie(
            list(expo.values()),
            labels=list(expo.keys()),
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            colors=couleurs, startangle=140, pctdistance=0.78
        )
        for t2 in texts:
            t2.set_color("white"); t2.set_fontsize(8)
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(7)
        ax.set_title(titre, color="white", fontsize=10, pad=15)

    plt.tight_layout()
    plt.savefig(graph("exposition_sectorielle.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Graphique sauvegarde : {graph('exposition_sectorielle.png')}")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  OPTIMISATION AVEC CONTRAINTES REELLES")
    print("=" * 60)

    prix       = telecharger_prix([])
    rendements = calculer_rendements(prix)
    mu, cov, sigma = statistiques_annuelles(rendements)
    tickers = mu.index.tolist()
    n = len(tickers)

    # Portefeuille sans contraintes
    w_libre = portefeuille_max_sharpe(mu.values, cov.values, TAUX_SANS_RISQUE)

    # Portefeuille actuel (equi-pondere)
    w_actuel = np.ones(n) / n

    print("\n[1] Optimisation avec contraintes reelles...")
    print(f"    Contraintes : poids 1%-15% | max 30%/secteur | turnover max 40%")
    w_contraint, succes = optimiser_contraint(
        mu.values, cov.values, tickers,
        poids_max=0.15,
        poids_min=0.01,
        max_secteur=0.30,
        poids_actuels=w_actuel,
        max_turnover=0.40,
    )
    print(f"    Optimisation {'reussie' if succes else 'echouee (solution approchee)'}")

    # Performances comparees
    def perf(w):
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        return r, v, (r - TAUX_SANS_RISQUE) / v

    r_l, v_l, s_l = perf(w_libre)
    r_c, v_c, s_c = perf(w_contraint)

    print(f"\n  {'':30} {'Sans contraintes':>18} {'Avec contraintes':>18}")
    print(f"  {'─'*68}")
    print(f"  {'Rendement annuel':30} {r_l:>17.2%} {r_c:>17.2%}")
    print(f"  {'Volatilite annuelle':30} {v_l:>17.2%} {v_c:>17.2%}")
    print(f"  {'Ratio de Sharpe':30} {s_l:>17.2f} {s_c:>17.2f}")
    print(f"  {'Nb de positions':30} {(w_libre>0.005).sum():>17} {(w_contraint>0.005).sum():>17}")

    # Exposition sectorielle
    print("\n[2] Exposition sectorielle (avec contraintes) :")
    expo = {}
    for i, t in enumerate(tickers):
        s = get_secteur(t)
        if w_contraint[i] > 0.005:
            expo[s] = expo.get(s, 0) + w_contraint[i]
    for secteur, poids in sorted(expo.items(), key=lambda x: x[1], reverse=True):
        barre = "█" * int(poids * 50)
        print(f"  {secteur:<28} {poids:>5.1%}  {barre}")

    # Liquidite
    print("\n[3] Analyse de liquidite (positions > 0.1%) :")
    analyser_liquidite(prix, w_contraint, tickers, NAV)

    # Turnover
    print("\n[4] Analyse du turnover :")
    analyser_turnover(w_actuel, w_contraint, tickers, NAV)

    # Allocation finale
    print("\n[5] Allocation finale avec contraintes :")
    alloc = pd.DataFrame({
        "Poids (%)":    (w_contraint * 100).round(2),
        "Montant ($)":  (w_contraint * NAV).round(0).astype(int),
        "Secteur":      [get_secteur(t) for t in tickers],
    }, index=tickers)
    alloc = alloc[alloc["Poids (%)"] > 0.5].sort_values("Poids (%)", ascending=False)
    print(alloc.to_string())
    alloc.to_csv(data("allocation_contrainte.csv"))
    print(f"\n  Sauvegarde : {data('allocation_contrainte.csv')}")

    # Visualisations
    print("\n[6] Generation des graphiques...")
    visualiser_contraintes(w_libre, w_contraint, tickers, mu.values, cov.values)
    visualiser_secteurs(w_libre, w_contraint, tickers)

    print("\nTermine.")
