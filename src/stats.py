"""
stats.py — Calcul des statistiques du portefeuille
===================================================
Rendements logarithmiques annualises, matrice de covariance,
et fonctions utilitaires pour evaluer un portefeuille donne.
"""

import numpy as np
import pandas as pd

JOURS_AN = 252   # jours de bourse par an


def calculer_rendements(prix):
    """
    Calcule les rendements logarithmiques journaliers.
    log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})

    Retourne un DataFrame de meme forme que prix (moins la 1ere ligne).
    """
    return np.log(prix / prix.shift(1)).dropna()


def statistiques_annuelles(rendements):
    """
    Annualise les rendements et la matrice de covariance.

    Retourne
    --------
    mu    : pd.Series  — rendement moyen annuel par action
    cov   : pd.DataFrame — matrice de covariance annuelle (variance-covariance)
    sigma : pd.Series  — volatilite annuelle (ecart-type) par action
    """
    mu    = rendements.mean() * JOURS_AN
    cov   = rendements.cov()  * JOURS_AN
    sigma = np.sqrt(np.diag(cov.values))
    sigma = pd.Series(sigma, index=cov.columns)

    return mu, cov, sigma


def perf_portefeuille(poids, mu, cov):
    """
    Calcule le rendement et la volatilite d'un portefeuille
    defini par son vecteur de poids.

    Parametres
    ----------
    poids : array-like (n,) — poids des actifs (doit sommer a 1)
    mu    : array-like (n,) — rendements annuels
    cov   : array-like (n,n) — matrice de covariance annuelle

    Retourne : (rendement, volatilite, ratio_sharpe)
    """
    w = np.array(poids)
    rendement  = w @ mu
    variance   = w @ cov @ w
    volatilite = np.sqrt(variance)
    sharpe     = rendement / volatilite   # taux sans risque = 0 pour simplifier

    return rendement, volatilite, sharpe


def afficher_resume(mu, sigma, top_n=10):
    """Affiche les actions avec le meilleur rendement ajuste au risque."""
    sharpe_indiv = mu / sigma
    resume = pd.DataFrame({
        "Rendement (%)":   (mu * 100).round(2),
        "Volatilite (%)":  (sigma * 100).round(2),
        "Sharpe indiv.":   sharpe_indiv.round(3),
    }).sort_values("Sharpe indiv.", ascending=False)

    print(f"\nTop {top_n} actions (ratio Sharpe individuel) :")
    print(resume.head(top_n).to_string())
    return resume


if __name__ == "__main__":
    from donnees import telecharger_prix

    prix = telecharger_prix([])   # charge depuis le cache
    rend = calculer_rendements(prix)
    mu, cov, sigma = statistiques_annuelles(rend)

    print(f"Univers : {len(mu)} actifs")
    afficher_resume(mu, sigma)
