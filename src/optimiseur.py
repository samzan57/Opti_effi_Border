"""
optimiseur.py — Construction de la Frontiere Efficiente (Markowitz)
====================================================================
Utilise scipy.optimize.minimize pour resoudre :

  Pour chaque cible de rendement mu* :
    min  w' * Sigma * w          (minimiser la variance)
    s.t. w' * mu  = mu*           (atteindre le rendement cible)
         sum(w)   = 1             (budget investi a 100%)
         w_i >= 0                 (pas de vente a decouvert)

Le lieu geometrique de toutes ces solutions est la Frontiere Efficiente.
"""

import numpy as np
from scipy.optimize import minimize


def _variance(poids, cov):
    """Fonction objectif : variance du portefeuille."""
    w = np.array(poids)
    return w @ cov @ w


def portefeuille_min_variance(mu, cov):
    """
    Trouve le portefeuille de variance minimale (pointe gauche de la frontiere).
    Aucune contrainte sur le rendement, juste min variance avec sum(w)=1, w>=0.
    """
    n = len(mu)
    w0 = np.ones(n) / n   # point de depart : equi-pondere

    contraintes = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bornes = [(0, 1)] * n

    res = minimize(
        _variance,
        w0,
        args=(cov,),
        method="SLSQP",
        bounds=bornes,
        constraints=contraintes,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x


def portefeuille_max_sharpe(mu, cov, taux_sans_risque=0.04):
    """
    Trouve le portefeuille qui maximise le ratio de Sharpe.
    Revient a maximiser (mu - rf) / sigma.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        rendement  = w @ mu
        volatilite = np.sqrt(w @ cov @ w)
        return -(rendement - taux_sans_risque) / volatilite

    contraintes = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bornes = [(0, 1)] * n

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bornes,
        constraints=contraintes,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x


def tracer_frontiere(mu, cov, n_points=80):
    """
    Calcule n_points de la frontiere efficiente.
    Pour chaque rendement cible entre mu_min et mu_max,
    resout le probleme de minimisation de variance.

    Retourne
    --------
    vols  : np.ndarray (n_points,) — volatilites des portefeuilles frontiere
    rends : np.ndarray (n_points,) — rendements correspondants
    poids_list : list de np.ndarray — vecteurs de poids
    """
    mu_arr = np.array(mu)
    cov_arr = np.array(cov)
    n = len(mu_arr)

    # Bornes de rendement atteignables
    poids_mv  = portefeuille_min_variance(mu_arr, cov_arr)
    mu_min    = poids_mv @ mu_arr
    mu_max    = mu_arr.max() * 0.95   # plafond pour eviter les solutions extremes

    cibles = np.linspace(mu_min, mu_max, n_points)
    vols, rends, poids_list = [], [], []

    print(f"Calcul de la frontiere ({n_points} points)...")
    for i, mu_cible in enumerate(cibles):
        contraintes = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, m=mu_cible: w @ mu_arr - m},
        ]
        bornes = [(0, 1)] * n
        w0 = np.ones(n) / n

        res = minimize(
            _variance,
            w0,
            args=(cov_arr,),
            method="SLSQP",
            bounds=bornes,
            constraints=contraintes,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if res.success:
            vol  = np.sqrt(res.fun)
            vols.append(vol)
            rends.append(mu_cible)
            poids_list.append(res.x)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_points} points calcules...")

    print(f"  Frontiere tracee : {len(vols)} points valides")
    return np.array(vols), np.array(rends), poids_list


def portefeuilles_aleatoires(mu, cov, n=3000):
    """
    Simule n portefeuilles aleatoires pour illustrer l'espace risque-rendement
    et montrer visuellement pourquoi la frontiere est optimale.
    """
    mu_arr  = np.array(mu)
    cov_arr = np.array(cov)
    n_actifs = len(mu_arr)

    vols, rends, sharpes = [], [], []
    rng = np.random.default_rng(42)

    for _ in range(n):
        w = rng.dirichlet(np.ones(n_actifs))   # poids aleatoires qui somment a 1
        r = w @ mu_arr
        v = np.sqrt(w @ cov_arr @ w)
        vols.append(v)
        rends.append(r)
        sharpes.append(r / v)

    return np.array(vols), np.array(rends), np.array(sharpes)


if __name__ == "__main__":
    import pandas as pd
    from donnees import telecharger_prix
    from stats import calculer_rendements, statistiques_annuelles

    prix = telecharger_prix([])
    rend = calculer_rendements(prix)
    mu, cov, _ = statistiques_annuelles(rend)

    w_mv = portefeuille_min_variance(mu.values, cov.values)
    w_ms = portefeuille_max_sharpe(mu.values, cov.values)

    vol_mv  = np.sqrt(w_mv @ cov.values @ w_mv)
    rend_mv = w_mv @ mu.values
    vol_ms  = np.sqrt(w_ms @ cov.values @ w_ms)
    rend_ms = w_ms @ mu.values

    print(f"Portefeuille min variance  : rend={rend_mv:.2%}  vol={vol_mv:.2%}")
    print(f"Portefeuille max Sharpe    : rend={rend_ms:.2%}  vol={vol_ms:.2%}  Sharpe={rend_ms/vol_ms:.2f}")

    top5_mv = pd.Series(w_mv, index=mu.index).nlargest(5)
    top5_ms = pd.Series(w_ms, index=mu.index).nlargest(5)
    print(f"\nTop 5 poids min variance :\n{top5_mv.apply('{:.1%}'.format)}")
    print(f"\nTop 5 poids max Sharpe :\n{top5_ms.apply('{:.1%}'.format)}")
