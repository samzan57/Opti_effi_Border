"""
Unit tests for the portfolio statistics, Markowitz optimizer, and
VaR/CVaR/drawdown risk metrics (src/stats.py, src/optimiseur.py, src/risque.py).

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stats import calculer_rendements, statistiques_annuelles, perf_portefeuille
from optimiseur import portefeuille_min_variance, portefeuille_max_sharpe
from risque import var_historique, cvar_historique, calculer_drawdown, max_drawdown


def _prix_synthetiques():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    n_actifs = 4
    rendements = rng.normal(0.0003, 0.01, size=(500, n_actifs))
    prix = 100 * np.exp(np.cumsum(rendements, axis=0))
    return pd.DataFrame(prix, index=dates, columns=list("ABCD"))


def test_statistiques_annuelles_shapes_and_positivity():
    prix = _prix_synthetiques()
    rendements = calculer_rendements(prix)
    mu, cov, sigma = statistiques_annuelles(rendements)

    assert len(mu) == 4
    assert cov.shape == (4, 4)
    assert (sigma > 0).all()
    # La diagonale de la covariance doit correspondre a sigma^2
    assert np.allclose(np.diag(cov.values), sigma.values ** 2)


def test_portefeuille_min_variance_weights_sum_to_one_and_are_non_negative():
    prix = _prix_synthetiques()
    rendements = calculer_rendements(prix)
    mu, cov, _ = statistiques_annuelles(rendements)

    w = portefeuille_min_variance(mu.values, cov.values)

    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-8).all()  # pas de vente a decouvert


def test_max_sharpe_portfolio_has_at_least_min_variance_sharpe():
    # Par construction, le portefeuille max Sharpe ne peut pas avoir un
    # ratio de Sharpe strictement inferieur au portefeuille min variance.
    prix = _prix_synthetiques()
    rendements = calculer_rendements(prix)
    mu, cov, _ = statistiques_annuelles(rendements)

    w_minvar = portefeuille_min_variance(mu.values, cov.values)
    w_maxsharpe = portefeuille_max_sharpe(mu.values, cov.values, taux_sans_risque=0.04)

    _, _, sharpe_minvar = perf_portefeuille(w_minvar, mu.values, cov.values)
    _, _, sharpe_maxsharpe = perf_portefeuille(w_maxsharpe, mu.values, cov.values)

    assert sharpe_maxsharpe >= sharpe_minvar - 1e-6


def test_cvar_is_at_least_as_large_as_var():
    rng = np.random.default_rng(1)
    rendements_port = pd.Series(rng.standard_normal(2000) * 0.01)
    nav = 1_000_000

    var = var_historique(rendements_port, nav, confiance=0.95)
    cvar = cvar_historique(rendements_port, nav, confiance=0.95)

    assert cvar >= var


def test_max_drawdown_on_known_series():
    # Valeur : 100 -> 120 (sommet) -> 90 (creux) -> 110
    valeur = pd.Series([100, 120, 90, 110])

    dd = calculer_drawdown(valeur)
    mdd = max_drawdown(valeur)

    assert dd.iloc[1] == 0  # au sommet, drawdown nul
    assert mdd == pytest.approx((90 - 120) / 120)
