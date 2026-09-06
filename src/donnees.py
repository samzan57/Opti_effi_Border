"""
donnees.py — Telechargement et nettoyage des donnees de marche
==============================================================
Recupere les cours historiques via yfinance pour tous les tickers
du portefeuille IBKR, gere les erreurs et sauvegarde en cache CSV.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import urllib3

# Fix SSL : sur certains systèmes (chemin certifi avec caractères non-ASCII),
# yfinance exige curl_cffi — on crée une session curl_cffi sans vérif SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from curl_cffi import requests as cffi_requests
_SESSION = cffi_requests.Session(impersonate="chrome", verify=False)

# Tickers exclus de l'optimisation (ETFs devises, levier, doublons)
EXCLUSIONS = {
    "FXC", "FXE", "FXF", "FXY",   # ETFs devises pures
    "SPXL",                         # ETF levier 3x
    "GOOG",                         # doublon de GOOGL
    "LILA",                         # doublon de LILAK
    "LLYVA",                        # doublon de LLYVK
}

# Corrections manuelles pour les tickers mal convertis
CORRECTIONS = {
    "NESN":   "NESN.SW",    # Nestle (SIX Zurich)
    "PGHN":   "PGHN.SW",    # Partners Group (SIX Zurich)
    "ROP":    "ROP",        # Roper Technologies (NYSE USD — erreur devise IBKR)
    "SQ":     "SQ",         # Block Inc (NYSE USD — erreur devise IBKR)
}

from paths import cache
FICHIER_CACHE = cache("prix_historiques.csv")


def nettoyer_tickers(positions):
    """
    Prend la liste de positions IBKR, applique les exclusions et corrections.
    Retourne une liste de tickers yfinance propres.
    """
    tickers = []
    for p in positions:
        ticker_ibkr = p["ticker_ibkr"]
        ticker_yf   = p["ticker_yf"]

        if ticker_ibkr in EXCLUSIONS:
            continue

        if ticker_ibkr in CORRECTIONS:
            ticker_yf = CORRECTIONS[ticker_ibkr]

        tickers.append(ticker_yf)

    return list(dict.fromkeys(tickers))  # supprime les doublons en conservant l'ordre


def telecharger_prix(tickers, periode="2y", forcer=False):
    """
    Telecharge les cours de cloture ajustes pour tous les tickers.
    Utilise un cache CSV pour eviter de re-telecharger a chaque fois.

    Retourne un DataFrame : index = dates, colonnes = tickers
    """
    if os.path.exists(FICHIER_CACHE) and not forcer:
        print(f"Chargement depuis le cache : {FICHIER_CACHE}")
        prix = pd.read_csv(FICHIER_CACHE, index_col=0, parse_dates=True)
        return prix

    print(f"Telechargement de {len(tickers)} tickers via yfinance ({periode})...")
    prix_brut = yf.download(
        tickers,
        period=periode,
        auto_adjust=True,
        progress=True,
        threads=True,
        session=_SESSION,
    )

    # Extraire uniquement les cours de cloture
    if isinstance(prix_brut.columns, pd.MultiIndex):
        prix = prix_brut["Close"]
    else:
        prix = prix_brut[["Close"]]
        prix.columns = tickers

    # Rapport sur les tickers manquants ou incomplets
    seuil_donnees = 0.7  # au moins 70% des jours doivent avoir des donnees
    n_jours = len(prix)
    tickers_ok      = []
    tickers_manquants = []

    for col in prix.columns:
        taux_remplissage = prix[col].notna().mean()
        if taux_remplissage >= seuil_donnees:
            tickers_ok.append(col)
        else:
            tickers_manquants.append((col, f"{taux_remplissage:.0%}"))

    if tickers_manquants:
        print(f"\n  Tickers ecartés (données insuffisantes) :")
        for t, taux in tickers_manquants:
            print(f"    {t:<16} {taux} de données disponibles")

    prix = prix[tickers_ok].dropna(how="all")

    # Interpolation limitee pour les quelques NaN restants
    prix = prix.ffill().dropna()

    print(f"\n  {len(tickers_ok)} tickers conserves | {len(prix)} jours de cotation")
    prix.to_csv(FICHIER_CACHE)
    print(f"  Cache sauvegarde : {FICHIER_CACHE}")

    return prix


if __name__ == "__main__":
    from recuperer_positions import recuperer_portefeuille

    print("Recuperation du portefeuille IBKR...")
    positions = recuperer_portefeuille()

    tickers = nettoyer_tickers(positions)
    print(f"\n{len(tickers)} tickers apres nettoyage :")
    print(tickers)

    prix = telecharger_prix(tickers, forcer=True)
    print(prix.tail())
