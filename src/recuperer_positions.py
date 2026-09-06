"""
Recuperation des positions du portefeuille IBKR via TWS API
===========================================================
Prerequis : TWS doit etre ouvert et l'API activee.
Host/port/client ID sont lus depuis les variables d'environnement
IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID (voir .env.example).
"""

import os
import time
import threading
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

load_dotenv()

class IBKRPortfolio(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.positions = []
        self.fini = threading.Event()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in (2104, 2106, 2158, 2119):  # messages info normaux
            print(f"  [Erreur {errorCode}] {errorString}")

    def updatePortfolio(self, contract, position, marketPrice, marketValue,
                        averageCost, unrealizedPNL, realizedPNL, accountName):
        if position != 0 and contract.secType == "STK":
            ticker_ibkr = contract.symbol
            exchange    = contract.primaryExchange or contract.exchange
            currency    = contract.currency

            # Conversion ticker IBKR -> ticker yfinance
            ticker_yf = convertir_ticker(ticker_ibkr, exchange, currency)

            self.positions.append({
                "ticker_ibkr": ticker_ibkr,
                "ticker_yf":   ticker_yf,
                "exchange":    exchange,
                "currency":    currency,
                "position":    position,
                "valeur":      marketValue,
            })

    def accountDownloadEnd(self, accountName):
        self.fini.set()

    def nextValidId(self, orderId):
        # Demande les donnees du portefeuille des la connexion etablie
        self.reqAccountUpdates(True, "")


def convertir_ticker(symbol, exchange, currency):
    """
    Convertit un ticker IBKR en ticker yfinance.
    - Actions europeennes : ajoute le suffixe de place boursiere
    - Actions US : ticker brut
    """
    suffixes = {
        "XETRA":  ".DE",   # Allemagne
        "SBF":    ".PA",   # France (Euronext Paris)
        "AEB":    ".AS",   # Pays-Bas (Euronext Amsterdam)
        "BVME":   ".MI",   # Italie (Borsa Italiana)
        "BM":     ".MC",   # Espagne (Bolsa Madrid)
        "LSE":    ".L",    # Royaume-Uni
        "VSE":    ".VI",   # Autriche
        "SIX":    ".SW",   # Suisse
        "IBIS":   ".DE",   # Xetra alternatif
        "XTSX":   ".TO",   # Canada
    }

    # Actions US : pas de suffixe
    if currency == "USD" and exchange in ("NASDAQ", "NYSE", "ARCA", "BATS", "SMART"):
        return symbol

    # Symboles europeens avec suffixe connu
    for exch_key, suffixe in suffixes.items():
        if exch_key in exchange.upper():
            return symbol + suffixe

    # Cas SMART avec devise EUR → supposer Paris ou Frankfurt
    if currency == "EUR":
        return symbol + ".PA"  # a affiner si besoin

    return symbol  # fallback


def recuperer_portefeuille(host=None, port=None, client_id=None):
    """
    Se connecte a TWS et recupere toutes les positions en actions.
    Retourne une liste de dicts avec ticker_yf, position, valeur.

    Par defaut, host/port/client_id sont lus depuis les variables
    d'environnement IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID (voir .env.example).
    """
    host = host or os.getenv("IBKR_HOST", "127.0.0.1")
    client_id = client_id if client_id is not None else int(os.getenv("IBKR_CLIENT_ID", "10"))
    if port is None:
        env_port = os.getenv("IBKR_PORT")
        if not env_port:
            raise ValueError(
                "IBKR_PORT non defini — passe `port=` ou renseigne-le dans ton .env "
                "(voir .env.example)."
            )
        port = int(env_port)

    app = IBKRPortfolio()
    app.connect(host, port, clientId=client_id)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    # Attente max 10 secondes
    recu = app.fini.wait(timeout=10)

    app.reqAccountUpdates(False, "")
    app.disconnect()

    if not recu:
        print("  Attention : timeout — TWS n'a pas repondu dans les temps.")

    return app.positions


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Connexion a TWS...")
    positions = recuperer_portefeuille()

    if not positions:
        print("Aucune position trouvee. Verifie que TWS est ouvert et l'API activee.")
    else:
        print(f"\n{len(positions)} action(s) trouvee(s) dans le portefeuille :\n")
        print(f"  {'Ticker IBKR':<14} {'Ticker yfinance':<16} {'Devise':<8} {'Position':>10} {'Valeur ($)':>12}")
        print("  " + "-" * 64)
        for p in positions:
            print(f"  {p['ticker_ibkr']:<14} {p['ticker_yf']:<16} {p['currency']:<8} "
                  f"{p['position']:>10.0f} {p['valeur']:>12.2f}")

        tickers_yf = [p["ticker_yf"] for p in positions]
        print(f"\nTickers yfinance : {tickers_yf}")
