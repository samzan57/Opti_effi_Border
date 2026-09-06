"""
execution.py — Application concrete de la strategie quant
==========================================================
Etapes :
  1. Recupere les positions actuelles depuis IBKR TWS
  2. Calcule le portefeuille cible (alpha-tilt + contraintes)
  3. Affiche les ordres precis (action, quantite, montant)
  4. [Optionnel] Envoie les ordres au compte paper trading IBKR
"""

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

from paths import data
import time
import threading
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── IBKR API ─────────────────────────────────────────────────────────────────
from ibapi.client   import EClient
from ibapi.wrapper  import EWrapper
from ibapi.contract import Contract
from ibapi.order    import Order

# ── Nos modules ──────────────────────────────────────────────────────────────
from recuperer_positions import recuperer_portefeuille, convertir_ticker
from donnees  import telecharger_prix, nettoyer_tickers, CORRECTIONS
from stats    import calculer_rendements, statistiques_annuelles
from alpha    import calculer_alpha

# ─────────────────────────────────────────────────────────────────────────────
NAV             = 1_472_484      # NAV totale du compte ($)
RF              = 0.04
POIDS_MIN       = 0.01           # 1% minimum par position
POIDS_MAX       = 0.15           # 15% maximum par position
SECTEUR_MAX     = 0.30           # 30% max par secteur
ALPHA_INTENSITY = 0.5
SEUIL_ORDRE     = 500            # ignorer les ordres < 500$

# Tickers a VENDRE (losers alpha confirmes)
LOSERS = ["DSY.PA", "CHTR", "POOL", "UNH", "STLAP.PA"]

# Tickers a SURPONDERER (top alpha multi-confirme)
STARS = {
    "ENGI.PA": 0.15,   # conviction max
    "JNJ":     0.12,
    "ASML.AS": 0.12,
    "GOOGL":   0.10,
    "GLD":     0.08,
}

print("=" * 62)
print("  EXECUTION DE LA STRATEGIE QUANT")
print("=" * 62)

# ── Etape 1 : Positions actuelles ────────────────────────────────────────────
print("\n[1] Recuperation des positions IBKR...")
positions = recuperer_portefeuille()

portefeuille_actuel = {}
for p in positions:
    t = CORRECTIONS.get(p["ticker_ibkr"], p["ticker_yf"])
    portefeuille_actuel[t] = {
        "valeur":    p["valeur"],
        "quantite":  p["position"],
        "ticker_ibkr": p["ticker_ibkr"],
        "currency":  p["currency"],
        "exchange":  p["exchange"],
    }

nav_actuelle = sum(v["valeur"] for v in portefeuille_actuel.values())
print(f"    {len(portefeuille_actuel)} positions | NAV actuelle : {nav_actuelle:,.0f} $")

# ── Etape 2 : Calcul du portefeuille cible ───────────────────────────────────
print("\n[2] Calcul du portefeuille cible (Alpha-tilt + contraintes)...")

tickers_actifs = nettoyer_tickers(positions)
prix_df = telecharger_prix(tickers_actifs)
rend_df = calculer_rendements(prix_df)
mu, cov, sigma = statistiques_annuelles(rend_df)

tickers = mu.index.tolist()
n = len(tickers)

# Scores alpha
_, alpha_series = calculer_alpha(prix_df, rend_df)
alpha = alpha_series.to_dict()

# mu tiltee
mu_alpha = mu.copy()
for t in tickers:
    a = alpha.get(t, 0.0)
    mu_alpha[t] = mu[t] + ALPHA_INTENSITY * a * sigma[t]

# Forcer poids = 0 pour les losers
losers_idx = [i for i, t in enumerate(tickers) if t in LOSERS]

def neg_sharpe_alpha(w):
    r = w @ mu_alpha.values
    v = np.sqrt(w @ cov.values @ w)
    return -(r - RF) / (v + 1e-10)

# Bornes : 0 pour losers, [min, max] pour les autres
bornes = []
for i, t in enumerate(tickers):
    if t in LOSERS:
        bornes.append((0.0, 0.0))          # forcer vente totale
    elif t in STARS:
        bornes.append((STARS[t] * 0.8, STARS[t]))  # surponderer
    else:
        bornes.append((0.0, POIDS_MAX))

contraintes = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

res = minimize(neg_sharpe_alpha, np.ones(n)/n, method="SLSQP",
               bounds=bornes, constraints=contraintes,
               options={"maxiter": 1000, "ftol": 1e-10})

if not res.success:
    print("    Optimisation echouee — utilisation des poids STARS uniquement")
    w_cible = np.zeros(n)
    reste = 1.0 - sum(STARS.values())
    autres = [i for i, t in enumerate(tickers) if t not in LOSERS and t not in STARS]
    for t, p in STARS.items():
        if t in tickers:
            w_cible[tickers.index(t)] = p
    if autres:
        for i in autres:
            w_cible[i] = reste / len(autres)
else:
    w_cible = res.x

# Stats du portefeuille cible
r_cible = w_cible @ mu_alpha.values
v_cible = np.sqrt(w_cible @ cov.values @ w_cible)
s_cible = (r_cible - RF) / v_cible
nb_pos  = (w_cible > 0.005).sum()

print(f"    Rendement cible   : {r_cible:.2%}")
print(f"    Volatilite cible  : {v_cible:.2%}")
print(f"    Ratio de Sharpe   : {s_cible:.2f}")
print(f"    Nb de positions   : {nb_pos}")

# ── Etape 3 : Calcul des ordres ───────────────────────────────────────────────
print("\n[3] Calcul des ordres :")

# Prix actuels pour estimer les quantites
derniers_prix = prix_df.iloc[-1]

ordres = []
tous_tickers = set(tickers) | set(portefeuille_actuel.keys())

for t in tous_tickers:
    val_actuelle = portefeuille_actuel.get(t, {}).get("valeur", 0.0)
    qtite_actuelle = portefeuille_actuel.get(t, {}).get("quantite", 0)

    if t in tickers:
        idx = tickers.index(t)
        val_cible = w_cible[idx] * NAV
    else:
        val_cible = 0.0  # pas dans le portefeuille cible → vendre tout

    delta = val_cible - val_actuelle

    if abs(delta) < SEUIL_ORDRE:
        continue

    # Estimer la quantite en actions
    prix_action = derniers_prix.get(t, None)
    if prix_action and prix_action > 0:
        qtite_delta = int(abs(delta) / prix_action)
    else:
        qtite_delta = None

    info_ibkr = portefeuille_actuel.get(t, {})
    ordres.append({
        "Ticker":          t,
        "Ticker_IBKR":     info_ibkr.get("ticker_ibkr", t.replace(".PA","").replace(".AS","").replace(".SW","")),
        "Action":          "ACHAT" if delta > 0 else "VENTE",
        "Val actuelle ($)": round(val_actuelle),
        "Val cible ($)":    round(val_cible),
        "Montant ($)":     round(abs(delta)),
        "Quantite":        qtite_delta,
        "Currency":        info_ibkr.get("currency", "USD"),
        "Exchange":        info_ibkr.get("exchange", "SMART"),
        "Priorite":        1 if t in LOSERS else (2 if t in STARS else 3),
    })

df_ordres = pd.DataFrame(ordres).sort_values(["Priorite", "Action", "Montant ($)"],
                                              ascending=[True, True, False])
ventes = df_ordres[df_ordres["Action"] == "VENTE"]
achats = df_ordres[df_ordres["Action"] == "ACHAT"]

print(f"\n  {'─'*60}")
print(f"  VENTES ({len(ventes)} ordres — {ventes['Montant ($)'].sum():,.0f} $)")
print(f"  {'─'*60}")
for _, r in ventes.iterrows():
    star = " ← LOSER ALPHA" if r["Ticker"] in LOSERS else ""
    qtite_str = f"{r['Quantite']} actions" if r["Quantite"] else "quantite a confirmer"
    print(f"  VENDRE  {r['Ticker']:<12}  {qtite_str:<20}  {r['Montant ($)']:>8,.0f} ${star}")

print(f"\n  {'─'*60}")
print(f"  ACHATS ({len(achats)} ordres — {achats['Montant ($)'].sum():,.0f} $)")
print(f"  {'─'*60}")
for _, r in achats.iterrows():
    star = " ← STAR ALPHA" if r["Ticker"] in STARS else ""
    qtite_str = f"{r['Quantite']} actions" if r["Quantite"] else "quantite a confirmer"
    print(f"  ACHETER {r['Ticker']:<12}  {qtite_str:<20}  {r['Montant ($)']:>8,.0f} ${star}")

df_ordres.to_csv(data("ordres_execution.csv"), index=False)
print(f"\n  Ordres sauvegardes : {data('ordres_execution.csv')}")

# ── Etape 4 : Envoi des ordres via IBKR ──────────────────────────────────────
print("\n" + "=" * 62)
reponse = input("  Envoyer les ordres au compte IBKR paper trading ? (oui/non) : ").strip().lower()

if reponse != "oui":
    print(f"\n  Ordres NON envoyes. Le fichier {data('ordres_execution.csv')} contient")
    print("  tous les ordres que tu peux saisir manuellement dans TWS.")
    print("\nTermine.")
    exit(0)


# ── Classe IBKR pour passer les ordres ───────────────────────────────────────
class IBKRExecution(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.next_order_id = None
        self.ordres_envoyes = []
        self._pret = threading.Event()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in (2104, 2106, 2158, 2119):
            print(f"  [Erreur {errorCode}] {errorString}")

    def nextValidId(self, orderId):
        self.next_order_id = orderId
        self._pret.set()

    def orderStatus(self, orderId, status, filled, remaining,
                    avgFillPrice, permId, parentId, lastFillPrice,
                    clientId, whyHeld, mktCapPrice):
        print(f"  Ordre {orderId} : {status} | rempli={filled} @ {avgFillPrice:.2f}")

    def openOrder(self, orderId, contract, order, orderState):
        pass


def creer_contrat(ticker_ibkr, currency, exchange):
    c = Contract()
    c.symbol   = ticker_ibkr
    c.secType  = "STK"
    c.currency = currency
    c.exchange = "SMART"
    if currency != "USD":
        c.primaryExch = exchange if exchange else "IBIS"
    return c


def creer_ordre_marche(action, quantite):
    o = Order()
    o.action         = action          # "BUY" ou "SELL"
    o.totalQuantity  = quantite
    o.orderType      = "MKT"
    o.tif            = "DAY"
    o.eTradeOnly     = False
    o.firmQuoteOnly  = False
    return o


# Connexion et envoi — host/port/client ID configurables via .env (voir .env.example)
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = os.getenv("IBKR_PORT")
IBKR_CLIENT_ID = int(os.getenv("IBKR_EXECUTION_CLIENT_ID", "20"))
if not IBKR_PORT:
    raise ValueError("IBKR_PORT non defini — renseigne-le dans ton .env (voir .env.example).")

print(f"\n  Connexion a TWS ({IBKR_HOST})...")
app = IBKRExecution()
app.connect(IBKR_HOST, int(IBKR_PORT), clientId=IBKR_CLIENT_ID)

thread = threading.Thread(target=app.run, daemon=True)
thread.start()

if not app._pret.wait(timeout=10):
    print("  Erreur : TWS ne repond pas. Verifie que TWS est ouvert.")
    exit(1)

print(f"  Connecte. Premier ordre ID : {app.next_order_id}")
print(f"  Envoi de {len(df_ordres)} ordres...\n")

# Envoyer VENTES d'abord (liberer du cash), puis ACHATS
ordre_id = app.next_order_id

for _, row in df_ordres.sort_values("Action", ascending=False).iterrows():
    if pd.isna(row["Quantite"]) or row["Quantite"] <= 0:
        print(f"  Skip {row['Ticker']} : quantite non calculable")
        continue

    action_ibkr = "SELL" if row["Action"] == "VENTE" else "BUY"
    contrat = creer_contrat(row["Ticker_IBKR"], row["Currency"], row["Exchange"])
    ordre   = creer_ordre_marche(action_ibkr, int(row["Quantite"]))

    app.placeOrder(ordre_id, contrat, ordre)
    print(f"  [{ordre_id}] {action_ibkr:<4} {row['Quantite']:>6} x {row['Ticker']:<12}  ~{row['Montant ($)']:>8,.0f} $")
    app.ordres_envoyes.append({"id": ordre_id, "ticker": row["Ticker"], "action": action_ibkr})

    ordre_id += 1
    time.sleep(0.4)   # respecter le rate limit IBKR (50 ordres/s max)

print(f"\n  {len(app.ordres_envoyes)} ordres envoyes a TWS.")
print("  Verifie l'onglet 'Ordres' dans TWS pour le suivi.")
time.sleep(3)
app.disconnect()
print("\nTermine.")
