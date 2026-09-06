"""
Point 5 — Backtest Walk-Forward sur 10 ans
Strategie : Max Sharpe reoptimise chaque trimestre
Benchmark  : Equal-weight + SPY buy-and-hold
"""

import warnings
warnings.filterwarnings("ignore")

from paths import graph, data, cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os, sys

# ── SSL fix (meme chemin special que donnees.py) ─────────────────────────────
try:
    from curl_cffi import requests as cffi_requests
    _SESSION = cffi_requests.Session(impersonate="chrome", verify=False)
    import urllib3
    urllib3.disable_warnings()
except ImportError:
    _SESSION = None

import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
TICKERS_PORTEFEUILLE = [
    "AAPL","AGN.AS","AI.PA","AIR.PA","ALLY","AMZN","AON","ASML.AS","AXP",
    "BAC","BLK","BN.PA","CAC.PA","CAP.PA","CB","CHTR","COF","DEO","DPZ",
    "DSY.PA","DVA","ENGI.PA","EQIX","F","FDX","FWONK","GD","GE","GLD",
    "GLE.PA","GOOGL","GS","GWW","HD","IBKR","IBM","JEF","JNJ","JPM",
    "KER.PA","KHC","KLAC","KO","KR","LILAK","LLYVK","LPX","MA","MCK",
    "MCO","META","MSFT","NESN.SW","NFLX","NUAI","NVDA","NVR","OR.PA",
    "ORCL","OXY","PGHN.SW","PHIA.AS","PLTR","POOL","QQQ","ROP","SIRI",
    "SPY","STLAP.PA","STZ","TEM","TMUS","TSLA","UNH","V","VGT","VRSN",
    "VST","XOM",
]

FENETRE_TRAIN = 252       # 1 an d'historique pour calibrer
FENETRE_TEST  = 63        # 3 mois hors-echantillon
RF_ANNUEL     = 0.04
PERIODE_10ANS = 10        # ans

print("=" * 60)
print("  BACKTEST WALK-FORWARD — 10 ANS")
print("=" * 60)

# ── 1. Telechargement 10 ans ──────────────────────────────────────────────────
date_fin   = datetime.today()
date_debut = date_fin - timedelta(days=365 * PERIODE_10ANS + 30)

cache_file = cache("prix_backtest_10ans.csv")

if os.path.exists(cache_file):
    print(f"\nChargement depuis le cache : {cache_file}")
    prix = pd.read_csv(cache_file, index_col=0, parse_dates=True)
else:
    print(f"\n[1] Telechargement 10 ans ({date_debut.date()} -> {date_fin.date()})...")
    print("    Cela peut prendre 1-2 minutes...")

    dl_kwargs = dict(
        tickers=TICKERS_PORTEFEUILLE,
        start=date_debut.strftime("%Y-%m-%d"),
        end=date_fin.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if _SESSION is not None:
        dl_kwargs["session"] = _SESSION

    data = yf.download(**dl_kwargs)
    prix = data["Close"] if "Close" in data.columns else data

    # Nettoyer : garder colonnes avec assez de donnees
    seuil = len(prix) * 0.5
    prix = prix.dropna(axis=1, thresh=int(seuil))
    prix.to_csv(cache_file)
    print(f"    {prix.shape[1]} actions, {len(prix)} jours. Cache : {cache_file}")

# S'assurer que SPY est present pour le benchmark
if "SPY" not in prix.columns:
    print("    Telechargement SPY benchmark...")
    kw = dict(tickers=["SPY"], start=date_debut.strftime("%Y-%m-%d"),
              end=date_fin.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if _SESSION:
        kw["session"] = _SESSION
    spy_data = yf.download(**kw)
    prix["SPY"] = spy_data["Close"] if "Close" in spy_data.columns else spy_data

prix = prix.dropna(how="all")
tickers = [c for c in prix.columns if c != "SPY"]
print(f"    Actions retenues : {len(tickers)}")


# ── Fonctions utilitaires ────────────────────────────────────────────────────

def rendements_log(df):
    return np.log(df / df.shift(1)).dropna()

def max_sharpe(mu, cov, rf=RF_ANNUEL/252):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 0.20)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - rf) / (v + 1e-10)

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})
    if res.success:
        return res.x
    return w0   # fallback equi-pondere


def perf_periode(rendements_df, poids):
    """Retourne la serie de valeur du portefeuille sur la periode."""
    r = rendements_df @ poids
    return (1 + r).cumprod()


# ── 2. Walk-forward ──────────────────────────────────────────────────────────
print(f"\n[2] Walk-forward : fenetre train={FENETRE_TRAIN}j, test={FENETRE_TEST}j")

rets_all = rendements_log(prix[tickers + (["SPY"] if "SPY" in prix.columns else [])])
rets_actifs = rets_all[tickers]
rets_spy    = rets_all["SPY"] if "SPY" in rets_all.columns else None

n_actifs = len(tickers)
dates    = rets_actifs.index

resultats = []      # (date_debut_test, date_fin_test, ret_port, ret_bench, sharpe)
val_port   = [1.0]
val_bench  = [1.0]
val_spy    = [1.0]
dates_val  = [dates[FENETRE_TRAIN]]

poids_courants = np.ones(n_actifs) / n_actifs  # equi-pondere initial

debut_idx = 0
n_fenetres = 0

while debut_idx + FENETRE_TRAIN + FENETRE_TEST <= len(dates):
    # Fenetre d'entrainement
    train = rets_actifs.iloc[debut_idx : debut_idx + FENETRE_TRAIN]

    # Supprimer colonnes avec trop de NaN dans cette fenetre
    valides = train.columns[train.isna().mean() < 0.1]
    train   = train[valides].fillna(0)

    mu  = train.mean().values * 252
    cov = train.cov().values  * 252

    # Optimisation Max Sharpe
    try:
        poids_opt_valides = max_sharpe(mu, cov)
        # Remettre dans l'espace complet (0 pour actifs exclus)
        poids_opt = np.zeros(n_actifs)
        for i, t in enumerate(valides):
            idx_global = tickers.index(t)
            poids_opt[idx_global] = poids_opt_valides[i]
    except Exception:
        poids_opt = np.ones(n_actifs) / n_actifs

    poids_courants = poids_opt

    # Fenetre de test (hors-echantillon)
    test_idx_debut = debut_idx + FENETRE_TRAIN
    test_idx_fin   = test_idx_debut + FENETRE_TEST
    test = rets_actifs.iloc[test_idx_debut:test_idx_fin].fillna(0)

    r_port  = (test @ poids_courants)
    r_bench = test.mean(axis=1)  # equi-pondere

    # Accumuler valeur
    for rp, rb in zip(r_port, r_bench):
        val_port.append(val_port[-1] * (1 + rp))
        val_bench.append(val_bench[-1] * (1 + rb))

    if rets_spy is not None:
        spy_test = rets_spy.iloc[test_idx_debut:test_idx_fin].fillna(0)
        for rs in spy_test:
            val_spy.append(val_spy[-1] * (1 + rs))

    dates_val.extend(dates[test_idx_debut:test_idx_fin].tolist())

    # Stats de la periode
    sharpe_test = (r_port.mean() * 252 - RF_ANNUEL) / (r_port.std() * np.sqrt(252) + 1e-10)
    resultats.append({
        "date_debut": dates[test_idx_debut],
        "date_fin"  : dates[test_idx_fin - 1],
        "ret_port"  : r_port.sum(),
        "ret_bench" : r_bench.sum(),
        "sharpe"    : sharpe_test,
        "n_positions": (poids_courants > 0.005).sum(),
    })

    debut_idx += FENETRE_TEST
    n_fenetres += 1
    if n_fenetres % 4 == 0:
        annee = dates[test_idx_debut].year
        print(f"    ... {annee} ({n_fenetres} fenetres traitees)")

print(f"    Total : {n_fenetres} fenetres de rebalancement")


# ── 3. Statistiques globales ─────────────────────────────────────────────────
print("\n[3] Statistiques globales du backtest :")

df_res = pd.DataFrame(resultats)
df_res.to_csv(data("resultats_backtest.csv"), index=False)

val_port_s  = pd.Series(val_port,  index=dates_val[:len(val_port)])
val_bench_s = pd.Series(val_bench, index=dates_val[:len(val_bench)])
val_spy_s   = pd.Series(val_spy,   index=dates_val[:len(val_spy)])

def stats_globales(vals):
    rets = vals.pct_change().dropna()
    ret_ann   = (vals.iloc[-1] / vals.iloc[0]) ** (252 / max(len(rets), 1)) - 1
    vol_ann   = rets.std() * np.sqrt(252)
    sharpe    = (ret_ann - RF_ANNUEL) / (vol_ann + 1e-10)
    rolling_max = vals.cummax()
    drawdown  = (vals - rolling_max) / rolling_max
    max_dd    = drawdown.min()
    return ret_ann, vol_ann, sharpe, max_dd

r_p, v_p, s_p, dd_p = stats_globales(val_port_s)
r_b, v_b, s_b, dd_b = stats_globales(val_bench_s)
r_s, v_s, s_s, dd_s = stats_globales(val_spy_s)

print(f"\n  {'':35s} {'Max Sharpe':>12s}  {'Equi-pond.':>12s}  {'SPY':>10s}")
print("  " + "─" * 75)
print(f"  {'Rendement annuel':35s} {r_p*100:>11.2f}%  {r_b*100:>11.2f}%  {r_s*100:>9.2f}%")
print(f"  {'Volatilite annuelle':35s} {v_p*100:>11.2f}%  {v_b*100:>11.2f}%  {v_s*100:>9.2f}%")
print(f"  {'Ratio de Sharpe':35s} {s_p:>12.2f}  {s_b:>12.2f}  {s_s:>10.2f}")
print(f"  {'Max Drawdown':35s} {dd_p*100:>11.2f}%  {dd_b*100:>11.2f}%  {dd_s*100:>9.2f}%")
print(f"  {'Performance totale':35s} {(val_port_s.iloc[-1]-1)*100:>11.1f}%  {(val_bench_s.iloc[-1]-1)*100:>11.1f}%  {(val_spy_s.iloc[-1]-1)*100:>9.1f}%")
print(f"\n  Resultats sauvegardes : {data('resultats_backtest.csv')}")


# ── 4. Graphiques ────────────────────────────────────────────────────────────
print("\n[4] Generation des graphiques...")

BG   = "#0f0f1a"
FG   = "#e0e0e0"
BLUE = "#00d4ff"
ORG  = "#ff7043"
GRN  = "#69ff47"

fig = plt.figure(figsize=(18, 14), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.3)
ax_style = dict(facecolor="#1a1a2e", labelcolor=FG, titlecolor=FG)

def style_ax(ax, title):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color=FG, fontsize=11, pad=8)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.grid(True, color="#1e1e3a", linewidth=0.5)

# ── Panel 1 : Performance cumulee ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(val_port_s.index,  val_port_s.values,  color=BLUE, linewidth=1.8, label=f"Max Sharpe Walk-Forward  +{(val_port_s.iloc[-1]-1)*100:.0f}%")
ax1.plot(val_bench_s.index, val_bench_s.values, color=ORG,  linewidth=1.2, alpha=0.8, label=f"Equi-pondere            +{(val_bench_s.iloc[-1]-1)*100:.0f}%")
ax1.plot(val_spy_s.index,   val_spy_s.values,   color=GRN,  linewidth=1.2, alpha=0.8, label=f"SPY (benchmark)         +{(val_spy_s.iloc[-1]-1)*100:.0f}%")
ax1.axhline(1, color="#555577", linewidth=0.8, linestyle="--")
ax1.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=9)
style_ax(ax1, "Performance cumulee — Walk-Forward 10 ans (1$ investi)")
ax1.set_ylabel("Valeur ($)")

# ── Panel 2 : Drawdown ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
dd_port  = (val_port_s  - val_port_s.cummax())  / val_port_s.cummax()  * 100
dd_bench = (val_bench_s - val_bench_s.cummax()) / val_bench_s.cummax() * 100
dd_spy_v = (val_spy_s   - val_spy_s.cummax())   / val_spy_s.cummax()   * 100
ax2.fill_between(dd_port.index,  dd_port,  0, alpha=0.4, color=BLUE)
ax2.fill_between(dd_bench.index, dd_bench, 0, alpha=0.3, color=ORG)
ax2.plot(dd_spy_v.index, dd_spy_v, color=GRN, linewidth=0.8, alpha=0.7, label="SPY")
ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax2, "Drawdown (%)")
ax2.set_ylabel("Drawdown (%)")

# ── Panel 3 : Sharpe glissant ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
sharpe_roll = pd.Series(
    [r["sharpe"] for r in resultats],
    index=[r["date_debut"] for r in resultats]
)
ax3.bar(sharpe_roll.index, sharpe_roll.values,
        color=[BLUE if v >= 0 else ORG for v in sharpe_roll.values],
        width=40, alpha=0.8)
ax3.axhline(0, color="#555577", linewidth=0.8)
ax3.axhline(sharpe_roll.mean(), color=GRN, linewidth=1.2, linestyle="--",
            label=f"Moy={sharpe_roll.mean():.2f}")
ax3.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax3, "Sharpe trimestriel hors-echantillon")
ax3.set_ylabel("Sharpe")

# ── Panel 4 : Rendement Port vs Bench par fenetre ───────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
dates_plot = [r["date_debut"] for r in resultats]
ret_p_list = [r["ret_port"]  * 100 for r in resultats]
ret_b_list = [r["ret_bench"] * 100 for r in resultats]
x = np.arange(len(dates_plot))
w = 0.4
ax4.bar(x - w/2, ret_p_list, w, color=BLUE,  alpha=0.8, label="Max Sharpe")
ax4.bar(x + w/2, ret_b_list, w, color=ORG,   alpha=0.8, label="Equi-pond.")
ax4.axhline(0, color="#555577", linewidth=0.6)
ax4.set_xticks(x[::4])
ax4.set_xticklabels([str(d)[:7] for d in dates_plot[::4]], rotation=45, fontsize=7)
ax4.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax4, "Rendement par trimestre (%)")
ax4.set_ylabel("Ret. trim. (%)")

# ── Panel 5 : Nb positions ───────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
nb_pos = [r["n_positions"] for r in resultats]
ax5.plot(dates_plot, nb_pos, color=BLUE, linewidth=1.5, marker="o", markersize=3)
ax5.axhline(np.mean(nb_pos), color=GRN, linestyle="--", linewidth=1,
            label=f"Moy={np.mean(nb_pos):.1f}")
ax5.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax5, "Nombre de positions par rebalancement")
ax5.set_ylabel("Nb positions")

# Titre global
fig.suptitle(
    "BACKTEST WALK-FORWARD — MAX SHARPE vs BENCHMARKS (10 ANS)",
    color=FG, fontsize=14, fontweight="bold", y=0.98
)

plt.savefig(graph("backtest_walkforward.png"), dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"  Graphique sauvegarde : {graph('backtest_walkforward.png')}")

print("\nTermine.")
