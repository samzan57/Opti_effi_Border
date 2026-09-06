"""
simulation.py — Dry-run complet avant execution reelle
=======================================================
Simule la strategie sur les 6 derniers mois SANS toucher IBKR.
Repond a la question : "Si j'avais applique cette strategie
il y a 6 mois, qu'est-ce qui se serait passe ?"

Etapes :
  1. Charge les donnees historiques (cache)
  2. Simule le portefeuille AVANT rebalancement (actuel equi-pondere)
  3. Simule le portefeuille APRES rebalancement (alpha-tilt)
  4. Compare jour par jour : P&L, Sharpe, Drawdown
  5. Montre les ordres qui auraient ete passes
  6. Checklist de validation avant passage en paper trading
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
import os

from donnees import telecharger_prix, CORRECTIONS
from stats   import calculer_rendements, statistiques_annuelles
from alpha   import signal_momentum, signal_quality, calculer_alpha

# ─────────────────────────────────────────────────────────────────────────────
NAV             = 1_472_484
RF              = 0.04
POIDS_MAX       = 0.15
ALPHA_INTENSITY = 0.5
FENETRE_SIMU    = 126   # 6 mois de simulation (jours de bourse)
SEUIL_ORDRE     = 500

LOSERS = ["DSY.PA", "CHTR", "POOL", "UNH", "STLAP.PA"]
STARS  = {"ENGI.PA": 0.15, "JNJ": 0.12, "ASML.AS": 0.12,
          "GOOGL": 0.10, "GLD": 0.08}

BG   = "#0f0f1a"
FG   = "#e0e0e0"
BLUE = "#00d4ff"
ORG  = "#ff7043"
GRN  = "#69ff47"
RED  = "#ff4444"

print("=" * 62)
print("  SIMULATION DRY-RUN — 6 MOIS")
print("=" * 62)

# ── 1. Chargement des donnees ─────────────────────────────────────────────────
print("\n[1] Chargement des donnees...")

# Essayer le cache 10 ans d'abord, sinon 2 ans
if os.path.exists(cache("prix_backtest_10ans.csv")):
    prix_all = pd.read_csv(cache("prix_backtest_10ans.csv"), index_col=0, parse_dates=True)
    print(f"    Source : {cache('prix_backtest_10ans.csv')}")
else:
    prix_all = pd.read_csv(cache("prix_historiques.csv"), index_col=0, parse_dates=True)
    print(f"    Source : {cache('prix_historiques.csv')}")

# Garder seulement les colonnes avec assez de donnees
prix_all = prix_all.dropna(axis=1, thresh=int(len(prix_all) * 0.7))
tickers  = [c for c in prix_all.columns if c != "SPY"]

print(f"    {len(tickers)} actions disponibles | {len(prix_all)} jours")

# Decoupage : periode de calibration + periode de simulation
# Calibration : tout sauf les 6 derniers mois
# Simulation  : les 6 derniers mois (hors-echantillon)
prix_calib = prix_all.iloc[:-FENETRE_SIMU]
prix_simu  = prix_all.iloc[-FENETRE_SIMU:]

date_debut_simu = prix_simu.index[0].strftime("%Y-%m-%d")
date_fin_simu   = prix_simu.index[-1].strftime("%Y-%m-%d")
print(f"    Calibration : jusqu'au {prix_calib.index[-1].strftime('%Y-%m-%d')}")
print(f"    Simulation  : {date_debut_simu} → {date_fin_simu}")

# ── 2. Optimisation sur la periode de calibration ────────────────────────────
print("\n[2] Optimisation sur la periode de calibration...")

rend_calib = calculer_rendements(prix_calib[tickers])
mu, cov, sigma = statistiques_annuelles(rend_calib)
tickers_ok = mu.index.tolist()
n = len(tickers_ok)

# Scores alpha (calcules sur la calibration)
df_alpha, alpha_series = calculer_alpha(prix_calib[tickers_ok], rend_calib)
alpha_dict = alpha_series.to_dict()

# mu tiltee
mu_alpha = mu.copy()
for t in tickers_ok:
    a = alpha_dict.get(t, 0.0)
    mu_alpha[t] = mu[t] + ALPHA_INTENSITY * a * sigma[t]

# Optimisation avec bornes STARS / LOSERS
bornes = []
for t in tickers_ok:
    if t in LOSERS:
        bornes.append((0.0, 0.0))
    elif t in STARS:
        bornes.append((STARS[t] * 0.8, STARS[t]))
    else:
        bornes.append((0.0, POIDS_MAX))

def neg_sharpe(w):
    r = w @ mu_alpha.values
    v = np.sqrt(w @ cov.values @ w)
    return -(r - RF) / (v + 1e-10)

res = minimize(neg_sharpe, np.ones(n)/n, method="SLSQP",
               bounds=bornes,
               constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
               options={"maxiter": 1000, "ftol": 1e-10})

w_opt   = res.x if res.success else np.ones(n)/n
w_equi  = np.ones(n) / n   # portefeuille actuel simplifie (equi-pondere)

# Top positions du portefeuille optimise
top_pos = sorted(zip(tickers_ok, w_opt), key=lambda x: -x[1])[:10]
print(f"    Optimisation reussie : Sharpe calibration = {-neg_sharpe(w_opt):.2f}")
print(f"    Top 5 positions cibles :")
for t, w in top_pos[:5]:
    tag = " ← STAR" if t in STARS else ""
    print(f"      {t:<14} {w*100:>6.1f}%{tag}")

# ── 3. Simulation hors-echantillon ───────────────────────────────────────────
print(f"\n[3] Simulation sur 6 mois ({date_debut_simu} → {date_fin_simu})...")

# Rendements journaliers pendant la simulation
prix_simu_ok = prix_simu[tickers_ok].ffill().dropna(how="all")
rend_simu = prix_simu_ok.pct_change().dropna().fillna(0)

# Portefeuille optimise (alpha-tilt)
r_opt_daily  = rend_simu @ w_opt
val_opt       = NAV * (1 + r_opt_daily).cumprod()

# Portefeuille actuel (equi-pondere)
r_equi_daily = rend_simu @ w_equi
val_equi      = NAV * (1 + r_equi_daily).cumprod()

# SPY benchmark
if "SPY" in prix_simu.columns:
    r_spy_daily = prix_simu["SPY"].pct_change().dropna().fillna(0)
    r_spy_daily = r_spy_daily.reindex(rend_simu.index).fillna(0)
    val_spy     = NAV * (1 + r_spy_daily).cumprod()
else:
    val_spy = None

# ── 4. Statistiques de simulation ────────────────────────────────────────────
def stats_simu(vals, rets):
    ret_tot  = (vals.iloc[-1] / NAV - 1) * 100
    ret_ann  = ((vals.iloc[-1] / NAV) ** (252/len(rets)) - 1) * 100
    vol_ann  = rets.std() * np.sqrt(252) * 100
    sharpe   = (ret_ann/100 - RF) / (vol_ann/100 + 1e-10)
    rolling_max = vals.cummax()
    drawdown = ((vals - rolling_max) / rolling_max * 100).min()
    pnl_usd  = vals.iloc[-1] - NAV
    return ret_tot, ret_ann, vol_ann, sharpe, drawdown, pnl_usd

r_tot_o, r_ann_o, v_o, s_o, dd_o, pnl_o = stats_simu(val_opt,  r_opt_daily)
r_tot_e, r_ann_e, v_e, s_e, dd_e, pnl_e = stats_simu(val_equi, r_equi_daily)

print(f"\n  {'':30s} {'Optimise':>12s}  {'Actuel':>12s}", end="")
if val_spy is not None:
    print(f"  {'SPY':>10s}")
else:
    print()
sep = "  " + "─" * 60
print(sep)
print(f"  {'Rendement total (6 mois)':30s} {r_tot_o:>+11.2f}%  {r_tot_e:>+11.2f}%")
print(f"  {'Rendement annualise':30s} {r_ann_o:>+11.2f}%  {r_ann_e:>+11.2f}%")
print(f"  {'Volatilite annualisee':30s} {v_o:>11.2f}%  {v_e:>11.2f}%")
print(f"  {'Ratio de Sharpe':30s} {s_o:>12.2f}  {s_e:>12.2f}")
print(f"  {'Max Drawdown':30s} {dd_o:>+11.2f}%  {dd_e:>+11.2f}%")
print(f"  {'P&L en dollars':30s} {pnl_o:>+11,.0f}$  {pnl_e:>+11,.0f}$")

if val_spy is not None:
    r_tot_s, r_ann_s, v_s, s_s, dd_s, pnl_s = stats_simu(val_spy, r_spy_daily)
    print(f"\n  SPY (benchmark) : {r_tot_s:+.2f}% total | Sharpe {s_s:.2f} | P&L {pnl_s:+,.0f}$")

avantage = pnl_o - pnl_e
print(f"\n  Avantage de la strategie vs actuel : {avantage:+,.0f} $")

# ── 5. Ordres simules (dry-run) ───────────────────────────────────────────────
print(f"\n[4] Ordres qui auraient ete passes le {date_debut_simu} :")
print(f"    (simulation pure — aucun ordre reel envoye)")

derniers_prix_calib = prix_calib[tickers_ok].iloc[-1]
ordres_sim = []

for i, t in enumerate(tickers_ok):
    val_actuelle = NAV / n    # equi-pondere
    val_cible    = w_opt[i] * NAV
    delta        = val_cible - val_actuelle

    if abs(delta) < SEUIL_ORDRE:
        continue

    prix_action = derniers_prix_calib.get(t, None)
    qtite = int(abs(delta) / prix_action) if (prix_action and prix_action > 0) else None

    ordres_sim.append({
        "Ticker":   t,
        "Action":   "ACHAT" if delta > 0 else "VENTE",
        "Montant":  round(abs(delta)),
        "Quantite": qtite,
        "Tag":      "LOSER" if t in LOSERS else ("STAR" if t in STARS else ""),
    })

df_sim = pd.DataFrame(ordres_sim).sort_values(["Action", "Montant"], ascending=[True, False])
ventes_sim = df_sim[df_sim["Action"] == "VENTE"]
achats_sim  = df_sim[df_sim["Action"] == "ACHAT"]

print(f"\n    VENTES simulees ({len(ventes_sim)} ordres — {ventes_sim['Montant'].sum():,.0f} $) :")
for _, r in ventes_sim.head(10).iterrows():
    tag = f" ← {r['Tag']}" if r["Tag"] else ""
    print(f"    VENTE  {r['Ticker']:<12} {r['Montant']:>8,.0f} ${tag}")

print(f"\n    ACHATS simules ({len(achats_sim)} ordres — {achats_sim['Montant'].sum():,.0f} $) :")
for _, r in achats_sim.head(10).iterrows():
    tag = f" ← {r['Tag']}" if r["Tag"] else ""
    print(f"    ACHAT  {r['Ticker']:<12} {r['Montant']:>8,.0f} ${tag}")

df_sim.to_csv(data("simulation_ordres.csv"), index=False)
print(f"\n    Ordres sauvegardes : {data('simulation_ordres.csv')}")

# ── 6. Checklist validation ───────────────────────────────────────────────────
print(f"\n[5] CHECKLIST AVANT PAPER TRADING :")
checks = []

# Check 1 : strategie profitable ?
ok1 = r_tot_o > 0
checks.append((ok1, f"Strategie profitable sur 6 mois ? {r_tot_o:+.2f}%"))

# Check 2 : bat le portefeuille actuel ?
ok2 = r_tot_o > r_tot_e
checks.append((ok2, f"Bat le portefeuille actuel ? {r_tot_o:+.2f}% vs {r_tot_e:+.2f}%"))

# Check 3 : bat SPY ?
if val_spy is not None:
    ok3 = r_tot_o > r_tot_s
    checks.append((ok3, f"Bat SPY ? {r_tot_o:+.2f}% vs {r_tot_s:+.2f}%"))

# Check 4 : drawdown acceptable (<15%) ?
ok4 = dd_o > -15
checks.append((ok4, f"Drawdown max acceptable (<15%) ? {dd_o:.2f}%"))

# Check 5 : Sharpe > 1 ?
ok5 = s_o > 1.0
checks.append((ok5, f"Sharpe > 1 ? {s_o:.2f}"))

# Check 6 : losers vendus ?
losers_presents = [t for t in LOSERS if t in tickers_ok and w_opt[tickers_ok.index(t)] < 0.001]
ok6 = len(losers_presents) == len(LOSERS)
checks.append((ok6, f"Losers elimines ({len(losers_presents)}/{len(LOSERS)}) ? {', '.join(losers_presents)}"))

# Check 7 : stars surponderes ?
stars_ok = [t for t in STARS if t in tickers_ok and w_opt[tickers_ok.index(t)] >= STARS[t] * 0.7]
ok7 = len(stars_ok) >= 3
checks.append((ok7, f"Stars surponderes ({len(stars_ok)}/{len(STARS)}) ? {', '.join(stars_ok)}"))

tous_ok = all(c[0] for c in checks)
print()
for ok, msg in checks:
    symbole = "OK" if ok else "!!"
    print(f"    [{symbole}] {msg}")

print()
if tous_ok:
    print("    => TOUS LES CHECKS PASSES — Strategie validee pour paper trading")
    verdict = "VALIDE"
else:
    nb_fail = sum(1 for ok, _ in checks if not ok)
    print(f"    => {nb_fail} check(s) echoue(s) — Revoir la strategie avant paper trading")
    verdict = "A_REVOIR"

# ── 7. Graphiques ────────────────────────────────────────────────────────────
print("\n[6] Generation des graphiques...")

fig = plt.figure(figsize=(16, 12), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

def style_ax(ax, title):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.grid(True, color="#1e1e3a", linewidth=0.5)

# Panel 1 : Performance cumulee
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(val_opt.index,  val_opt.values,  color=BLUE, linewidth=2.2,
         label=f"Strategie optimisee  {r_tot_o:+.1f}%  (P&L: {pnl_o:+,.0f}$)")
ax1.plot(val_equi.index, val_equi.values, color=ORG, linewidth=1.5, alpha=0.85,
         label=f"Portefeuille actuel  {r_tot_e:+.1f}%  (P&L: {pnl_e:+,.0f}$)")
if val_spy is not None:
    ax1.plot(val_spy.index, val_spy.values, color=GRN, linewidth=1.2, alpha=0.7,
             label=f"SPY benchmark        {r_tot_s:+.1f}%  (P&L: {pnl_s:+,.0f}$)")
ax1.axhline(NAV, color="#555577", linewidth=0.8, linestyle="--", label="Capital initial")
ax1.fill_between(val_opt.index,
                 val_opt.values, val_equi.values,
                 where=(val_opt.values >= val_equi.values),
                 alpha=0.12, color=BLUE, label="Surperformance")
ax1.fill_between(val_opt.index,
                 val_opt.values, val_equi.values,
                 where=(val_opt.values < val_equi.values),
                 alpha=0.12, color=RED)
ax1.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax1, f"SIMULATION 6 MOIS : Performance cumulee ({date_debut_simu} → {date_fin_simu})")
ax1.set_ylabel("Valeur du portefeuille ($)")

# Panel 2 : Drawdown compare
ax2 = fig.add_subplot(gs[1, 0])
dd_opt_s  = (val_opt  - val_opt.cummax())  / val_opt.cummax()  * 100
dd_equi_s = (val_equi - val_equi.cummax()) / val_equi.cummax() * 100
ax2.fill_between(dd_opt_s.index,  dd_opt_s,  0, alpha=0.5, color=BLUE, label=f"Optimise  min={dd_o:.1f}%")
ax2.fill_between(dd_equi_s.index, dd_equi_s, 0, alpha=0.35, color=ORG,  label=f"Actuel    min={dd_e:.1f}%")
ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax2, "Drawdown quotidien (%)")
ax2.set_ylabel("Drawdown (%)")

# Panel 3 : P&L journalier
ax3 = fig.add_subplot(gs[1, 1])
pnl_daily = (val_opt - val_opt.shift(1)).fillna(0)
colors_pnl = [BLUE if v >= 0 else RED for v in pnl_daily]
ax3.bar(pnl_daily.index, pnl_daily.values, color=colors_pnl, alpha=0.8, width=1)
ax3.axhline(0, color="#555577", linewidth=0.8)
pnl_moy = pnl_daily.mean()
ax3.axhline(pnl_moy, color=GRN, linewidth=1, linestyle="--",
            label=f"Moy: {pnl_moy:+,.0f}$/j")
ax3.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor=FG, fontsize=8)
style_ax(ax3, "P&L journalier ($) — Strategie optimisee")
ax3.set_ylabel("P&L ($)")

# Panel 4 : Allocation portefeuille cible (pie)
ax4 = fig.add_subplot(gs[2, 0])
pos_significatives = [(t, w) for t, w in zip(tickers_ok, w_opt) if w > 0.01]
pos_significatives.sort(key=lambda x: -x[1])
labels_pie = [t for t, _ in pos_significatives[:12]]
sizes_pie  = [w for _, w in pos_significatives[:12]]
if sum(sizes_pie) < 0.999:
    labels_pie.append("Autres")
    sizes_pie.append(1 - sum(sizes_pie))
colors_pie = plt.cm.plasma(np.linspace(0.15, 0.9, len(labels_pie)))
wedges, texts, autotexts = ax4.pie(sizes_pie, labels=labels_pie, autopct="%1.1f%%",
                                    colors=colors_pie, startangle=90,
                                    textprops={"color": FG, "fontsize": 7})
for at in autotexts:
    at.set_color(FG)
    at.set_fontsize(7)
style_ax(ax4, "Allocation portefeuille optimise")

# Panel 5 : Checklist visuelle
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, len(checks) + 1)
ax5.axis("off")
ax5.set_facecolor("#1a1a2e")
ax5.set_title("Checklist de validation", color=FG, fontsize=10, pad=8)

for i, (ok, msg) in enumerate(reversed(checks)):
    y = i + 0.5
    color = GRN if ok else RED
    symbole = "✓" if ok else "✗"
    msg_court = msg[:48] + "..." if len(msg) > 48 else msg
    ax5.text(0.3, y, symbole, color=color, fontsize=14, va="center", fontweight="bold")
    ax5.text(1.2, y, msg_court, color=FG if ok else ORG, fontsize=7.5, va="center")

verdict_color = GRN if tous_ok else RED
verdict_txt   = "STRATEGIE VALIDEE" if tous_ok else "A REVOIR AVANT LIVE"
ax5.text(5, len(checks) + 0.5, verdict_txt, color=verdict_color,
         fontsize=11, fontweight="bold", ha="center", va="center")
ax5.axhline(len(checks), color="#333355", linewidth=0.8)

fig.suptitle(
    f"DRY-RUN SIMULATION — {date_debut_simu} → {date_fin_simu}   |   Verdict : {verdict_txt}",
    color=FG, fontsize=12, fontweight="bold", y=0.99
)

plt.savefig(graph("simulation_dryrun.png"), dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"    Graphique sauvegarde : {graph('simulation_dryrun.png')}")

print(f"\n{'='*62}")
print(f"  VERDICT FINAL : {verdict_txt}")
print(f"  P&L simule sur 6 mois : {pnl_o:+,.0f} $ (vs actuel : {pnl_e:+,.0f} $)")
print(f"  Avantage de la strategie : {avantage:+,.0f} $")
if tous_ok:
    print(f"\n  => Tu peux lancer execution.py pour passer en paper trading.")
else:
    print(f"\n  => Examine les checks echoues avant de lancer execution.py.")
print("=" * 62)
print("\nTermine.")
