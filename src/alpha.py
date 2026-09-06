"""
alpha.py — Signaux Alpha : Momentum + Quality
==============================================
Markowitz optimise le RISQUE mais utilise les rendements historiques
comme proxy du futur — ce qui est naif.

Un vrai quant ajoute des SIGNAUX qui predisent quelles actions vont
surperformer. On construit deux facteurs classiques :

MOMENTUM (suivre la tendance)
    → Les actions qui ont monte ces 12 derniers mois continuent souvent
      de monter (inertie des prix). Signal = rendement 12-1 mois
      (on exclut le dernier mois car il y a souvent un retournement court terme)

QUALITY (choisir les meilleures entreprises)
    → Les entreprises "solides" surperforment sur le long terme.
      Proxy prix : Sharpe ratio + regularite des gains + faible drawdown

Ces deux scores sont combines en un SCORE ALPHA qui vient MODIFIER
les rendements attendus avant de les passer a l'optimiseur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from donnees import telecharger_prix
from stats import calculer_rendements, statistiques_annuelles
from optimiseur import portefeuille_max_sharpe
from paths import graph, data

TAUX_SANS_RISQUE = 0.04
NAV              = 1_472_484


# ---------------------------------------------------------------------------
# 1. SIGNAL MOMENTUM
# ---------------------------------------------------------------------------

def signal_momentum(prix, mois_retour=12, mois_skip=1):
    """
    Momentum cross-sectionnel : rendement sur les (mois_retour - mois_skip)
    derniers mois, en sautant le dernier mois (evite le retournement court terme).

    Retourne un score entre -1 et +1 pour chaque action (rang normalise).
    """
    # Rendement sur 12 mois en sautant le dernier mois
    jours_retour = mois_retour * 21
    jours_skip   = mois_skip   * 21

    if len(prix) < jours_retour + jours_skip:
        print(f"  Attention : pas assez de donnees pour le momentum complet")
        jours_retour = len(prix) - jours_skip - 5

    debut = prix.iloc[-(jours_retour + jours_skip)]
    fin   = prix.iloc[-jours_skip]

    rendement_momentum = (fin / debut) - 1

    # Normalisation en rang : -1 (pire) → +1 (meilleur)
    rang = rendement_momentum.rank(pct=True)   # rang entre 0 et 1
    score = 2 * rang - 1                        # rescaler entre -1 et +1

    return score, rendement_momentum


# ---------------------------------------------------------------------------
# 2. SIGNAL QUALITY
# ---------------------------------------------------------------------------

def signal_quality(rendements, fenetre_jours=252):
    """
    Quality score base sur 3 composantes prix :
      1. Sharpe ratio     → recompense le rendement ajuste au risque
      2. Regularite       → % de mois positifs (consistance)
      3. Faible drawdown  → penalise les actions qui ont beaucoup chute

    Retourne un score normalise entre -1 et +1.
    """
    rend = rendements.iloc[-fenetre_jours:]

    # Composante 1 : Sharpe individuel
    mu_j    = rend.mean()
    sigma_j = rend.std()
    sharpe  = mu_j / (sigma_j + 1e-10)

    # Composante 2 : regularite (% de jours positifs)
    regularite = (rend > 0).mean()

    # Composante 3 : faible drawdown (on prend l'inverse)
    prix_simule = (1 + rend).cumprod()
    rolling_max = prix_simule.cummax()
    drawdowns   = (prix_simule - rolling_max) / rolling_max
    max_dd      = drawdowns.min()           # negatif
    score_dd    = 1 + max_dd               # entre 0 et 1 (1 = pas de drawdown)

    # Score composite (ponderation egale des 3 composantes)
    score_brut = (sharpe.rank(pct=True) +
                  regularite.rank(pct=True) +
                  score_dd.rank(pct=True)) / 3

    score = 2 * score_brut - 1   # rescaler entre -1 et +1
    return score, sharpe, regularite, max_dd


# ---------------------------------------------------------------------------
# 3. SCORE ALPHA COMBINE
# ---------------------------------------------------------------------------

def calculer_alpha(prix, rendements,
                   poids_momentum=0.5,
                   poids_quality=0.5):
    """
    Combine momentum et quality en un score alpha unique.
    Retourne un DataFrame avec tous les scores par action.
    """
    score_mom, rend_mom = signal_momentum(prix)
    score_qual, sharpe, regularite, max_dd = signal_quality(rendements)

    # Alignement sur les memes tickers
    tickers_communs = score_mom.index.intersection(score_qual.index)
    score_mom  = score_mom[tickers_communs]
    score_qual = score_qual[tickers_communs]

    alpha = poids_momentum * score_mom + poids_quality * score_qual

    df = pd.DataFrame({
        "Score Momentum":  score_mom.round(3),
        "Score Quality":   score_qual.round(3),
        "Score Alpha":     alpha.round(3),
        "Rend 12m (%)":    (rend_mom[tickers_communs] * 100).round(1),
        "Sharpe":          sharpe[tickers_communs].round(3),
        "Regularite (%)":  (regularite[tickers_communs] * 100).round(1),
        "Max DD (%)":      (max_dd[tickers_communs] * 100).round(1),
    }).sort_values("Score Alpha", ascending=False)

    return df, alpha


# ---------------------------------------------------------------------------
# 4. OPTIMISATION ALPHA-TILTEE
# ---------------------------------------------------------------------------

def optimiser_avec_alpha(mu, cov, alpha_scores, tickers,
                         intensite_alpha=0.5,
                         taux_rf=TAUX_SANS_RISQUE):
    """
    Modifie les rendements attendus en les tilting vers le score alpha.

    mu_alpha[i] = mu[i] + intensite * alpha[i] * vol[i]

    L'idee : les actions avec bon alpha voient leur rendement attendu
    augmenter, ce qui pousse l'optimiseur a les surponderer.
    """
    alpha_arr = np.array([alpha_scores.get(t, 0) for t in tickers])
    vol_arr   = np.sqrt(np.diag(cov))

    # Tilt des rendements : on ajoute une prime proportionnelle au score alpha
    mu_alpha = mu + intensite_alpha * alpha_arr * vol_arr

    # Optimisation Max Sharpe sur les rendements tiltes
    w = portefeuille_max_sharpe(mu_alpha, cov, taux_rf)

    return w, mu_alpha


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------

def visualiser_alpha(df_scores, w_markowitz, w_alpha, tickers,
                     mu, cov, prix, rendements):

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Signaux Alpha : Momentum + Quality",
                 color="white", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # --- 1. Top/Flop momentum ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0f0f1a")
    top10  = df_scores.nlargest(10,  "Score Momentum")
    flop5  = df_scores.nsmallest(5,  "Score Momentum")
    afficher = pd.concat([top10, flop5])
    couleurs = ["#00e5ff" if v > 0 else "#ff6b6b"
                for v in afficher["Score Momentum"]]
    ax1.barh(afficher.index, afficher["Score Momentum"],
             color=couleurs, edgecolor="#333355", linewidth=0.4)
    ax1.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax1.set_title("Top/Flop Momentum", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=7)
    ax1.invert_yaxis()
    for sp in ax1.spines.values(): sp.set_edgecolor("#333355")

    # --- 2. Top/Flop quality ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0f0f1a")
    top10q = df_scores.nlargest(10, "Score Quality")
    flop5q = df_scores.nsmallest(5, "Score Quality")
    afficherq = pd.concat([top10q, flop5q])
    couleursq = ["#00ff88" if v > 0 else "#ff6b35"
                 for v in afficherq["Score Quality"]]
    ax2.barh(afficherq.index, afficherq["Score Quality"],
             color=couleursq, edgecolor="#333355", linewidth=0.4)
    ax2.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax2.set_title("Top/Flop Quality", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=7)
    ax2.invert_yaxis()
    for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

    # --- 3. Score Alpha final ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#0f0f1a")
    top15 = df_scores.nlargest(15, "Score Alpha")
    couleurs3 = plt.cm.plasma(np.linspace(0.2, 0.9, len(top15)))
    ax3.barh(top15.index, top15["Score Alpha"],
             color=couleurs3, edgecolor="#333355", linewidth=0.4)
    ax3.set_title("Top 15 Score Alpha (Momentum + Quality)",
                  color="white", fontsize=9)
    ax3.tick_params(colors="white", labelsize=7)
    ax3.invert_yaxis()
    for sp in ax3.spines.values(): sp.set_edgecolor("#333355")

    # --- 4. Scatter Momentum vs Quality ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#0f0f1a")
    sc = ax4.scatter(df_scores["Score Momentum"],
                     df_scores["Score Quality"],
                     c=df_scores["Score Alpha"],
                     cmap="plasma", s=40, alpha=0.8)
    # Labels pour les meilleurs
    for t in df_scores.nlargest(8, "Score Alpha").index:
        ax4.annotate(t, (df_scores.loc[t, "Score Momentum"],
                         df_scores.loc[t, "Score Quality"]),
                     fontsize=6, color="white", alpha=0.9,
                     xytext=(3, 3), textcoords="offset points")
    ax4.axvline(0, color="white", alpha=0.3, linewidth=0.8)
    ax4.axhline(0, color="white", alpha=0.3, linewidth=0.8)
    ax4.set_xlabel("Momentum", color="white", fontsize=8)
    ax4.set_ylabel("Quality",  color="white", fontsize=8)
    ax4.set_title("Momentum vs Quality\n(couleur = Score Alpha)",
                  color="white", fontsize=9)
    ax4.tick_params(colors="white", labelsize=7)
    plt.colorbar(sc, ax=ax4).ax.yaxis.set_tick_params(color="white")
    for sp in ax4.spines.values(): sp.set_edgecolor("#333355")

    # --- 5. Comparaison allocations ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#0f0f1a")
    s_mark = pd.Series(w_markowitz, index=tickers).nlargest(12)
    s_alph = pd.Series(w_alpha,     index=tickers).nlargest(12)
    tous   = list(dict.fromkeys(list(s_mark.index) + list(s_alph.index)))[:15]
    x      = np.arange(len(tous))
    w_m_v  = [w_markowitz[tickers.index(t)] if t in tickers else 0 for t in tous]
    w_a_v  = [w_alpha[tickers.index(t)]     if t in tickers else 0 for t in tous]
    ax5.bar(x - 0.2, np.array(w_m_v)*100, 0.38,
            color="#4fc3f7", label="Markowitz pur", alpha=0.85)
    ax5.bar(x + 0.2, np.array(w_a_v)*100, 0.38,
            color="#ffd700", label="Alpha-tilt",    alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(tous, rotation=45, ha="right", fontsize=7, color="white")
    ax5.set_ylabel("Poids (%)", color="white", fontsize=8)
    ax5.set_title("Poids : Markowitz pur vs Alpha-tilt",
                  color="white", fontsize=9)
    ax5.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax5.tick_params(colors="white")
    for sp in ax5.spines.values(): sp.set_edgecolor("#333355")

    # --- 6. Tableau recap ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#0f0f1a")
    ax6.axis("off")

    def perf(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return r, v, (r - TAUX_SANS_RISQUE) / v

    r_m, v_m, s_m = perf(w_markowitz)
    r_a, v_a, s_a = perf(w_alpha)

    data = [
        ["Metrique",          "Markowitz pur", "Alpha-tilt"],
        ["Rendement annuel",  f"{r_m:.2%}",    f"{r_a:.2%}"],
        ["Volatilite",        f"{v_m:.2%}",    f"{v_a:.2%}"],
        ["Sharpe",            f"{s_m:.2f}",    f"{s_a:.2f}"],
        ["Nb positions",      f"{(w_markowitz>0.005).sum()}", f"{(w_alpha>0.005).sum()}"],
    ]
    table = ax6.table(cellText=data[1:], colLabels=data[0],
                      cellLoc="center", loc="center",
                      bbox=[0, 0.25, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1a1a2e" if r == 0 else "#0f0f1a")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333355")
        if r > 0 and c == 2:
            cell.set_facecolor("#0d2b0d")
    ax6.set_title("Comparaison performances", color="white", fontsize=10)

    plt.savefig(graph("signaux_alpha.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Graphique sauvegarde : {graph('signaux_alpha.png')}")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  SIGNAUX ALPHA — MOMENTUM + QUALITY")
    print("=" * 60)

    prix       = telecharger_prix([])
    rendements = calculer_rendements(prix)
    mu, cov, sigma = statistiques_annuelles(rendements)
    tickers = mu.index.tolist()

    # --- Calcul des scores ---
    print("\n[1] Calcul des scores Momentum et Quality...")
    df_scores, alpha_series = calculer_alpha(prix, rendements,
                                              poids_momentum=0.5,
                                              poids_quality=0.5)

    print(f"\n  Top 10 actions par score alpha :")
    print(f"  {'Ticker':<12} {'Alpha':>8} {'Momentum':>10} {'Quality':>9} "
          f"{'Rend 12m':>10} {'Sharpe':>8}")
    print(f"  {'─'*60}")
    for t, row in df_scores.head(10).iterrows():
        print(f"  {t:<12} {row['Score Alpha']:>8.3f} {row['Score Momentum']:>10.3f} "
              f"{row['Score Quality']:>9.3f} {row['Rend 12m (%)']:>9.1f}% "
              f"{row['Sharpe']:>8.3f}")

    print(f"\n  Flop 5 (actions a eviter) :")
    for t, row in df_scores.tail(5).iterrows():
        print(f"  {t:<12} {row['Score Alpha']:>8.3f} {row['Score Momentum']:>10.3f} "
              f"{row['Score Quality']:>9.3f} {row['Rend 12m (%)']:>9.1f}%")

    # --- Optimisation Markowitz pure ---
    print("\n[2] Optimisation Markowitz pure (reference)...")
    w_markowitz = portefeuille_max_sharpe(mu.values, cov.values, TAUX_SANS_RISQUE)

    # --- Optimisation Alpha-tiltee ---
    print("\n[3] Optimisation Alpha-tiltee (momentum + quality)...")
    alpha_dict = alpha_series.to_dict()
    w_alpha, mu_alpha = optimiser_avec_alpha(
        mu.values, cov.values, alpha_dict, tickers,
        intensite_alpha=0.5
    )

    # --- Comparaison ---
    def perf(w):
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        return r, v, (r - TAUX_SANS_RISQUE) / v

    r_m, v_m, s_m = perf(w_markowitz)
    r_a, v_a, s_a = perf(w_alpha)

    print(f"\n  {'':30} {'Markowitz pur':>15} {'Alpha-tilt':>15}")
    print(f"  {'─'*62}")
    print(f"  {'Rendement annuel':30} {r_m:>14.2%} {r_a:>14.2%}")
    print(f"  {'Volatilite annuelle':30} {v_m:>14.2%} {v_a:>14.2%}")
    print(f"  {'Ratio de Sharpe':30} {s_m:>14.2f} {s_a:>14.2f}")
    print(f"  {'Nb de positions':30} {(w_markowitz>0.005).sum():>14} "
          f"{(w_alpha>0.005).sum():>14}")

    # Top positions du portefeuille alpha
    print(f"\n  Top 10 positions du portefeuille Alpha-tilt :")
    top10_alpha = pd.Series(w_alpha, index=tickers).nlargest(10)
    for t, p in top10_alpha.items():
        score = alpha_dict.get(t, 0)
        print(f"    {t:<14} {p:.2%}  (score alpha: {score:+.3f})")

    # Sauvegarde
    df_scores.to_csv(data("scores_alpha.csv"))
    print(f"\n  Scores sauvegardes : {data('scores_alpha.csv')}")

    # --- Visualisation ---
    print("\n[4] Generation des graphiques...")
    visualiser_alpha(df_scores, w_markowitz, w_alpha,
                     tickers, mu.values, cov.values, prix, rendements)

    print("\nTermine.")
