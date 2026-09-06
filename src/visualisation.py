"""
visualisation.py — Trace de la Frontiere Efficiente
====================================================
Produit un graphique complet avec :
  - Nuage de portefeuilles aleatoires (colores par ratio Sharpe)
  - Courbe de la frontiere efficiente
  - Portefeuille de variance minimale (etoile bleue)
  - Portefeuille max Sharpe (etoile or) + droite du marche des capitaux (CML)
  - Chaque action individuelle (points gris)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from paths import graph, data

def tracer(vols_front, rends_front,
           vols_alea,  rends_alea, sharpes_alea,
           w_mv, w_ms, mu, sigma,
           taux_sans_risque=0.04,
           sauvegarder=None):
    if sauvegarder is None:
        sauvegarder = graph("frontiere_efficiente.png")

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # --- Nuage aleatoire ---
    sc = ax.scatter(
        vols_alea * 100, rends_alea * 100,
        c=sharpes_alea, cmap="plasma",
        alpha=0.35, s=8, linewidths=0,
        label="Portefeuilles aleatoires"
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Ratio de Sharpe", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # --- Frontiere efficiente ---
    ax.plot(
        vols_front * 100, rends_front * 100,
        color="#00e5ff", linewidth=3,
        label="Frontiere efficiente", zorder=5
    )

    # --- Actions individuelles ---
    ax.scatter(
        sigma * 100, mu * 100,
        color="white", s=20, alpha=0.5, zorder=4,
        label="Actions individuelles"
    )
    # Labels pour les actions les plus visibles
    tickers_labels = mu.nlargest(5).index.tolist() + (mu / sigma).nlargest(5).index.tolist()
    tickers_labels = list(set(tickers_labels))
    for t in tickers_labels:
        ax.annotate(
            t,
            xy=(sigma[t] * 100, mu[t] * 100),
            xytext=(4, 4), textcoords="offset points",
            color="white", fontsize=7, alpha=0.85
        )

    # --- Portefeuille min variance ---
    mu_arr  = np.array(mu)
    cov_vals = None   # non utilise ici directement
    vol_mv  = vols_front[0]
    rend_mv = rends_front[0]
    ax.scatter(
        vol_mv * 100, rend_mv * 100,
        color="#4fc3f7", s=200, zorder=6, marker="*",
        label=f"Min Variance  rend={rend_mv:.1%}  vol={vol_mv:.1%}"
    )

    # --- Portefeuille max Sharpe ---
    sharpes_front = rends_front / vols_front
    idx_ms = np.argmax(sharpes_front)
    vol_ms  = vols_front[idx_ms]
    rend_ms = rends_front[idx_ms]
    sharpe_ms = sharpes_front[idx_ms]
    ax.scatter(
        vol_ms * 100, rend_ms * 100,
        color="#ffd700", s=200, zorder=6, marker="*",
        label=f"Max Sharpe  rend={rend_ms:.1%}  vol={vol_ms:.1%}  S={sharpe_ms:.2f}"
    )

    # --- Droite du marche des capitaux (CML) ---
    vol_cml  = np.array([0, max(vols_front) * 1.2])
    rend_cml = taux_sans_risque + sharpe_ms * vol_cml
    ax.plot(
        vol_cml * 100, rend_cml * 100,
        "--", color="#ffd700", linewidth=1.5, alpha=0.7,
        label=f"CML (rf={taux_sans_risque:.0%})"
    )

    # --- Mise en forme ---
    ax.set_xlabel("Volatilite annuelle (%)", color="white", fontsize=12)
    ax.set_ylabel("Rendement annuel (%)",    color="white", fontsize=12)
    ax.set_title("Frontiere Efficiente de Markowitz\nPortefeuille IBKR Demo",
                 color="white", fontsize=14, fontweight="bold")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#333355",
                       labelcolor="white", fontsize=9, loc="upper left")

    ax.grid(True, linestyle="--", alpha=0.2, color="white")

    plt.tight_layout()
    plt.savefig(sauvegarder, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Graphique sauvegarde : {sauvegarder}")
    plt.show()


def afficher_composition(w, tickers, titre="Composition du portefeuille", top_n=15):
    """Trace un camembert des top N positions d'un portefeuille optimal."""
    poids = pd.Series(w, index=tickers).sort_values(ascending=False)
    top   = poids.head(top_n)
    reste = poids.iloc[top_n:].sum()
    if reste > 0.001:
        top["Autres"] = reste

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    couleurs = cm.plasma(np.linspace(0.1, 0.9, len(top)))
    wedges, texts, autotexts = ax.pie(
        top.values,
        labels=top.index,
        autopct=lambda p: f"{p:.1f}%" if p > 1.5 else "",
        colors=couleurs,
        startangle=140,
        pctdistance=0.82,
    )
    for t in texts:
        t.set_color("white")
        t.set_fontsize(9)
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)

    ax.set_title(titre, color="white", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    nom = graph(titre.replace(" ", "_").lower() + ".png")
    plt.savefig(nom, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Composition sauvegardee : {nom}")
    plt.show()
