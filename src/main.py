"""
main.py — Point d'entree unique du projet
==========================================
Lance le menu principal et execute le module choisi.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MENU = {
    "1": ("Frontiere efficiente Markowitz",         "optimiseur.py"),
    "2": ("Analyse du risque (VaR / CVaR / Stress)","risque.py"),
    "3": ("Contraintes reelles (secteur / turnover)","contraintes.py"),
    "4": ("Signaux Alpha (Momentum + Quality)",      "alpha.py"),
    "5": ("Backtest walk-forward 10 ans",            "backtest.py"),
    "6": ("Simulation dry-run 6 mois",               "simulation.py"),
    "7": ("Execution des ordres (paper trading)",    "execution.py"),
}

def afficher_menu():
    print()
    print("=" * 54)
    print("   SYSTEME QUANT — PORTEFEUILLE IBKR")
    print("=" * 54)
    for k, (label, _) in MENU.items():
        print(f"   [{k}]  {label}")
    print("   [0]  Quitter")
    print("=" * 54)

def lancer(script):
    print(f"\n>>> Lancement de {script}...\n")
    script_path = os.path.join(SCRIPT_DIR, script)
    repo_root = os.path.dirname(SCRIPT_DIR)  # data/ et graphs/ vivent a la racine du repo
    subprocess.run([sys.executable, "-X", "utf8", script_path], cwd=repo_root, check=False)

if __name__ == "__main__":
    while True:
        afficher_menu()
        choix = input("\n   Votre choix : ").strip()
        if choix == "0":
            print("\nAu revoir.\n")
            break
        elif choix in MENU:
            lancer(MENU[choix][1])
            input("\n   [Appuie sur Entree pour revenir au menu]")
        else:
            print("   Choix invalide, recommence.")
