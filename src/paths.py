"""
paths.py — Chemins centraux du projet
"""
import os

GRAPHS = "graphs"
DATA   = "data"
CACHE  = os.path.join("data", "cache")

for _d in [GRAPHS, DATA, CACHE]:
    os.makedirs(_d, exist_ok=True)

def graph(filename):  return os.path.join(GRAPHS, filename)
def data(filename):   return os.path.join(DATA,   filename)
def cache(filename):  return os.path.join(CACHE,  filename)
