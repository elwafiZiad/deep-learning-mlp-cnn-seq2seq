# Deep Learning from Scratch — MLP · CNN · RNN/Seq2Seq

Projet de fin de module Deep Learning (EMSI, 2025–2026).

Ce projet implémente et compare trois grandes familles d'architectures de réseaux de neurones profonds, chacune appliquée à un type de données différent.

---

## Ce que contient ce projet

### Partie I — MLP sur données tabulaires
- Dataset : **Dry Bean** (13 611 graines, 7 variétés, 16 features morphologiques)
- Implémentation from scratch avec NumPy (forward pass, backprop, mise à jour des poids)
- Réimplémentation avec PyTorch (`nn.Sequential` + classe `nn.Module`)
- Étude comparative de 3 stratégies d'initialisation : Constante, Gaussienne, Xavier
- Résultat : **91.4% accuracy** avec initialisation Xavier

### Partie II — CNN sur images
- Dataset : **Fashion-MNIST** (70 000 images 28×28, 10 classes de vêtements)
- Implémentation manuelle de la corrélation croisée 2D et du pooling (NumPy)
- Architecture inspirée de LeNet, améliorée avec BatchNorm et Dropout
- Étude de l'impact du padding, stride, pooling, nombre de filtres, convolution 1×1
- Visualisation des feature maps pour interpréter ce que le réseau apprend
- Résultat : **90.54% accuracy** (CNN) vs 89.26% (MLP) — réduction d'erreur de 36%

### Partie III — RNN / LSTM / GRU / Seq2Seq sur séquences textuelles
- Dataset : **Tatoeba fra-eng** (~50 000 paires de traduction français-anglais)
- Comparaison de trois cellules récurrentes : RNN simple, LSTM, GRU
- Architecture Seq2Seq (encodeur-décodeur) avec Teacher Forcing
- Décodage Greedy vs Beam Search (k=3)
- Évaluation par perplexité et score BLEU
- Résultats :
  - LSTM : **PPL = 7.6** vs RNN : PPL = 24.3 (3× meilleur)
  - Beam Search : **BLEU = 12.23** vs Greedy : BLEU = 11.11

---

## Structure du projet

```
├── notebooks/
│   ├── partie1_mlp.ipynb       # Notebook Partie I
│   ├── partie2_cnn.ipynb       # Notebook Partie II
│   ├── partie3_rnn.ipynb       # Notebook Partie III
│   ├── figures/                # Graphiques comparatifs générés
│   └── models/                 # Modèles entraînés (.pth)
├── data/
│   └── DryBeanDataset/         # Dataset Dry Bean (inclus)

```

> **Note :** Les données Fashion-MNIST et Tatoeba sont téléchargées automatiquement à l'exécution des notebooks (torchvision / requests).

---

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn requests
```

Python 3.10+ recommandé. Les notebooks sont auto-suffisants et peuvent être exécutés dans l'ordre.

---

## Outils utilisés

| Outil | Usage |
|---|---|
| Python 3.10 | Langage principal |
| NumPy | Implémentations from scratch |
| PyTorch 2.1 | Framework Deep Learning |
| torchvision | Chargement Fashion-MNIST |
| scikit-learn | Métriques, prétraitement |
| Matplotlib | Visualisations |

---

## Auteur

Projet réalisé dans le cadre du module **Deep Learning** — EMSI, Année universitaire 2025–2026.
