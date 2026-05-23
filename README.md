# 🧠 Débruitage d'Images — Deep Learning (DnCNN-Lite v4)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green?logo=nvidia)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Projet de Deep Learning pour le **débruitage d'images** par réseau de neurones convolutif (DnCNN-Lite), entraîné sur 50 000 images CIFAR-10 avec un pipeline mémoire optimisé.

---

## 📊 Résultats d'entraînement — Loss & MAE

![Training Loss and MAE](https://github.com/user-attachments/assets/857077ea-0669-4372-ae7c-ba9b988f7e80)

*Loss combinée (MSE + SSIM + Gradient) et MAE sur 50 epochs — Train vs Validation*

---

## 🖼️ Résultats visuels — Images débruitées (64×64)

![DnCNN-Lite v4 Results](https://github.com/user-attachments/assets/90ea7aa2-e92c-4ca5-b96d-9a20fc10e836)

*Ligne 1 : images bruitées — Ligne 2 : images débruitées — Ligne 3 : images originales*

---

## 🐶 Test sur image réelle — PSNR : 18.9 dB

![Real Image Test](https://github.com/user-attachments/assets/bdd63fde-a85f-462c-b463-647bc96ba065)

*Gauche : image bruitée (input) — Centre : image débruitée (output) — Droite : bruit supprimé (×5)*

---

## 🏗️ Architecture du modèle — DnCNN-Lite v4

```
Input (64×64×3)
      ↓
Conv2D(64, 3×3) + ReLU
      ↓
Conv2D(64, 3×3) + BatchNorm + ReLU  ×6
      ↓
Conv2D(3, 3×3)
      ↓
Residual : Input - Bruit prédit
      ↓
Output débruité (64×64×3)
```

- **Paramètres** : 211 923
- **Mémoire modèle** : ~0.8 MB
- **GPU** : Tesla T4 (~13 757 MB disponibles)

---

## ⚡ Optimisation mémoire

| Approche | RAM utilisée | Statut |
|----------|-------------|--------|
| ❌ float32 · 50 000 images · 64×64 | **2 457 MB** | CRASH |
| ✅ uint8 · tf.data pipeline · bruit à la volée | **~200 MB** | OK (×12 moins) |

**Stratégie** : les images restent en `uint8` et sont upscalées + bruitées **à la volée** par batch via `tf.data`, évitant de tout charger en RAM.

---

## 🔧 Stack technique

| Composant | Détail |
|-----------|--------|
| Framework | TensorFlow 2.19 / Keras |
| Dataset | CIFAR-10 (50 000 images train) |
| Résolution | 64×64 px (upscalé depuis 32×32) |
| Batch size | 32 |
| Epochs | 50 |
| Loss | MSE + SSIM + Gradient |
| Optimizer | Adam + ReduceLROnPlateau |
| Environnement | Google Colab (GPU Tesla T4) |

---

## 🚀 Lancer le projet

### Sur Google Colab (recommandé)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TON_USERNAME/Debruitage-Images-Deep-Learning/blob/main/projet_DL.ipynb)

### En local

```bash
git clone https://github.com/TON_USERNAME/Debruitage-Images-Deep-Learning.git
cd Debruitage-Images-Deep-Learning
pip install tensorflow numpy matplotlib
jupyter notebook projet_DL.ipynb
```

---

## 📁 Structure du projet

```
Debruitage-Images-Deep-Learning/
├── projet_DL.ipynb                              # Notebook principal
├── rapport_deep.pdf                             # Rapport du projet
├── presentation_projet_Debruitage-dImages.pdf   # Présentation slides
├── docs/
│   ├── output_image_0.png                       # Courbes Loss & MAE
│   ├── output_image_1.png                       # Grille résultats 64×64
│   └── output_image_2.png                       # Test image réelle
└── README.md
```

---

## 📄 Documents

- 📘 [Rapport complet](./rapport_deep.pdf)
- 📊 [Présentation](./presentation_projet_Debruitage-dImages.pdf)

---

## 👤 Auteur

**firdawsM7** · [GitHub](https://github.com/firdawsM7)
