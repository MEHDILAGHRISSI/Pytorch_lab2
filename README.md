# Lab 2 – Deep Learning avec PyTorch (MNIST)

Université Abdelmalek Essaadi – FST Tanger – Cycle LSI – Pr. ELAACHAK Lotfi
Auteur : Mehdi Laghrissi

## Objectif
Prise en main de PyTorch (mode GPU) : CNN, Faster R-CNN, transfer learning (VGG16 / AlexNet) et Vision Transformer, tous évalués sur MNIST.

## Structure
```
utils.py                         # métriques, boucle d'entraînement, sauvegarde des résultats
part1/part1_custom_cnn.py        # 1. CNN personnalisé
part1/part1_faster_rcnn.py       # 2. Faster R-CNN
part1/part1_transfer_learning.py # 4. VGG16 / AlexNet fine-tunés
part1/part1_comparison.py        # 3. tableau comparatif
part2/part2_vit.py               # Partie 2 : ViT from scratch
results/                         # métriques JSON + comparison.md
```

## Exécution (Colab / Kaggle, GPU activé)
```bash
pip install -r requirements.txt
python part1/part1_custom_cnn.py
python part1/part1_faster_rcnn.py
python part1/part1_transfer_learning.py --model alexnet
python part1/part1_transfer_learning.py --model vgg16
python part2/part2_vit.py
python part1/part1_comparison.py
```

## Partie 1 – CNN
- Architecture : Conv(1→16, k=3, p=1, s=1) → ReLU → MaxPool(2) → Conv(16→32, k=3, p=1, s=1) → ReLU → MaxPool(2) → Dropout(0.25) → FC(1568→128) → ReLU → FC(128→10)
- Hyper-paramètres : Adam, lr = 1e-3, weight decay (L2) = 1e-4, batch = 64, 5 époques
- Résultats : *à compléter*

## Partie 1 – Faster R-CNN
- Faster R-CNN ResNet50-FPN pré-entraîné, tête remplacée (11 classes = 10 chiffres + fond).
- Boîte englobante calculée à partir des pixels non nuls ; prédiction = label de la détection au score maximal.
- Entraîné sur un sous-ensemble (images redimensionnées à 128 px) pour des raisons de temps.
- Résultats : *à compléter*

## Partie 1 – Transfer learning (VGG16 / AlexNet)
- Couches convolutives gelées, dernière couche FC remplacée (1000 → 10), images 224×224 × 3 canaux.
- Résultats : *à compléter*

## Comparaison
*Coller ici le contenu de `results/comparison.md`.*

| Modèle | Accuracy | F1 (macro) | Loss | Temps d'entraînement (s) |
|---|---|---|---|---|
| custom_cnn | | | | |
| faster_rcnn | | | | |
| transfer_alexnet | | | | |
| transfer_vgg16 | | | | |
| vit_scratch | | | | |

**Conclusion** : *à rédiger (précision vs temps, intérêt du transfer learning, pertinence d'un détecteur pour une tâche de classification…).*

## Partie 2 – Vision Transformer
- Patchs 7×7 (16 patchs), dim 64, 4 blocs, 4 têtes, MLP 128, token [CLS] + embeddings positionnels appris.
- Résultats et comparaison avec la Partie 1 : *à compléter*

## Synthèse personnelle
*Brève synthèse de ce que j'ai appris pendant ce lab.*