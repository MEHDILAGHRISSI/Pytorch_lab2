# Lab 2 – Deep Learning avec PyTorch (MNIST)

Université Abdelmalek Essaadi – FST Tanger – Département Génie Informatique – Cycle LSI
Module : Deep Learning – Pr. ELAACHAK Lotfi
Auteur : Mehdi Laghrissi

**Notebook Kaggle (exécuté en mode GPU) : [https://www.kaggle.com/code/laghrissimehdi/notebookcaf6de140a/](LIEN_KAGGLE)**

## Objectif

Prise en main de PyTorch en mode GPU : construire et comparer un CNN, un Faster R-CNN, des modèles pré-entraînés (VGG16, AlexNet) et un Vision Transformer (ViT) pour classifier le dataset MNIST.

## Structure du dépôt

```
utils.py                          # chemins, métriques, boucle d'entraînement/évaluation, sauvegarde JSON
part1/part1_custom_cnn.py         # 1. CNN personnalisé
part1/part1_faster_rcnn.py        # 2. Faster R-CNN
part1/part1_transfer_learning.py  # 4. VGG16 / AlexNet fine-tunés
part1/part1_comparison.py         # 3. tableau comparatif
part2/part2_vit.py                # Partie 2 : ViT from scratch
results/                          # métriques (JSON) + comparison.md
requirements.txt
```

## Exécution

Environnement : Kaggle, GPU (mode `cuda`), dataset MNIST téléchargé automatiquement par `torchvision`.

```bash
git clone https://github.com/MEHDILAGHRISSI/Pytorch_lab2.git
cd Pytorch_lab2
pip install -r requirements.txt

python part1/part1_custom_cnn.py
python part1/part1_faster_rcnn.py --train-subset 60000 --test-subset 10000 --epochs 2
python part1/part1_transfer_learning.py --model alexnet --train-subset 0 --test-subset 0
python part1/part1_transfer_learning.py --model vgg16   --train-subset 0 --test-subset 0
python part2/part2_vit.py
python part1/part1_comparison.py
```

Chaque script sauvegarde ses métriques dans `results/<modèle>.json` ; le script de comparaison les lit et génère le tableau.

---

## Partie 1

### 1. CNN personnalisé

| Élément | Choix |
|---|---|
| Architecture | Conv(1→16, k=3, p=1, s=1) → ReLU → MaxPool(2, s=2) → Conv(16→32, k=3, p=1, s=1) → ReLU → MaxPool(2, s=2) → Dropout(0,25) → FC(1568→128) → ReLU → FC(128→10) |
| Optimiseur | Adam, lr = 1e-3 |
| Régularisation | Dropout 0,25 + L2 (weight decay = 1e-4) |
| Entraînement | 5 époques, batch = 64, jeu complet (60 000 images) |
| Perte | CrossEntropyLoss |

### 2. Faster R-CNN

- Faster R-CNN ResNet50-FPN pré-entraîné (COCO), dont la tête de prédiction est remplacée (11 classes = 10 chiffres + fond).
- MNIST n'ayant pas de boîtes englobantes, celles-ci sont calculées à partir des pixels non nuls de chaque image. Comme le modèle est un détecteur, la classe prédite pour une image est le label de la détection ayant le score le plus élevé (aucune détection = prédiction fausse).
- Les images sont redimensionnées à 128 px (au lieu de 800 par défaut) pour accélérer l'entraînement.
- Entraînement sur le jeu complet : **60 000 images d'entraînement, 10 000 de test, 2 époques**, batch = 8, SGD (lr = 0,005, momentum = 0,9, weight decay = 5e-4).

### 3. Transfer learning (VGG16 et AlexNet)

- Poids ImageNet, couches convolutives gelées, dernière couche FC remplacée (1000 → 10). Seul le classifieur est entraîné (Adam, lr = 1e-3, 3 époques, jeu complet).
- Images MNIST redimensionnées en 224×224 et converties en 3 canaux, normalisées avec les statistiques ImageNet.

### 4. Comparaison

Tous les modèles sont entraînés sur les 60 000 images et évalués sur les 10 000 images de test de MNIST, en mode GPU (Kaggle).

| Modèle | Accuracy | F1 (macro) | Loss (test) | Temps d'entraînement (s) |
|---|---|---|---|---|
| CNN personnalisé | **99,06 %** | **0,9905** | 0,0282 | **70,3** |
| Faster R-CNN | 98,40 % | 0,9839 | 0,0125 (*) | 3554,9 |
| AlexNet (fine-tuné) | 98,25 % | 0,9825 | 0,0610 | 384,1 |
| VGG16 (fine-tuné) | 98,57 % | 0,9856 | 0,1087 | 1222,8 |
| ViT from scratch | 97,81 % | 0,9780 | 0,0712 | 155,9 |

(\*) La loss de Faster R-CNN est la somme des pertes de détection (RPN + régression des boîtes + classification) et n'est **pas comparable** à l'entropie croisée des autres modèles : il ne faut pas en conclure qu'il est « meilleur » en loss.

**Limites de la comparaison :** le nombre d'époques diffère selon les modèles (CNN : 5, Faster R-CNN : 2, AlexNet et VGG16 : 3, ViT : 10) et chaque modèle n'a été entraîné qu'une seule fois (sans graine fixée). Un écart de moins d'un point d'accuracy n'est donc pas concluant.

### Analyse et conclusion

Le CNN personnalisé obtient les meilleurs résultats (99,06 % d'accuracy, F1 = 0,9905) avec le temps d'entraînement le plus court (70 s) : MNIST est une tâche simple et une petite architecture adaptée suffit.

Les modèles pré-entraînés (AlexNet 98,25 %, VGG16 98,57 %) n'apportent pas de gain : ils ont été entraînés sur des images naturelles en couleur, et il faut agrandir les images 28×28 en 224×224, ce qui coûte cher en calcul (384 s et 1223 s, soit 5 à 17 fois plus que le CNN) pour un résultat inférieur. VGG16 a une loss de test plus élevée (0,1087) qu'AlexNet malgré une meilleure accuracy : avec les couches convolutives gelées, le classifieur n'a probablement pas totalement convergé après 3 époques.

Faster R-CNN atteint 98,40 %, ce qui le place devant AlexNet et le ViT, mais il reste en dessous du CNN personnalisé et de VGG16, et il est environ **50 fois plus lent** que le CNN (3555 s contre 70 s). C'est un détecteur d'objets utilisé ici pour une simple classification : il est donc l'outil le moins adapté, car il calcule des boîtes englobantes dont on n'a pas besoin pour MNIST, pour un coût très élevé.

**Conclusion :** pour une tâche de classification simple comme MNIST, un CNN léger offre le meilleur compromis précision / temps / complexité. Le transfer learning est surtout utile quand les données sont rares ou proches d'ImageNet, et Faster R-CNN quand on a réellement besoin de localiser des objets.

---

## Partie 2 – Vision Transformer (ViT)

| Élément | Choix |
|---|---|
| Patchs | 7×7 → 16 patchs par image (28×28), projetés par une convolution |
| Dimension | 64, avec token `[CLS]` et embeddings positionnels appris |
| Encodeur | 4 blocs (LayerNorm → attention multi-têtes à 4 têtes → MLP de 128 avec GELU, connexions résiduelles) |
| Paramètres | 139 018 |
| Entraînement | Adam (lr = 1e-3, weight decay = 1e-4), 10 époques, batch = 128 |


**Écarts avec le tutoriel.** L'architecture suit le tutoriel de Brian Pulfer (patchs → projection linéaire → token `[CLS]` → encodage positionnel → blocs encodeur LN-MSA-résiduel / LN-MLP-résiduel → MLP de classification), avec les adaptations suivantes :
- découpage en patchs par une `Conv2d` (équivalente à `patchify` + couche linéaire, mais bien plus rapide) ;
- 16 patchs de 7×7 au lieu de 49 patchs de 4×4 ;
- embeddings positionnels appris au lieu de sinus/cosinus fixes ;
- modèle plus grand (dim = 64, 4 blocs, 4 têtes, 139 k paramètres au lieu de dim = 8, 2 blocs, 2 têtes) et 10 époques ;
- attention multi-têtes vectorisée (une projection q/k/v fusionnée) avec dropout ;
- le modèle renvoie des logits : le `Softmax` final du tutoriel est retiré car `CrossEntropyLoss` l'applique déjà (le double softmax ralentit l'apprentissage, ce qui explique en partie les ~80 % du tutoriel contre 97,81 % ici).

**Résultats :** 97,81 % d'accuracy, F1 = 0,9780, loss = 0,0712, 155,9 s d'entraînement.

**Interprétation :** le ViT entraîné from scratch est environ 1,2 point en dessous du CNN (99,06 %) et environ 2 fois plus lent (156 s contre 70 s), mais il reste proche des modèles pré-entraînés et de Faster R-CNN, sans aucun pré-entraînement et avec seulement 139 k paramètres. Sa loss d'entraînement diminuait encore à la 10ᵉ époque (0,0668) : plus d'époques ou de l'augmentation de données l'amélioreraient probablement. Contrairement au CNN, le ViT n'a pas de biais inductif de localité ni de partage de poids : il doit apprendre ces structures à partir des données, ce qui est moins efficace sur un petit dataset comme MNIST. Les ViT prennent l'avantage avec de grandes quantités de données.

---

