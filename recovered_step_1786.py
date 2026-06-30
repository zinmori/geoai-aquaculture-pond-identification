# Plan d'implémentation — S1/S2 Split Features & Robust Self-Training

Ce plan vise à corriger les faiblesses des approches précédentes en exploitant les observations S1 (radar) indépendantes et en résolvant le bug de duplication des pseudo-labels dans le self-training pour franchir le palier des 0.90+ sur le Leaderboard.

## 🔬 Analyses et Découvertes Clés

1. **S1/S2 Split Features (+0.0024 CV)**: 
   - 26.5% des lignes du test set contiennent des mois où S1 (VH/VV) est disponible mais S2 (optique) est masqué par les nuages.
   - En extrayant les caractéristiques séparément pour chaque capteur (au lieu d'éliminer le mois complet s'il manque un des capteurs), on augmente considérablement la couverture temporelle des signatures radar (cruciales pour discriminer l'eau).
2. **Correctif Self-Training (+0.0016 CV)**:
   - L'ancienne approche B+ accumulait les pseudo-labels de façon redondante à chaque itération, provoquant un surapprentissage massif (effondrement à 0.84 LB).
   - En fixant un seuil de confiance optimal de **0.90** en 1 seule itération transductive et en évitant les doublons, on améliore systématiquement les performances locales.
3. **Cross-Validation Infiltrée (Transductive CV)**:
   - Pour chaque pli de la 5-fold CV, les pseudo-labels du test set sont ajoutés uniquement aux 80% d'entraînement, préservant la propreté absolue des 20% de validation.

---

## Proposed Changes

### [Feature Engineering & Model Pipeline]

#### [NEW] [run_final_improved_pipeline.py](file:///c:/Users/BigZ/Downloads/geoai-aquaculture-pond-identification-challenge20260529-12149-ccxi78/run_final_improved_pipeline.py)
Un script complet autonome pour :
- Extraire les caractéristiques en séparant la disponibilité S1 et S2.
- Intégrer les nouvelles caractéristiques physiques validées (statistiques VH/VV bruts en dB et linéaire, corrélations temporelles MNDWI-NDVI).
- Entraîner un ensemble robuste de 3 modèles (LightGBM, CatBoost, XGBoost) par pli de validation (5 folds).
- Appliquer le self-training transductif corrigé à un seuil de 0.90 de confiance sur les prédictions OOF moyennées.
- Calibrer les probabilités finales avec une régression isotonique.
- Sauvegarder la nouvelle soumission optimisée sous le nom `submission_improved_final.csv`.

---

## Verification Plan

### Automated Tests
- Exécuter le script `run_final_improved_pipeline.py` et vérifier la progression des scores de validation croisée (CV) par pli.
- Le score CV combiné attendu (F1/AUC) après self-training transductif est d'environ **0.956**.

### Manual Verification
- Vérifier que la distribution des prédictions positives dans `submission_improved_final.csv` est réaliste (~400 à 435 échantillons positifs, soit ~40% du test set).
