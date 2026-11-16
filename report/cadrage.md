# Note de Cadrage - Système Multi-Agents IA pour la Souscription en Assurance

**Date** : 30 Octobre 2026
**Phase** : Phase 0 - Cadrage

> **Note de cadrage initiale, pas un rapport final.** Les résultats et hypothèses de cette version ne sont pas tous validés sur un jeu de test indépendant. Consulter le [README](../README.md) et le [suivi du projet](../PROGRESS.md) pour les résultats corrigés et les limites méthodologiques. Le prototype ne revendique aucune conformité réglementaire.

---

## 1. Contexte et Objectifs

### 1.1 Contexte
La souscription (underwriting) en assurance est un processus lent, coûteux et peu homogène. Les souscripteurs consultent manuellement des formulaires, des pièces justificatives, des grilles internes et des historiques de sinistres.

### 1.2 Objectif Principal
Concevoir et prototyper un système multi-agents IA pour la souscription en assurance automobile, capable de :
- Automatiser l'extraction et l'analyse des informations
- Produire une recommandation de décision avec score de risque
- Quantifier l'incertitude et déclencher l'escalade si nécessaire
- Mesurer et limiter les biais
- Fournir des explications compréhensibles
- Journaliser chaque étape pour l'audit

### 1.3 Contribution de Recherche
Étudier l'apport du multi-agents face à un agent unique sur 5 dimensions :
1. Collaboration multi-agents
2. Incertitude
3. Équité et biais
4. Explicabilité
5. Conformité réglementaire

---

## 2. Choix de la Branche d'Assurance

### 2.1 Choix : Assurance Automobile

**Justification pour la phase de recherche** :
- Un benchmark public français de responsabilité civile automobile est catalogué sur OpenML : fréquence (`freMTPL2freq`, identifiant 41214) et sévérité (`freMTPL2sev`, identifiant 41215).
- Le catalogue OpenML indique une licence CC0 pour ces deux jeux et les rattache au dépôt actuariel CASdatasets.
- Ces données permettent d’étudier une base actuarielle de fréquence/coût des sinistres, mais ne fournissent ni décisions réelles de souscription, ni dossiers contractuels, ni intégrations d’assureur.

La France est donc un **périmètre candidat pour le benchmark de recherche**, choisi selon les sources publiques actuellement repérées. Ce choix ne vaut pas validation d’un marché de déploiement : un pilote exploitable par un assureur exige ses données autorisées, ses règles et ses validations métier.

### 2.2 Exclusions
- Vie, santé, grands risques industriels (hors périmètre)
- Gestion des sinistres post-souscription (hors périmètre)

---

## 3. Données

### 3.1 Jeux présents et provenance

**Source retenue** : le prototype s’appuie sur le benchmark public **freMTPL2** (responsabilité civile auto française), diffusé par OpenML sous licence **CC0** (voir § 3.2). Le modèle livré est entraîné et évalué dessus (`experiments/real_data/train_fremtpl2_model.py`, `experiments/baselines/evaluate_holdout.py`).

**Héritage retiré** : l’ancien jeu de recherche synthétique (deux fichiers de 5 000 enregistrements sous `data/processed/`) et les scripts qui l’exploitaient (baselines B1–B4, ablation, collaboration, explainability, statistical, robustness, générateur `utils/data_processing*.py`) ont été **supprimés** du dépôt. Ils reposaient sur des variables fictives (score de crédit, prime, montant de sinistre, score de fraude) et ne sont plus référencés par le pipeline, les agents, l’API ni l’évaluation courante.

### 3.2 Benchmark public retenu — responsabilité civile auto française

Les fiches OpenML repérées sont :

- [`freMTPL2freq` (OpenML 41214)](https://www.openml.org/d/41214) : variables de risque de polices auto et nombre de sinistres. La fiche décrit 677 991 polices, alors que la qualité du catalogue indique 678 013 instances ; le nombre exact doit être contrôlé sur le fichier téléchargé avant toute expérience.
- [`freMTPL2sev` (OpenML 41215)](https://www.openml.org/d/41215) : 26 639 montants de sinistres.
- OpenML indique CC0 pour les deux fiches et cite Dutang et Charpentier (2018), CASdatasets ; le dépôt source est [dutangc/CASdatasets](https://github.com/dutangc/CASdatasets).

Ce benchmark est une piste pour construire et évaluer des modèles actuariels de fréquence/sévérité. Il ne contient pas les décisions historiques d’acceptation/refus, le texte des polices de l’assureur, les résultats des voies de recours ni les données nécessaires pour certifier un modèle de tarification ou de souscription. Il n’autorise donc pas à conclure qu’un système est prêt pour un assureur.

### 3.3 Labels expérimentaux — déprécié (contexte historique)

La version initiale dérivait des pseudo-labels d’un `fraud_risk_score` (champ qui **n’existe pas** dans freMTPL2) :
- `fraud_risk_score > 70` → `Reject`, `> 50` → `Accept_with_Surcharge`, sinon `Accept_Standard`.

Cette convention a été **abandonnée**. Le prototype actuel entraîne un modèle **binaire** sur une cible réelle observée de freMTPL2 (la police a-t-elle déclaré au moins un sinistre ?) et dérive la recommandation par seuils métier documentés (surprime, revue). freMTPL2 ne fournissant aucune décision de souscription, cette transformation reste une convention de recherche, pas une politique d’assureur validée.

**Distribution historique rapportée dans la version initiale (obsolète)** : Accept_Standard 49,9 % ; Accept_with_Surcharge 20,5 % ; Reject 29,7 %. La répartition réelle sur le hold-out freMTPL2 figure dans le [README](../README.md).

### 3.4 Catalogue des Variables (freMTPL2)

Le contrat de features est défini une seule fois dans `utils/actuarial_features.py` (partagé entraînement / inférence) :

| Variable | Type | Sensible | Description |
|-----------|------|----------|-------------|
| driver_age | Numérique | ⚠️ proxy | Âge du conducteur (`DrivAge`) |
| bonus_malus | Numérique | ❌ | Coefficient bonus-malus (`BonusMalus`) |
| vehicle_age | Numérique | ❌ | Ancienneté du véhicule (`VehAge`) |
| vehicle_power | Numérique | ❌ | Puissance du véhicule (`VehPower`) |
| vehicle_gas | Catégorielle | ❌ | Carburant (`Regular` / `Diesel`) |
| density | Numérique | ❌ | Densité de population de la commune (proxy géographique) |
| area | Catégorielle | ❌ | Zone géographique codée de `A` à `F` (proxy géographique) |

**Groupes sensibles / proxys** : `driver_age` et `bonus_malus` (indirectement `density` / `area`) servent de proxys d’équité. L’analyse d’équité rapportée est une **parité descriptive du taux de sélection** ; faute de décisions réelles d’acceptation/refus dans freMTPL2, l’égalité des chances ne peut pas être mesurée honnêtement.

---

## 4. Questions de Recherche

### 4.1 Collaboration Multi-Agents
**H1** : Une architecture à débat ou à vote pondéré réduit les erreurs sur les dossiers ambigus par rapport à un agent unique.

**Méthode** : Comparer 4 schémas (pipeline, orchestrateur, débat, vote pondéré) contre baseline agent unique.

**Mesures** : Exactitude, F1, latence, taux de désaccord résolu.

### 4.2 Incertitude
**H2** : La prise en compte de l'incertitude (abstention + escalade) réduit fortement le taux d'erreur des décisions automatiques, au prix d'un taux d'automatisation plus faible.

**Méthode** : Calibration (MAPIE), ensembles, escalade si confiance < seuil.

**Mesures** : ECE, Brier score, courbe risque-couverture.

### 4.3 Équité et Biais
**H3** : Un agent équité dédié réduit les écarts de traitement sans dégrader fortement la performance.

**Méthode** : Tests de parité, tests contrefactuels, atténuation (post-processing).

**Mesures** : Parité démographique, égalité des chances, impact disproportionné.

### 4.4 Explicabilité
**H4** : Des explications ancrées dans des sources citées sont jugées plus utiles que des explications génériques.

**Méthode** : SHAP, DiCE (contrefactuels), citations RAG, évaluation humaine.

**Mesures** : Fidélité, stabilité, score de compréhension (panel 30-50 décisions).

### 4.5 Conformité Réglementaire
**H5** : Le système satisfait aux exigences de transparence, supervision humaine et protection des données.

**Méthode** : Journal d'audit, cartographie exigences → fonctions, tests de conformité.

**Mesures** : Couverture des exigences, complétude de la traçabilité.

---

## 5. Architecture Multi-Agents

### 5.1 Agents Spécialisés (6)

1. **Intake/Validation** : Réception, validation de complétude, normalisation
2. **Agent Client** : Analyse du profil, cohérence des déclarations, score de profil
3. **Agent Documents** : RAG sur conditions générales, vérification d'éligibilité
4. **Agent Risque** : Scoring ML (XGBoost) + règles actuarielles
5. **Agent Externe** : Données météo, zones à risque, registres publics
6. **Agent Sinistres** : Historique, fréquence/sévérité, signaux de fraude

### 5.2 Agents de Décision (2)

7. **Orchestrateur** : Planification, routage, détection de conflits
8. **Agent Décideur** : Agrégation, arbitrage, calcul d'incertitude

### 5.3 Agents de Gouvernance (3)

9. **Agent Équité** : Tests de biais, blocage si seuil dépassé
10. **Agent Conformité** : Vérification légale, RGPD, IA Act
11. **Agent Explication** : Explications globales et locales, contrefactuels

---

## 6. Baselines et Variantes

### 6.1 Baselines

- **B1** : Règles de souscription seules (if-then sur seuils)
- **B2** : Modèle ML seul (XGBoost sur variables tabulaires)
- **B3** : Agent LLM unique avec outils
- **B4** : Pipeline séquentiel d'agents

### 6.2 Variantes à Tester

- Avec/sans module d'incertitude
- Avec/sans agent équité
- Avec/sans agent d'explication
- 4 schémas de collaboration (pipeline, orchestrateur, débat, vote)

---

## 7. Protocole Expérimental

### 7.1 Split des Données

- Entraînement / hold-out : **80 % / 20 %** (séparation stratifiée, graine 42), sur les 678 013 polices freMTPL2 retenues.
- L’évaluation hold-out (`experiments/baselines/evaluate_holdout.py`) reproduit exactement cette partition (mêmes taille, stratification et graine).
- Un split temporel n’est pas possible : freMTPL2 ne fournit pas d’horodatage exploitable des polices.

### 7.2 Mesures par Axe

| Axe | Mesures |
|-----|---------|
| Qualité de décision | ROC-AUC, PR-AUC, exactitude (seuil 0,5) |
| Robustesse | Résultats sous bruit, champs manquants |
| Incertitude | ECE top-label, Brier, taux d’escalade, courbe risque-couverture |
| Équité | Parité descriptive du taux de sélection (âge, bonus-malus) — pas d’égalité des chances (absence de ground-truth accept/refus) |
| Explicabilité | Fidélité, évaluation humaine |
| Efficacité | Latence, coût en jetons, taux d’automatisation |
| Conformité | Couverture exigences, complétude journal |

### 7.3 Rigueur Statistique
- Plusieurs exécutions par configuration (LLM non déterministes)
- Moyenne et écart-type
- Tests de significativité
- Graine fixée quand possible
- Étude d'ablation

---

## 8. Stack Technique

- **Orchestration** : LangGraph
- **LLM** : OpenAI GPT-4 (prototypage) / Open source local (confidentialité)
- **RAG** : Chroma + sentence-transformers
- **Scoring** : scikit-learn, XGBoost
- **Incertitude** : MAPIE (prédiction conforme)
- **Équité** : Fairlearn
- **Explicabilité** : SHAP, DiCE
- **Interface** : Streamlit
- **Backend** : SQLite
- **MLOps** : MLflow

---

## 9. Planning (Accéléré)

**Phase actuelle** : Phase 0 - Cadrage ✅

**Prochaines phases** :
- Phase 1 : Données ✅ (terminé)
- Phase 2 : Baselines (en cours)
- Phase 3 : Agents spécialisés
- Phase 4 : Orchestration
- Phase 5 : Gouvernance
- Phase 6 : Expérimentation
- Phase 7 : Rédaction

---

## 10. Risques et Mitigation

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Données non représentatives | Élevée | Élevé | Reconnaître la limite dans le rapport |
| Hallucinations LLM | Moyenne | Élevé | RAG avec citations, vérifications croisées |
| Coût LLM élevé | Moyenne | Moyen | Budget de jetons, cache |
| Périmètre trop large | Élevée | Élevé | Focus sur auto, MVP simplifié |
| Biais difficiles à mesurer | Moyenne | Moyen | Tests contrefactuels, variables proxy |

---

## 11. Livrables

1. **Rapport** : État de l'art, conception, expériences, résultats
2. **Code** : Versionné, reproductible, scripts d'expériences
3. **Prototype** : Streamlit fonctionnel
4. **Données** : Dataset avec documentation
5. **Journal d'audit** : Exemple de traçabilité
6. **Rapport d'équité** : Mesures et analyse

---

**Statut** : Cadrage validé, prêt pour Phase 2 (Baselines)
