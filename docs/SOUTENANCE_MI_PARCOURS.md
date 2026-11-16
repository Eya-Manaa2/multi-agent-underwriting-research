# Soutenance de mi-parcours — système multi-agents pour la souscription

> **Objet de ce document.** Reconstituer, dans l’ordre, le travail matérialisé
> dans le dépôt, décrire les micro-étapes, les données, les technologies, les
> résultats vérifiables et les limites, puis proposer une suite concrète.
>
> **Règle de lecture.** Les dates exactes de chaque développement ne sont pas
> toutes journalisées. La chronologie ci-dessous est reconstruite à partir du
> cadrage, des métadonnées datées, du suivi et des dépendances du code. Elle
> n’attribue donc pas de date précise aux étapes dont le dépôt ne fournit pas.
> « Implémenté » ne signifie pas « validé par un assureur » ni « prêt pour la
> production ».

## 1. Résumé à présenter au jury

Le projet étudie comment des agents spécialisés peuvent aider un souscripteur
à analyser un dossier d’assurance. Le périmètre réellement développé est
l’assurance automobile. Le résultat est un **prototype académique local** :
une application de démonstration, une API, une orchestration multi-agents, un
modèle de risque entraîné sur un jeu automobile public, des expériences de
recherche et un volet de tarification actuarielle.

La conclusion à annoncer sans ambiguïté est la suivante :

> Le système produit une analyse et une recommandation explicables qui passent
> par une revue humaine. Le jeu freMTPL2 permet d’apprendre et d’évaluer la
> survenance et le coût de sinistres, mais pas de retrouver les décisions
> historiques d’acceptation ou de refus d’un assureur. Les résultats actuels
> sont donc des résultats de recherche sur des sinistres observés, et non la
> preuve d’une qualité de décision de souscription en production.

Le volet de tarification pure (fréquence × sévérité) a été entraîné sur
freMTPL2 et possède des artefacts versionnés. Un second volet, le benchmark
Prudential sur de vraies classes de risque attribuées par des souscripteurs,
est préparé dans le code mais reste **bloqué tant que le téléchargement Kaggle,
l’acceptation des règles, l’entraînement et les tests dédiés n’ont pas été
effectués**.

## 2. Chronologie reconstituée et micro-étapes

### Étape 1 — Cadrer la question de recherche

Le cadrage initial, daté du 30 octobre 2026 dans
[`report/cadrage.md`](../report/cadrage.md), pose comme question générale :
comment coordonner plusieurs agents pour analyser un dossier, mesurer le
risque, signaler l’incertitude, surveiller l’équité et expliquer une
recommandation.

Micro-étapes :

1. Définir le processus métier visé : collecte des informations, lecture des
   pièces, analyse du risque, recommandation et contrôle humain.
2. Retenir l’automobile comme première ligne d’assurance, car un jeu public
   actuariel français est disponible.
3. Séparer les axes de recherche : collaboration, incertitude, équité,
   explicabilité et gouvernance réglementaire.
4. Formuler des comparaisons expérimentales avec des baselines : règles,
   modèle seul, agent LLM unique et pipeline multi-agents.
5. Poser les limites dès le cadrage : jeu public non équivalent aux dossiers
   d’un assureur, labels de décisions manquants et conformité non certifiée.

**Livrable :** cadrage de recherche, objectifs, hypothèses et protocole
préliminaire. Le plan du cadrage est une intention de recherche : les
technologies citées dans cette première version ne sont pas toutes intégrées
au prototype final.

### Étape 2 — Sélectionner, récupérer et contrôler les données automobiles

Le jeu retenu est **freMTPL2**, une base publique de responsabilité civile
automobile française associée à CASdatasets (Dutang et Charpentier, 2018) et
référencée sur OpenML :

- `freMTPL2freq`, OpenML 41214 : exposition des polices, caractéristiques du
  risque et nombre de sinistres ; 678 013 enregistrements dans le fichier
  utilisé.
- `freMTPL2sev`, OpenML 41215 : 26 639 montants de sinistres.
- Licence affichée par OpenML : CC0. La provenance et les empreintes des
  sources sont enregistrées par les scripts d’ingestion.

Micro-étapes :

1. Télécharger les deux fichiers ARFF avec
   [`experiments/data/fetch_fremtpl2.py`](../experiments/data/fetch_fremtpl2.py).
2. Contrôler le schéma, les enregistrements et les empreintes attendues avant
   l’utilisation.
3. Enregistrer les fichiers bruts et un manifeste de provenance sous
   `data/raw/` (données brutes ignorées par Git).
4. Transformer chaque police vers un contrat de variables unique dans
   [`utils/actuarial_features.py`](../utils/actuarial_features.py).
5. Construire des jeux de référence inspectables pour les agents à partir des
   données source, avec
   [`experiments/real_data/build_reference_datasets.py`](../experiments/real_data/build_reference_datasets.py).
6. Versionner les références locales et leurs informations de provenance
   sous `data/reference/`.

Variables principales utilisées par le modèle : âge du conducteur, coefficient
bonus-malus, âge et puissance du véhicule, carburant, densité et zone. L’âge
et certaines variables géographiques sont aussi des proxys à examiner pour
l’équité.

**Limite déterminante :** freMTPL2 fournit des observations de sinistres, pas
les décisions d’acceptation, de refus ou de surprime prises par un assureur.
Il ne contient pas non plus les dossiers contractuels ou les motifs de
décision nécessaires pour certifier une politique de souscription.

### Étape 3 — Construire une première estimation du risque

Un classifieur XGBoost est entraîné sur une cible réelle du jeu : la police
a-t-elle déclaré au moins un sinistre (`ClaimNb >= 1`) ? Le script
[`experiments/real_data/train_fremtpl2_model.py`](../experiments/real_data/train_fremtpl2_model.py)
entraîne le modèle ; l’agent de risque le charge ensuite pour scorer une
demande.

Micro-étapes :

1. Lire les lignes ARFF et exclure les lignes dont le schéma ou l’exposition
   ne convient pas.
2. Encoder les variables catégorielles et garder l’ordre de variables défini
   dans le contrat partagé.
3. Séparer les données en entraînement/test : 80/20, stratifié, graine 42.
   Le test représente 135 603 polices.
4. Entraîner le classifieur sur la survenance observée d’au moins un sinistre.
5. Sauvegarder le modèle XGBoost et ses métadonnées : définition de la cible,
   variables, source, partition, seuils et métriques.
6. À l’inférence, produire une probabilité de sinistre puis une recommandation
   de recherche à l’aide de seuils documentés.

Métriques présentes dans
[`models/fremtpl2_underwriting_metadata.json`](../models/fremtpl2_underwriting_metadata.json) :

| Mesure sur le hold-out | Résultat consigné |
|---|---:|
| ROC-AUC | 0,6460 |
| PR-AUC / average precision | 0,1094 |
| Exactitude au seuil 0,5 | 0,9497 |
| Taux de base sinistre | 0,0502 |
| Brier binaire | 0,0466 (README) |
| ECE top-label | 0,0011 (README) |

L’exactitude élevée ne doit pas être présentée seule : environ 95 % des
polices du test ne déclarent pas de sinistre. Un modèle qui prédit presque
toujours la classe majoritaire peut ainsi afficher une forte exactitude. La
ROC-AUC, la PR-AUC, la calibration et le taux de base apportent le contexte.

La correspondance de probabilité vers `Accept_Standard`,
`Accept_with_Surcharge` ou `Reject` reste une **convention de recherche**,
pas une décision historique apprise sur des décisions d’assureur. Le cadrage
et les notes de suivi signalent que les anciennes pseudo-étiquettes synthétiques
ont été abandonnées.

### Étape 4 — Ajouter une tarification actuarielle fréquence-sévérité

Le levier A améliore l’estimation économique en remplaçant le seuil de
probabilité seul par une **prime pure attendue** :

`prime pure = fréquence attendue des sinistres × coût moyen attendu par sinistre`

Le module
[`experiments/pricing/pure_premium.py`](../experiments/pricing/pure_premium.py)
entraîne deux modèles XGBoost :

1. **Fréquence** : régression Poisson sur le nombre de sinistres, avec
   exposition (`Exposure`) prise en compte comme offset.
2. **Sévérité** : régression Gamma sur les montants de sinistres observés,
   joints aux caractéristiques de police via `IDpol`.
3. **Prime pure** : multiplication des prédictions de fréquence et de
   sévérité, exprimée en euros.
4. **Politique économique versionnée** : seuil de surprime égal à une fois la
   prime pure moyenne de référence et seuil de refus égal à trois fois cette
   moyenne. Le ratio de sinistralité cible de 0,70 sert à convertir la prime
   pure en prime technique.
5. **Évaluation** : comparaison sur le même hold-out reproductible que le
   reste du projet ; génération des modèles et métadonnées.

Chiffres consignés dans
[`models/fremtpl2_pricing_metadata.json`](../models/fremtpl2_pricing_metadata.json)
et [`data/underwriting/pricing_policy_v1.json`](../data/underwriting/pricing_policy_v1.json) :

| Résultat | Valeur |
|---|---:|
| Polices train / test | 542 410 / 135 603 |
| Prime pure moyenne de référence train | 123,01 € |
| Seuil de surprime | 123,01 € |
| Seuil de refus économique | 369,04 € |
| Seuils équivalents de probabilité de sinistre | 0,0532 / 0,1596 |
| Déviance Poisson fréquence | 0,3055 |
| Fréquence prédite / observée en moyenne | 0,05345 / 0,05336 |
| Déviance Gamma sévérité | 1,7168 |
| Sévérité moyenne des sinistres | 2 311,83 € |
| Coût total prédit / observé sur le test | 11 951 916 € / 11 055 690,67 € |
| Ratio coût prédit / observé | 1,0811 |

Interprétation : la fréquence est bien alignée en moyenne sur ce test, mais
le coût total est surestimé d’environ 8,1 %. La sévérité est entraînée sur les
seuls sinistres observés ; le taux de couverture des sinistres est d’environ
3,69 % dans les métadonnées, ce qui doit être pris en compte dans
l’interprétation du coût total. Ces résultats ne démontrent ni rentabilité
commerciale, ni adéquation à un tarif réel, ni validité d’un refus d’assurance.

Tests du module : [`tests/test_pricing.py`](../tests/test_pricing.py) couvre les
fonctions économiques, les déviances, les artefacts et leurs cohérences. Le
journal de travail partagé rapporte une exécution antérieure de **19 tests
réussis** pour le levier A ; ce document ne prétend pas avoir relancé la suite
pendant sa rédaction.

### Étape 5 — Mettre en place les agents et l’orchestration

L’orchestration repose sur **LangGraph** et un état partagé entre nœuds. Les
onze agents métiers/gouvernance du graphe sont :

1. Intake — validation de la demande et complétude.
2. Customer — synthèse du profil client.
3. Risk — score issu du modèle.
4. Uncertainty — mesure d’ambiguïté du score et proposition d’escalade.
5. Document — contrôles documentaires locaux.
6. External — facteurs externes et géographiques.
7. Claims — éléments de références d’antécédents.
8. Decision — agrégation et recommandation.
9. Fairness — indicateurs de surveillance d’équité.
10. Compliance — signaux et limites de conformité.
11. Explanation — explication structurée.

L’agent de document LLM optionnel est également un composant du flux, mais ne
remplace pas les onze rôles ci-dessus et ne prend jamais la décision. Il
analyse, après opt-in, des extraits de pièces fournies et retourne des constats
et citations contrôlées dans le texte source.

Micro-étapes :

1. Instancier les agents Python spécialisés dans
   [`orchestrator/langgraph_orchestrator.py`](../orchestrator/langgraph_orchestrator.py).
2. Définir un état de dossier contenant les résultats intermédiaires, le mode
   de décision et la provenance.
3. Relier les nœuds dans les modes de collaboration disponibles (`sequential`
   et `parallel`).
4. Faire agréger les résultats dans
   [`orchestrator/underwriting_report.py`](../orchestrator/underwriting_report.py),
   qui construit un bilan consolidé.
5. Traiter l’incertitude prédictive : entropie binaire normalisée entre 0 et
   1 ; niveaux low/moderate/high ; forte ambiguïté ou score absent = revue
   humaine recommandée.
6. Enregistrer les résultats et les sources exploitées dans la provenance du
   rapport.

**Important :** l’entropie est une mesure d’ambiguïté du score, pas une preuve
que la probabilité est bien calibrée. La calibration est mesurée séparément
dans les expériences.

### Étape 6 — Ajouter pièces, références locales et garde-fous

Le parcours d’intake accepte plusieurs documents et s’appuie sur le pack
produit automobile [`products/auto/rules.py`](../products/auto/rules.py).
L’extraction PDF/TXT est disponible ; l’OCR d’images dépend de composants
optionnels. La checklist produit peut bloquer la recommandation si une pièce
obligatoire manque ou ne peut pas être traitée.

Les agents sinistres et externe lisent des jeux JSON versionnés par
[`connectors/reference_data.py`](../connectors/reference_data.py), situés dans
`data/reference/`. Le connecteur expose source, licence, version, checksum et
nombre d’enregistrements. En cas de source absente ou invalide, le système
échoue en mode fermé plutôt que d’inventer une information.

Le connecteur [`connectors/agira.py`](../connectors/agira.py) prépare une
intégration selon des modes de rejeu, réel ou désactivé. La fixture de démonstration
AGIRA est explicitement fictive ; une connexion réelle exige autorisation,
convention et accès d’un assureur. Elle ne doit pas être présentée comme une
connexion en production.

Les pièces de démonstration sont sous `examples/test_documents/`. Les documents
ne sont pas conservés comme pièces dans le parcours décrit par le README ; des
métadonnées d’intake sont enregistrées. Si l’option LLM est activée, des
extraits pertinents peuvent être transmis au fournisseur configuré : seules
des données fictives doivent être utilisées pour la démonstration.

### Étape 7 — Exposer le prototype à un utilisateur

Technologies de démonstration :

- **Streamlit** : parcours de dépôt, analyse, file de revue humaine et audit.
- **FastAPI / Uvicorn** : API de connexion, analyse, dossiers en attente,
  décisions, audit et gestion d’utilisateurs.
- **SQLite** : persistance locale des dossiers et de l’audit sous
  `data/runtime/` (non versionnée).
- **RBAC pilote** : rôles souscripteur, manager, conformité et administrateur.
- **Audit append-only** : export JSON et contrôle d’une chaîne de hachage.

Le parcours est configuré en `human_in_the_loop` : chaque recommandation est
soumise à la file de revue, le refus nécessite une revue humaine et le LLM ne
décide pas. Le prototype inclut également un brouillon DPIA dans
[`docs/DPIA_DRAFT.md`](./DPIA_DRAFT.md), non validé par un DPO.

### Étape 8 — Comparer, analyser et documenter les limites

Le dépôt contient des scripts pour :

- Baselines B1–B4 : règles seules, ML seul, agent LLM seul et pipeline
  multi-agents.
- Évaluation sur hold-out commun et reproductible.
- Robustesse aux perturbations et aux champs manquants.
- Ablation d’agents.
- Évaluation de l’incertitude et de l’équité.
- Comparaison de schémas de collaboration.

Le README rapporte notamment une parité descriptive de taux de sélection
(ratio min/max) de 0,9953 pour les groupes d’âge utilisés, et de 0,5529 pour
les groupes bonus-malus. Ces chiffres ne sont pas des tests de discrimination
légale : en l’absence de décisions réelles, ils ne mesurent pas l’égalité des
chances entre décisions d’acceptation/refus.

Les expériences de collaboration LLM (agent unique, débat, vote pondéré) ne
sont pas toutes exécutées, selon le suivi disponible : le quota fournisseur
est un blocage consigné. Aucun résultat indisponible ne doit être remplacé par
une estimation ou être présenté comme acquis. Les scripts exécutés, les tailles
d’échantillon et les statuts `partial`/`unavailable` doivent accompagner toute
comparaison montrée au jury.

## 3. Levier B — ce qui est préparé et ce qui reste à faire

Le levier B vise une question différente du levier A : mesurer la capacité à
reproduire des **classes ordinales réelles de risque attribuées par des
souscripteurs**. Le jeu Prudential Life Insurance Assessment contient des
demandes d’assurance vie et une cible `Response` allant de 1 à 8.

Le code présent :

- [`experiments/data/fetch_prudential.py`](../experiments/data/fetch_prudential.py)
  : téléchargement Kaggle, vérification de 129 colonnes, validation de la
  cible et création d’un manifeste de provenance.
- [`experiments/underwriting/prudential_decision.py`](../experiments/underwriting/prudential_decision.py)
  : préparation de données, modèle ordinal XGBoost, optimisation de seuils et
  calcul du QWK, avec comparaison à une baseline majoritaire.

À la date de préparation de cette note, le dépôt ne contient pas
`data/raw/prudential/train.csv`, ses artefacts de modèle/métadonnées ni
`tests/test_prudential_decision.py`. Le téléchargement, l’entraînement, le
calcul du QWK et les tests dédiés ne sont donc pas des résultats acquis.
Le nombre de dossiers annoncé par la compétition (59 381) est une information
de description du jeu, à confirmer après téléchargement par le manifeste.

Micro-étapes restantes du levier B :

1. Obtenir l’accès Kaggle **sur le poste local uniquement** et accepter les
   conditions de la compétition sur le site.
2. Télécharger le CSV avec le script d’ingestion ; ne jamais placer un token
   dans Git, une commande partagée, un fichier de rapport ou une capture.
3. Vérifier le manifeste, la taille, les 129 colonnes (`Id`, 127 variables,
   `Response`), le nombre de lignes et les classes 1–8.
4. Examiner valeurs manquantes, variables catégorielles et distribution des
   classes ; conserver la provenance et les conditions d’usage non commercial.
5. Exécuter le protocole défini par le script : séparation train/validation/test,
   entraînement XGBoost ordinal et optimisation des sept seuils sur validation.
   Le test final ne doit pas servir à optimiser ces seuils.
6. Calculer le QWK (métrique officielle ordinale) et comparer à la baseline
   classe majoritaire ; compléter par accuracy, MAE, macro-F1 et matrice de
   confusion pour rendre les erreurs interprétables.
7. Ajouter les tests unitaires pour les seuils, métriques, encodages,
   provenance, artefacts et cas limites ; exécuter les tests et enregistrer
   les résultats réels.
8. Documenter les limites : jeu vie et non automobile, contexte de compétition,
   décisions de classes qui ne représentent pas nécessairement les pratiques
   actuelles d’un assureur, et conditions Kaggle à respecter.

**Précaution de sécurité :** un jeton Kaggle a été copié dans la conversation
partagée avec le projet. Il doit être révoqué depuis Kaggle puis régénéré.
Ne pas le recopier dans le dépôt ni dans cette soutenance ; utiliser une
variable d’environnement locale ou le mécanisme de stockage privé proposé
par Kaggle.

## 4. Arborescence et fichiers mobilisés

Extrait ciblé des fichiers pertinents pour le projet (les environnements
`.venv/`, caches, bases locales, données brutes et artefacts temporaires sont
volontairement omis) :

```text
.
├── README.md
├── PROGRESS.md
├── requirements.txt
├── requirements-research.txt
├── agents/
│   ├── intake_agent.py
│   ├── customer_agent.py
│   ├── risk_agent.py
│   ├── uncertainty_agent.py
│   ├── document_agent.py
│   ├── llm_document_agent.py
│   ├── external_agent.py
│   ├── claims_agent.py
│   ├── decision_agent_enhanced.py
│   ├── fairness_agent.py
│   ├── compliance_agent.py
│   └── explanation_agent.py
├── api/
│   └── app.py
├── connectors/
│   ├── reference_data.py
│   └── agira.py
├── data/
│   ├── demo/real_cases.json
│   ├── reference/
│   │   ├── claims_history.json
│   │   ├── geographic_risk.json
│   │   └── agira_fixture.json
│   └── underwriting/pricing_policy_v1.json
├── docs/
│   ├── COMPLIANCE_DOCUMENTATION.md
│   ├── DPIA_DRAFT.md
│   └── SOUTENANCE_MI_PARCOURS.md
├── examples/test_documents/
├── experiments/
│   ├── data/
│   │   ├── fetch_fremtpl2.py
│   │   └── fetch_prudential.py
│   ├── real_data/
│   │   ├── train_fremtpl2_model.py
│   │   ├── build_reference_datasets.py
│   │   └── build_demo_cases.py
│   ├── pricing/pure_premium.py
│   ├── underwriting/prudential_decision.py
│   ├── baselines/
│   │   ├── baseline_b1_rules.py
│   │   ├── baseline_b2_ml.py
│   │   ├── baseline_b3_llm_agent.py
│   │   ├── baseline_b4_pipeline.py
│   │   ├── evaluation.py
│   │   └── run_baselines.py
│   ├── ablation/ablation_study.py
│   ├── collaboration/collaboration_schemes.py
│   ├── fairness/
│   ├── robustness/robustness_tests.py
│   └── uncertainty/
├── models/
│   ├── fremtpl2_underwriting_model.json
│   ├── fremtpl2_underwriting_metadata.json
│   ├── fremtpl2_frequency_model.json
│   ├── fremtpl2_severity_model.json
│   └── fremtpl2_pricing_metadata.json
├── orchestrator/
│   ├── langgraph_orchestrator.py
│   └── underwriting_report.py
├── products/auto/rules.py
├── report/cadrage.md
├── streamlit/app.py
├── tests/
│   ├── test_core.py
│   ├── test_api.py
│   ├── test_baselines.py
│   ├── test_pricing.py
│   ├── test_reference_data.py
│   ├── test_research_experiments.py
│   └── test_underwriting_report.py
└── utils/
    ├── actuarial_features.py
    ├── case_store.py
    ├── document_bundle.py
    └── document_upload.py
```

### Rôle des groupes de fichiers

| Groupe | Rôle dans le projet |
|---|---|
| `agents/` | Logique des agents spécialisés, chacun avec une responsabilité bornée. |
| `orchestrator/` | Coordination LangGraph et production du rapport consolidé. |
| `experiments/data/`, `experiments/real_data/` | Ingestion, préparation des sources et entraînement reproductible. |
| `experiments/pricing/`, `experiments/underwriting/` | Modèle de prime pure automobile et benchmark de décisions Prudential à venir. |
| `experiments/baselines/`, `fairness/`, `robustness/`, `uncertainty/`, `ablation/`, `collaboration/` | Protocole comparatif et axes d’évaluation de la recherche. |
| `models/` | Modèles entraînés et métadonnées de provenance/performance. |
| `data/reference/` | Références locales versionnées utilisées par des agents ; la fixture AGIRA est fictive. |
| `products/auto/` | Règles opérationnelles de démonstration propres au produit automobile. |
| `connectors/` | Accès abstrait aux sources de référence et point d’intégration AGIRA. |
| `api/`, `streamlit/`, `utils/` | Interface, API, validation/intake et persistance du pilote local. |
| `tests/` | Tests automatisés des composants présents. Aucun test Prudential dédié n’est encore listé. |
| `report/`, `docs/`, `README.md`, `PROGRESS.md` | Cadrage, limites, gouvernance, état de projet et instructions. |

`data/raw/` n’est pas montré ici parce qu’il est ignoré par Git : il doit contenir
les téléchargements locaux. En particulier, les fichiers Prudential n’y sont
pas encore présents d’après l’inspection du dépôt. Certains anciens fichiers
ou résultats supprimés apparaissent comme suppressions dans l’état Git local ;
ils ne sont donc pas considérés comme des livrables actuels dans cette
arborescence.

## 5. Technologies et sources — synthèse

### Technologies réellement présentes

| Besoin | Technologie / composant |
|---|---|
| Langage et manipulation numérique | Python, NumPy |
| Modèles tabulaires | XGBoost ; scikit-learn pour partitionnement et métriques |
| Coordination d’agents | LangGraph |
| API de démonstration | FastAPI, Uvicorn |
| Interface locale | Streamlit |
| Persistance pilote et audit | SQLite, export JSON, empreinte chaînée |
| Pièces | Extraction PDF/TXT ; OCR optionnel selon dépendances installées |
| LLM documentaire | Groq facultatif, uniquement sur opt-in ; non nécessaire au mode local par défaut |
| Tests | `unittest` dans les tests de pricing et d’ingestion ; autres tests présents dans `tests/` |

`requirements-research.txt` déclare en plus des outils optionnels pour les
expériences, notamment SciPy, Fairlearn, MAPIE, SHAP, pandas et les outils
d’analyse. Une dépendance inscrite dans le fichier ne signifie pas que toutes
les méthodes correspondantes sont utilisées dans le parcours utilisateur.

### Sources de données et de référence

1. OpenML `freMTPL2freq` (41214) et `freMTPL2sev` (41215), associés à
   CASdatasets, Dutang & Charpentier (2018), licence annoncée CC0.
2. Jeux locaux de références `data/reference/`, dérivés pour le prototype de
   freMTPL2 et contrôlés par provenance/checksum.
3. Prudential Life Insurance Assessment sur Kaggle : source prévue pour le
   levier B ; accès soumis à l’authentification, à l’acceptation des règles et
   aux conditions de la compétition.
4. AGIRA : intégration seulement préparée ; les données de fixture sont
   fictives et l’accès réel nécessite un cadre contractuel et des identifiants
   autorisés.

## 6. Prochaines étapes, dans l’ordre et avec critères de sortie

### Priorité 1 — Sécuriser et figer la démonstration de mi-parcours

- Révoquer le jeton Kaggle partagé et n’utiliser qu’un secret local.
- Exécuter les tests ciblés existants et archiver la sortie réelle, le commit,
  l’environnement Python et la date.
- Préparer une démonstration reproductible avec les seuls dossiers fictifs.
- Vérifier que les résultats présentés distinguent clairement données
  observées, fixtures fictives et références dérivées.
- Corriger dans le pack produit automobile toute description devenue obsolète :
  `products/auto/rules.py` contient encore une note indiquant que le classifieur
  utilise des pseudo-labels, alors que les métadonnées du modèle indiquent une
  cible réelle `ClaimNb >= 1`.

**Critère de sortie :** aucun secret affiché, démonstration sans données
personnelles, résultats reproductibles et slides cohérents avec les artefacts.

### Priorité 2 — Terminer et qualifier le levier B

- Obtenir l’accès et vérifier les conditions de réutilisation Kaggle.
- Télécharger, vérifier, versionner le manifeste (pas le secret) et consigner
  les empreintes des données.
- Compléter le test unitaire Prudential manquant.
- Exécuter le train/validation/test sans fuite : optimiser les seuils sur
  validation et annoncer le QWK du test une seule fois.
- Rapporter baseline, métriques, matrice de confusion et distribution des
  classes ; documenter les limites d’une étiquette de risque héritée d’une
  compétition.

**Critère de sortie :** les performances et la provenance sont vérifiables par
un tiers, et les chiffres n’apparaissent dans la présentation qu’après
exécution réelle.

### Priorité 3 — Renforcer l’évaluation de la recherche

- Relancer B1–B4 à configuration et échantillons identiques ; conserver les
  statuts des runs LLM non exécutés.
- Ajouter des répétitions et intervalles d’incertitude si les ressources le
  permettent.
- Mesurer la calibration et le compromis risque-couverture de l’escalade.
- Faire analyser l’équité par groupes pertinents avec des labels de décision
  appropriés ; ne pas assimiler parité de sélection et absence de discrimination.
- Faire évaluer les explications par des souscripteurs / experts, dès qu’un
  panel métier est disponible.
- Réunir au moins 200 dossiers annotés par des experts métiers, comme indiqué
  dans `PROGRESS.md`, en définissant les règles d’annotation et l’accord entre
  annotateurs.

**Critère de sortie :** résultats comparables, limites quantifiées, protocole
documenté et conclusions adaptées à la qualité réelle des labels.

### Priorité 4 — Extension vers santé et autres branches

L’architecture peut être réutilisée, mais **les données, les variables, les
règles, les agents métier, les métriques et les validations ne peuvent pas être
copiés tels quels depuis l’automobile**. Il faut développer une ligne produit
distincte et avancer par étapes :

1. **Cadrage métier et juridique.** Choisir un sous-produit précis (ex. santé
   individuelle, habitation), le pays, les utilisateurs et l’usage autorisé.
   Faire valider les finalités, la base légale, les catégories de données, les
   règles d’accès et les exigences réglementaires par des spécialistes. Pour
   la santé, traiter les données de santé comme particulièrement sensibles :
   minimisation stricte, accès restreint, durée de conservation définie,
   chiffrement et environnement de test fictif/sécurisé.
2. **Accès aux données autorisées.** Identifier les sources métier et leur
   propriétaire, documenter consentement/base juridique, licence, qualité,
   biais et période de validité. Ne pas récupérer de dossiers patients ou
   clients réels pour une démonstration non autorisée.
3. **Définir les labels propres à la branche.** Exemples : décision
   d’acceptation, niveau de couverture, exclusions ou demande d’informations
   complémentaires. Définir les motifs et le processus de revue avec les
   experts ; ne pas substituer un label de sinistre à une décision de
   souscription.
4. **Créer un pack produit indépendant.** Ajouter par exemple
   `products/health/rules.py`, un schéma d’entrée, les pièces requises, les
   exclusions/garanties, les états de dossier et les règles HITL. Conserver
   `products/auto/` inchangé et tester la sélection de ligne produit.
5. **Créer un adaptateur de données spécialisé.** Valider types, plages,
   champs manquants et provenance ; isoler les connecteurs et jeux de test.
   Les documents ne doivent pas circuler vers un fournisseur LLM par défaut.
6. **Entraîner un modèle spécifique.** Faire un split adapté (temporel ou
   groupé si nécessaire), contrôler les fuites, calibrer les probabilités,
   établir une baseline simple et documenter les variables exclues/proxys.
7. **Adapter les agents.** Réutiliser les interfaces génériques (intake,
   incertitude, explication, audit), mais développer les agents de garanties,
   exclusions et règles santé/habitation comme modules propres au produit.
   L’agent de risque automobile ne doit pas être chargé pour scorer un dossier
   santé.
8. **Évaluer avant toute interface réelle.** Mesurer performance, calibration,
   robustesse, équité, taux d’abstention, qualité de l’explication, sécurité et
   comportement face aux données manquantes. Faire valider les seuils par les
   responsables métier et conformité.
9. **Piloter en mode silencieux puis HITL.** Comparer les recommandations aux
   décisions des experts sans influencer d’abord le traitement ; analyser
   écarts et erreurs ; conduire un pilote limité avec supervision humaine,
   procédure de recours et suivi des incidents.
10. **Autoriser le déploiement seulement après les portes de contrôle.**
    DPIA/évaluations nécessaires, revue juridique et conformité, tests de
    sécurité, validation modèle et données, procédures de contrôle humain,
    monitoring de dérive, plan de retrait et responsabilité clairement
    attribuée.

Ordre d’extension recommandé : **automobile stabilisée → benchmark B terminé
→ annotation métier → prototype d’une deuxième branche en données fictives
ou autorisées → pilote contrôlé**. La santé n’est pas un simple changement de
nom de produit : c’est un nouveau périmètre à cadrer et à valider.

### Autres branches envisageables

| Branche | Variables / documents spécifiques à cadrer | Risques particuliers à traiter |
|---|---|---|
| Habitation | caractéristiques du logement, zone, garanties, matériaux, historique autorisé, diagnostic | données géographiques/proxys, périls locaux, exclusions contractuelles |
| Santé | garanties demandées, éléments strictement nécessaires et légalement autorisés, règles de produit | données sensibles, minimisation, accès, discrimination et contrôle juridique renforcé |
| Vie | âge et critères permis selon juridiction, montant et durée de garantie, pièces réglementaires | limites d’usage des données personnelles et validation actuarielle/juridique |
| PME / risques professionnels | secteur, activité, biens, mesures de prévention, historique de sinistres | hétérogénéité métier, qualité des évaluations et validation par spécialistes |

Ces exemples sont des pistes de cadrage ; ils ne signifient pas que les données
ou connecteurs correspondants sont déjà disponibles dans le dépôt.

## 7. Proposition de déroulé oral (10 à 12 minutes)

1. **Contexte et objectif (1 min)** : rendre l’analyse de dossier plus
   structurée, traçable et supervisée.
2. **Données et limites (2 min)** : présenter freMTPL2, OpenML, cible sinistre
   et absence de décisions d’assureur.
3. **Modèle de risque et tarification (2 min)** : classifier la fréquence,
   puis expliquer fréquence × sévérité et les seuils économiques de recherche.
4. **Architecture multi-agents (2 min)** : montrer le flux LangGraph, le
   rapport consolidé, l’incertitude et l’escalade humaine.
5. **Prototype et gouvernance (1 min)** : démonstration Streamlit/API,
   références locales, audit, opt-in LLM et limites des intégrations.
6. **Résultats et prudence d’interprétation (1 min)** : métriques hold-out,
   robustesse/équité consignées et statut des expériences LLM.
7. **Levier B et prochaines étapes (1 à 2 min)** : distinguer code prêt de
   résultats non encore produits, puis présenter le plan de validation et
   l’extension vers une deuxième branche.

### Formulation de conclusion

> À mi-parcours, nous avons construit un prototype automobile intégrant une
> orchestration multi-agents, un modèle entraîné sur des sinistres observés,
> une estimation actuarielle de prime pure, une revue humaine et une trace
> d’audit. Les résultats sont reproductibles sur un benchmark public, mais ils
> ne remplacent ni les décisions réelles d’un assureur, ni une validation
> réglementaire et métier. Notre prochaine étape est de terminer le benchmark
> Prudential sans fuite de données, d’obtenir des annotations métier, puis de
> réutiliser l’architecture pour une deuxième branche avec ses propres données
> autorisées, ses règles et ses critères de validation.
