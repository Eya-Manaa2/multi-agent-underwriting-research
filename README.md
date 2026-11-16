# Multi-Agent Underwriting Research System - Academic Prototype

🎓 **Projet de Fin d'Études - Prototype Académique de Recherche en Souscription Multi-Agents**

Prototype local de recherche sur l’aide à la souscription automobile par agents IA. Le dépôt fournit une démonstration Streamlit, un classifieur XGBoost, une orchestration LangGraph et des scripts d’évaluation. **Ce logiciel n’est ni un produit d’assurance, ni un système de décision automatisée prêt pour la production.**

## Lancer la démonstration

Prérequis : Python 3.10 ou supérieur. Depuis la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit/app.py
```

Connexion pilote requise (comptes démo à changer) : `uw` / `uw-demo-change-me` (souscripteur), `manager` / `manager-demo-change-me`, `compliance` / `compliance-demo-change-me` (lecture seule) ou `admin` / `admin-demo-change-me`. Les accès sont gouvernés par un RBAC pilote (capacités `analyze`, `view_queue`, `decide`, `override`, `view_audit`, `manage_users`). L’interface propose une nouvelle analyse, une **file HITL** (décision humaine + motif) et un **export d’audit JSON** dont la chaîne de hachage append-only est vérifiée. Les dossiers sont stockés localement en SQLite sous `data/runtime/` (gitignoré). La ligne active est `auto` ; d’autres lignes (`home`, …) sont réservées pour une extension ultérieure.

API pilote (même store SQLite) :

```powershell
python -m uvicorn api.app:app --reload --port 8000
```

Endpoints principaux : `POST /v1/auth/login`, `POST /v1/auth/logout`, `POST /v1/cases/analyze`, `GET /v1/cases/pending`, `POST /v1/cases/{id}/decision`, `GET /v1/cases/{id}/audit`, `GET /v1/admin/users` (admin). Documentation interactive : `http://127.0.0.1:8000/docs`. Brouillon DPIA : [docs/DPIA_DRAFT.md](docs/DPIA_DRAFT.md).

Par défaut, le parcours reste local et n’a besoin d’aucune clé API fournisseur. La revue LLM documentaire est facultative et désactivée tant que l’utilisateur ne coche pas explicitement son activation. Toute recommandation entre en file HITL ; le LLM ne décide jamais.

Ne saisir aucune donnée personnelle réelle. Les pièces ne sont pas sauvegardées par l’application : seules les métadonnées d’intake (nom de fichier, statut d’extraction, taille) et la checklist produit sont conservées pour l’audit ; le texte extrait est utilisé en mémoire puis retiré du résultat. Si la revue LLM Groq est activée, des extraits pertinents du document sont transmis à un fournisseur externe : utiliser uniquement des documents fictifs.

## Parcours du prototype

1. **Validation** : vérifie la présence, le type et les bornes des entrées obligatoires ; aucune valeur manquante n’est imputée silencieusement.
2. **Profil et risque** : combine des règles de profil avec le modèle XGBoost fourni.
3. **Documents, externe et sinistres** : l’intake documentaire est réel (extraction PDF/TXT + OCR images, checklist produit) ; les agents **externe** (risque de zone) et **antécédents sinistres** lisent un jeu de référence **versionné local** via le connecteur `connectors/` (données inspectables, provenance + checksum, fail-closed). Aucun accès réel à AGIRA, à un registre ou à une API assureur — cet accès exige un partenariat.
4. **Décision** : agrège le résultat du modèle et certains signaux. En mode supervisé, toute recommandation de refus mène à une revue humaine.
5. **Gouvernance** : produit des signaux de revue d’équité, un statut de conformité explicitement non certifié, une explication et une trace exportable.
6. **Pièces jointes (multi-documents)** : plusieurs pièces peuvent être déposées et typées (par ex. relevé d’information, carte grise, permis). L’intake est un pipeline réel : extraction texte pour PDF/TXT (10 Mo maximum, PDF limité à 50 pages par pièce) et **OCR** pour les images (png/jpg/tiff/bmp/webp) lorsque les dépendances optionnelles sont installées. La **checklist de pièces est dérivée du pack produit versionné** (`products/auto/rules.py`), pas codée en dur : une pièce obligatoire manquante **bloque la recommandation** (fail-closed), et une pièce lue mais non exploitable est signalée comme affectant la décision. Une pièce lue sans texte est marquée « joint, aucun texte détecté » plutôt que d’être ignorée silencieusement. Le contrôle local repère quelques termes indicatifs ; le mode LLM facultatif effectue une recherche lexicale locale dans les pièces puis analyse uniquement les extraits retrouvés, avec citations vérifiées dans le texte source. Un document non automobile ou dont le périmètre ne peut être vérifié bloque la recommandation et n’est pas envoyé au LLM. Ni le repérage lexical ni le LLM ne prennent la décision de souscription. Des pièces fictives sont disponibles dans [examples/test_documents](examples/test_documents).

L’interface de démonstration destinée à l’utilisateur présente le dépôt, une synthèse lisible et les étapes de suivi. À l’issue de l’analyse, le dépôt de pièces produit un **bilan de souscription consolidé** (`orchestrator/underwriting_report.py`) : un rapport unique agrégeant les onze agents (recommandation et revue humaine requise, profil, risque, incertitude du modèle, antécédents, environnement de zone, checklist des pièces, exclusions et conditions repérées, gouvernance, sources de référence et suite proposée). Ce bilan est renvoyé par l’API (`report`), affiché en tête de la synthèse Streamlit et conservé dans le dossier, donc exportable en audit. Les métriques expérimentales, les détails d’équité/incertitude et les traces techniques restent dans les scripts et la documentation de recherche, pas dans le parcours utilisateur.

Les probabilités affichées sont les scores du classifieur, pas une garantie de justesse calibrée. L’indice de risque est heuristique, pas un tarif.

## Ce que le prototype exécute réellement

L’orchestrateur LangGraph exécute des classes Python spécialisées. Par défaut, l’agent de risque utilise le modèle XGBoost fourni ; les agents de profil et d’explication reposent sur des règles ; l’agent **incertitude** convertit la probabilité du modèle en entropie prédictive et recommande une revue humaine quand l’estimation est trop ambiguë ; les agents **sinistres** et **facteurs externes** lisent de vraies données de référence versionnées via `connectors/` (`data/reference/`), avec provenance et empreinte, et échouent en mode fermé (aucun signal inventé) si la source est absente. Le choix du mode de collaboration (`sequential` ou `parallel`) est exposé dans l’interface, l’API (`collaboration_mode`) et enregistré dans la provenance du bilan. Lorsque l’utilisateur active la revue documentaire, un agent LLM Groq est appelé après une recherche lexicale locale dans la pièce jointe : il extrait des constats et citations, sans émettre ni modifier de décision de souscription. Ce RAG est limité au document fourni ; il n’interroge aucun corpus contractuel externe, registre ou source juridique.

Pour activer cette revue, installer la dépendance incluse dans `requirements.txt`, définir localement `GROQ_API_KEY` (et éventuellement `GROQ_MODEL`, qui vaut par défaut `openai/gpt-oss-20b`), redémarrer Streamlit puis cocher l’option correspondante. Ne jamais déposer la clé dans le dépôt. La revue n’est pas exécutée sans opt-in et une clé configurée. Les agents LLM spécialisés pour le risque, les sinistres et les sources externes ne sont pas encore intégrés.

En particulier, le contenu d’une police reconnue comme automobile et les constats du LLM ne sont pas utilisés pour calculer le score de risque. Les règles de périmètre servent uniquement à empêcher une recommandation quand une pièce semble hors sujet ou ne peut pas être vérifiée. Le modèle a été entraîné sur une cible réelle observée (la survenance d’au moins un sinistre dans freMTPL2) et des variables automobiles réelles ; la transformation du score en recommandation de souscription reste une convention de recherche, pas une politique validée métier.

## Évaluation reproductible

```powershell
python -m unittest discover -s tests -p "test_*.py"
python experiments/data/fetch_fremtpl2.py
python experiments/baselines/evaluate_holdout.py
python experiments/uncertainty/evaluate_uncertainty.py
python experiments/fairness/evaluate_fairness.py
```

Le modèle livré (`models/fremtpl2_underwriting_model.json`) est évalué sur la partition **hold-out** du jeu public réel **freMTPL2** (CC0), en reproduisant exactement la partition d’entraînement (mêmes taille, stratification et graine) :

- Dataset : **678 013 polices réelles** ; partition hold-out : **135 603 polices**.
- Cible réelle observée : la police a-t-elle déclaré **au moins un sinistre** ? Taux de base (test) : **5,02 %**.

| Mesure sur le test tenu à l’écart | Résultat |
|---|---:|
| ROC-AUC | 0,6460 |
| PR-AUC (average precision) | 0,1094 |
| Exactitude (seuil 0,5) | 0,9497 |

Le modèle est binaire et calibré ; les recommandations sont dérivées du score par des seuils métier (surprime si p ≥ 0,05, revue humaine si p ≥ 0,15). Sur la partition de test, la répartition est : **90 952** `Accept_Standard`, **40 889** `Accept_with_Surcharge`, **3 762** `Reject`.

### Comparaison des baselines (B1–B4)

```powershell
python -m experiments.baselines.run_baselines
```

Ce runner exécute les quatre baselines de l’énoncé (`report/cadrage.md` § 6.1) sur la **même** partition hold-out reproductible et écrit le rapport complet sous `experiments/baselines/results/`. B1 (règles) et B2 (ML seul) sont évaluées sur les 135 603 polices tenues à l’écart ; B3 (agent LLM unique) et B4 (pipeline multi-agents) tournent sur un échantillon stratifié borné, car elles sont bien plus lourdes par dossier.

| Baseline | ROC-AUC | PR-AUC | Taux de signalement | Précision / Rappel (signal) |
|---|---:|---:|---:|---:|
| B1 — règles seules | 0,5089 | 0,0553 | 5,6 % | 0,076 / 0,084 |
| B2 — modèle ML seul | 0,6460 | 0,1094 | 32,9 % | 0,080 / 0,522 |
| B3 — agent LLM unique | 0,7057 † | 0,1243 † | 14,0 % † | 0,143 / 0,333 † |
| B4 — pipeline multi-agents | 0,6821 * | 0,1807 * | 23,0 % * | 0,087 / 0,400 * |

\* B4 est mesurée sur un échantillon stratifié de 100 polices ; † B3 sur un échantillon stratifié de 50 polices × 3 passes (`gemini-3.1-flash-lite`, 151 appels, stabilité de décision 1,00) ; B1/B2 sur l’intégralité du hold-out. Les échantillons de B3 et B4 diffèrent (50 vs 100 polices), aussi leur comparaison directe n’est qu’indicative. Les modes `sequential` et `parallel` de B4 produisent des décisions **identiques** (accord 1,00) : les agents étant déterministes, le mode de collaboration ne change pas la décision sur ces données. B3 est **fail-closed** : sans fournisseur LLM configuré, elle renvoie `unavailable` et ne produit aucun résultat inventé.

Exécuter B3 consomme du quota fournisseur, d’où des garde-fous explicites :

- **Preflight** : un appel minimal valide la clé et le nom de modèle avant la campagne ; en cas d’échec, rien d’autre n’est appelé.
- **Budget** : au-delà de 300 appels planifiés, la baseline refuse de tourner (`refused`) tant que `--yes-b3` n’est pas fourni ; la taille d’échantillon B3 par défaut est réduite (50 polices) et distincte de celle de B4.
- **Cadence** : les appels sont espacés pour respecter la limite du fournisseur (`--b3-rpm`, 15 appels/min par défaut) et un délai d’expiration de 30 s est appliqué.
- La clé n’est **jamais** journalisée. Gemini (`GEMINI_API_KEY` + `LLM_PROVIDER=gemini`) nécessite le paquet optionnel `google-genai` ; le modèle se surcharge via `GEMINI_MODEL` (la valeur par défaut du projet peut ne pas exister sur un compte donné — le preflight le signale alors).

```powershell
$env:LLM_PROVIDER = "gemini"; $env:GEMINI_API_KEY = "<clé>"; $env:GEMINI_MODEL = "gemini-3.1-flash-lite"
python -m experiments.baselines.run_baselines --b3-sample 50 --b3-runs 3 --skip-b4
```

freMTPL2 ne contenant aucune décision d’acceptation/refus, le « taux de signalement » (décision autre que `Accept_Standard`) et sa précision/rappel sont des mesures **proxy** contre le fait observé « la police a déclaré un sinistre » — pas une politique de souscription.

| Incertitude (test) | Résultat |
|---|---:|
| ECE top-label | 0,0011 |
| Brier (probabilité binaire) | 0,0466 |
| Exactitude sélective @0,8 (couverture) | 0,9523 (98,80 %) |
| Taux d’escalade @0,9 | 6,39 % |

Équité — **parité descriptive du taux de sélection** (le jeu ne contient aucune décision réelle d’acceptation/refus, seule cette parité de sélection est donc rapportée) :

| Groupement (proxy) | Taux groupe 0 | Taux groupe 1 | Ratio min/max |
|---|---:|---:|---:|
| Âge (<35 / ≥35) | 0,9688 | 0,9734 | 0,9953 |
| Bonus-malus (>100 / ≤100) | 0,5403 | 0,9773 | 0,5529 |

**Interprétation indispensable :** freMTPL2 ne contient **pas** de décisions de souscription d’assureur (acceptation, refus, surprime). La cible entraînée et évaluée est la survenance d’un sinistre, un **fait observé**, pas une décision d’assureur ; les seuils qui transforment le score en recommandation sont une convention de recherche. Ces chiffres mesurent la discrimination d’un modèle actuariel de fréquence sur un hold-out réel, pas sa rentabilité, sa conformité ni son équité légale. L’équité n’est qu’une parité de taux de sélection sur des proxys (âge, bonus-malus) : faute de ground-truth accept/refus, l’égalité des chances ne peut pas être mesurée honnêtement. Une analyse juridique de discrimination exige d’autres sources et une revue spécialisée.

## Expériences académiques sur freMTPL2

Les trois axes expérimentaux du cadrage (`report/cadrage.md` § 4.1, 6.2 et 7) tournent sur la **même** partition hold-out reproductible que les baselines, via le harnais d’évaluation commun (`experiments/baselines/evaluation.py`).

```powershell
python -m experiments.robustness.robustness_tests
python -m experiments.ablation.ablation_study
python -m experiments.collaboration.collaboration_schemes
```

### Robustesse aux dégradations

Échantillon stratifié borné de **10 000 polices** (graine 42) ; ROC-AUC de référence **0,6356**. Le modèle se dégrade **graduellement**, sans effondrement, sous bruit et champs manquants (les champs catégoriels ne sont pas bruités ; les valeurs numériques restent dans les bornes observées).

| Dégradation | ROC-AUC | Δ vs référence |
|---|---:|---:|
| Bruit gaussien 1 % | 0,6102 | −0,0254 |
| Bruit gaussien 5 % | 0,6063 | −0,0293 |
| Bruit gaussien 10 % | 0,6134 | −0,0222 |
| Bruit gaussien 20 % | 0,5972 | −0,0384 |
| Champs manquants 10 % | 0,5868 | −0,0487 |
| Champs manquants 30 % | 0,5518 | −0,0838 |

Champ retiré (leave-one-out, un champ à la fois) :

| Champ retiré | ROC-AUC | Δ |
|---|---:|---:|
| `vehicle_power` | 0,5463 | −0,0893 |
| `driver_age` | 0,5527 | −0,0829 |
| `vehicle_age` | 0,5724 | −0,0631 |
| `bonus_malus` | 0,5728 | −0,0628 |
| `density` | 0,6130 | −0,0226 |

### Ablation des agents (leave-one-out)

**100 polices** (graine 42), chaque configuration relancée dans les **deux** modes de décision. Les agents structurels (`intake`, `risk`, `decision`) ne sont pas ablatables.

| Mode de décision | Décisions de référence (11 agents) | Ablation | Δ taux de décision changée |
|---|---|---|---|
| `automated` | 77 `Accept_Standard`, 21 `Accept_with_Surcharge`, 2 `Reject` | les 8 agents non structurels | **0,0000** |
| `human_in_the_loop` | 77 `Accept_Standard`, 20 `Accept_with_Surcharge`, 3 `Escalate_to_Human` | `−uncertainty` | **0,0100** |
| `human_in_the_loop` | idem | les 7 autres | 0,0000 |

Lecture : en mode `automated`, l’agent décision suit **mécaniquement** le score ML (aucune ablation ne déplace la décision) ; en mode `human_in_the_loop`, retirer l’agent **incertitude** re-classe 1 % des dossiers (les escalades ne sont plus déclenchées), ce qui confirme le rôle propre de cet agent dans le parcours HITL.

freMTPL2 ne fournissant aucune décision accept/refus, toutes les métriques de décision sont des mesures **proxy** contre le fait observé « la police a déclaré un sinistre ».

### Collaboration (schémas)

**15 polices** (graine 42), quatre schémas de collaboration du cadrage (§ 6.2) comparés sur le même hold-out. Deux schémas **déterministes** s'exécutent sans fournisseur LLM ; les schémas à base d'agents LLM (`single_agent`, `debate`, `weighted_vote`) héritent des garde-fous de la baseline B3 et restent **fail-closed**.

| Schéma | Type | ROC-AUC | Décisions (proxy) | Latence |
|---|---|---:|---|---:|
| `pipeline` | déterministe (séquentiel) | 0,5714 | 13 `Accept_Standard`, 2 `Accept_with_Surcharge` | 9 ms/dossier |
| `orchestrator` | déterministe (parallèle) | 0,5714 | 13 `Accept_Standard`, 2 `Accept_with_Surcharge` | 9 ms/dossier |
| `single_agent` / `debate` / `weighted_vote` | LLM | — | non exécutés (quota fournisseur) | — |

Les schémas LLM ne sont **pas** exécutés tant que le fournisseur n'a pas de quota : le preflight (un appel) échoue et **aucun résultat n'est inventé** (`status: partial`). Sur le palier gratuit Groq, `openai/gpt-oss-20b` — un modèle de raisonnement — est plafonné à **8 000 tokens/minute** et **200 000 tokens/jour**, budget épuisé par la campagne de diagnostic ; les valeurs réelles sont donc enregistrées comme indisponibles. Le rejeu du run complet (`single_agent` → `weighted_vote` → `debate`) suppose un quota rechargé et un cadencement en **tokens** (non en appels), le TPM étant la contrainte liante.

## Données et reproductibilité

Le prototype s’appuie sur le benchmark public **freMTPL2** (responsabilité civile auto française), diffusé par OpenML sous licence **CC0** et rattaché au dépôt actuariel CASdatasets (Dutang & Charpentier, 2018) :

- [`freMTPL2freq` (OpenML 41214)](https://www.openml.org/d/41214) : variables de risque et nombre de sinistres de **678 013 polices** (fréquence).
- [`freMTPL2sev` (OpenML 41215)](https://www.openml.org/d/41215) : **26 639 montants** de sinistres (sévérité).

Récupération et contrôle de provenance (empreintes MD5 OpenML, schéma, valeurs manquantes, manifeste local) :

```powershell
python experiments/data/fetch_fremtpl2.py
```

Entraînement du modèle livré :

```powershell
python experiments/real_data/train_fremtpl2_model.py
```

Les fichiers ARFF téléchargés (`data/raw/freMTPL2*.arff`) et le manifeste ne sont pas versionnés (voir `.gitignore`). Le jeu ne contient **aucune** décision de souscription ni donnée contractuelle d’assureur : il permet des expériences actuarielles de fréquence/sévérité, pas l’entraînement d’une politique de souscription.

Les jeux de référence utilisés par les agents **antécédents** et **externe** sont eux aussi dérivés de freMTPL2 et versionnés sous `data/reference/` (`geographic_risk.json`, `claims_history.json`), avec provenance et empreinte contrôlées par le connecteur `connectors/reference_data.py`.

L’agent **externe** interroge aussi le connecteur **AGIRA** (`connectors/agira.py`) pour le statut d’assurance et les antécédents d’un véhicule à partir de son immatriculation. AGIRA ne diffuse **aucune API publique** : l’accès est réservé aux assureurs adhérents (convention signée + identifiants). Le connecteur est donc *prêt à l’intégration*, avec trois modes pilotés par `AGIRA_MODE` :

- `replay` (défaut) : répond depuis la fixture `data/reference/agira_fixture.json`, au format d’échange AGIRA, **fictive et libellée comme telle** dans sa provenance (`real_data: false`) ;
- `live` : exécute l’échange HTTPS réel vers `AGIRA_API_URL` avec `AGIRA_API_TOKEN` (endpoint, authentification et format à confirmer sous convention avant activation) ;
- `disabled` : aucune interrogation.

Sans immatriculation, sans configuration ou en cas de réponse invalide, le connecteur échoue en mode fermé (aucun statut inventé). L’immatriculation provient de la demande ou est lue dans les pièces jointes.

## Dépendances

- `requirements.txt` : uniquement le prototype et ses évaluations.
- `requirements-research.txt` : dépendances optionnelles pour certaines expériences, OCR, explications et intégrations de recherche. Elles ne sont pas nécessaires pour lancer l’application.

Le fichier modèle XGBoost est `models/fremtpl2_underwriting_model.json` (métadonnées d’entraînement : `models/fremtpl2_underwriting_metadata.json`). Le contrat de features (ordre et encodage attendus, labels de classes, seuils métier) est défini une seule fois dans `utils/actuarial_features.py`, partagé par l’entraînement et l’inférence. `pypdf` est inclus dans `requirements.txt` pour lire les PDF textuels. L’**OCR des pièces images** repose sur `pytesseract` + `Pillow` (et le binaire Tesseract) déclarés dans `requirements-research.txt` ; sans ces dépendances, les images sont acceptées mais marquées « OCR indisponible » et n’alimentent pas la recommandation.

## Structure principale

```text
agents/                 Agents de validation, risque, incertitude, décision et gouvernance
connectors/             Connecteurs de données (référence versionnée, AGIRA) — fail-closed
data/raw/               Fichiers freMTPL2 téléchargés (non versionnés)
data/reference/         Jeux de référence versionnés (zones, antécédents) + fixture AGIRA (fictive)
docs/                   Documentation de conformité et de limites
experiments/            Baselines et scripts d’évaluation
models/                 Modèle XGBoost freMTPL2 livré + métadonnées
orchestrator/           Workflow LangGraph
products/               Packs produits versionnés (règles, pièces, intégrations)
report/                 Note de cadrage académique
streamlit/app.py        Interface de démonstration
tests/                  Tests unitaires
utils/                  Contrat de features actuarielles, intake documentaire
```

Le répertoire des jeux de référence est paramétrable via `AURELIA_REFERENCE_DATA_DIR` (par défaut `data/reference/`). Remplacer la source réelle (AGIRA, export CRM, système sinistres) revient à alimenter ce répertoire ou à substituer le connecteur, sans toucher aux agents. Le connecteur AGIRA est paramétrable via `AGIRA_MODE`, `AGIRA_API_URL` et `AGIRA_API_TOKEN`.

### Benchmark actuariel français

Les scripts d’expérience s’appuient sur freMTPL2, benchmark public **CC0** catalogué sur OpenML (voir « Données et reproductibilité »). La cible est un fait observé (survenance d’un sinistre) ; le jeu ne contient **aucune** décision d’acceptation d’un assureur et ne doit pas servir à entraîner une politique de souscription.

> Les scripts de recherche historiques (ablation, collaboration, explainability, statistical, robustness) et l’ancien jeu synthétique de 5 000 lignes ont été **retirés** du dépôt : ils reposaient sur des variables fictives (score de crédit, score de fraude, prime) et ne constituaient pas une preuve sur le modèle freMTPL2 réel. Les baselines **B1–B4** ont été **reconstruites** sur le benchmark réel (`experiments/baselines/`, voir « Comparaison des baselines ») ; l’ablation, la collaboration et la robustness restent à refaire sur freMTPL2.

## Limites avant tout dépôt ou usage réel

- Le prototype ne dispose d’**aucune décision réelle de souscription** (acceptation/refus/surprime). La cible est la survenance d’un sinistre : entraîner une politique de souscription exige des labels métier annotés par des experts et un protocole d’évaluation hors échantillon validé.
- Faire valider par des spécialistes les seuils (surprime, revue) qui transforment le score en recommandation.
- Examiner l’utilisation de l’âge et du bonus-malus (proxys), la discrimination directe et indirecte, les performances par sous-groupe et les voies de recours : la parité de sélection rapportée n’est pas une analyse juridique.
- Brancher de vraies sources d’assureur derrière les agents documentaires/externe/sinistres (AGIRA / export CRM / système sinistres) : les jeux `data/reference/` sont réels mais dérivés d’un benchmark public, pas d’un SI assureur. Le connecteur AGIRA est en place avec un mode `live` ; il reste à obtenir la convention AGIRA, les identifiants et à confirmer l’endpoint/le format d’échange. Aucune donnée AGIRA réelle n’est utilisée aujourd’hui (fixture de démonstration uniquement).
- Mettre en place une revue humaine réelle, une journalisation sécurisée et durable, une analyse de protection des données, une sécurité applicative et une validation réglementaire adaptées au contexte juridique.
- Faire relire le rapport de recherche, les résultats et les exigences juridiques par le jury et les responsables compétents ; le prototype ne revendique aucune certification RGPD ou AI Act.
