# État du projet — pilote souscription (hors production)

**Mise à jour :** 10 novembre 2026  
**État :** checklist 30 j bien avancée (HITL, audit immuable, auth + RBAC, API, DPIA brouillon) ; toujours **pas** un système de production.

## Checklist 30 jours — suivi

| # | Action | Statut |
|---|---|---|
| 1 | Positionner le produit : recommandation + HITL obligatoire | **Fait** (UI + API ; mode `human_in_the_loop` figé) |
| 2 | Grille UW v0 / règles produit versionnées | **Partiel** — pack `products/auto/rules.py` v0.1.0-pilot |
| 3 | ≥ 200 dossiers annotés métier | **À faire** (partenaire / expert) |
| 4 | Workbench HITL (file, décision, motif, override) | **Fait** (Streamlit + API) |
| 5 | Audit durable exportable | **Fait** (SQLite + export JSON UI/API, chaîne de hachage append-only vérifiée) |
| 6 | Auth + rôles (uw / manager / compliance / admin) | **Fait** (RBAC par capacités, Streamlit + Bearer API, logout) |
| 7 | LLM opt-in + avertissement PII | **Fait** |
| 8 | DPIA / registre traitements brouillon | **Fait** — `docs/DPIA_DRAFT.md` (non validé DPO) |
| 9 | Matrice intégrations stub → réel | **Partiel** — champ `integrations` dans le pack auto (claims/external passés en `local_reference_dataset`) |
| — | Skeleton `product_line` (auto actif, home réservé) | **Fait** |
| — | API FastAPI pilote | **Fait** — `api/app.py` |
| — | Intake multi-pièces + OCR + checklist produit | **Fait** — `utils/document_bundle.py` (blocage si pièce obligatoire manquante) |
| — | Connecteur source réelle (antécédents + risque externe) | **Fait** — `connectors/reference_data.py` + `data/reference/*.json` (versionnés, checksum, fail-closed) ; agents `claims`/`external` réécrits dessus |
| — | Bilan de souscription consolidé (dépôt de pièces → rapport unique) | **Fait** — `orchestrator/underwriting_report.py` agrège les 11 agents (recommandation, risque, incertitude, profil, antécédents, environnement, pièces, exclusions, gouvernance, sources, suite) ; exposé UI + API |
| — | Agent Incertitude (entropie prédictive) | **Fait** — `agents/uncertainty_agent.py` : qualifie la fiabilité du modèle (niveaux low/moderate/high), escalade HITL au-delà du seuil haut, fail-closed sans probabilité |
| — | Mode de collaboration exposé | **Fait** — graphes `sequential`/`parallel` sélectionnables (UI + `collaboration_mode` API), mode consigné dans la provenance du bilan |
| — | Connecteur AGIRA (prêt à l’intégration) | **Fait** — `connectors/agira.py` (modes `replay`/`live`/`disabled`, `AGIRA_MODE`), fixture fictive `data/reference/agira_fixture.json` libellée `real_data: false`, provenance + fail-closed ; branché sur l’agent Externe (statut d’assurance, résiliation, antécédents 5 ans) |
| — | Baselines comparatives B1–B4 (benchmark réel) | **Fait** — `experiments/baselines/` : B1 règles seules, B2 ML seul, B3 agent LLM unique (fail-closed, opt-in), B4 pipeline multi-agents (séquentiel/parallèle) + runner reproductible `run_baselines.py` et harnais d’évaluation commun |

Prochaines actions : labels métier (partenaire), brancher la vraie source (AGIRA / export CRM) derrière le connecteur, exécuter les schémas de collaboration **LLM** (débat/vote) sur freMTPL2 quand le quota du fournisseur le permet (robustesse et ablation sont faites), durcir sécurité (MFA/SSO), CI/Docker.

## Livrables présents

- Interface Streamlit : login, nouvelle analyse, file HITL, audit/export (navigation selon le rôle).
- API FastAPI : login, logout, products, analyze, pending, decision, audit, admin/users.
- RBAC pilote par capacités (`uw`, `manager`, `compliance`, `admin`).
- Persistance locale SQLite (`data/runtime/aurelia_cases.db`, gitignoré), audit append-only à chaîne de hachage.
- Pack produit `auto` actif ; `home` réservé pour extension ultérieure.
- Intake documentaire multi-pièces : extraction PDF/TXT + OCR images (optionnel), checklist produit versionnée, blocage si pièce obligatoire manquante.
- Connecteur de données de référence (`connectors/`) : jeux versionnés + provenance/checksum (`data/reference/`), fail-closed ; agents antécédents/externe branchés dessus.
- Bilan de souscription consolidé (`orchestrator/underwriting_report.py`) : un rapport unique agrégé sur les 11 agents, stocké dans le dossier et exportable en audit.
- Brouillon DPIA + registre traitements.
- Orchestration LangGraph (11 agents ; modes séquentiel/parallèle ; agent Incertitude), modèle XGBoost recherche, scripts d’évaluation.
- Tests unitaires store / product_line / API / RBAC / audit / documents.

## Lancement

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit/app.py
# API (autre terminal) :
python -m uvicorn api.app:app --reload --port 8000
```

Comptes démo (à changer) — mot de passe par défaut `<rôle>-demo-change-me` :
- `uw` / `uw-demo-change-me` (souscripteur : analyse, file, décision)
- `manager` / `manager-demo-change-me` (décision + override)
- `compliance` / `compliance-demo-change-me` (lecture seule : file, audit)
- `admin` / `admin-demo-change-me` (gestion des comptes)

Variables optionnelles : `AURELIA_UW_PASSWORD`, `AURELIA_MANAGER_PASSWORD`, `AURELIA_COMPLIANCE_PASSWORD`, `AURELIA_ADMIN_PASSWORD`, `AURELIA_DB_PATH`.

## Modèle et données réelles (freMTPL2)

Le modèle livré (`models/fremtpl2_underwriting_model.json`) est entraîné et évalué sur le jeu public réel **freMTPL2** (responsabilité civile auto française, licence **CC0**, OpenML 41214/41215, dépôt CASdatasets — Dutang & Charpentier 2018) : **678 013 polices**, cible = survenance d’au moins un sinistre.

- Hold-out : **135 603 polices** ; taux de base (test) **5,02 %**.
- ROC-AUC **0,6460**, PR-AUC **0,1094**, exactitude @0,5 **0,9497**.
- Incertitude : ECE top-label **0,0011**, Brier binaire **0,0466**.
- Équité (parité de taux de sélection) : âge (<35/≥35) **0,9953** ; bonus-malus (>100/≤100) **0,5529**.

freMTPL2 **ne contient aucune décision de souscription** (acceptation/refus/surprime) : la cible est un fait observé, et la transformation du score en recommandation (surprime p ≥ 0,05, revue p ≥ 0,15) est une convention de recherche. Les jeux de référence `data/reference/` (zones, antécédents) sont dérivés de freMTPL2 via `experiments/real_data/build_reference_datasets.py`.

## Points académiques encore ouverts

1. Labels de souscription annotés par experts (freMTPL2 n’en fournit pas).
2. Baselines comparatives B1–B4 **reconstruites** sur le benchmark réel (`experiments/baselines/`). Robustesse et ablation sont **refaites** sur freMTPL2 (`experiments/robustness/`, `experiments/ablation/`) ; voir `README.md` § « Expériences académiques sur freMTPL2 ». Pour la collaboration, les schémas **déterministes** (`pipeline`, `orchestrator`) sont exécutés ; les schémas **LLM** (`single_agent`, `debate`, `weighted_vote`) restent **à exécuter** : le palier gratuit du fournisseur (Groq, `openai/gpt-oss-20b`) est plafonné à 8 000 tokens/minute et 200 000 tokens/jour, ce qui rend le run complet impossible en une journée et impose un cadencement en tokens. Le résultat est consigné en `partial` (fail-closed) tant que le quota n’est pas rechargé (les scripts historiques et l’ancien jeu synthétique de 5 000 lignes ont été retirés du dépôt car ils reposaient sur des variables fictives).
3. Rapport final au-delà de `report/cadrage.md`.
4. Validation juridique équité / conformité.
