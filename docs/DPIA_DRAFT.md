# Brouillon — Analyse d’impact relative à la protection des données (DPIA)

> **Statut :** brouillon opérationnel pour le pilote AURELIA.  
> **Ce document n’est pas une DPIA finalisée ni un avis du DPO.**  
> À faire valider par un DPO / juriste avant tout traitement de données personnelles réelles.

**Date :** 8 novembre 2026  
**Version :** 0.1-draft  
**Responsable rédaction :** équipe projet (à compléter)  
**Juridiction cible :** France / UE

---

## 1. Description du traitement

| Élément | Description pilote |
|---|---|
| Finalité | Aide à la souscription (recommandation + revue humaine HITL) |
| Base légale envisagée | Intérêt légitime **ou** exécution de mesures précontractuelles — **à confirmer juridiquement** |
| Personnes concernées | Prospects / assurés (en pilote : **données fictives uniquement**) |
| Décision | Recommandation IA ; **décision humaine obligatoire** avant suite opérationnelle |
| Conservation | SQLite local `data/runtime/` ; durée cible pilote ≤ 90 jours — **à formaliser** |
| Destinataires | Rôles pilotes authentifiés (`uw`, `manager`, `compliance`, `admin`) ; Groq **uniquement** si revue LLM opt-in |

## 2. Données traitées (catégories)

- Identifiants de dossier (`application_id`, `case_id`)
- Données de risque saisies : âge, score de crédit, valeur véhicule, prime, sinistre, zone, indicateur fraude
- Pièces jointes (PDF texte / TXT) et extraits éventuels
- Recommandation IA, décision humaine, motifs, journal d’audit
- Identifiants de connexion (hash de mot de passe)

**Interdit en pilote actuel :** données personnelles réelles, documents nominatifs réels, envoi LLM hors opt-in explicite.

## 3. Nécessité et proportionnalité

- Le scoring est une **aide** ; le refus recommandé entraîne une revue humaine.
- Pas d’émission de police / binding dans le pilote.
- LLM désactivé par défaut ; citations limitées aux extraits locaux.
- Multi-produit : seule la ligne `auto` est active ; pas d’extension sans nouvelle analyse.

## 4. Risques identifiés (ébauche)

| Risque | Gravité | Mesure actuelle | Mesure à renforcer |
|---|---|---|---|
| Fuite de dossier via export JSON | Moyenne | Auth locale, DB gitignorée | Chiffrement at-rest, ACL export |
| Envoi d’extraits à un LLM externe | Élevée si PII réelle | Opt-in + disclaimer | Interdiction PII / DPA / LLM on-prem |
| Décision perçue comme automatisée | Élevée | HITL file + messages UI/API | Formation UW + traces de conseil |
| Accès non autorisé | Moyenne | Login + RBAC (uw/manager/compliance/admin) + token API 12 h | SSO, MFA, rotation secrets |
| Biais / discrimination | Élevée | Flags exploratoires seulement | Étude juridique + labels métier |

## 5. Droits des personnes (à opérationnaliser)

- Information préalable claire (« aide à la décision », pas décision finale IA)
- Accès / rectification / effacement : **procédure manuelle à écrire** (pas d’API self-service encore)
- Opposition / limitation : possible via non-utilisation du pilote
- Recours / information BCT en cas de difficulté d’assurance : rappel UI à conserver

## 6. Transferts hors UE

- Groq (si activé) : transfert potentiel hors UE — **bloquer sur données réelles** tant qu’aucune analyse de transfert / DPA n’est validée.

## 7. Avis / validation (cases à cocher)

- [ ] DPO / conseil juridique
- [ ] RSSI / sécurité
- [ ] Métier souscription
- [ ] Décision de go pilote avec données anonymisées / fictives uniquement

## 8. Registre des traitements (extrait pilote)

| Traitement | Finalité | Base | Durée | Support |
|---|---|---|---|---|
| Analyse dossier UW | Recommandation + HITL | À confirmer | ≤ 90 j pilote | SQLite local + Streamlit/API |
| Authentification | Contrôle d’accès | Intérêt légitime / contrat | Compte actif | Hash PBKDF2 |
| Revue LLM optionnelle | Extraction documentaire assistée | Consentement / contrat — à confirmer | Session + traces case | Groq (opt-in) |

---

**Prochaine révision :** avant tout passage de données non fictives, et avant ouverture d’une nouvelle `product_line`.
