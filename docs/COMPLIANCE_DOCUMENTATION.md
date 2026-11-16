# Gouvernance, conformité et audit — état réel du prototype

> **Aucune conformité réglementaire n’est certifiée.** Ce document décrit les mécanismes de démonstration présents dans le dépôt et les contrôles qui restent à réaliser. Il ne constitue pas un avis juridique ni une déclaration de conformité RGPD, AI Act ou assurance.

## Mécanismes présents

- **Validation** : champs requis, types, valeurs finies et bornes contrôlés par `IntakeAgent`. Les valeurs manquantes ou invalides bloquent le workflow ; elles ne sont pas remplacées par une valeur par défaut.
- **Supervision** : le mode humain est le défaut dans l’interface et dans l’orchestrateur. Une recommandation de refus est transformée en demande de revue humaine dans ce mode.
- **Accès et rôles** : authentification locale et RBAC pilote (`uw`, `manager`, `compliance`, `admin`) par capacités explicites. Le rôle conformité est en lecture seule ; l’administrateur gère les comptes. Il n’y a ni SSO, ni MFA, ni annuaire d’entreprise.
- **Explication** : le prototype produit une synthèse utilisateur, des facteurs et une trace des sorties des agents. Ces textes ne constituent pas une explication juridique validée.
- **Équité** : l’agent individuel signale les cas à examiner, sans déclarer qu’un cas isolé prouve ou exclut un biais. L’analyse statistique hors échantillon rapporte des taux d’approbation par groupes d’âge et de score de crédit.
- **Trace** : l’application expose les résultats des agents, journalise chaque dossier et chaque décision humaine dans un stockage SQLite local append-only à chaîne de hachage, et permet de télécharger un objet JSON avec l’identifiant d’exécution, l’horodatage et un verdict d’intégrité de la chaîne.

## Ce que le prototype ne fait pas

- Le statut de conformité est volontairement `Prototype - not certified`.
- Les agents documentaires, externes et sinistres utilisent des règles/données simulées ; il n’y a ni OCR réel, ni RAG avec sources juridiques vérifiées, ni consultation d’API ou registre.
- La trace est append-only et vérifiable par chaîne de hachage (détection d’altération), mais elle n’est ni signée cryptographiquement, ni conservée dans un stockage sécurisé externe.
- L’application implémente une authentification locale et un RBAC pilote, mais ni gestion du consentement, ni politiques de conservation/suppression, ni chiffrement applicatif, ni processus réel de réclamation ou de recours.
- Les contrôles affichés ne vérifient pas la base légale, l’information des personnes, l’analyse d’impact, les transferts de données, les obligations de documentation, ni l’applicabilité de règles réglementaires précises.
- L’âge est une variable du modèle ; le score de crédit peut servir de proxy socio-économique. Les mesures sur ces variables ne permettent pas de conclure à l’équité ou à la légalité.

## Données et export

Les dossiers saisis sont enregistrés localement dans SQLite (`data/runtime/`, gitignoré) au titre du pilote, pour la revue HITL et l’audit ; ils restent sur le poste et ne sont pas transmis à un service tiers. Le téléchargement de la trace inclut toutes les valeurs soumises ; il ne faut donc pas saisir de données personnelles réelles ni partager un export sans revue.

Les pièces jointes PDF texte et TXT sont traitées en mémoire. Seul du texte UTF-8 peut être extrait d’un TXT ; un PDF texte nécessite le paquet `pypdf`, et un PDF scanné ne peut pas être traité sans OCR. Quand le lecteur PDF n’est pas disponible, le fichier peut rester attaché au dossier, mais l’interface indique explicitement que son contenu n’a pas été lu. Les expressions repérées par le prototype sont lexicales, non contractuelles, et n’influencent pas la recommandation.

Le jeu d’évaluation contient 5 000 dossiers de recherche et des pseudo-labels générés par règles. La source initiale n’est pas distribuée dans ce dépôt. Sa provenance, sa licence et ses conditions d’usage doivent être établies avant diffusion académique ou réutilisation.

## Conditions préalables à un usage réel

1. Obtenir une revue juridique et conformité dans la juridiction et le produit visés.
2. Documenter les sources, licences, catégories de données, bases légales, finalités, durées et droits des personnes.
3. Évaluer les performances et les erreurs sur des données indépendantes représentatives, annotées par des experts métier.
4. Réaliser une analyse d’impact et des tests de discrimination directe/indirecte avec groupes appropriés, effectifs suffisants et intervalles d’incertitude.
5. Définir la responsabilité, le contrôle, la contestation, l’intervention humaine effective et la journalisation sécurisée.
6. Faire valider les explications, les seuils et le protocole de surveillance par les parties prenantes compétentes.

Le prototype est conçu pour la démonstration et la discussion de ces exigences, pas pour souscrire, tarifer, refuser ou modifier une police réelle.
