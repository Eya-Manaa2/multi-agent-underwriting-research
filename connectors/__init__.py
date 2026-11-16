"""Connecteurs de données du pilote.

Ces connecteurs lisent des jeux de référence réels, versionnés et inspectables
(provenance + empreinte SHA-256). Ils remplacent la simulation codée en dur et
exposent la même interface qu'une source réelle (AGIRA, export CRM, système
sinistres) : brancher la vraie source ne touche pas les agents.

Principe : fail-closed. Si un jeu est absent, illisible ou invalide, le connecteur
renvoie ``available=False`` et l'agent ne doit pas fabriquer de signal.
"""

from connectors.reference_data import ReferenceDataset, load_reference_dataset

__all__ = ["ReferenceDataset", "load_reference_dataset"]
