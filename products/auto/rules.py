"""Automobile underwriting rules pack v0 (pilot).

This is an operational skeleton for document checklists and decision vocabulary.
It is not a certified insurer pricing or acceptance policy.
"""

AUTO_PRODUCT_PACK = {
    "product_line": "auto",
    "label_fr": "Automobile",
    "status": "active",
    "version": "0.1.0-pilot",
    "required_documents": [
        {
            "code": "releve_information",
            "label_fr": "Relevé d'information / antécédents",
            "mandatory": True,
        },
        {
            "code": "carte_grise",
            "label_fr": "Carte grise (certificat d'immatriculation)",
            "mandatory": True,
        },
        {
            "code": "permis_conduire",
            "label_fr": "Permis de conduire",
            "mandatory": True,
        },
        {
            "code": "devis_ou_police",
            "label_fr": "Devis, conditions particulières ou police",
            "mandatory": False,
        },
    ],
    "decision_types": [
        "Accept_Standard",
        "Accept_with_Surcharge",
        "Accept_with_Exclusion",
        "Reject",
        "Escalate_to_Human",
        "Request_More_Info",
    ],
    "hitl_policy": {
        "all_recommendations_enter_queue": True,
        "reject_requires_human": True,
        "llm_never_decides": True,
    },
    "scoring": {
        "engine": "xgboost_research_model",
        "note": "Research classifier trained on pseudo-labels; not a production tariff.",
    },
    "integrations": {
        "claims_history": "local_reference_dataset (data/reference/claims_history.json) — remplaçable par export CRM / AGIRA",
        "external_risk": "local_reference_dataset (data/reference/geographic_risk.json) — remplaçable par API géo / sinistralité",
        "agira_or_equivalent": "planned (accès assureur requis)",
        "pas_binding": "planned",
    },
}
