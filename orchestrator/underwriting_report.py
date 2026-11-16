"""Consolidated underwriting report ("bilan de souscription").

Aggregates the outputs of every agent into a single ordered, auditable report:
recommendation, risk, profile, claims history, environment, documents, exclusions,
governance and next steps. The report is descriptive only: it never decides, and the
human decision recorded in the HITL queue remains the source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DECISION_LABELS_FR = {
    'Accept_Standard': 'Recommandation favorable (standard)',
    'Accept_with_Surcharge': 'Recommandation favorable avec surprime',
    'Accept_with_Exclusion': 'Recommandation favorable avec exclusion',
    'Reject': 'Refus recommandé — revue humaine obligatoire',
    'Escalate_to_Human': 'Revue humaine requise',
    'Request_More_Info': 'Pièces ou informations complémentaires requises',
}

AGENT_ORDER = (
    'intake',
    'customer',
    'risk',
    'uncertainty',
    'document',
    'external',
    'claims',
    'decision',
    'fairness',
    'compliance',
    'explanation',
)
EXTRACTED_STATUSES = {'text_extracted', 'ocr_extracted'}


def _result(state: dict[str, Any], name: str) -> dict[str, Any]:
    value = state.get(f'{name}_result')
    return value if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def build_underwriting_report(state: dict[str, Any]) -> dict[str, Any]:
    """Build the consolidated report from a completed workflow state."""
    application = state.get('application') or {}
    risk = _result(state, 'risk')
    uncertainty = _result(state, 'uncertainty')
    document = _result(state, 'document')
    llm_document = _result(state, 'llm_document')
    external = _result(state, 'external')
    claims = _result(state, 'claims')
    decision = _result(state, 'decision')
    fairness = _result(state, 'fairness')
    compliance = _result(state, 'compliance')
    explanation = _result(state, 'explanation')

    final_decision = decision.get('final_decision') or risk.get('decision')
    escalation = (
        bool(decision.get('escalation_triggered'))
        or final_decision == 'Escalate_to_Human'
    )
    checklist = document.get('document_checklist') or {}
    uploaded = document.get('uploaded_document') or {}
    claims_analysis = claims.get('claims_analysis') or {}

    missing_documents = list(checklist.get('missing_mandatory') or [])
    unreadable_documents = [
        item['filename']
        for item in (document.get('documents') or [])
        if item.get('extraction_status') not in EXTRACTED_STATUSES
    ]

    # --- Narrative lines ---------------------------------------------------
    if claims.get('history_source_status') == 'ok':
        claims_line = (
            f"Antécédents sinistres : {claims_analysis.get('prior_claims_count')} "
            "au fichier de référence"
        )
        if claims.get('fraud_indicators'):
            claims_line += f" · {len(claims['fraud_indicators'])} signal(aux) de vigilance"
        claims_line += '.'
    else:
        claims_line = "Antécédents sinistres : source de référence indisponible."

    if external.get('source_status') == 'ok':
        environment_line = (
            f"Environnement : {len(external.get('external_factors') or [])} "
            "facteur(s) issu(s) du jeu de référence"
        )
        environment_line += (
            ' (' + ', '.join(external.get('data_sources') or []) + ').'
            if external.get('data_sources')
            else '.'
        )
    else:
        environment_line = "Environnement : source de référence indisponible."

    if checklist:
        provided = sum(
            1 for item in checklist.get('required', []) if item.get('provided')
        )
        total = len(checklist.get('required', []))
        documents_line = f"Pièces : {provided}/{total} présentes"
        if missing_documents:
            documents_line += (
                " · obligatoires manquantes : " + ', '.join(missing_documents)
            )
        documents_line += '.'
    else:
        documents_line = "Pièces : aucune pièce rattachée à ce dossier."

    governance_line = (
        f"Gouvernance : {compliance.get('compliance_status', 'non évalué')}"
        + (" · revue d’équité requise" if fairness.get('review_required') else "")
        + " · décision humaine requise (HITL)."
    )

    summary_lines = [
        f"Recommandation : {DECISION_LABELS_FR.get(final_decision, final_decision)}.",
    ]
    if risk:
        model_line = (
            f"Modèle : {risk.get('decision')} · indice de risque "
            f"{float(risk.get('risk_score', 0)):.1f}/100 · confiance "
            f"{float(risk.get('confidence', 0)) * 100:.0f}%"
        )
        if isinstance(uncertainty.get('predictive_uncertainty'), (int, float)):
            model_line += (
                f" · incertitude {float(uncertainty['predictive_uncertainty']):.2f} "
                f"({uncertainty.get('level')})"
            )
        summary_lines.append(model_line + '.')
    summary_lines.extend([claims_line, environment_line, documents_line, governance_line])

    next_steps = []
    if missing_documents:
        next_steps.append(
            'Compléter les pièces obligatoires manquantes : '
            + ', '.join(missing_documents)
            + '.'
        )
    if unreadable_documents:
        next_steps.append(
            'Revoir l’original des pièces non exploitables : '
            + ', '.join(unreadable_documents)
            + '.'
        )
    next_steps.append('Trancher le dossier dans la file HITL et motiver la décision humaine.')
    if escalation or final_decision == 'Reject':
        next_steps.append(
            'Si un refus est confirmé, informer le client de la voie de recours '
            'ou de réclamation applicable.'
        )
    next_steps.extend(
        explanation.get('explanations', {}).get('customer', {}).get('next_steps', [])
    )
    next_steps = _dedupe(next_steps)

    sources = [
        descriptor
        for descriptor in (external.get('source'), claims.get('history_source'))
        if isinstance(descriptor, dict)
    ]
    agira = external.get('agira')
    if isinstance(agira, dict) and agira.get('available'):
        sources.append(agira)

    llm_review = None
    if llm_document.get('status') in {'analyzed', 'no_relevant_passages'}:
        llm_review = {
            'status': llm_document.get('status'),
            'document_type': llm_document.get('document_type'),
            'summary': llm_document.get('summary'),
            'findings': llm_document.get('findings') or [],
            'uncertainties': llm_document.get('uncertainties') or [],
            'model': llm_document.get('model'),
        }

    return {
        'product_line': application.get('product_line'),
        'application_id': application.get('application_id'),
        'generated_at': _utc_now(),
        'headline': DECISION_LABELS_FR.get(final_decision, final_decision),
        'recommendation': {
            'code': final_decision,
            'label_fr': DECISION_LABELS_FR.get(final_decision, final_decision),
            'escalation_required': escalation,
            'human_review_required': True,
            'confidence': risk.get('confidence'),
            'risk_score': decision.get('final_risk_score', risk.get('risk_score')),
        },
        'summary_lines': summary_lines,
        'risk_assessment': {
            'final_risk_score': decision.get('final_risk_score', risk.get('risk_score')),
            'model_risk_score': risk.get('risk_score'),
            'confidence': risk.get('confidence'),
            'claim_probability': risk.get('claim_probability'),
            'probabilities': risk.get('ml_probabilities') or {},
            'model_decision': risk.get('decision'),
            'predictive_uncertainty': uncertainty.get('predictive_uncertainty'),
            'uncertainty_level': uncertainty.get('level'),
            'reasons': decision.get('reasons') or [],
        },
        'uncertainty': {
            'status': uncertainty.get('status'),
            'claim_probability': uncertainty.get('claim_probability'),
            'predictive_uncertainty': uncertainty.get('predictive_uncertainty'),
            'level': uncertainty.get('level'),
            'escalation_recommended': uncertainty.get('escalation_recommended'),
            'thresholds': uncertainty.get('thresholds') or {},
            'notes': uncertainty.get('notes') or [],
        },
        'profile': {
            'bonus_malus_band': _result(state, 'customer').get('bonus_malus_band'),
            'risk_factors': _result(state, 'customer').get('risk_factors') or [],
        },
        'claims_history': {
            'source': claims.get('history_source'),
            'status': claims.get('history_source_status'),
            'prior_claims_count': claims_analysis.get('prior_claims_count'),
            'prior_claims_total_amount': claims_analysis.get('prior_claims_total_amount'),
            'severity_assessment': claims.get('severity_assessment'),
            'similarity_score': claims.get('similarity_score'),
            'fraud_indicators': claims.get('fraud_indicators') or [],
        },
        'environment': {
            'source': external.get('source'),
            'status': external.get('source_status'),
            'factors': external.get('external_factors') or [],
            'adjustments': external.get('risk_adjustments') or [],
            'agira': external.get('agira'),
        },
        'documents': {
            'checklist': checklist or None,
            'items': [
                {
                    'filename': item.get('filename'),
                    'status': item.get('extraction_status'),
                    'characters': item.get('characters'),
                    'declared_type_label': item.get('declared_type_label'),
                }
                for item in (document.get('documents') or [])
            ],
            'review_terms': uploaded.get('possible_terms_for_human_review') or [],
            'complete': checklist.get('complete') if checklist else None,
            'missing_mandatory': missing_documents,
            'unreadable': unreadable_documents,
        },
        'document_findings': {
            'exclusions': document.get('exclusions') or [],
            'conditions': document.get('conditions') or [],
            'warnings': document.get('warnings') or [],
            'llm_review': llm_review,
        },
        'governance': {
            'human_review_required': True,
            'escalation_triggered': escalation,
            'compliance_status': compliance.get('compliance_status'),
            'compliance_checks': compliance.get('compliance_checks') or [],
            'fairness_review_required': bool(fairness.get('review_required')),
            'fairness_notes': fairness.get('fairness_checks') or [],
            'uncertainty_escalation_recommended': bool(
                uncertainty.get('escalation_recommended')
            ),
            'decision_mode': decision.get('mode'),
        },
        'next_steps': next_steps,
        'explanation': explanation.get('explanations') or {},
        'provenance': {
            'agents': [name for name in AGENT_ORDER if _result(state, name)],
            'collaboration_mode': state.get('collaboration_mode'),
            'sources': sources,
        },
    }
