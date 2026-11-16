"""
Explanation Agent: Generates explanations for decisions
Audience-specific explanations (customer, regulator, auditor)
"""

import json
from pathlib import Path

class ExplanationAgent:
    """Generates explanations for underwriting decisions"""

    def __init__(self):
        print("[OK] Explanation Agent initialized")

    def process(self, application, decision_result, all_agent_results):
        """Generate explanation for this decision"""

        decision = decision_result['final_decision']
        risk_score = decision_result.get('final_risk_score', decision_result.get('risk_score', 0))
        confidence = decision_result['confidence']
        reasons = decision_result.get('reasons', [])

        result = {
            'agent': 'Explanation',
            'explanations': {}
        }

        # Customer-facing explanation (simple, non-technical)
        result['explanations']['customer'] = self._generate_customer_explanation(
            decision, risk_score, reasons
        )
        uploaded_document = all_agent_results.get('document', {}).get(
            'uploaded_document', {}
        )
        if uploaded_document.get('provided') and uploaded_document.get(
            'scope_status'
        ) in {'unsupported', 'unverified'}:
            result['explanations']['customer'] = {
                'title': 'Vérification du document requise',
                'summary': uploaded_document.get(
                    'scope_reason',
                    'Le document doit être vérifié par un souscripteur avant toute analyse.',
                ),
                'details': [
                    'Le document fourni ne permet pas d’établir qu’il concerne '
                    'l’assurance automobile.',
                    'Aucune recommandation de souscription ne doit être tirée de '
                    'cette analyse.',
                ],
                'next_steps': [
                    'Vérifier que le dossier concerne bien une assurance automobile.',
                    'Joindre une pièce automobile lisible puis relancer l’analyse.',
                ],
            }

        # Regulator-facing explanation (detailed, includes fairness)
        result['explanations']['regulator'] = self._generate_regulator_explanation(
            decision, risk_score, confidence, all_agent_results
        )

        # Auditor-facing explanation (technical, includes audit trail)
        result['explanations']['auditor'] = self._generate_auditor_explanation(
            decision, all_agent_results
        )

        return result

    def _generate_customer_explanation(self, decision, risk_score, reasons):
        """Generate simple explanation for customer"""

        if decision == 'Accept_Standard':
            return {
                'title': 'Votre dossier reçoit une recommandation favorable',
                'summary': 'Le modèle estime que le profil correspond aux critères standards du prototype.',
                'details': [
                    'Cette recommandation est expérimentale et ne vaut pas décision contractuelle.',
                    'Le résultat doit être vérifié par une personne habilitée.'
                ],
                'next_steps': [
                    'Vérifier les informations déclarées et les conditions du contrat.',
                    'Confirmer la décision par une revue humaine.'
                ]
            }

        elif decision == 'Accept_with_Surcharge':
            return {
                'title': 'Recommandation favorable avec surprime',
                'summary': 'Le prototype suggère une surprime au regard des facteurs analysés.',
                'details': [
                    f'Indice de risque heuristique : {risk_score:.1f}/100.',
                    'Cet indice n’est pas un tarif et doit être interprété avec prudence.'
                ],
                'next_steps': [
                    'Faire vérifier les facteurs et le montant par un souscripteur.',
                    'Ne pas utiliser ce résultat seul pour fixer une prime.'
                ]
            }

        elif decision == 'Escalate_to_Human':
            return {
                'title': 'Revue humaine requise',
                'summary': 'Le prototype ne prend pas de décision finale et demande une vérification par un souscripteur.',
                'details': [
                    f'Indice de risque heuristique : {risk_score:.1f}/100.',
                    'Une recommandation de refus est toujours transmise à une revue humaine.'
                ],
                'next_steps': [
                    'Examiner les facteurs et les données du dossier.',
                    'Documenter la décision humaine et ses motifs.'
                ]
            }

        elif decision == 'Accept_with_Exclusion':
            return {
                'title': 'Revue des exclusions requise',
                'summary': 'Le prototype a signalé une exclusion potentielle à vérifier dans les documents contractuels.',
                'details': [f'Indice de risque heuristique : {risk_score:.1f}/100.'],
                'next_steps': ['Vérifier le texte contractuel applicable avant toute décision.']
            }

        else:
            return {
                'title': 'Recommandation de refus — revue humaine obligatoire',
                'summary': 'Le prototype identifie des éléments justifiant une analyse approfondie, mais ne peut pas prononcer de refus.',
                'details': [
                    f'Indice de risque heuristique : {risk_score:.1f}/100.',
                    'Le refus ne peut être confirmé que par une personne habilitée.'
                ],
                'next_steps': [
                    'Revoir le dossier et les explications fournies.',
                    'Consigner les motifs et les voies de recours applicables.'
                ]
            }

    def _generate_regulator_explanation(self, decision, risk_score, confidence, all_agent_results):
        """Generate detailed explanation for regulator"""
        fairness_result = all_agent_results.get('fairness', {})
        compliance_result = all_agent_results.get('compliance', {})

        explanation = {
            'title': f'Regulatory explanation for {decision} decision',
            'decision': decision,
            'risk_score': risk_score,
            'confidence': confidence,
            'factors_considered': [],
            'fairness_assessment': (
                'Review required'
                if fairness_result.get('review_required')
                else 'Not conclusively assessed'
            ),
            'compliance_status': compliance_result.get(
                'compliance_status', 'Not assessed'
            ),
        }

        # List factors from each agent
        for agent_name, agent_result in all_agent_results.items():
            if isinstance(agent_result, dict):
                if 'risk_factors' in agent_result:
                    explanation['factors_considered'].extend([
                        f"{agent_name}: {rf}" for rf in agent_result['risk_factors']
                    ])
                if 'exclusions' in agent_result and agent_result['exclusions']:
                    explanation['factors_considered'].append(
                        f"{agent_name}: Exclusions - {', '.join(agent_result['exclusions'])}"
                    )

        return explanation

    def _generate_auditor_explanation(self, decision, all_agent_results):
        """Generate technical explanation for auditor"""

        explanation = {
            'title': f'Audit trail for {decision} decision',
            'decision_path': [],
            'agent_outputs': {},
            'traceability': 'Complete'
        }

        # Document each agent's contribution
        for agent_name, agent_result in all_agent_results.items():
            if isinstance(agent_result, dict):
                explanation['agent_outputs'][agent_name] = {
                    'key_output': str(agent_result)[:100],  # Truncated
                    'status': 'completed'
                }
                explanation['decision_path'].append(agent_name)

        return explanation
