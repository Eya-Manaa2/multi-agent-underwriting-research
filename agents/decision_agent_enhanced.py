"""
Enhanced Decision Agent: Implements all decision types and modes
Supports: Accept, Reject, Accept with Surcharge, Accept with Exclusion, Escalate to Human
Modes: Automated, Human-in-the-loop
"""

class EnhancedDecisionAgent:
    """Enhanced decision agent with all decision types and modes"""

    def __init__(self, mode="automated"):
        if mode not in {"automated", "human_in_the_loop"}:
            raise ValueError(f"Unknown decision mode: {mode}")
        self.mode = mode  # "automated" or "human_in_the_loop"
        self.escalation_thresholds = {
            'confidence': 0.7,  # Escalate if confidence below this
            'risk_score': 80,  # Escalate if risk score above this
        }
        print(f"[OK] Enhanced Decision Agent initialized (mode: {self.mode})")

    def process(self, application, all_agent_results):
        """Process with all decision types and modes"""

        risk_result = all_agent_results.get('risk', {})
        if risk_result.get('error'):
            raise RuntimeError(f"Risk evaluation failed: {risk_result['error']}")
        if risk_result.get('decision') not in {
            'Accept_Standard',
            'Accept_with_Surcharge',
            'Reject',
        }:
            raise ValueError('Risk evaluation did not return a valid underwriting recommendation')
        document_result = all_agent_results.get('document', {})
        uploaded_document = document_result.get('uploaded_document', {})
        claims_result = all_agent_results.get('claims', {})
        customer_result = all_agent_results.get('customer', {})
        uncertainty_result = all_agent_results.get('uncertainty', {})

        # Start with ML decision
        ml_decision = risk_result.get('decision', 'Accept_Standard')
        confidence = risk_result.get('confidence', 0.5)
        risk_score = risk_result.get('risk_score', 50)

        reasons = ['ML-based primary decision']
        final_decision = ml_decision
        document_scope_status = uploaded_document.get('scope_status', 'not_provided')

        if uploaded_document.get('provided') and document_scope_status != 'supported':
            final_decision = 'Escalate_to_Human'
            reasons.append(
                'Document scope is unsupported or unverified; no underwriting '
                'recommendation can be made from this attachment'
            )

        # Override based on document exclusions
        if (
            final_decision != 'Escalate_to_Human'
            and document_result.get('exclusions')
        ):
            # Check if exclusion warrants full rejection or exclusion only
            if 'Fraudulent claims' in document_result['exclusions']:
                final_decision = 'Reject'
                reasons.append("Fraud exclusion → Reject")
            else:
                final_decision = 'Accept_with_Exclusion'
                reasons.append(f"Exclusions: {', '.join(document_result['exclusions'])} → Accept with Exclusion")

        # Consider fraud indicators from claims
        if claims_result.get('fraud_indicators'):
            reasons.append(f"Claims fraud indicators: {', '.join(claims_result['fraud_indicators'])}")
            if len(claims_result['fraud_indicators']) >= 2:
                final_decision = 'Reject'
                reasons.append('Multiple fraud indicators → Reject')

        # Escalation logic for human-in-the-loop mode
        if self.mode == "human_in_the_loop":
            if final_decision == 'Reject':
                final_decision = 'Escalate_to_Human'
                reasons.append('A rejection recommendation requires human review')
            elif confidence < self.escalation_thresholds['confidence']:
                final_decision = 'Escalate_to_Human'
                reasons.append(f"Low confidence ({confidence:.2f} < {self.escalation_thresholds['confidence']}) → Escalate to Human")
            elif risk_score > self.escalation_thresholds['risk_score']:
                final_decision = 'Escalate_to_Human'
                reasons.append(f"High risk score ({risk_score:.1f} > {self.escalation_thresholds['risk_score']}) → Escalate to Human")
            elif uncertainty_result.get('escalation_recommended'):
                final_decision = 'Escalate_to_Human'
                uncertainty = uncertainty_result.get('predictive_uncertainty')
                if isinstance(uncertainty, (int, float)):
                    reasons.append(
                        f"Predictive uncertainty too high ({uncertainty:.2f}, level "
                        f"{uncertainty_result.get('level')}) → Escalate to Human"
                    )
                else:
                    reasons.append(
                        'Model predictive uncertainty could not be established → Escalate to Human'
                    )
            elif ml_decision == 'Reject' and customer_result.get('bonus_malus_band') == 'Excellent':
                final_decision = 'Escalate_to_Human'
                reasons.append('Favourable bonus-malus applicant rejected → Escalate to Human for review')

        # Calculate final risk score
        final_risk_score = risk_score

        # Adjust based on decision type
        if final_decision == 'Escalate_to_Human':
            final_risk_score = max(final_risk_score, 75)  # High risk for escalation
        elif final_decision == 'Accept_with_Exclusion':
            final_risk_score += 10
        elif final_decision == 'Reject':
            final_risk_score += 20

        final_risk_score = min(final_risk_score, 100)

        return {
            'agent': 'Decision_Enhanced',
            'final_decision': final_decision,
            'final_risk_score': final_risk_score,
            'confidence': confidence,
            'reasons': reasons,
            'ml_decision': ml_decision,
            'customer_band': customer_result.get('bonus_malus_band'),
            'exclusions_found': len(document_result.get('exclusions', [])),
            'fraud_indicators': len(claims_result.get('fraud_indicators', [])),
            'predictive_uncertainty': uncertainty_result.get('predictive_uncertainty'),
            'uncertainty_level': uncertainty_result.get('level'),
            'mode': self.mode,
            'escalation_triggered': final_decision == 'Escalate_to_Human'
        }
