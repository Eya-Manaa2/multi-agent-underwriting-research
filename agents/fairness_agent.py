"""
Fairness Agent: Checks for bias and ensures fair decisions
Implements fairness metrics and bias mitigation
"""


class FairnessAgent:
    """Monitors fairness and detects bias in decisions"""

    def __init__(self):
        self.fairness_thresholds = {
            'demographic_parity_diff': 0.1,  # Max 10% difference
            'equal_opportunity_diff': 0.1,
            'disparate_impact_ratio': 0.8  # Min 80% ratio
        }
        print("[OK] Fairness Agent initialized")

    def process(self, application, decision_result):
        """Check fairness of this decision"""

        result = {
            'agent': 'Fairness',
            'fairness_checks': [],
            'bias_detected': False,
            'review_required': False,
            'bias_factors': [],
            'recommendations': []
        }

        # One application cannot establish statistical discrimination.
        driver_age = application.get('driver_age', 40)
        decision = decision_result['final_decision']

        if driver_age < 25 or driver_age > 65:
            result['fairness_checks'].append(
                'Age is a model input; assess outcomes across groups before drawing conclusions'
            )
            result['review_required'] = True
            result['bias_factors'].append('Age-related impact cannot be assessed on a single application')

        # Geographic class (real freMTPL2 Area) can act as a socioeconomic proxy.
        area = str(application.get('area', '')).strip().upper()
        if area in {'E', 'F'} and decision in {'Reject', 'Escalate_to_Human'}:
            result['fairness_checks'].append(
                'Geographic risk class is a potential socioeconomic proxy and requires review'
            )
            result['review_required'] = True
            result['bias_factors'].append('Potential geographic/socioeconomic impact requires group-level analysis')

        if decision in {'Reject', 'Escalate_to_Human'}:
            result['review_required'] = True
            result['recommendations'].append(
                'Review the decision with group-level fairness evidence; a single case is insufficient'
            )
        if not result['fairness_checks']:
            result['fairness_checks'].append('No individual fairness conclusion can be drawn from this case')

        return result

    def calculate_batch_fairness(self, results):
        """Calculate fairness metrics across a batch of decisions"""

        decisions_by_age = {}
        decisions_by_area = {}

        for r in results:
            driver_age = r.get('driver_age', 40)
            area = str(r.get('area', 'A')).strip().upper()
            decision = r.get('final_decision')

            # Age groups
            if driver_age < 25:
                age_group = 'young'
            elif driver_age > 65:
                age_group = 'senior'
            else:
                age_group = 'adult'

            if age_group not in decisions_by_age:
                decisions_by_age[age_group] = []
            decisions_by_age[age_group].append(decision)

            # Geographic groups (dense vs. non-dense)
            area_group = 'dense' if area in {'E', 'F'} else 'non_dense'
            if area_group not in decisions_by_area:
                decisions_by_area[area_group] = []
            decisions_by_area[area_group].append(decision)

        # Calculate rejection rates
        fairness_metrics = {}

        for group, decisions in decisions_by_age.items():
            reject_rate = decisions.count('Reject') / len(decisions)
            fairness_metrics[f'age_{group}_reject_rate'] = reject_rate

        for group, decisions in decisions_by_area.items():
            reject_rate = decisions.count('Reject') / len(decisions)
            fairness_metrics[f'area_{group}_reject_rate'] = reject_rate

        # Demographic parity difference
        if 'young' in decisions_by_age and 'adult' in decisions_by_age:
            young_reject = decisions_by_age['young'].count('Reject') / len(decisions_by_age['young'])
            adult_reject = decisions_by_age['adult'].count('Reject') / len(decisions_by_age['adult'])
            fairness_metrics['demographic_parity_diff_age'] = abs(young_reject - adult_reject)

        return fairness_metrics
