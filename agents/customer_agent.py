"""
Customer Agent: driver and vehicle profile built from real freMTPL2 variables.

The former credit-score band was a fictional variable. It is replaced by the real
French bonus-malus coefficient (CRM), the actual driver-rating variable, plus the
real driver-age and vehicle characteristics.
"""


class CustomerAgent:
    """Analyzes the driver and vehicle profile"""

    def process(self, application, intake_result):
        """Analyze the real driver/vehicle profile."""
        driver_age = application.get('driver_age', 40)
        bonus_malus = application.get('bonus_malus', 100)
        vehicle_age = application.get('vehicle_age', 5)
        vehicle_power = application.get('vehicle_power', 6)

        result = {
            'agent': 'Customer',
            'risk_factors': [],
            'bonus_malus_band': 'Medium',
        }

        # Bonus-malus coefficient (CRM): the lower, the better the driver.
        if bonus_malus <= 60:
            result['bonus_malus_band'] = 'Excellent'
        elif bonus_malus <= 100:
            result['bonus_malus_band'] = 'Good'
        elif bonus_malus <= 150:
            result['bonus_malus_band'] = 'Medium'
        else:
            result['bonus_malus_band'] = 'Poor'
            result['risk_factors'].append(
                f'Coefficient bonus-malus élevé ({bonus_malus:.0f})'
            )

        # Driver-age risk (real freMTPL2 DrivAge).
        if driver_age < 25:
            result['risk_factors'].append('Jeune conducteur (< 25 ans)')
        elif driver_age > 65:
            result['risk_factors'].append('Conducteur senior (> 65 ans)')

        # Vehicle characteristics (real freMTPL2 VehAge / VehPower).
        if vehicle_age > 15:
            result['risk_factors'].append(f'Véhicule ancien ({vehicle_age} ans)')
        if vehicle_power >= 12:
            result['risk_factors'].append(f'Véhicule puissant ({vehicle_power} CV)')

        return result
