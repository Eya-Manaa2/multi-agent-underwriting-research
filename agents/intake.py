"""
Agent 1: Intake/Validation Agent
Validates application completeness and normalizes data
"""

from typing import Dict

from agents.intake_agent import IntakeAgent as ApplicationIntakeAgent
from utils.actuarial_features import AREA_CODES, VEHICLE_GAS_CODES


class IntakeAgent:
    """Intake and validation agent"""

    def __init__(self):
        self.validator = ApplicationIntakeAgent()
        self.valid_areas = list(AREA_CODES)
        self.valid_vehicle_gas = list(VEHICLE_GAS_CODES)

    def validate(self, application: Dict) -> Dict:
        """
        Validate an application

        Returns:
            dict: {is_valid, missing_fields, errors, normalized_data}
        """
        prepared = self.validator.process(application)
        errors = list(prepared['errors'])
        normalized = application.copy() if isinstance(application, dict) else {}

        for field, allowed in (
            ('area', self.valid_areas),
            ('vehicle_gas', self.valid_vehicle_gas),
        ):
            if field in normalized and normalized[field] not in allowed:
                errors.append(f"Invalid {field}: {normalized[field]}")

        is_valid = prepared['status'] == 'validated' and not errors

        return {
            'is_valid': is_valid,
            'missing_fields': prepared['missing_fields'],
            'errors': errors,
            'normalized_data': normalized,
            'agent': 'intake',
            'confidence': 1.0 if is_valid else 0.0
        }


def test_intake_agent():
    """Test the intake agent"""
    print("=" * 60)
    print("Testing Intake Agent")
    print("=" * 60)

    agent = IntakeAgent()

    # Test 1: Valid application (real freMTPL2 schema)
    valid_app = {
        'driver_age': 35,
        'bonus_malus': 60.0,
        'vehicle_age': 5,
        'vehicle_power': 6,
        'vehicle_gas': 'Regular',
        'density': 1200.0,
        'area': 'C',
    }

    result = agent.validate(valid_app)
    print(f"\nTest 1 - Valid application:")
    print(f"  Is valid: {result['is_valid']}")
    print(f"  Missing fields: {result['missing_fields']}")
    print(f"  Errors: {result['errors']}")

    # Test 2: Invalid application
    invalid_app = {
        'driver_age': 150,  # Invalid
        'bonus_malus': 60.0,
        'vehicle_age': 5,
        'vehicle_power': 6,
        'vehicle_gas': 'Steam',  # Invalid
        'density': 1200.0,
        'area': 'Z',  # Invalid
    }

    result = agent.validate(invalid_app)
    print(f"\nTest 2 - Invalid application:")
    print(f"  Is valid: {result['is_valid']}")
    print(f"  Missing fields: {result['missing_fields']}")
    print(f"  Errors: {result['errors']}")

    print("\n" + "=" * 60)
    print("Intake Agent Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_intake_agent()
