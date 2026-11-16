"""
Intake Agent: Validates and prepares application data.

The application carries the real variables of the public freMTPL2 motor dataset
(see ``utils.actuarial_features``): driver age, French bonus-malus coefficient,
vehicle age, vehicle power, fuel type, commune density and geographic risk class.
Validation is fail-closed: nothing is imputed.
"""

import math
from numbers import Real

from products.registry import DEFAULT_PRODUCT_LINE, validate_product_line
from utils.actuarial_features import (
    AREA_CODES,
    FIELD_BOUNDS,
    INTEGER_FIELDS,
    VEHICLE_GAS_CODES,
)


class IntakeAgent:
    """Validates and prepares application data"""

    def process(self, application):
        """Validate an application without silently imputing invalid values."""
        result = {
            'agent': 'Intake',
            'status': 'needs_review',
            'warnings': [],
            'errors': [],
            'missing_fields': [],
            'prepared_data': {}
        }

        if not isinstance(application, dict):
            result['errors'].append('Application must be a mapping')
            return result

        raw_product_line = application.get('product_line', DEFAULT_PRODUCT_LINE)
        try:
            product_line = validate_product_line(
                raw_product_line if raw_product_line is not None else DEFAULT_PRODUCT_LINE,
                require_active=True,
            )
            result['prepared_data']['product_line'] = product_line
        except ValueError as error:
            result['errors'].append(str(error))

        for field, (minimum, maximum) in FIELD_BOUNDS.items():
            value = application.get(field)
            if value is None:
                result['missing_fields'].append(field)
                result['errors'].append(f'Missing required field: {field}')
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                result['errors'].append(f'{field} must be a finite number')
                continue
            if field in INTEGER_FIELDS and not float(value).is_integer():
                result['errors'].append(f'{field} must be a whole number')
                continue
            if value < minimum or value > maximum:
                result['errors'].append(f'{field} is outside the permitted range')
                continue
            result['prepared_data'][field] = value

        vehicle_gas = application.get('vehicle_gas')
        gas_lookup = {code.lower(): code for code in VEHICLE_GAS_CODES}
        if vehicle_gas is None:
            result['missing_fields'].append('vehicle_gas')
            result['errors'].append('Missing required field: vehicle_gas')
        elif not isinstance(vehicle_gas, str) or vehicle_gas.strip().lower() not in gas_lookup:
            result['errors'].append(
                f'vehicle_gas must be one of {VEHICLE_GAS_CODES}'
            )
        else:
            result['prepared_data']['vehicle_gas'] = gas_lookup[vehicle_gas.strip().lower()]

        area = application.get('area')
        if area is None:
            result['missing_fields'].append('area')
            result['errors'].append('Missing required field: area')
        elif not isinstance(area, str) or area.strip().upper() not in AREA_CODES:
            result['errors'].append(f'area must be one of {AREA_CODES}')
        else:
            result['prepared_data']['area'] = area.strip().upper()

        if not result['errors']:
            result['status'] = 'validated'

        return result
