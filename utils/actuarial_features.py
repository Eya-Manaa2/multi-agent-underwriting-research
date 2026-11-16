"""Real French motor underwriting feature contract (freMTPL2, CC0).

Single source of truth for turning an application into the model feature vector.
Shared by training (experiments/real_data) and inference (agents/risk_agent) so the
encoding can never drift between the two.

The variables are the real ones carried by the public freMTPL2 dataset
(Christophe Dutang and Arthur Charpentier, CASdatasets; OpenML, CC0):

    driver_age   <- DrivAge    age of the driver
    bonus_malus  <- BonusMalus French bonus-malus coefficient (CRM)
    vehicle_age  <- VehAge     age of the vehicle
    vehicle_power<- VehPower   fiscal power of the vehicle
    vehicle_gas  <- VehGas     Regular | Diesel
    density      <- Density    population density of the commune
    area         <- Area       geographic risk class, A (rural) to F (urban)
"""

from __future__ import annotations

AREA_CODES = ('A', 'B', 'C', 'D', 'E', 'F')
VEHICLE_GAS_CODES = ('Regular', 'Diesel')

FEATURE_NAMES = (
    'driver_age',
    'bonus_malus',
    'vehicle_age',
    'vehicle_power',
    'vehicle_gas',
    'density',
    'area',
)

# Recommendation vocabulary of the pilot, unchanged downstream.
CLASS_LABELS = ('Accept_Standard', 'Accept_with_Surcharge', 'Reject')

# Business thresholds that map the real predicted claim probability onto the pilot
# recommendation vocabulary. They sit around the freMTPL2 base rate: about 5% of
# real policies file at least one claim.
SURCHARGE_THRESHOLD = 0.05
REVIEW_THRESHOLD = 0.15

POSITIVE_CLASS = 'claim_occurred'

# Real input bounds (observed freMTPL2 ranges, widened for robustness).
FIELD_BOUNDS = {
    'driver_age': (18, 100),
    'bonus_malus': (50.0, 350.0),
    'vehicle_age': (0, 40),
    'vehicle_power': (4, 20),
    'density': (0.0, 100000.0),
}
INTEGER_FIELDS = ('driver_age', 'vehicle_age', 'vehicle_power')


def encode_area(value) -> float:
    """Encode the geographic risk class A-F; fail closed on unknown values."""
    text = str(value).strip().upper()
    if text not in AREA_CODES:
        raise ValueError(f'area must be one of {AREA_CODES}; received {value!r}')
    return float(AREA_CODES.index(text))


def encode_vehicle_gas(value) -> float:
    """Encode the fuel type; fail closed on unknown values."""
    text = str(value).strip().lower()
    lookup = {code.lower(): index for index, code in enumerate(VEHICLE_GAS_CODES)}
    if text not in lookup:
        raise ValueError(
            f'vehicle_gas must be one of {VEHICLE_GAS_CODES}; received {value!r}'
        )
    return float(lookup[text])


def target_claim_occurred(claim_count) -> int:
    """Binary real target: did the policy file at least one claim?"""
    return 1 if claim_count >= 1 else 0


def decision_from_probability(probability: float) -> str:
    """Map the predicted claim probability to the pilot recommendation vocabulary."""
    if probability >= REVIEW_THRESHOLD:
        return CLASS_LABELS[2]
    if probability >= SURCHARGE_THRESHOLD:
        return CLASS_LABELS[1]
    return CLASS_LABELS[0]


def feature_vector(application: dict) -> list[float]:
    """Build the ordered numeric feature vector from a validated application."""
    return [
        float(application['driver_age']),
        float(application['bonus_malus']),
        float(application['vehicle_age']),
        float(application['vehicle_power']),
        encode_vehicle_gas(application['vehicle_gas']),
        float(application['density']),
        encode_area(application['area']),
    ]
