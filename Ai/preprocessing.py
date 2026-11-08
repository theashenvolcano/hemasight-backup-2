
"""
HemaSight questionnaire preprocessing.

Validates and cleans questionnaire answers.
This module does not calculate a medical score or diagnose anemia.
"""

import math
from collections.abc import Mapping
from typing import cast


# Accepted categorical answers
VALID_CHOICES = {
    "sex_assigned_at_birth": {
        "male",
        "female",
    },

    "natural_skin_tone": {
        "fair_pale",
        "light_medium",
        "olive_medium",
        "brown_tan",
        "dark_brown_deep",
        "prefer_not_to_answer",
    },

    "previous_bp_diagnosis": {
        "yes_high",
        "yes_low",
        "no",
        "not_sure",
    },

    "recent_bp_reading": {
        "yes",
        "no",
    },

    "bp_medications": {
        "yes_low_bp",
        "yes_high_bp",
        "no",
    },

    "bp_symptoms": {
        "yes",
        "no",
    },

    "family_bp_history": {
        "yes",
        "no",
        "not_sure",
    },

    "physical_activity": {
        "daily",
        "few_times_week",
        "rarely",
        "never",
    },

    "sodium_intake": {
        "low",
        "moderate",
        "high",
        "unsure",
    },

    "smoking_status": {
        "current",
        "former",
        "never",
    },

    "fasting_status": {
        "yes",
        "no",
    },

    "cold_hands_feet": {
        "yes",
        "no",
    },
}


# Numeric fields
NUMERIC_FIELDS = {
    "age",
    "bp_systolic_value",
    "bp_diastolic_value",
}


# Optional numeric fields
OPTIONAL_NUMERIC_FIELDS = {
    "bp_systolic_value",
    "bp_diastolic_value",
}


# Required questionnaire fields
REQUIRED_FIELDS = {
    "age",
    "sex_assigned_at_birth",
    "natural_skin_tone",
    "previous_bp_diagnosis",
    "recent_bp_reading",
    "bp_medications",
    "bp_symptoms",
    "family_bp_history",
    "physical_activity",
    "sodium_intake",
    "smoking_status",
    "fasting_status",
    "cold_hands_feet",
}


def clean_number(value: object, field_name: str) -> int | float | None:
    """Convert and validate numeric answers."""

    if value is None or value == "":
        if field_name in OPTIONAL_NUMERIC_FIELDS:
            return None

        raise ValueError(f"{field_name} is required.")

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(
            f"{field_name} must be a valid number."
        )

    try:
        number = float(value)

    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be a valid number."
        ) from None

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if field_name == "age":

        if not number.is_integer():
            raise ValueError(
                "Age must be a whole number."
            )

        if not 1 <= number <= 120:
            raise ValueError(
                "Age must be between 1 and 120."
            )

        return int(number)

    if field_name == "bp_systolic_value":

        if not number.is_integer():
            raise ValueError(
                "Systolic blood pressure must be a whole number."
            )

        if not 0 <= number <= 400:
            raise ValueError(
                "Systolic blood pressure must be between 0 and 400."
            )

    if field_name == "bp_diastolic_value":

        if not number.is_integer():
            raise ValueError(
                "Diastolic blood pressure must be a whole number."
            )

        if not 0 <= number <= 300:
            raise ValueError(
                "Diastolic blood pressure must be between 0 and 300."
            )

    return int(number) if field_name in OPTIONAL_NUMERIC_FIELDS else number


def get_age_group(age: int) -> str:
    """Categorize age into the questionnaire's age groups."""

    if age < 18:
        return "under_18"

    elif age <= 24:
        return "18_24"

    elif age <= 34:
        return "25_34"

    elif age <= 44:
        return "35_44"

    elif age <= 54:
        return "45_54"

    elif age <= 64:
        return "55_64"

    else:
        return "65_or_older"


def preprocess_answers(answers: object) -> dict[str, object]:
    """
    Validate questionnaire answers and return cleaned answers.

    Does not calculate a medical score or diagnosis.
    """

    if not isinstance(answers, Mapping):
        raise ValueError(
            "Questionnaire answers must be a dictionary."
        )

    cleaned: dict[str, object] = dict(cast(Mapping[str, object], answers))

    # Check required fields
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in cleaned
        or cleaned[field] is None
        or cleaned[field] == ""
    ]

    if missing_fields:
        raise ValueError(
            "Missing required answers: "
            + ", ".join(sorted(missing_fields))
        )

    # Validate categorical answers
    for field_name, valid_values in VALID_CHOICES.items():

        value = cleaned.get(field_name)

        if value is None:
            raise ValueError(
                f"{field_name} is required."
            )

        normalized_value = str(value).strip().lower()

        if normalized_value not in valid_values:
            allowed = ", ".join(sorted(valid_values))

            raise ValueError(
                f"Invalid answer for {field_name}: {value}. "
                f"Accepted values: {allowed}"
            )

        cleaned[field_name] = normalized_value

    # Validate numeric answers
    for field_name in NUMERIC_FIELDS:

        if (
            field_name in OPTIONAL_NUMERIC_FIELDS
            and field_name not in cleaned
        ):
            cleaned[field_name] = None

        else:
            cleaned[field_name] = clean_number(
                cleaned.get(field_name),
                field_name
            )

    # Clear BP readings if no recent measurement was taken
    if cleaned["recent_bp_reading"] == "no":
        cleaned["bp_systolic_value"] = None
        cleaned["bp_diastolic_value"] = None

    # Add calculated age group
    age = cleaned["age"]
    if not isinstance(age, int) or isinstance(age, bool):
        raise ValueError("Age must be a whole number.")

    cleaned["age_group"] = get_age_group(age)

    return cleaned
