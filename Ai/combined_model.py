"""Join the image estimate with questionnaire context and reference cutoffs."""

from __future__ import annotations

from collections.abc import Mapping
from Ai.image_model import estimate_hemoglobin


ANEMIA_SYMPTOMS = {
    "fatigue": "Unusual tiredness or low energy",
    "weakness": "General weakness",
    "dizziness": "Dizziness or lightheadedness",
    "shortness_of_breath": "Shortness of breath during ordinary activity",
    "pale_inner_eyelid": "Paler than usual inner lower eyelid",
    "cold_hands_feet": "Unusually cold hands or feet",
}


def reference_cutoff(answers: Mapping[str, object]) -> tuple[float | None, str]:
    """WHO 2024 reference cutoff for applicable age/sex/pregnancy groups."""
    age = float(answers["age_years"])
    sex = str(answers["sex_assigned_at_birth"])
    pregnancy = str(answers["pregnancy_status"])

    if 0.5 <= age < 2:
        return 10.5, "children 6–23 months"
    if 2 <= age < 5:
        return 11.0, "children 24–59 months"
    if 5 <= age < 12:
        return 11.5, "children 5–11 years"
    if 12 <= age < 15:
        return 12.0, "children 12–14 years"
    if age >= 15 and sex == "female" and pregnancy != "no":
        return None, "pregnancy status is not represented in the image-model training data"
    if age >= 15 and sex == "prefer_not_to_answer":
        return None, "sex-specific reference group not selected"
    if age >= 15 and sex == "female":
        return 12.0, "nonpregnant females 15 years and older"
    if age >= 15 and sex == "male":
        return 13.0, "males 15 years and older"
    return None, "model age/sex group not supported"


def screening_interpretation(
    estimate: float,
    typical_error: float,
    cutoff: float | None,
) -> tuple[str, str]:
    if cutoff is None:
        return (
            "estimate_only",
            "The model can show an image-based research estimate, but it cannot assign an anemia category for this age, sex, or pregnancy group.",
        )
    if estimate + typical_error < cutoff:
        return (
            "below_reference_signal",
            "The image model estimate is below the reference cutoff by more than the model's typical held-out error. A blood test is needed to check this result.",
        )
    if estimate - typical_error >= cutoff:
        return (
            "not_flagged_by_image_model",
            "The image model did not flag an estimate below the reference cutoff. This cannot rule out anemia.",
        )
    return (
        "inconclusive",
        "The estimate is close to the reference cutoff compared with the model's typical held-out error, so the image result is inconclusive.",
    )


def combine_screening(
    answers: Mapping[str, object],
    image_bytes: bytes,
) -> dict[str, object]:
    """Produce the complete, non-diagnostic result for one submitted scan."""
    age = float(answers["age_years"])
    sex = str(answers["sex_assigned_at_birth"])

    if 19 <= age <= 65:
        model_output = estimate_hemoglobin(
            image_bytes,
            cohort="adult",
            age_years=age,
            sex=sex,
        )
        estimate = float(model_output["estimated_hb_g_dl"])
        typical_error = float(model_output["heldout_mae_g_dl"])
        cutoff, cutoff_group = reference_cutoff(answers)
        status, message = screening_interpretation(estimate, typical_error, cutoff)
        screening_percent = None
        if cutoff is not None:
            residuals = model_output["heldout_residuals_g_dl"]
            hits = sum(
                estimate + float(residual) < cutoff
                for residual in residuals
            )
            # Light Laplace smoothing avoids reporting absolute 0% or 100%
            # from a small, 32-person holdout set.
            screening_percent = round(100 * (hits + 1) / (len(residuals) + 2))
            summary = (
                f"The model gives a {screening_percent}% research screening estimate that hemoglobin is below the "
                f"{cutoff:.1f} g/dL reference cutoff. Its hemoglobin estimate is {estimate:.1f} g/dL, and its "
                f"mean absolute error on {len(residuals)} held-out scans was {typical_error:.2f} g/dL. "
                "This percentage is not a diagnosis or a clinically validated personal probability; confirm with a blood test."
            )
        else:
            summary = (
                f"The image model estimates hemoglobin at {estimate:.1f} g/dL, but cannot calculate a percentage "
                f"for this reference group ({cutoff_group}). A blood test is needed for an anemia assessment."
            )
        result: dict[str, object] = {
            "model_available": True,
            "signal": status,
            "message": message,
            "summary": summary,
            "screening_percent": screening_percent,
            "estimated_hb_g_dl": estimate,
            "reference_cutoff_g_dl": cutoff,
            "reference_group": cutoff_group,
            "typical_heldout_error_g_dl": round(typical_error, 2),
            "heldout_test_images": int(model_output["heldout_test_images"]),
            "training_images": int(model_output["training_images"]),
        }
    else:
        result = {
            "model_available": False,
            "signal": "outside_model_scope",
            "message": (
                "The available image model has been evaluated only on adults aged 19–65. "
                "It does not return an estimate for this age group."
            ),
            "summary": "No percentage is available because the image model has not been evaluated for this age group.",
            "screening_percent": None,
            "estimated_hb_g_dl": None,
            "reference_cutoff_g_dl": None,
            "reference_group": None,
            "typical_heldout_error_g_dl": None,
            "heldout_test_images": None,
            "training_images": None,
        }

    symptoms = answers["symptoms"]
    assert isinstance(symptoms, Mapping)
    reported = [
        ANEMIA_SYMPTOMS[key]
        for key, value in symptoms.items()
        if value == "yes" and key in ANEMIA_SYMPTOMS
    ]
    result["reported_symptoms"] = reported
    result["symptom_count"] = len(reported)
    result["follow_up"] = (
        "You reported one or more symptoms. Consider discussing a complete blood count (CBC) with a healthcare professional, regardless of the image result."
        if reported
        else "No listed symptoms were reported. This does not rule out anemia; a blood test is needed if there is a concern."
    )
    result["notice"] = (
        "Research screening only. This model is not a diagnosis. The uploaded image is processed in memory and is not saved. "
        "Questionnaire symptoms are shown as context and do not change the image estimate because this dataset has no symptom labels. "
        "Reference cutoffs are not adjusted for altitude or smoking."
    )
    return result

