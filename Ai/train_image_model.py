"""Train compact Hb regression models from the supplied image archive.

Usage:
  python -m Ai.train_image_model "C:/path/Hemaglobin - Under-Eye Scans.zip"

The preassigned train, validation, and test folders are retained. Test images
are used only for the reported final evaluation; the deployable artifact is fit
on train+validation after hyperparameters are chosen.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from openpyxl import load_workbook
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from Ai.image_model import _region_features


def number(value: object) -> float:
    return float(str(value).strip().replace(",", "."))


def maybe_number(value: object) -> float | None:
    if value is None or str(value).strip() in {"", "_", "-"}:
        return None
    try:
        return number(value)
    except (TypeError, ValueError):
        return None


def split_name(path: str) -> str:
    normalized = path.lower()
    if "test set" in normalized:
        return "test"
    if "validation set" in normalized:
        return "validation"
    return "train"


def read_rows(archive: zipfile.ZipFile, member: str) -> list[dict[str, object]]:
    workbook = load_workbook(io.BytesIO(archive.read(member)), read_only=True, data_only=True)
    worksheet = workbook.active
    rows = worksheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(rows)]
    return [dict(zip(headers, row)) for row in rows if row and any(value is not None for value in row)]


def adult_metadata(archive: zipfile.ZipFile) -> dict[tuple[str, int], dict[str, float]]:
    base = "Hemaglobin - Under-Eye Scans/Full Eye/Meta Data/"
    result: dict[tuple[str, int], dict[str, float]] = {}
    for country, sheet in (("india", "India Meta Data.xlsx"), ("italy", "Italian Meta Data.xlsx")):
        for row in read_rows(archive, base + sheet):
            sample_id = maybe_number(row.get("Number"))
            hb = maybe_number(row.get("Hgb"))
            age = maybe_number(row.get("Age"))
            if sample_id is None or hb is None or age is None:
                continue
            sex = str(row.get("Gender", "")).strip().lower()
            if sex not in {"m", "f", "male", "female"}:
                continue
            result[(country, int(number(row["Number"])))] = {
                "hb": hb,
                "age": age,
                "sex": 1.0 if sex in {"m", "male"} else 0.0,
            }
    return result


def collect_adult(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    prefix = "Hemaglobin - Under-Eye Scans/Full Eye/AI Model Sets - Full Eye/"
    metadata = adult_metadata(archive)
    samples = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if not name.startswith(prefix) or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        if not path.stem.isdigit():
            continue
        country = "india" if "india" in name.lower() else "italy"
        label = metadata.get((country, int(path.stem)))
        if label is None or not 19 <= label["age"] <= 65:
            continue
        samples.append({
            "id": f"{country}-{path.stem}",
            "country": country,
            "name": name,
            "split": split_name(name),
            "cohort": "adult",
            **label,
        })
    return samples


def collect_child(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    base = "Hemaglobin - Under-Eye Scans/Non-Full Eye/"
    sheet = base + "Meta Data/Anemia_Data_Collection_Sheet.xlsx"
    metadata: dict[str, dict[str, float]] = {}
    for row in read_rows(archive, sheet):
        image_id = str(row.get("IMAGE_ID", "")).strip().lower()
        hb = maybe_number(row.get("HB_LEVEL"))
        age = maybe_number(row.get("Age(Months)"))
        if not image_id or hb is None or age is None:
            continue
        sex = str(row.get("GENDER", "")).strip().lower()
        if sex not in {"m", "f", "male", "female"}:
            continue
        metadata[image_id] = {
            "hb": hb,
            "age": age,
            "sex": 1.0 if sex in {"m", "male"} else 0.0,
        }

    prefix = base + "AI Model Sets - Non-Full Eye/"
    samples = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if not name.startswith(prefix) or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        key = path.stem.strip().lower()
        label = metadata.get(key)
        if label is None:
            match = re.fullmatch(r"image[_-]?0*(\d+)", key)
            if match:
                label = metadata.get(f"image_{int(match.group(1)):03d}")
        if label is None:
            continue
        samples.append({
            "id": key,
            "name": name,
            "split": split_name(name),
            "cohort": "child",
            **label,
        })
    return samples


def feature_for(archive: zipfile.ZipFile, sample: dict[str, Any]) -> list[float]:
    with Image.open(io.BytesIO(archive.read(sample["name"]))) as source:
        source = source.convert("RGB")
        source.thumbnail((128, 96), Image.Resampling.LANCZOS)
        rgb = np.asarray(source, dtype=np.float32) / 255.0
    values = _region_features(rgb, sample["cohort"])
    values.extend([float(sample["age"]), float(sample["sex"])])
    return values


def fit_and_report(
    archive: zipfile.ZipFile,
    samples: list[dict[str, Any]],
    cohort: str,
) -> dict[str, Any]:
    samples = [sample for sample in samples if sample["cohort"] == cohort]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["id"], []).append(sample)

    grouped_samples: list[dict[str, Any]] = []
    grouped_features: list[list[float]] = []
    for patient_id, patient_images in grouped.items():
        first = patient_images[0]
        if any(
            image["split"] != first["split"]
            or image["sex"] != first["sex"]
            or image["age"] != first["age"]
            or image["hb"] != first["hb"]
            for image in patient_images[1:]
        ):
            raise ValueError(f"{cohort}: patient {patient_id} crosses a data split or has conflicting metadata.")
        per_image_features = [feature_for(archive, image) for image in patient_images]
        grouped_features.append(np.mean(np.asarray(per_image_features), axis=0).tolist())
        grouped_samples.append(first)

    samples = grouped_samples
    X = np.asarray(grouped_features, dtype=np.float64)
    y = np.asarray([sample["hb"] for sample in samples], dtype=np.float64)
    splits = [sample["split"] for sample in samples]
    if not {"train", "validation", "test"}.issubset(set(splits)):
        raise ValueError(f"{cohort}: expected train, validation, and test partitions.")

    split_array = np.asarray(splits)
    train = split_array == "train"
    validation = split_array == "validation"
    test = split_array == "test"
    if min(int(train.sum()), int(validation.sum()), int(test.sum())) < 10:
        raise ValueError(f"{cohort}: one of the holdout partitions is too small.")

    # A small, predeclared Ridge grid uses validation data only.
    choices = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        scaler = StandardScaler().fit(X[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(X[train]), y[train])
        validation_prediction = model.predict(scaler.transform(X[validation]))
        choices.append((mean_absolute_error(y[validation], validation_prediction), alpha))
    validation_mae, alpha = min(choices)

    scaler = StandardScaler().fit(X[train | validation])
    model = Ridge(alpha=alpha).fit(scaler.transform(X[train | validation]), y[train | validation])
    prediction = model.predict(scaler.transform(X[test]))
    test_mae = float(mean_absolute_error(y[test], prediction))
    test_mape_percent = float(mean_absolute_percentage_error(y[test], prediction) * 100)
    test_rmse = float(mean_squared_error(y[test], prediction) ** 0.5)
    test_r2 = float(r2_score(y[test], prediction))
    heldout_residuals = (y[test] - prediction).astype(float).tolist()
    classification: dict[str, Any] = {}
    if cohort == "adult":
        test_samples = [sample for sample, keep in zip(samples, test) if keep]
        subgroup_mae: dict[str, dict[str, float | int]] = {}
        for group_name, group_values in (
            ("country", [str(sample.get("country", "unknown")) for sample in test_samples]),
            ("sex", ["male" if sample["sex"] == 1 else "female" for sample in test_samples]),
        ):
            subgroup_mae[group_name] = {}
            for group_value in sorted(set(group_values)):
                positions = np.asarray([i for i, value in enumerate(group_values) if value == group_value])
                subgroup_mae[group_name][group_value] = {
                    "n": int(len(positions)),
                    "mae_g_dl": round(float(mean_absolute_error(y[test][positions], prediction[positions])), 3),
                }
        cutoffs = np.asarray([13.0 if sample["sex"] == 1 else 12.0 for sample in test_samples])
        actual_positive = y[test] < cutoffs
        predicted_positive = prediction < cutoffs
        true_positive = int(np.logical_and(actual_positive, predicted_positive).sum())
        false_negative = int(np.logical_and(actual_positive, ~predicted_positive).sum())
        true_negative = int(np.logical_and(~actual_positive, ~predicted_positive).sum())
        false_positive = int(np.logical_and(~actual_positive, predicted_positive).sum())
        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        classification = {
            "who_cutoff_screen_accuracy": round((true_positive + true_negative) / len(test_samples), 3),
            "who_cutoff_screen_sensitivity": round(sensitivity, 3),
            "who_cutoff_screen_specificity": round(specificity, 3),
            "who_cutoff_screen_balanced_accuracy": round((sensitivity + specificity) / 2, 3),
            "who_cutoff_screen_test_n": len(test_samples),
            "test_mae_by_subgroup": subgroup_mae,
        }

    # Final artifact uses all labeled rows after the untouched test report.
    final_scaler = StandardScaler().fit(X)
    final_model = Ridge(alpha=alpha).fit(final_scaler.transform(X), y)
    return {
        "model": {
            "feature_means": final_scaler.mean_.tolist(),
            "feature_scales": final_scaler.scale_.tolist(),
            "coefficients": final_model.coef_.tolist(),
            "intercept": float(final_model.intercept_),
            "heldout_mae_g_dl": test_mae,
            "heldout_mape_percent": test_mape_percent,
            "heldout_rmse_g_dl": test_rmse,
            "heldout_r2": test_r2,
            "heldout_residuals_g_dl": heldout_residuals,
            "validation_mae_g_dl": float(validation_mae),
            "training_images": len(samples),
            "test_images": int(test.sum()),
            "ridge_alpha": alpha,
            **classification,
            "scope": "Research-only estimate; not a diagnostic result.",
        },
        "report": {
            "n": len(samples),
            "partitions": dict(Counter(splits)),
            "validation_mae_g_dl": round(float(validation_mae), 3),
            "test_mae_g_dl": round(test_mae, 3),
            "test_mape_percent": round(test_mape_percent, 3),
            "test_rmse_g_dl": round(test_rmse, 3),
            "test_r2": round(test_r2, 3),
            "ridge_alpha": alpha,
            **classification,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to Hemaglobin - Under-Eye Scans ZIP")
    parser.add_argument("--output", default="Ai/artifacts/hemoglobin_model.json")
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        all_samples = collect_adult(archive) + collect_child(archive)
        output: dict[str, Any] = {"models": {}, "training": {}}
        for cohort in ("adult", "child"):
            result = fit_and_report(archive, all_samples, cohort)
            output["training"][cohort] = result["report"]
            print(cohort, json.dumps(result["report"], sort_keys=True))
            # The pediatric color model performed worse than the mean-Hb
            # baseline on its held-out test partition, so it is not shipped.
            if cohort == "adult" and result["model"]["heldout_r2"] > 0:
                output["models"][cohort] = result["model"]

    output_path = __import__("pathlib").Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved model artifact: {output_path}")


if __name__ == "__main__":
    main()

