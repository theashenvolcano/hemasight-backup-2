"""HemaSight one-time temporary patient database.

Each patient gets an individual SQLite database file in the operating
system's temporary directory. The first successful get_patient_memory()
call consumes the patient ID, reads the questionnaire, and deletes that
patient's database file. Unused records expire after the configured TTL.

Python standard library only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


# =========================================================
# CONFIGURATION
# =========================================================

SESSION_TTL_SECONDS = 15 * 60
TEMP_DATABASE_DIR = Path(tempfile.gettempdir()) / "hemasight_patient_sessions"


# =========================================================
# QUESTIONNAIRE SCHEMA
# =========================================================

REQUIRED_FIELDS = {
    "age_years",
    "sex_assigned_at_birth",
    "natural_skin_tone",
    "hypertension_diagnosis",
    "bp_measured_past_30_days",
    "systolic_bp_recent",
    "diastolic_bp_recent",
    "bp_medications",
    "frequent_headaches_lightheadedness",
    "family_history_cvd_hypertension",
    "moderate_physical_activity_frequency",
    "daily_salt_consumption",
    "smoking_status",
    "fasting_state_8_hours",
    "cold_hands_feet",
}

CATEGORICAL_CHOICES = {
    "sex_assigned_at_birth": {"male", "female"},
    "natural_skin_tone": {
        "fair_pale", "light_medium", "olive_medium", "brown_tan",
        "dark_brown_deep", "prefer_not_to_answer",
    },
    "hypertension_diagnosis": {"yes_high", "yes_low", "no", "not_sure"},
    "bp_measured_past_30_days": {"yes", "no"},
    "bp_medications": {"yes_low_bp", "yes_high_bp", "no"},
    "frequent_headaches_lightheadedness": {"yes", "no"},
    "family_history_cvd_hypertension": {"yes", "no", "not_sure"},
    "moderate_physical_activity_frequency": {
        "daily", "few_times_week", "rarely", "never",
    },
    "daily_salt_consumption": {"low", "moderate", "high", "unsure"},
    "smoking_status": {"current", "former", "never"},
    "fasting_state_8_hours": {"yes", "no"},
    "cold_hands_feet": {"yes", "no"},
}

OPTIONAL_NUMERIC_FIELDS = {
    "systolic_bp_recent",
    "diastolic_bp_recent",
}

PatientData = dict[str, int | str | None]


class SavePatientResult(TypedDict):
    success: bool
    patient_id: str
    expires_in_seconds: int
    message: str


class PatientResult(TypedDict):
    patient_id: str
    data: PatientData
    created_at: str
    expires_in_seconds: int
    prompt_context: str


# =========================================================
# VALIDATION
# =========================================================

def normalize_answer(value: object) -> object:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value.strip().lower()
    return value


def validate_patient_data(data: Mapping[str, object]) -> PatientData:
    if not isinstance(data, Mapping):
        raise ValueError("Questionnaire data must be a dictionary.")

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            "Missing questionnaire fields: " + ", ".join(sorted(missing))
        )

    unexpected = set(data.keys()) - REQUIRED_FIELDS
    if unexpected:
        raise ValueError(
            "Unexpected questionnaire fields: " + ", ".join(sorted(unexpected))
        )

    cleaned: PatientData = {}

    age = data["age_years"]
    if isinstance(age, bool):
        raise ValueError("Age must be a whole number.")
    try:
        age_number = float(age)
    except (TypeError, ValueError):
        raise ValueError("Age must be a valid whole number.") from None
    if not age_number.is_integer():
        raise ValueError("Age must be a whole number.")
    age = int(age_number)
    if not 1 <= age <= 120:
        raise ValueError("Age must be between 1 and 120.")
    cleaned["age_years"] = age

    for field, choices in CATEGORICAL_CHOICES.items():
        value = normalize_answer(data[field])
        if not isinstance(value, str) or value not in choices:
            raise ValueError(
                f"Invalid answer for {field}. Accepted values: {sorted(choices)}"
            )
        cleaned[field] = value

    for field in OPTIONAL_NUMERIC_FIELDS:
        value = data[field]
        if value is None or value == "":
            cleaned[field] = None
            continue
        if isinstance(value, bool):
            raise ValueError(f"{field} must be numeric.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be numeric.") from None
        if not number.is_integer():
            raise ValueError(f"{field} must be a whole number.")
        number = int(number)
        if field == "systolic_bp_recent" and not 0 <= number <= 400:
            raise ValueError("Invalid systolic blood pressure value.")
        if field == "diastolic_bp_recent" and not 0 <= number <= 300:
            raise ValueError("Invalid diastolic blood pressure value.")
        cleaned[field] = number

    if cleaned["bp_measured_past_30_days"] == "no":
        cleaned["systolic_bp_recent"] = None
        cleaned["diastolic_bp_recent"] = None

    return cleaned


# =========================================================
# TEMPORARY PER-PATIENT DATABASE
# =========================================================

class PatientDatabase:
    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Expiration time must be greater than zero.")
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    @staticmethod
    def create_patient_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _normalize_patient_id(patient_id: str) -> str:
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("A valid patient ID is required.")
        try:
            # Parsing as UUID prevents path traversal or arbitrary file access.
            return str(uuid.UUID(patient_id.strip()))
        except (ValueError, AttributeError, TypeError):
            raise ValueError("Invalid patient ID format.") from None

    def _database_path(self, patient_id: str) -> Path:
        safe_id = self._normalize_patient_id(patient_id)
        return TEMP_DATABASE_DIR / f"{safe_id}.sqlite3"

    @staticmethod
    def _ensure_temp_directory() -> None:
        TEMP_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(TEMP_DATABASE_DIR, 0o700)
        except OSError:
            # Permission modes may not be supported on every platform.
            pass

    @contextmanager
    def _connect(self, path: Path):
        connection = sqlite3.connect(str(path), timeout=10)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _remove_empty_temp_directory() -> None:
        try:
            TEMP_DATABASE_DIR.rmdir()
        except OSError:
            # The directory is not empty, or another operation is using it.
            pass

    def _create_or_update_record(
        self,
        patient_id: str,
        questionnaire_data: Mapping[str, object],
    ) -> SavePatientResult:
        safe_id = self._normalize_patient_id(patient_id)
        cleaned = validate_patient_data(questionnaire_data)
        path = TEMP_DATABASE_DIR / f"{safe_id}.sqlite3"

        with self._lock:
            self._ensure_temp_directory()
            created_at = datetime.now(timezone.utc).isoformat()
            expires_at = time.time() + self.ttl_seconds
            serialized_data = json.dumps(cleaned, ensure_ascii=False)

            with self._connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patient_session (
                        patient_id TEXT PRIMARY KEY,
                        questionnaire_data TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO patient_session (
                        patient_id, questionnaire_data, created_at, expires_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        questionnaire_data = excluded.questionnaire_data,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at
                    """,
                    (safe_id, serialized_data, created_at, expires_at),
                )

        return {
            "success": True,
            "patient_id": safe_id,
            "expires_in_seconds": self.ttl_seconds,
            "message": "Temporary patient database created and answers saved.",
        }

    def create_patient_session(
        self,
        questionnaire_data: Mapping[str, object],
    ) -> SavePatientResult:
        """Generate a new ID and save the questionnaire in its own DB file."""
        patient_id = self.create_patient_id()
        return self._create_or_update_record(patient_id, questionnaire_data)

    def save_patient_memory(
        self,
        patient_id: str,
        data_dict: Mapping[str, object],
    ) -> SavePatientResult:
        """Save/update answers under an ID previously made by create_session_id()."""
        return self._create_or_update_record(patient_id, data_dict)

    @staticmethod
    def format_memory_context(patient_data: Mapping[str, object]) -> str:
        lines = [
            "HEMASIGHT PATIENT QUESTIONNAIRE",
            "",
            "The following information was provided by the patient.",
            "These are self-reported questionnaire responses, not verified findings.",
            "",
        ]
        for field, value in patient_data.items():
            lines.append(f"{field.replace('_', ' ').title()}: {value}")
        lines.extend([
            "",
            "IMPORTANT:",
            "This questionnaire does not establish an anemia diagnosis.",
            "Do not treat these responses as confirmed clinical findings.",
            "Any predictive model must be validated before clinical use.",
        ])
        return "\n".join(lines)

    def get_patient_memory(self, patient_id: str) -> PatientResult | None:
        """Consume the ID: read the record once, then delete its database file.

        A second call with the same ID returns None because its file is gone.
        """
        safe_id = self._normalize_patient_id(patient_id)
        source_path = TEMP_DATABASE_DIR / f"{safe_id}.sqlite3"
        claim_path = TEMP_DATABASE_DIR / f"{safe_id}.consuming-{uuid.uuid4().hex}.tmp"

        with self._lock:
            try:
                # Atomic rename: only one concurrent caller can claim this ID.
                source_path.rename(claim_path)
            except FileNotFoundError:
                return None
            except OSError:
                return None

        try:
            with self._connect(claim_path) as connection:
                row = connection.execute(
                    """
                    SELECT patient_id, questionnaire_data, created_at, expires_at
                    FROM patient_session
                    WHERE patient_id = ?
                    """,
                    (safe_id,),
                ).fetchone()

            if row is None:
                return None

            remaining = int(row[3] - time.time())
            if remaining <= 0:
                return None

            patient_data = json.loads(row[1])
            return {
                "patient_id": safe_id,
                "data": patient_data,
                "created_at": row[2],
                "expires_in_seconds": remaining,
                "prompt_context": self.format_memory_context(patient_data),
            }
        finally:
            # This is the one-time deletion: remove the per-patient DB file.
            try:
                claim_path.unlink(missing_ok=True)
            finally:
                self._remove_empty_temp_directory()

    def delete_patient_memory(self, patient_id: str) -> bool:
        safe_id = self._normalize_patient_id(patient_id)
        path = TEMP_DATABASE_DIR / f"{safe_id}.sqlite3"
        with self._lock:
            existed = path.exists()
            try:
                path.unlink(missing_ok=True)
            finally:
                self._remove_empty_temp_directory()
        return existed

    def cleanup_expired_sessions(self) -> int:
        """Delete unused patient DB files whose 15-minute TTL has elapsed."""
        if not TEMP_DATABASE_DIR.exists():
            return 0

        deleted = 0
        for path in TEMP_DATABASE_DIR.glob("*.sqlite3"):
            try:
                with self._connect(path) as connection:
                    row = connection.execute(
                        "SELECT expires_at FROM patient_session LIMIT 1"
                    ).fetchone()
                if row is None or float(row[0]) <= time.time():
                    path.unlink(missing_ok=True)
                    deleted += 1
            except (OSError, sqlite3.Error, TypeError, ValueError):
                # Remove an unreadable/corrupt temporary DB rather than retain it.
                try:
                    path.unlink(missing_ok=True)
                    deleted += 1
                except OSError:
                    pass

        self._remove_empty_temp_directory()
        return deleted

    def clear_all_sessions(self) -> int:
        if not TEMP_DATABASE_DIR.exists():
            return 0
        deleted = 0
        for path in TEMP_DATABASE_DIR.glob("*.sqlite3"):
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass
        self._remove_empty_temp_directory()
        return deleted

    def count_patients(self) -> int:
        self.cleanup_expired_sessions()
        if not TEMP_DATABASE_DIR.exists():
            return 0
        return sum(1 for _ in TEMP_DATABASE_DIR.glob("*.sqlite3"))


# =========================================================
# SHARED INSTANCE / FUNCTIONS FOR OTHER PROJECT FILES
# =========================================================

database = PatientDatabase()


def create_patient_session(questionnaire_data):
    """Preferred: create an ID and save all pre-questionnaire answers at once."""
    database.cleanup_expired_sessions()
    return database.create_patient_session(questionnaire_data)


def create_session_id():
    """Compatibility option: create an ID; no DB file exists until it is saved."""
    return database.create_patient_id()


def save_patient_memory(patient_id, data_dict):
    database.cleanup_expired_sessions()
    return database.save_patient_memory(patient_id, data_dict)


def get_patient_memory(patient_id):
    """One-time read: returns the answers and deletes that patient's DB file."""
    return database.get_patient_memory(patient_id)


def delete_patient_memory(patient_id):
    return database.delete_patient_memory(patient_id)


def cleanup_expired_sessions():
    return database.cleanup_expired_sessions()


def clear_all_sessions():
    return database.clear_all_sessions()


def count_patients():
    return database.count_patients()


# =========================================================
# DEMONSTRATION (SYNTHETIC DATA ONLY)
# =========================================================

if __name__ == "__main__":
    demo_answers = {
        "age_years": 25,
        "sex_assigned_at_birth": "female",
        "natural_skin_tone": "brown_tan",
        "hypertension_diagnosis": "no",
        "bp_measured_past_30_days": "yes",
        "systolic_bp_recent": 120,
        "diastolic_bp_recent": 80,
        "bp_medications": "no",
        "frequent_headaches_lightheadedness": "yes",
        "family_history_cvd_hypertension": "not_sure",
        "moderate_physical_activity_frequency": "few_times_week",
        "daily_salt_consumption": "moderate",
        "smoking_status": "never",
        "fasting_state_8_hours": "no",
        "cold_hands_feet": "yes",
    }

    print("1. CREATE TEMPORARY PATIENT DATABASE")
    saved = create_patient_session(demo_answers)
    patient_id = saved["patient_id"]
    print(saved)
    print("Active temporary databases:", count_patients())

    print("\n2. USE PATIENT ID (THIS CONSUMES IT)")
    patient = get_patient_memory(patient_id)
    print("Retrieved:", patient is not None)
    if patient:
        print("Age:", patient["data"]["age_years"])
        print("AI context available:", bool(patient["prompt_context"]))

    print("\n3. TRY THE SAME ID AGAIN")
    second_attempt = get_patient_memory(patient_id)
    print("Second retrieval:", second_attempt)  # None: database was deleted
    print("Active temporary databases:", count_patients())
