
import os
from secrets import token_urlsafe
from collections.abc import Mapping

from flask import Flask, jsonify, render_template, request
from itsdangerous import BadSignature, URLSafeTimedSerializer

from Ai.image_model import ImageModelError, ModelUnavailableError
from Ai.combined_model import ANEMIA_SYMPTOMS, combine_screening

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SECRET_KEY"] = os.environ.get("SCREENING_SESSION_SECRET") or token_urlsafe(32)
SCREENING_SESSION_TTL_SECONDS = 15 * 60
screening_session_signer = URLSafeTimedSerializer(
    app.config["SECRET_KEY"],
    salt="hemasight-screening-session",
)


def valid_screening_session(token: str) -> bool:
    """Accept a server-signed screening token for at most 15 minutes."""
    try:
        screening_session_signer.loads(token, max_age=SCREENING_SESSION_TTL_SECONDS)
    except BadSignature:
        return False
    return True


@app.after_request
def prevent_screening_page_cache(response):
    if request.endpoint == "questionnaire":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def normalize_screening_form(form: Mapping[str, str]) -> dict[str, object]:
    """Validate answers for the anemia-focused intake."""
    try:
        age = float(form.get("age_years", ""))
    except (TypeError, ValueError):
        raise ValueError("Enter the patient's age in years.") from None
    if not 0.5 <= age <= 120:
        raise ValueError("Age must be between 0.5 and 120 years.")

    sex = form.get("sex_assigned_at_birth", "").strip().lower()
    if sex not in {"male", "female", "prefer_not_to_answer"}:
        raise ValueError("Choose a sex assigned at birth or prefer not to answer.")

    pregnancy = form.get("pregnancy_status", "not_applicable").strip().lower()
    if age >= 15 and sex == "female":
        if pregnancy not in {"no", "yes", "unsure", "prefer_not_to_answer"}:
            raise ValueError("Answer the pregnancy question or choose prefer not to answer.")
    else:
        pregnancy = "not_applicable"

    symptoms: dict[str, str] = {}
    for key in ANEMIA_SYMPTOMS:
        answer = form.get(f"symptom_{key}", "").strip().lower()
        if answer not in {"yes", "no", "unsure"}:
            raise ValueError("Answer each symptom question.")
        symptoms[key] = answer

    return {
        "age_years": age,
        "sex_assigned_at_birth": sex,
        "pregnancy_status": pregnancy,
        "symptoms": symptoms,
    }


def submit_screening():
    """Accept the only supported form submission: answers plus an eye image."""
    if request.is_json:
        return jsonify({
            "success": False,
            "error": "This older questionnaire request is no longer supported. Reload the page and submit the answers with an eye image.",
        }), 410

    session_token = request.form.get("screening_session_token", "")
    if not valid_screening_session(session_token):
        return jsonify({
            "success": False,
            "error": "This screening session expired. Start a new screening and submit within 15 minutes.",
        }), 410

    try:
        answers = normalize_screening_form(request.form)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    uploaded = request.files.get("eye_image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"success": False, "error": "Upload an eye image to continue."}), 400

    image_bytes = uploaded.stream.read(app.config["MAX_CONTENT_LENGTH"] + 1)
    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"success": False, "error": "The image is too large. Maximum size is 12 MB."}), 413
    if not image_bytes:
        return jsonify({"success": False, "error": "The uploaded image is empty."}), 400

    try:
        result = combine_screening(answers, image_bytes)
    except ModelUnavailableError as exc:
        app.logger.error("Research image model unavailable: %s", exc)
        return jsonify({"success": False, "error": "The image screening model is unavailable."}), 503
    except ImageModelError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except (OSError, ValueError, KeyError, TypeError):
        app.logger.exception("Could not run the research image model")
        return jsonify({"success": False, "error": "The image model could not process this scan."}), 503

    return jsonify({"success": True, "result": result})


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# About Us page
@app.route("/about")
def about():
    return render_template("about.html")


# Frequently asked questions page
@app.route("/faqs")
def faqs():
    return render_template("faqs.html")


# Questionnaire page
@app.route("/questionnaire")
def questionnaire():
    return render_template("questionnaire.html")


@app.route("/screening", methods=["POST"])
def screening():
    return submit_screening()


@app.route("/screening/session", methods=["POST"])
def create_screening_session():
    """Issue a signed, expiring token without storing patient answers."""
    response = jsonify({
        "success": True,
        "token": screening_session_signer.dumps({"nonce": token_urlsafe(18)}),
        "expires_in_seconds": SCREENING_SESSION_TTL_SECONDS,
    })
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/accessibility")
def accessibility():
    return render_template("accessibility.html")


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "success": False,
        "error": "The image is too large. Maximum upload size is 12 MB.",
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        # Use a separate local port from stale PyCharm Flask runs that often
        # remain bound to 5000. Hosted environments still provide PORT.
        port=int(os.environ.get("PORT", "5001")),
        debug=False,
    )
