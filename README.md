# HemaSight

HemaSight is an educational research prototype. It accepts anemia-related symptom answers and an eye image, then returns an image-based hemoglobin estimate when the patient is within the model's evaluated adult age range. It is not a medical device and cannot diagnose or rule out anemia. A blood test is required to measure hemoglobin.

## Run the app

1. Install Python 3.14.
2. Create and activate a virtual environment.
3. Install runtime packages with `pip install -r requirements.txt`.
4. Start the app with `python app.py` and open `http://127.0.0.1:5001`.

If an older local copy is already running, stop it in its terminal with `Ctrl+C` before restarting. The app now uses port 5001 locally to avoid showing an old page still bound to port 5000.

The trained compact model is included at `Ai/artifacts/hemoglobin_model.json`; app startup does not download research data or model weights. Uploads are read and analyzed in memory and are not written to disk.

## Model scope and evaluation

The deployed estimator is a ridge regression trained on labeled full-eye images and paired hemoglobin metadata from the supplied India and Italy cohorts. It uses image color features, age, and sex assigned at birth. Evaluation used the archive's separate train, validation, and test folders. Two images belonging to one patient were averaged within their existing training partition, so they are not counted as independent people. The final model was refit on all eligible labeled adults after test evaluation.

- Evaluated ages: 19–65 years.
- Training cohort: 197 people.
- Held-out test cohort: 32 people.
- Held-out mean absolute error: 1.239 g/dL; mean absolute percentage error (MAPE): 9.981%; R2: 0.582.
- At WHO adult cutoffs on the 32-person held-out set: classification accuracy 81.2%, balanced accuracy 83.6%, sensitivity 90.0%, and specificity 77.3%. These estimates are preliminary because the holdout set is small. Held-out MAE was 1.197 g/dL for India (n=14), 1.271 for Italy (n=18), 1.321 for females (n=14), and 1.174 for males (n=18).
- The pediatric color-feature model had negative held-out R² and is intentionally not included in the app.

The app reports an estimate and, for nonpregnant patients in the evaluated adult range, compares it with the corresponding WHO hemoglobin cutoff. If the estimate is within the model's held-out mean absolute error of the cutoff, it reports an inconclusive result. A result above the cutoff does not rule out anemia. If pregnancy is reported or uncertain, the app shows an estimate only because pregnancy was not represented or evaluated separately in this image cohort.

The questionnaire asks about common symptoms. The source image dataset does not contain symptom labels, so symptoms are displayed as context and can prompt a CBC follow-up recommendation; they do not change the numeric image estimate. Age and sex do affect the trained image estimate and the reference cutoff. The model does not use blood-pressure, smoking, sodium, or skin-tone answers as invented anemia weights.

## Retrain from the scan archive

Install training-only packages with `pip install -r requirements-training.txt`, then run:

```powershell
python -m Ai.train_image_model "C:\path\to\Hemaglobin - Under-Eye Scans-20261107T204913Z-1-001.zip"
```

The script reads the metadata sheets and image partitions directly from the ZIP and writes `Ai/artifacts/hemoglobin_model.json`. It reports adult and pediatric holdout metrics, but only ships the adult model when its held-out R² is above zero. The pediatric model is not shipped because the current supplied dataset did not meet that bar. Do not commit the source image archive or private patient files to the repository.

## WHO reference cutoffs

The app uses the 2024 WHO hemoglobin cutoffs for age and sex groups where shown. Pregnancy-specific interpretation is not enabled because the image cohort was not evaluated for pregnancy. The app does not adjust reference cutoffs for altitude or smoking. The image estimate remains a research output; it does not replace laboratory measurement or clinical evaluation. See the [WHO guideline on hemoglobin cutoffs](https://www.who.int/publications/i/item/9789240088542).

## Deployment

The included Render Blueprint starts Flask with Gunicorn and checks `/healthz`. Push the repository to GitHub, connect it as a Render Blueprint, and deploy the `hemasight` service. The model JSON is a small project artifact; the source scan archive is not needed at runtime.

## Privacy

Each screening session expires 15 minutes after the patient clicks Continue. The page shows a countdown and clears the answers and selected image at expiry; the server rejects submissions with an expired session token. The token contains no patient answers. Questionnaire responses and the uploaded image are processed in memory for the screening request and are not saved as a patient record or written to disk. Avoid including names, faces, or other identifying details in uploaded images.
