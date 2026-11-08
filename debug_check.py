from app import app

client = app.test_client()
health = client.get('/healthz')
print('HEALTH', health.status_code, health.get_data(as_text=True))

payload = {
    'age': 30,
    'sex_assigned_at_birth': 'female',
    'natural_skin_tone': 'olive_medium',
    'previous_bp_diagnosis': 'no',
    'recent_bp_reading': 'yes',
    'bp_systolic_value': 120,
    'bp_diastolic_value': 80,
    'bp_medications': 'no',
    'bp_symptoms': 'no',
    'family_bp_history': 'no',
    'physical_activity': 'few_times_week',
    'sodium_intake': 'moderate',
    'smoking_status': 'never',
    'fasting_status': 'no',
    'cold_hands_feet': 'no',
}

response = client.post('/questionnaire', json={'answers': payload})
print('QUESTIONNAIRE', response.status_code, response.get_json())
