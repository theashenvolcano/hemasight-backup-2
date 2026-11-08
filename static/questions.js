/*
 * Screening content kept in sync with templates/questionnaire.html and the
 * field names accepted by app.py. This file is a data source for questionnaire
 * interfaces; questionnaire.html currently renders its fields directly.
 */
window.QUESTION_SECTIONS = [
    {
        id: "patient-information",
        title: "Section 1: About the Patient",
        description: "Basic information used to select the appropriate research reference range.",
        questions: [
            {
                id: "age_years",
                label: "What is your age in years?",
                type: "number",
                placeholder: "Enter your age",
                min: "0.5",
                max: "120",
                step: "0.5",
                required: true
            },
            {
                id: "sex_assigned_at_birth",
                label: "What sex were you assigned at birth?",
                type: "choice",
                required: true,
                options: [
                    { label: "Female", value: "female" },
                    { label: "Male", value: "male" },
                    { label: "Prefer not to answer", value: "prefer_not_to_answer" }
                ]
            },
            {
                id: "pregnancy_status",
                label: "Are you currently pregnant, or could you be?",
                helpText: "Shown for patients age 15 or older who were assigned female at birth.",
                type: "conditional-choice",
                conditionalOn: {
                    questionId: "sex_assigned_at_birth",
                    value: "female",
                    minimumAge: 15
                },
                options: [
                    { label: "No", value: "no" },
                    { label: "Yes", value: "yes" },
                    { label: "Unsure", value: "unsure" },
                    { label: "Prefer not to answer", value: "prefer_not_to_answer" }
                ]
            }
        ]
    },
    {
        id: "symptoms",
        title: "Section 2: Symptoms",
        description: "For each item, choose the response that best describes how you feel.",
        questions: [
            {
                id: "symptom_fatigue",
                label: "Have you felt unusually tired or low in energy?",
                type: "choice",
                required: true,
                options: [
                    { label: "Yes", value: "yes" },
                    { label: "No", value: "no" },
                    { label: "Unsure", value: "unsure" }
                ]
            },
            {
                id: "symptom_weakness",
                label: "Have you felt generally weak?",
                type: "choice",
                required: true,
                options: [
                    { label: "Yes", value: "yes" },
                    { label: "No", value: "no" },
                    { label: "Unsure", value: "unsure" }
                ]
            },
            {
                id: "symptom_dizziness",
                label: "Have you felt dizzy or lightheaded?",
                type: "choice",
                required: true,
                options: [
                    { label: "Yes", value: "yes" },
                    { label: "No", value: "no" },
                    { label: "Unsure", value: "unsure" }
                ]
            },
            {
                id: "symptom_shortness_of_breath",
                label: "Have you been short of breath during ordinary activity?",
                type: "choice",
                required: true,
                options: [
                    { label: "Yes", value: "yes" },
                    { label: "No", value: "no" },
                    { label: "Unsure", value: "unsure" }
                ]
            },
            {
                id: "symptom_pale_inner_eyelid",
                label: "Does the inside of your lower eyelid look paler than usual?",
                type: "choice",
                required: true,
                options: [
                    { label: "Yes", value: "yes" },
                    { label: "No", value: "no" },
                    { label: "Unsure", value: "unsure" }
                ]
            },
            {
                id: "symptom_cold_hands_feet",
                label: "Have your hands or feet felt unusually cold?",
                type: "choice",
                required: true,
                options: [
                    { label: "Yes", value: "yes" },
                    { label: "No", value: "no" },
                    { label: "Unsure", value: "unsure" }
                ]
            }
        ]
    },
    {
        id: "eye-image",
        title: "Section 3: Eye Image",
        description: "Upload a clear image of the inner lower eyelid. The image estimate is research-only and cannot diagnose anemia.",
        questions: [
            {
                id: "eye_image",
                label: "Upload an eye image",
                type: "file",
                accept: "image/jpeg,image/png,image/webp",
                maxSizeBytes: 12 * 1024 * 1024,
                required: true
            }
        ]
    }
];
