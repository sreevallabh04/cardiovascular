# Cardiovascular Risk Prediction Dataset Documentation

## Overview
This dataset is designed for building an AI-driven risk prediction engine for chronic care patients, specifically focusing on cardiovascular health. The dataset includes time series data over 180 days for 50 patients with various chronic conditions.

## Dataset Structure

### 1. Patient Demographics (`patient_demographics.csv`)
**Purpose**: Baseline patient characteristics and risk factors

**Columns**:
- `patient_id`: Unique patient identifier
- `age`: Patient age in years
- `gender`: Male/Female
- `race`: White, Black, Hispanic, Asian
- `smoking_status`: Never, Former, Current
- `diabetes_type`: None, Type1, Type2
- `hypertension`: Yes/No
- `previous_heart_attack`: Yes/No
- `family_history_cvd`: Yes/No
- `bmi_baseline`: Baseline BMI
- `cholesterol_baseline`: Baseline total cholesterol (mg/dL)
- `creatinine_baseline`: Baseline creatinine (mg/dL)
- `enrollment_date`: Date of enrollment in study
- `risk_category`: High, Medium, Low (based on baseline risk factors)

### 2. Daily Vitals (`daily_vitals.csv`)
**Purpose**: Daily physiological measurements and activity data

**Columns**:
- `patient_id`: Unique patient identifier
- `date`: Date of measurement
- `systolic_bp`: Systolic blood pressure (mmHg)
- `diastolic_bp`: Diastolic blood pressure (mmHg)
- `heart_rate`: Heart rate (bpm)
- `weight_kg`: Weight in kilograms
- `oxygen_saturation`: Oxygen saturation percentage
- `temperature_c`: Body temperature in Celsius
- `steps_count`: Daily step count
- `sleep_hours`: Hours of sleep
- `stress_level`: Self-reported stress level (1-10 scale)

### 3. Medication Adherence (`medication_adherence.csv`)
**Purpose**: Medication compliance and side effects tracking

**Columns**:
- `patient_id`: Unique patient identifier
- `date`: Date of medication tracking
- `medication_name`: Name of medication
- `prescribed_dose`: Prescribed dosage
- `actual_dose_taken`: Actual dose taken by patient
- `adherence_percentage`: Percentage of prescribed dose taken
- `side_effects`: Reported side effects (None, Mild, Severe)
- `medication_type`: Diabetes, Hypertension, etc.

### 4. Lifestyle Monitoring (`lifestyle_monitoring.csv`)
**Purpose**: Daily lifestyle and behavioral factors

**Columns**:
- `patient_id`: Unique patient identifier
- `date`: Date of monitoring
- `diet_quality_score`: Diet quality score (1-10)
- `exercise_minutes`: Minutes of exercise
- `alcohol_consumption`: Alcohol consumption units
- `smoking_cigarettes`: Number of cigarettes smoked
- `stress_level`: Stress level (1-10 scale)
- `meditation_minutes`: Minutes of meditation
- `social_activity_score`: Social activity score (1-10)
- `sleep_quality_score`: Sleep quality score (1-10)
- `mood_score`: Mood score (1-10)

### 5. Lab Results (`lab_results.csv`)
**Purpose**: Periodic laboratory test results

**Columns**:
- `patient_id`: Unique patient identifier
- `date`: Date of lab test
- `glucose_mg_dl`: Blood glucose (mg/dL)
- `hba1c_percent`: HbA1c percentage
- `total_cholesterol_mg_dl`: Total cholesterol (mg/dL)
- `ldl_cholesterol_mg_dl`: LDL cholesterol (mg/dL)
- `hdl_cholesterol_mg_dl`: HDL cholesterol (mg/dL)
- `triglycerides_mg_dl`: Triglycerides (mg/dL)
- `creatinine_mg_dl`: Creatinine (mg/dL)
- `bun_mg_dl`: Blood urea nitrogen (mg/dL)
- `alt_u_l`: Alanine aminotransferase (U/L)
- `ast_u_l`: Aspartate aminotransferase (U/L)

### 6. Deterioration Events (`deterioration_events.csv`)
**Purpose**: Clinical events and outcomes for risk labeling

**Columns**:
- `patient_id`: Unique patient identifier
- `event_date`: Date of clinical event
- `event_type`: Type of event (Heart_Failure, Stroke, etc.)
- `severity`: Event severity (Mild, Moderate, Severe)
- `days_to_event`: Days from enrollment to event
- `risk_score_30_days_prior`: Risk score 30 days before event
- `risk_score_7_days_prior`: Risk score 7 days before event
- `risk_score_1_day_prior`: Risk score 1 day before event
- `intervention_required`: Whether intervention was required
- `hospitalization_days`: Days of hospitalization
- `outcome`: Event outcome (Stable, Recovered, Disability)

## Data Characteristics

### Time Series Structure
- **Duration**: 180 days per patient
- **Frequency**: Daily measurements for vitals, lifestyle, and medication
- **Lab Frequency**: Weekly lab results
- **Events**: Sparse, occurring in high-risk patients

### Risk Patterns
- **High Risk Patients**: Show deteriorating trends in vitals, poor medication adherence, declining lifestyle factors
- **Low Risk Patients**: Stable vitals, good adherence, healthy lifestyle patterns
- **Medium Risk Patients**: Mixed patterns with some concerning trends

### Missing Data Strategy
- Forward fill for lab results (weekly to daily)
- Interpolation for occasional missing vitals
- Zero-fill for missing lifestyle data (assumes no activity)

## Feature Engineering

### Time Series Features
- **Rolling Statistics**: Mean, standard deviation, trends over 7, 14, 30-day windows
- **Trend Analysis**: Linear regression slopes for key metrics
- **Change Detection**: Rate of change in critical parameters

### Risk Scoring
- **Vital Risk Score**: Based on blood pressure, heart rate, oxygen saturation
- **Medication Risk Score**: Based on adherence rates and side effects
- **Lifestyle Risk Score**: Based on diet, exercise, sleep, stress
- **Lab Risk Score**: Based on abnormal lab values

### Explainability Features
- **Risk Categories**: Clear categorization of risk factors
- **Thresholds**: Clinical thresholds for each metric
- **Trend Indicators**: Direction and magnitude of changes

## Model Requirements

### Input Requirements
- **Time Window**: 30-180 days of patient data
- **Features**: 100+ engineered features from raw data
- **Target**: Binary risk label (90-day deterioration risk)

### Output Requirements
- **Prediction**: Probability of deterioration within 90 days
- **Explanation**: Key risk factors and their contributions
- **Actionability**: Specific recommendations for intervention

## Clinical Significance

### Key Risk Factors
1. **Blood Pressure Trends**: Rising systolic/diastolic pressure
2. **Medication Adherence**: Declining compliance rates
3. **Lifestyle Deterioration**: Reduced exercise, poor diet, increased stress
4. **Lab Abnormalities**: Rising glucose, cholesterol, creatinine
5. **Vital Signs**: Declining oxygen saturation, increasing heart rate

### Intervention Points
- **Early Warning**: 30-60 days before potential deterioration
- **Critical Thresholds**: Specific values that trigger alerts
- **Lifestyle Interventions**: Diet, exercise, stress management
- **Medication Adjustments**: Dosage changes, side effect management

## Usage for Hackathon

### Model Development
1. Use `feature_engineering.py` to create processed dataset
2. Train time series models (LSTM, GRU, Transformer)
3. Implement explainability methods (SHAP, LIME)
4. Create risk scoring algorithms

### Dashboard Development
1. **Cohort View**: Risk scores for all patients
2. **Patient Detail**: Individual trends and risk factors
3. **Alerts**: Real-time risk notifications
4. **Recommendations**: Actionable intervention suggestions

### Evaluation Metrics
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **Calibration**: Reliability of probability estimates
- **Confusion Matrix**: Classification performance

## Data Quality Notes

### Realistic Patterns
- **Deterioration**: Gradual decline in high-risk patients
- **Stability**: Consistent patterns in low-risk patients
- **Variability**: Natural day-to-day fluctuations
- **Correlations**: Realistic relationships between variables

### Clinical Validity
- **Thresholds**: Based on clinical guidelines
- **Trends**: Realistic progression patterns
- **Events**: Clinically relevant deterioration events
- **Outcomes**: Realistic clinical outcomes

## File Dependencies

```
cardiovascular_dataset/
├── patient_demographics.csv
├── daily_vitals.csv
├── medication_adherence.csv
├── lifestyle_monitoring.csv
├── lab_results.csv
├── deterioration_events.csv
├── feature_engineering.py
├── processed_cardiovascular_data.csv (generated)
├── feature_importance.csv (generated)
├── risk_thresholds.csv (generated)
└── data_documentation.md
```

## Next Steps

1. **Run Feature Engineering**: Execute `feature_engineering.py`
2. **Model Development**: Build prediction models
3. **Explainability**: Implement SHAP/LIME explanations
4. **Dashboard**: Create visualization interface
5. **Validation**: Test on held-out data
6. **Deployment**: Prepare for production use
