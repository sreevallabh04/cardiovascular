"""
Feature Engineering and Preprocessing Script for Cardiovascular Risk Prediction
This script processes the time series data and creates features for the AI model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CardiovascularFeatureEngine:
    def __init__(self):
        self.patient_data = None
        self.vitals_data = None
        self.medication_data = None
        self.lifestyle_data = None
        self.lab_data = None
        self.events_data = None
        
    def load_data(self):
        """Load all CSV files"""
        print("Loading data files...")
        self.patient_data = pd.read_csv('patient_demographics.csv')
        self.vitals_data = pd.read_csv('daily_vitals.csv')
        self.medication_data = pd.read_csv('medication_adherence.csv')
        self.lifestyle_data = pd.read_csv('lifestyle_monitoring.csv')
        self.lab_data = pd.read_csv('lab_results.csv')
        self.events_data = pd.read_csv('deterioration_events.csv')
        
        # Convert date columns
        for df in [self.vitals_data, self.medication_data, self.lifestyle_data, self.lab_data, self.events_data]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        print("Data loaded successfully!")
        
    def create_time_series_features(self):
        """Create time series features from vitals data"""
        print("Creating time series features...")
        
        # Sort by patient and date
        self.vitals_data = self.vitals_data.sort_values(['patient_id', 'date'])
        
        # Create rolling window features
        window_sizes = [7, 14, 30]
        features = []
        
        for patient in self.vitals_data['patient_id'].unique():
            patient_vitals = self.vitals_data[self.vitals_data['patient_id'] == patient].copy()
            
            for window in window_sizes:
                # Rolling statistics for key vitals
                patient_vitals[f'systolic_bp_mean_{window}d'] = patient_vitals['systolic_bp'].rolling(window=window).mean()
                patient_vitals[f'systolic_bp_std_{window}d'] = patient_vitals['systolic_bp'].rolling(window=window).std()
                patient_vitals[f'systolic_bp_trend_{window}d'] = patient_vitals['systolic_bp'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
                
                patient_vitals[f'heart_rate_mean_{window}d'] = patient_vitals['heart_rate'].rolling(window=window).mean()
                patient_vitals[f'heart_rate_std_{window}d'] = patient_vitals['heart_rate'].rolling(window=window).std()
                patient_vitals[f'heart_rate_trend_{window}d'] = patient_vitals['heart_rate'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
                
                patient_vitals[f'weight_trend_{window}d'] = patient_vitals['weight_kg'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
                
                # Oxygen saturation decline
                patient_vitals[f'oxygen_sat_decline_{window}d'] = patient_vitals['oxygen_saturation'].rolling(window=window).apply(
                    lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else np.nan
                )
                
                # Steps count decline
                patient_vitals[f'steps_decline_{window}d'] = patient_vitals['steps_count'].rolling(window=window).apply(
                    lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else np.nan
                )
                
                # Sleep hours decline
                patient_vitals[f'sleep_decline_{window}d'] = patient_vitals['sleep_hours'].rolling(window=window).apply(
                    lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else np.nan
                )
                
                # Stress level increase
                patient_vitals[f'stress_increase_{window}d'] = patient_vitals['stress_level'].rolling(window=window).apply(
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else np.nan
                )
            
            features.append(patient_vitals)
        
        self.vitals_features = pd.concat(features, ignore_index=True)
        print("Time series features created!")
        
    def create_medication_features(self):
        """Create medication adherence features"""
        print("Creating medication features...")
        
        med_features = []
        
        for patient in self.medication_data['patient_id'].unique():
            patient_meds = self.medication_data[self.medication_data['patient_id'] == patient].copy()
            
            # Calculate adherence rates by medication type
            for med_type in patient_meds['medication_type'].unique():
                med_type_data = patient_meds[patient_meds['medication_type'] == med_type]
                
                # Rolling adherence rates
                for window in [7, 14, 30]:
                    med_type_data[f'{med_type.lower()}_adherence_{window}d'] = med_type_data['adherence_percentage'].rolling(window=window).mean()
                    med_type_data[f'{med_type.lower()}_adherence_trend_{window}d'] = med_type_data['adherence_percentage'].rolling(window=window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                    )
                
                # Side effects count
                med_type_data[f'{med_type.lower()}_side_effects_count'] = (med_type_data['side_effects'] != 'None').rolling(window=7).sum()
                
                med_features.append(med_type_data)
        
        self.medication_features = pd.concat(med_features, ignore_index=True)
        print("Medication features created!")
        
    def create_lifestyle_features(self):
        """Create lifestyle monitoring features"""
        print("Creating lifestyle features...")
        
        lifestyle_features = []
        
        for patient in self.lifestyle_data['patient_id'].unique():
            patient_lifestyle = self.lifestyle_data[self.lifestyle_data['patient_id'] == patient].copy()
            
            # Rolling lifestyle metrics
            for window in [7, 14, 30]:
                patient_lifestyle[f'diet_quality_mean_{window}d'] = patient_lifestyle['diet_quality_score'].rolling(window=window).mean()
                patient_lifestyle[f'exercise_mean_{window}d'] = patient_lifestyle['exercise_minutes'].rolling(window=window).mean()
                patient_lifestyle[f'sleep_quality_mean_{window}d'] = patient_lifestyle['sleep_quality_score'].rolling(window=window).mean()
                patient_lifestyle[f'mood_mean_{window}d'] = patient_lifestyle['mood_score'].rolling(window=window).mean()
                
                # Declining trends
                patient_lifestyle[f'diet_decline_{window}d'] = patient_lifestyle['diet_quality_score'].rolling(window=window).apply(
                    lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else np.nan
                )
                patient_lifestyle[f'exercise_decline_{window}d'] = patient_lifestyle['exercise_minutes'].rolling(window=window).apply(
                    lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else np.nan
                )
                patient_lifestyle[f'mood_decline_{window}d'] = patient_lifestyle['mood_score'].rolling(window=window).apply(
                    lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else np.nan
                )
            
            lifestyle_features.append(patient_lifestyle)
        
        self.lifestyle_features = pd.concat(lifestyle_features, ignore_index=True)
        print("Lifestyle features created!")
        
    def create_lab_features(self):
        """Create lab results features"""
        print("Creating lab features...")
        
        lab_features = []
        
        for patient in self.lab_data['patient_id'].unique():
            patient_labs = self.lab_data[self.lab_data['patient_id'] == patient].copy()
            
            # Calculate lab trends and abnormalities
            lab_columns = ['glucose_mg_dl', 'hba1c_percent', 'total_cholesterol_mg_dl', 
                          'ldl_cholesterol_mg_dl', 'hdl_cholesterol_mg_dl', 'triglycerides_mg_dl',
                          'creatinine_mg_dl', 'bun_mg_dl', 'alt_u_l', 'ast_u_l']
            
            for lab in lab_columns:
                # Trend calculation
                patient_labs[f'{lab}_trend'] = patient_labs[lab].diff()
                patient_labs[f'{lab}_trend_7d'] = patient_labs[lab].rolling(window=2).apply(
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 2 else np.nan
                )
                
                # Abnormal values (simplified thresholds)
                if lab == 'glucose_mg_dl':
                    patient_labs[f'{lab}_abnormal'] = (patient_labs[lab] > 126).astype(int)
                elif lab == 'hba1c_percent':
                    patient_labs[f'{lab}_abnormal'] = (patient_labs[lab] > 6.5).astype(int)
                elif lab == 'total_cholesterol_mg_dl':
                    patient_labs[f'{lab}_abnormal'] = (patient_labs[lab] > 200).astype(int)
                elif lab == 'creatinine_mg_dl':
                    patient_labs[f'{lab}_abnormal'] = (patient_labs[lab] > 1.2).astype(int)
            
            lab_features.append(patient_labs)
        
        self.lab_features = pd.concat(lab_features, ignore_index=True)
        print("Lab features created!")
        
    def create_risk_labels(self):
        """Create risk labels for the next 90 days"""
        print("Creating risk labels...")
        
        # Create a comprehensive dataset with all features
        self.create_combined_dataset()
        
        # Add risk labels based on deterioration events
        self.combined_data['risk_label_90d'] = 0
        self.combined_data['days_to_event'] = np.nan
        
        for _, event in self.events_data.iterrows():
            if event['event_type'] != 'None':
                # Find the patient's data
                patient_mask = self.combined_data['patient_id'] == event['patient_id']
                
                # Mark risk for 90 days before the event
                event_date = event['event_date']
                risk_start_date = event_date - timedelta(days=90)
                
                date_mask = (self.combined_data['date'] >= risk_start_date) & (self.combined_data['date'] < event_date)
                
                self.combined_data.loc[patient_mask & date_mask, 'risk_label_90d'] = 1
                self.combined_data.loc[patient_mask & date_mask, 'days_to_event'] = (event_date - self.combined_data.loc[patient_mask & date_mask, 'date']).dt.days
        
        print("Risk labels created!")
        
    def create_combined_dataset(self):
        """Combine all features into a single dataset"""
        print("Combining all features...")
        
        # Start with vitals data
        self.combined_data = self.vitals_features.copy()
        
        # Merge medication features
        if hasattr(self, 'medication_features'):
            med_agg = self.medication_features.groupby(['patient_id', 'date']).agg({
                'adherence_percentage': 'mean',
                'side_effects': lambda x: (x != 'None').sum()
            }).reset_index()
            med_agg.columns = ['patient_id', 'date', 'avg_adherence', 'side_effects_count']
            self.combined_data = self.combined_data.merge(med_agg, on=['patient_id', 'date'], how='left')
        
        # Merge lifestyle features
        if hasattr(self, 'lifestyle_features'):
            lifestyle_agg = self.lifestyle_features.groupby(['patient_id', 'date']).agg({
                'diet_quality_score': 'mean',
                'exercise_minutes': 'mean',
                'sleep_quality_score': 'mean',
                'mood_score': 'mean',
                'stress_level': 'mean'
            }).reset_index()
            self.combined_data = self.combined_data.merge(lifestyle_agg, on=['patient_id', 'date'], how='left')
        
        # Merge lab features (forward fill for missing dates)
        if hasattr(self, 'lab_features'):
            lab_agg = self.lab_features.groupby(['patient_id', 'date']).mean().reset_index()
            self.combined_data = self.combined_data.merge(lab_agg, on=['patient_id', 'date'], how='left')
            
            # Forward fill lab values
            lab_columns = [col for col in lab_agg.columns if col not in ['patient_id', 'date']]
            self.combined_data[lab_columns] = self.combined_data.groupby('patient_id')[lab_columns].fillna(method='ffill')
        
        # Merge patient demographics
        self.combined_data = self.combined_data.merge(self.patient_data, on='patient_id', how='left')
        
        print("Combined dataset created!")
        
    def create_explainability_features(self):
        """Create features specifically for explainability"""
        print("Creating explainability features...")
        
        # Risk factor categories
        self.combined_data['vital_risk_score'] = (
            (self.combined_data['systolic_bp'] > 140).astype(int) * 0.3 +
            (self.combined_data['diastolic_bp'] > 90).astype(int) * 0.2 +
            (self.combined_data['heart_rate'] > 100).astype(int) * 0.2 +
            (self.combined_data['oxygen_saturation'] < 95).astype(int) * 0.3
        )
        
        self.combined_data['medication_risk_score'] = (
            (self.combined_data['avg_adherence'] < 80).astype(int) * 0.5 +
            (self.combined_data['side_effects_count'] > 0).astype(int) * 0.5
        )
        
        self.combined_data['lifestyle_risk_score'] = (
            (self.combined_data['diet_quality_score'] < 5).astype(int) * 0.25 +
            (self.combined_data['exercise_minutes'] < 150).astype(int) * 0.25 +
            (self.combined_data['sleep_quality_score'] < 5).astype(int) * 0.25 +
            (self.combined_data['stress_level'] > 7).astype(int) * 0.25
        )
        
        self.combined_data['lab_risk_score'] = (
            (self.combined_data['glucose_mg_dl'] > 126).astype(int) * 0.2 +
            (self.combined_data['hba1c_percent'] > 6.5).astype(int) * 0.2 +
            (self.combined_data['total_cholesterol_mg_dl'] > 200).astype(int) * 0.2 +
            (self.combined_data['creatinine_mg_dl'] > 1.2).astype(int) * 0.2 +
            (self.combined_data['alt_u_l'] > 40).astype(int) * 0.1 +
            (self.combined_data['ast_u_l'] > 40).astype(int) * 0.1
        )
        
        # Overall risk score
        self.combined_data['overall_risk_score'] = (
            self.combined_data['vital_risk_score'] * 0.4 +
            self.combined_data['medication_risk_score'] * 0.3 +
            self.combined_data['lifestyle_risk_score'] * 0.2 +
            self.combined_data['lab_risk_score'] * 0.1
        )
        
        print("Explainability features created!")
        
    def save_processed_data(self):
        """Save the processed dataset"""
        print("Saving processed data...")
        
        # Save the combined dataset
        self.combined_data.to_csv('processed_cardiovascular_data.csv', index=False)
        
        # Save feature importance data for explainability
        feature_importance = pd.DataFrame({
            'feature_category': ['Vitals', 'Medication', 'Lifestyle', 'Lab Results'],
            'importance_score': [0.4, 0.3, 0.2, 0.1],
            'description': [
                'Blood pressure, heart rate, oxygen saturation, weight trends',
                'Medication adherence rates and side effects',
                'Diet, exercise, sleep quality, stress levels',
                'Glucose, cholesterol, kidney function markers'
            ]
        })
        feature_importance.to_csv('feature_importance.csv', index=False)
        
        # Save risk thresholds for explainability
        risk_thresholds = pd.DataFrame({
            'metric': ['Systolic BP', 'Diastolic BP', 'Heart Rate', 'Oxygen Saturation', 
                      'Glucose', 'HbA1c', 'Total Cholesterol', 'Creatinine'],
            'normal_range': ['<140', '<90', '<100', '>95%', '<126', '<6.5%', '<200', '<1.2'],
            'risk_threshold': ['>140', '>90', '>100', '<95%', '>126', '>6.5%', '>200', '>1.2'],
            'clinical_significance': [
                'Hypertension risk', 'Hypertension risk', 'Tachycardia', 'Hypoxemia risk',
                'Diabetes risk', 'Diabetes risk', 'Cardiovascular risk', 'Kidney function decline'
            ]
        })
        risk_thresholds.to_csv('risk_thresholds.csv', index=False)
        
        print("Processed data saved!")
        
    def run_full_pipeline(self):
        """Run the complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        
        self.load_data()
        self.create_time_series_features()
        self.create_medication_features()
        self.create_lifestyle_features()
        self.create_lab_features()
        self.create_risk_labels()
        self.create_explainability_features()
        self.save_processed_data()
        
        print("Feature engineering pipeline completed!")
        print(f"Final dataset shape: {self.combined_data.shape}")
        print(f"Features created: {len(self.combined_data.columns)}")
        
        return self.combined_data

if __name__ == "__main__":
    # Run the feature engineering pipeline
    feature_engine = CardiovascularFeatureEngine()
    processed_data = feature_engine.run_full_pipeline()
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(processed_data.head())
    
    # Display risk label distribution
    print(f"\nRisk label distribution:")
    print(processed_data['risk_label_90d'].value_counts())
