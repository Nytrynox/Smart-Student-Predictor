import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class MLEngine:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade',
            'ExtracurricularActivities', 'ParentalSupport', 'StudyHours',
            'AttendancePercent', 'OnlineClasses'
        ]
        self.target_column = 'FinalGrade'
        self.is_trained = False
        
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)

    def load_data(self, filepath):
        """Load and preprocess data from CSV"""
        try:
            self.df = pd.read_csv(filepath)
            # Ensure all required columns exist
            missing_cols = [col for col in self.feature_columns + [self.target_column] 
                          if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in CSV: {missing_cols}")
            
            self.X = self.df[self.feature_columns]
            self.y = self.df[self.target_column]
            return True, f"Loaded {len(self.df)} records successfully."
        except Exception as e:
            return False, str(e)

    def train_model(self, model_type='Random Forest', test_size=0.2):
        """Train a specific model type"""
        if not hasattr(self, 'X'):
            return False, "No data loaded. Please load data first."

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Initialize model
            if model_type == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'Gradient Boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == 'Neural Network':
                model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            else:
                return False, f"Unknown model type: {model_type}"

            # Train
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Save model and metrics
            self.models[model_type] = {
                'model': model,
                'metrics': {'MSE': mse, 'R2': r2, 'MAE': mae}
            }
            self.current_model_type = model_type
            self.is_trained = True

            return True, {
                'metrics': {'MSE': f"{mse:.4f}", 'R2': f"{r2:.4f}", 'MAE': f"{mae:.4f}"},
                'message': f"{model_type} trained successfully."
            }

        except Exception as e:
            return False, str(e)

    def predict(self, input_data):
        """Make prediction for single student"""
        if not self.is_trained:
            return None, "Model not trained yet."

        try:
            # Convert input dict to dataframe ensuring order
            input_df = pd.DataFrame([input_data], columns=self.feature_columns)
            
            # Scale input
            input_scaled = self.scaler.transform(input_df)
            
            # Predict
            model = self.models[self.current_model_type]['model']
            prediction = model.predict(input_scaled)[0]
            
            # Get feature importance if available
            importance = {}
            if hasattr(model, 'feature_importances_'):
                imp_vals = model.feature_importances_
                importance = dict(zip(self.feature_columns, imp_vals))
            
            return True, {
                'prediction': prediction,
                'importance': importance
            }

        except Exception as e:
            return False, str(e)

    def get_feature_stats(self):
        """Get statistics for all features"""
        if not hasattr(self, 'df'):
            return {}
        
        stats = {}
        for col in self.feature_columns:
            stats[col] = {
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std())
            }
        return stats

    def save_model(self, filepath):
        """Save current model to file"""
        if not self.is_trained:
            return False, "No trained model to save."
        try:
            data = {
                'model': self.models[self.current_model_type]['model'],
                'scaler': self.scaler,
                'model_type': self.current_model_type,
                'feature_columns': self.feature_columns
            }
            joblib.dump(data, filepath)
            return True, "Model saved successfully."
        except Exception as e:
            return False, str(e)

    def load_model(self, filepath):
        """Load model from file"""
        try:
            data = joblib.load(filepath)
            self.current_model_type = data['model_type']
            self.models[self.current_model_type] = {'model': data['model']}
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.is_trained = True
            return True, "Model loaded successfully."
        except Exception as e:
            return False, str(e)

    def get_decision_tree(self):
        """Get a single decision tree from Random Forest for visualization"""
        if not self.is_trained:
            return None
        
        model_data = self.models.get('Random Forest')
        if not model_data:
            return None
            
        model = model_data['model']
        if hasattr(model, 'estimators_'):
            return model.estimators_[0]
        return None
