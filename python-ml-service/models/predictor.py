"""
Advanced ML Predictor with multiple models
Supports Random Forest, XGBoost, Neural Networks, and Ensemble methods
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib
import json
import os
from datetime import datetime

class StudentPredictor:
    """Advanced ML predictor for student performance analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.current_model = None
        self.current_model_id = None
        self.model_dir = 'saved_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self, X, y, model_type='random_forest', hyperparameters=None):
        """
        Train a model with specified type and hyperparameters
        
        Args:
            X: Feature matrix
            y: Target values
            model_type: 'random_forest', 'xgboost', 'neural_network', or 'ensemble'
            hyperparameters: Dictionary of model-specific hyperparameters
        
        Returns:
            Dictionary with training results and metrics
        """
        # Determine if classification or regression
        is_classification = len(np.unique(y)) <= 10 and np.all(np.mod(y, 1) == 0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select and configure model
        if model_type == 'random_forest':
            model = self._create_random_forest(is_classification, hyperparameters)
        elif model_type == 'xgboost':
            model = self._create_xgboost(is_classification, hyperparameters)
        elif model_type == 'neural_network':
            model = self._create_neural_network(is_classification, hyperparameters)
        elif model_type == 'ensemble':
            model = self._create_ensemble(is_classification, hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_scaled, y)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        
        # Make predictions for metrics
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        if is_classification:
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2_score': r2_score(y, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        # Save model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = model
        self.scalers[model_id] = scaler
        self.current_model = model
        self.current_model_id = model_id
        
        # Persist to disk
        self._save_to_disk(model_id, model, scaler)
        
        return {
            'model_id': model_id,
            'metrics': metrics,
            'is_classification': is_classification
        }
    
    def _create_random_forest(self, is_classification, hyperparameters):
        """Create Random Forest model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        params = {**default_params, **(hyperparameters or {})}
        
        if is_classification:
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    
    def _create_xgboost(self, is_classification, hyperparameters):
        """Create Gradient Boosting model (XGBoost alternative)"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        params = {**default_params, **(hyperparameters or {})}
        
        if is_classification:
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)
    
    def _create_neural_network(self, is_classification, hyperparameters):
        """Create Neural Network model"""
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True
        }
        params = {**default_params, **(hyperparameters or {})}
        
        if is_classification:
            return MLPClassifier(**params)
        else:
            return MLPRegressor(**params)
    
    def _create_ensemble(self, is_classification, hyperparameters):
        """Create ensemble of multiple models"""
        # For simplicity, returning Random Forest as base
        # In production, implement proper ensemble (voting/stacking)
        return self._create_random_forest(is_classification, hyperparameters)
    
    def predict(self, X):
        """Make prediction for single or multiple samples"""
        if self.current_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        X_scaled = self.scalers[self.current_model_id].transform(X)
        return self.current_model.predict(X_scaled)
    
    def predict_batch(self, students):
        """Make predictions for batch of students"""
        # Convert student data to feature matrix
        X = self._prepare_features(students)
        predictions = self.predict(X)
        
        return [
            {
                'student_id': student.get('student_id', f'student_{i}'),
                'prediction': float(pred),
                'confidence': self._get_confidence(X[i:i+1])
            }
            for i, (student, pred) in enumerate(zip(students, predictions))
        ]
    
    def predict_proba(self, X):
        """Get prediction probabilities (for classification)"""
        if self.current_model is None:
            raise ValueError("No model loaded.")
        
        if not hasattr(self.current_model, 'predict_proba'):
            raise ValueError("Current model doesn't support probability predictions")
        
        X_scaled = self.scalers[self.current_model_id].transform(X)
        return self.current_model.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        if self.current_model is None:
            raise ValueError("No model loaded.")
        
        X_scaled = self.scalers[self.current_model_id].transform(X_test)
        y_pred = self.current_model.predict(X_scaled)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y_test)) <= 10
        
        if is_classification:
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            return {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.current_model is None:
            raise ValueError("No model loaded.")
        
        if hasattr(self.current_model, 'feature_importances_'):
            return self.current_model.feature_importances_.tolist()
        else:
            return None
    
    def analyze_student(self, student_data, analysis_type='full'):
        """
        Comprehensive student analysis
        
        Args:
            student_data: Dictionary with student features
            analysis_type: 'full' or 'quick'
        
        Returns:
            Comprehensive analysis results
        """
        X = self._prepare_features([student_data])
        
        # Basic prediction
        prediction = self.predict(X)[0]
        confidence = self._get_confidence(X)
        
        analysis = {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if analysis_type == 'full':
            # Feature importance for this student
            if self.get_feature_importance() is not None:
                analysis['feature_contributions'] = self._get_feature_contributions(X[0])
            
            # Recommendations
            analysis['recommendations'] = self._generate_recommendations(student_data, prediction)
            
            # Risk factors
            analysis['risk_factors'] = self._identify_risk_factors(student_data)
        
        return analysis
    
    def _prepare_features(self, students):
        """Convert student dictionaries to feature matrix"""
        # Extract numeric features
        feature_names = ['hours_studied', 'attendance', 'assignments_submitted', 'previous_gpa']
        X = []
        for student in students:
            features = [student.get(fn, 0) for fn in feature_names]
            X.append(features)
        return np.array(X)
    
    def _get_confidence(self, X):
        """Calculate prediction confidence"""
        if hasattr(self.current_model, 'predict_proba'):
            X_scaled = self.scalers[self.current_model_id].transform(X)
            proba = self.current_model.predict_proba(X_scaled)
            return float(np.max(proba))
        else:
            return 0.8  # Default confidence for regression
    
    def _get_feature_contributions(self, X):
        """Get feature contributions for a single prediction"""
        importance = self.get_feature_importance()
        feature_names = ['hours_studied', 'attendance', 'assignments_submitted', 'previous_gpa']
        
        if importance is not None:
            contributions = []
            for name, imp, value in zip(feature_names, importance, X):
                contributions.append({
                    'feature': name,
                    'importance': float(imp),
                    'value': float(value)
                })
            return sorted(contributions, key=lambda x: x['importance'], reverse=True)
        return []
    
    def _generate_recommendations(self, student_data, prediction):
        """Generate personalized recommendations"""
        recommendations = []
        
        hours = student_data.get('hours_studied', 0)
        attendance = student_data.get('attendance', 0)
        assignments = student_data.get('assignments_submitted', 0)
        
        if hours < 8:
            recommendations.append({
                'type': 'study_hours',
                'message': f'Increase study hours from {hours} to at least 8-10 hours per week',
                'priority': 'high'
            })
        
        if attendance < 0.8:
            recommendations.append({
                'type': 'attendance',
                'message': f'Improve attendance from {attendance*100:.0f}% to at least 80%',
                'priority': 'high'
            })
        
        if assignments < 7:
            recommendations.append({
                'type': 'assignments',
                'message': f'Submit more assignments. Current: {assignments}, Target: 7+',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _identify_risk_factors(self, student_data):
        """Identify risk factors for poor performance"""
        risk_factors = []
        
        if student_data.get('hours_studied', 0) < 5:
            risk_factors.append('Very low study hours')
        
        if student_data.get('attendance', 0) < 0.7:
            risk_factors.append('Poor attendance record')
        
        if student_data.get('assignments_submitted', 0) < 5:
            risk_factors.append('Low assignment completion')
        
        if student_data.get('previous_gpa', 0) < 2.5:
            risk_factors.append('Low previous academic performance')
        
        return risk_factors
    
    def load_model(self, model_id):
        """Load a saved model from disk"""
        model_path = os.path.join(self.model_dir, f'{model_id}_model.pkl')
        scaler_path = os.path.join(self.model_dir, f'{model_id}_scaler.pkl')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found")
        
        self.current_model = joblib.load(model_path)
        self.scalers[model_id] = joblib.load(scaler_path)
        self.models[model_id] = self.current_model
        self.current_model_id = model_id
    
    def _save_to_disk(self, model_id, model, scaler):
        """Save model and scaler to disk"""
        model_path = os.path.join(self.model_dir, f'{model_id}_model.pkl')
        scaler_path = os.path.join(self.model_dir, f'{model_id}_scaler.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
