"""
Smart Student Analysis Predictor - Python ML Service
Flask-based ML backend with advanced models and MongoDB integration
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import logging
import os
from models.predictor import StudentPredictor
from services.data_service import DataService
from services.model_service import ModelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'smart_student_db')

# Initialize services
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
data_service = DataService(db)
model_service = ModelService(db)
predictor = StudentPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'python-ml-service',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/ml/train', methods=['POST'])
def train_model():
    """
    Train ML model with data from MongoDB
    Request body: {
        "model_type": "random_forest|xgboost|neural_network|ensemble",
        "features": ["hours_studied", "attendance", ...],
        "target": "final_score" or "passed",
        "hyperparameters": {...}
    }
    """
    try:
        data = request.json
        model_type = data.get('model_type', 'random_forest')
        features = data.get('features')
        target = data.get('target', 'final_score')
        hyperparameters = data.get('hyperparameters', {})
        
        # Fetch training data from MongoDB
        training_data = data_service.get_training_data(features, target)
        
        # Train model
        result = predictor.train(
            training_data['X'],
            training_data['y'],
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        
        # Save model metadata to MongoDB
        model_id = model_service.save_model_metadata(
            model_type=model_type,
            features=features,
            target=target,
            metrics=result['metrics'],
            hyperparameters=hyperparameters
        )
        
        logger.info(f"Model trained successfully: {model_id}")
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'metrics': result['metrics'],
            'model_type': model_type
        })
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/predict', methods=['POST'])
def predict():
    """
    Make predictions using trained model
    Request body: {
        "model_id": "model_123",
        "students": [{student_data}, ...]
    }
    """
    try:
        data = request.json
        model_id = data.get('model_id')
        students = data.get('students', [])
        
        # Load model
        model_metadata = model_service.get_model_metadata(model_id)
        predictor.load_model(model_id)
        
        # Make predictions
        predictions = predictor.predict_batch(students)
        
        # Save predictions to MongoDB
        prediction_ids = data_service.save_predictions(
            model_id=model_id,
            students=students,
            predictions=predictions
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'prediction_ids': prediction_ids
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/evaluate', methods=['POST'])
def evaluate_model():
    """
    Evaluate model performance on test data
    Request body: {
        "model_id": "model_123"
    }
    """
    try:
        data = request.json
        model_id = data.get('model_id')
        
        # Load model and test data
        model_metadata = model_service.get_model_metadata(model_id)
        test_data = data_service.get_test_data(
            model_metadata['features'],
            model_metadata['target']
        )
        
        predictor.load_model(model_id)
        
        # Evaluate
        metrics = predictor.evaluate(test_data['X'], test_data['y'])
        
        # Update model metadata
        model_service.update_metrics(model_id, metrics)
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance for a trained model"""
    try:
        model_id = request.args.get('model_id')
        
        predictor.load_model(model_id)
        importance = predictor.get_feature_importance()
        
        return jsonify({
            'success': True,
            'feature_importance': importance
        })
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/models', methods=['GET'])
def list_models():
    """List all trained models"""
    try:
        models = model_service.list_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/analyze', methods=['POST'])
def analyze_student():
    """
    Comprehensive student analysis with multiple models
    Request body: {
        "student_data": {...},
        "analysis_type": "full|quick"
    }
    """
    try:
        data = request.json
        student_data = data.get('student_data')
        analysis_type = data.get('analysis_type', 'full')
        
        # Run comprehensive analysis
        analysis = predictor.analyze_student(student_data, analysis_type)
        
        # Save analysis to MongoDB
        analysis_id = data_service.save_analysis(student_data, analysis)
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
