"""
Data Service for MongoDB operations
Handles student data, predictions, and analytics
"""
import pandas as pd
import numpy as np
from datetime import datetime
from bson import ObjectId

class DataService:
    """Service for managing student data in MongoDB"""
    
    def __init__(self, db):
        self.db = db
        self.students = db.students
        self.predictions = db.predictions
        self.analyses = db.analyses
        
        # Create indexes for better query performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes"""
        self.students.create_index('student_id', unique=True)
        self.students.create_index('created_at')
        self.predictions.create_index('model_id')
        self.predictions.create_index('student_id')
        self.predictions.create_index('created_at')
        self.analyses.create_index('student_id')
        self.analyses.create_index('created_at')
    
    def get_training_data(self, features, target):
        """
        Fetch training data from MongoDB
        
        Args:
            features: List of feature names
            target: Target variable name
        
        Returns:
            Dictionary with X (features) and y (target) arrays
        """
        # Fetch all students from MongoDB
        students = list(self.students.find({}))
        
        if not students:
            raise ValueError("No training data available in database")
        
        # Convert to DataFrame
        df = pd.DataFrame(students)
        
        # Extract features and target
        X = df[features].values
        y = df[target].values
        
        return {
            'X': X,
            'y': y,
            'feature_names': features,
            'target_name': target
        }
    
    def get_test_data(self, features, target, test_split=0.2):
        """Get test data split from MongoDB"""
        data = self.get_training_data(features, target)
        
        # Simple split (last 20% as test)
        split_idx = int(len(data['X']) * (1 - test_split))
        
        return {
            'X': data['X'][split_idx:],
            'y': data['y'][split_idx:],
            'feature_names': features,
            'target_name': target
        }
    
    def save_predictions(self, model_id, students, predictions):
        """
        Save predictions to MongoDB
        
        Args:
            model_id: ID of the model used
            students: List of student data dictionaries
            predictions: List of prediction results
        
        Returns:
            List of prediction document IDs
        """
        prediction_docs = []
        
        for student, pred in zip(students, predictions):
            doc = {
                'model_id': model_id,
                'student_id': student.get('student_id', str(ObjectId())),
                'student_data': student,
                'prediction': pred['prediction'],
                'confidence': pred.get('confidence', 0.0),
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            prediction_docs.append(doc)
        
        result = self.predictions.insert_many(prediction_docs)
        return [str(id) for id in result.inserted_ids]
    
    def save_analysis(self, student_data, analysis):
        """
        Save comprehensive student analysis to MongoDB
        
        Args:
            student_data: Student information
            analysis: Analysis results
        
        Returns:
            Analysis document ID
        """
        doc = {
            'student_id': student_data.get('student_id', str(ObjectId())),
            'student_data': student_data,
            'analysis': analysis,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = self.analyses.insert_one(doc)
        return str(result.inserted_id)
    
    def get_student(self, student_id):
        """Get student by ID"""
        return self.students.find_one({'student_id': student_id})
    
    def get_predictions_by_model(self, model_id, limit=100):
        """Get predictions for a specific model"""
        return list(self.predictions.find({'model_id': model_id}).limit(limit))
    
    def get_student_history(self, student_id):
        """Get all predictions and analyses for a student"""
        predictions = list(self.predictions.find({'student_id': student_id}))
        analyses = list(self.analyses.find({'student_id': student_id}))
        
        return {
            'student_id': student_id,
            'predictions': predictions,
            'analyses': analyses
        }
    
    def add_student(self, student_data):
        """Add new student to database"""
        doc = {
            **student_data,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = self.students.insert_one(doc)
        return str(result.inserted_id)
    
    def update_student(self, student_id, updates):
        """Update student information"""
        updates['updated_at'] = datetime.utcnow()
        
        result = self.students.update_one(
            {'student_id': student_id},
            {'$set': updates}
        )
        
        return result.modified_count > 0
    
    def get_analytics_summary(self):
        """Get analytics summary from MongoDB"""
        total_students = self.students.count_documents({})
        total_predictions = self.predictions.count_documents({})
        total_analyses = self.analyses.count_documents({})
        
        # Calculate average metrics
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'avg_hours_studied': {'$avg': '$hours_studied'},
                    'avg_attendance': {'$avg': '$attendance'},
                    'avg_final_score': {'$avg': '$final_score'},
                    'pass_rate': {'$avg': '$passed'}
                }
            }
        ]
        
        avg_metrics = list(self.students.aggregate(pipeline))
        
        return {
            'total_students': total_students,
            'total_predictions': total_predictions,
            'total_analyses': total_analyses,
            'average_metrics': avg_metrics[0] if avg_metrics else {}
        }
