"""
Model Service for managing ML models metadata in MongoDB
"""
from datetime import datetime
from bson import ObjectId

class ModelService:
    """Service for managing ML model metadata"""
    
    def __init__(self, db):
        self.db = db
        self.models = db.models
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes"""
        self.models.create_index('model_id', unique=True)
        self.models.create_index('created_at')
        self.models.create_index('model_type')
    
    def save_model_metadata(self, model_type, features, target, metrics, hyperparameters):
        """
        Save model metadata to MongoDB
        
        Args:
            model_type: Type of ML model
            features: List of feature names
            target: Target variable name
            metrics: Dictionary of performance metrics
            hyperparameters: Model hyperparameters
        
        Returns:
            Model document ID
        """
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        doc = {
            'model_id': model_id,
            'model_type': model_type,
            'features': features,
            'target': target,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'status': 'active',
            'version': '1.0',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        self.models.insert_one(doc)
        return model_id
    
    def get_model_metadata(self, model_id):
        """Get model metadata by ID"""
        model = self.models.find_one({'model_id': model_id})
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        return model
    
    def list_models(self, limit=50):
        """List all models"""
        models = list(self.models.find({}).sort('created_at', -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for model in models:
            model['_id'] = str(model['_id'])
        
        return models
    
    def update_metrics(self, model_id, new_metrics):
        """Update model metrics"""
        result = self.models.update_one(
            {'model_id': model_id},
            {
                '$set': {
                    'metrics': new_metrics,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
    
    def deactivate_model(self, model_id):
        """Deactivate a model"""
        result = self.models.update_one(
            {'model_id': model_id},
            {
                '$set': {
                    'status': 'inactive',
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
    
    def get_best_model(self, model_type=None, metric='accuracy'):
        """Get the best performing model"""
        query = {'status': 'active'}
        if model_type:
            query['model_type'] = model_type
        
        models = list(self.models.find(query))
        
        if not models:
            return None
        
        # Sort by specified metric
        best_model = max(models, key=lambda m: m.get('metrics', {}).get(metric, 0))
        best_model['_id'] = str(best_model['_id'])
        
        return best_model
    
    def compare_models(self, model_ids):
        """Compare multiple models"""
        models = []
        
        for model_id in model_ids:
            model = self.get_model_metadata(model_id)
            if model:
                model['_id'] = str(model['_id'])
                models.append(model)
        
        return models
