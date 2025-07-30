import os
import numpy as np
import logging
from PIL import Image
from django.conf import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CLASSES = ["Bengin", "Malignant", "Normal"]
CLASS_LABELS = {cls: i for i, cls in enumerate(CLASSES)}
IMG_SIZE = (64, 64)

# Model paths
MODEL_PATHS = {
    'ResNet50': 'best_ResNet50.keras',
    'DenseNet121': 'best_DenseNet121.keras',
    'EnhancedCNN': 'best_EnhancedCNN.keras'
}

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

class LungCancerPredictor:
    def __init__(self):
        self.models = {}
        self.models_available = False
        self.load_models()

    def load_models(self):
        """Load all available models"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Model loading skipped.")
            return False

        try:
            for model_name, model_file in MODEL_PATHS.items():
                model_path = os.path.join(settings.BASE_DIR, model_file)
                if os.path.exists(model_path):
                    logger.info(f"Loading {model_name} from {model_path}")
                    self.models[model_name] = load_model(model_path)
                    logger.info(f"✓ Successfully loaded {model_name}")
                else:
                    logger.warning(f"✗ Model file not found: {model_path}")

            self.models_available = len(self.models) > 0
            logger.info(f"Loaded {len(self.models)} models successfully")
            return self.models_available

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Use PIL instead of OpenCV for basic image processing
            img = Image.open(image_path)

            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')

            # Resize to target size
            img = img.resize(IMG_SIZE)

            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Convert grayscale to RGB by repeating channels
            img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

            # Add batch dimension
            img_batch = np.expand_dims(img_rgb, axis=0)

            return img_batch

        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict_single_model(self, image_path, model_name):
        """Make prediction using a single model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning(f"TensorFlow not available. Returning mock prediction for {model_name}")
            # Return mock prediction for demonstration
            mock_probabilities = [0.2, 0.3, 0.5]  # Mock probabilities for Bengin, Malignant, Normal
            predicted_class_idx = np.argmax(mock_probabilities)
            predicted_class = CLASSES[predicted_class_idx]
            confidence = mock_probabilities[predicted_class_idx]

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': mock_probabilities,
                'class_names': CLASSES,
                'note': 'This is a mock prediction. Install TensorFlow to use real models.'
            }

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)

            # Make prediction
            model = self.models[model_name]
            predictions = model.predict(processed_image, verbose=0)

            # Get probabilities for each class
            probabilities = predictions[0].tolist()

            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'class_names': CLASSES
            }

        except Exception as e:
            raise Exception(f"Error making prediction with {model_name}: {str(e)}")

    def predict_ensemble(self, image_path):
        """Make predictions using all available models and ensemble them"""
        if not TENSORFLOW_AVAILABLE or not self.models:
            logger.warning("TensorFlow not available or no models loaded. Returning mock ensemble prediction")

            results = {}
            # Generate mock predictions for each model
            for model_name in MODEL_PATHS.keys():
                try:
                    result = self.predict_single_model(image_path, model_name)
                    results[model_name] = result
                except Exception as e:
                    logger.error(f"Error with {model_name}: {str(e)}")
                    continue

            if not results:
                raise Exception("No successful predictions from any model")

            # Mock ensemble prediction
            ensemble_probs = [0.25, 0.35, 0.4]  # Mock ensemble probabilities
            ensemble_class_idx = np.argmax(ensemble_probs)
            ensemble_class = CLASSES[ensemble_class_idx]
            ensemble_confidence = ensemble_probs[ensemble_class_idx]

            # Pick first model as "best"
            best_model = list(results.keys())[0] if results else 'ResNet50'

            return {
                'ensemble_prediction': {
                    'predicted_class': ensemble_class,
                    'confidence': ensemble_confidence,
                    'probabilities': ensemble_probs,
                    'class_names': CLASSES,
                    'note': 'This is a mock prediction. Install TensorFlow to use real models.'
                },
                'individual_predictions': results,
                'best_model': best_model,
                'models_used': list(results.keys()),
                'warning': 'TensorFlow not installed. These are mock predictions for demonstration.'
            }

        # Real prediction with loaded models
        results = {}
        all_predictions = []

        # Get predictions from all models
        for model_name in self.models.keys():
            try:
                result = self.predict_single_model(image_path, model_name)
                results[model_name] = result
                all_predictions.append(result['probabilities'])
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}")
                continue

        if not all_predictions:
            raise Exception("No successful predictions from any model")

        # Calculate ensemble prediction (average of all model predictions)
        ensemble_probs = np.mean(all_predictions, axis=0).tolist()
        ensemble_class_idx = np.argmax(ensemble_probs)
        ensemble_class = CLASSES[ensemble_class_idx]
        ensemble_confidence = ensemble_probs[ensemble_class_idx]

        # Find best performing model (highest confidence)
        best_model = max(results.keys(), key=lambda k: results[k]['confidence'])

        return {
            'ensemble_prediction': {
                'predicted_class': ensemble_class,
                'confidence': ensemble_confidence,
                'probabilities': ensemble_probs,
                'class_names': CLASSES
            },
            'individual_predictions': results,
            'best_model': best_model,
            'models_used': list(results.keys())
        }

    def get_model_info(self):
        """Get information about loaded models"""
        model_info = {}
        for model_name in MODEL_PATHS.keys():
            model_info[model_name] = {
                'loaded': model_name in self.models,
                'classes': CLASSES,
                'input_size': IMG_SIZE,
                'status': 'Loaded' if model_name in self.models else ('TensorFlow required' if not TENSORFLOW_AVAILABLE else 'Model file not found')
            }
        return model_info

# Global predictor instance
predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        predictor = LungCancerPredictor()
    return predictor

def predict_image(image_path):
    """Convenience function to predict image using ensemble"""
    predictor = get_predictor()
    return predictor.predict_ensemble(image_path)

def get_available_models():
    """Get list of available models"""
    predictor = get_predictor()
    return list(predictor.models.keys())

def get_model_info():
    """Get model information"""
    predictor = get_predictor()
    return predictor.get_model_info()
