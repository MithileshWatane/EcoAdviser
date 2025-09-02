import joblib
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

class MaterialClassifier:
    """AI model for classifying waste materials with version compatibility fixes"""
    
    def __init__(self, model_path="C:/Users/shree/Downloads/EcoAdvisor/EcoAdvisor/models/material_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train it first.")
        
        # Try multiple loading approaches for compatibility
        self.vectorizer = None
        self.classifier = None
        self.material_types = None
        
        success = self._load_model_safe(model_path)
        if not success:
            raise ValueError("Could not load the model due to version compatibility issues. Please retrain the model.")
    
    def _load_model_safe(self, model_path):
        """Safely load model with multiple fallback approaches"""
        
        # Approach 1: Try joblib with error suppression
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                saved_data = joblib.load(model_path)
                self.vectorizer = saved_data["vectorizer"]
                self.classifier = saved_data["classifier"]
                self.material_types = saved_data["material_types"]
                print("✅ Model loaded successfully with joblib")
                return True
        except Exception as e:
            print(f"❌ Joblib loading failed: {str(e)}")
        
        # Approach 2: Try with pickle directly
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.vectorizer = saved_data["vectorizer"]
                self.classifier = saved_data["classifier"]
                self.material_types = saved_data["material_types"]
                print("✅ Model loaded successfully with pickle")
                return True
        except Exception as e:
            print(f"❌ Pickle loading failed: {str(e)}")
        
        # Approach 3: Try loading individual components
        try:
            saved_data = {}
            # Try to extract what we can
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import sklearn
                print(f"Current scikit-learn version: {sklearn.__version__}")
                
                # Load with allow_pickle=True and handle errors
                saved_data = joblib.load(model_path)
                
                # Extract components that work
                if "vectorizer" in saved_data:
                    self.vectorizer = saved_data["vectorizer"]
                if "material_types" in saved_data:
                    self.material_types = saved_data["material_types"]
                
                # For classifier, try to rebuild if necessary
                if "classifier" in saved_data:
                    self.classifier = saved_data["classifier"]
                
                print("✅ Model loaded with partial recovery")
                return True
                
        except Exception as e:
            print(f"❌ Partial recovery failed: {str(e)}")
        
        return False
    
    def classify_material(self, description):
        """Classify a material based on its description"""
        if not self.classifier or not self.vectorizer:
            return "unknown"
            
        if not description or not description.strip():
            return "unknown"
            
        try:
            cleaned = self._clean_text(description)
            X = self.vectorizer.transform([cleaned])
            return self.classifier.predict(X)[0]
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return "unknown"
    
    def get_top_predictions(self, description, n=3):
        """Get top N predictions with confidence scores"""
        if not self.classifier or not self.vectorizer:
            return []
            
        if not description or not description.strip():
            return []
            
        try:
            cleaned = self._clean_text(description)
            X = self.vectorizer.transform([cleaned])
            
            # Check if classifier supports predict_proba
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X)[0]
                classes = self.classifier.classes_
                sorted_indices = probabilities.argsort()[::-1][:n]
                return [{"material": classes[i], "confidence": float(probabilities[i])} for i in sorted_indices]
            else:
                # Fallback for classifiers without probability prediction
                prediction = self.classifier.predict(X)[0]
                return [{"material": prediction, "confidence": 1.0}]
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return []
    
    def get_confidence_score(self, description):
        """Return the confidence score of the top prediction"""
        preds = self.get_top_predictions(description, n=1)
        return preds[0]["confidence"] if preds else 0.0


    def get_classification_details(self, material_type):
        """Get detailed information about a material classification"""
    
        details = {
            "plastic": {
                "category": "Recyclable",
                "disposal_method": "Clean and place in recycling bin",
                "environmental_impact": "Takes 450-1000 years to decompose",
                "recycling_tips": "Remove caps and labels, rinse clean"
            },
            "metal": {
                "category": "Recyclable",
                "disposal_method": "Place in recycling bin",
                "environmental_impact": "Highly recyclable, saves energy",
                "recycling_tips": "Clean containers, aluminum cans are valuable"
            },
            "paper": {
                "category": "Recyclable",
                "disposal_method": "Place in paper recycling",
                "environmental_impact": "Biodegradable but deforestation concern",
                "recycling_tips": "Keep dry, remove staples and tape"
            },
            "glass": {
                "category": "Recyclable",
                "disposal_method": "Glass recycling bin",
                "environmental_impact": "100% recyclable indefinitely",
                "recycling_tips": "Separate by color if required"
            },
            "electronics": {
                "category": "Special Waste",
                "disposal_method": "Take to electronics recycling center",
                "environmental_impact": "Contains toxic materials",
                "recycling_tips": "Remove batteries, find certified e-waste recycler"
            },
            "textile": {
                "category": "Reusable/Recyclable",
                "disposal_method": "Donate or textile recycling",
                "environmental_impact": "Slow to decompose, water-intensive production",
                "recycling_tips": "Donate wearable items, recycle damaged textiles"
            },
            "organic": {
                "category": "Compostable",
                "disposal_method": "Compost bin or organic waste",
                "environmental_impact": "Biodegradable, creates methane in landfills",
                "recycling_tips": "Home composting or municipal organic waste program"
            }
        }
        
        return details.get(material_type.lower(), {
            "category": "Unknown",
            "disposal_method": "Check local waste guidelines",
            "environmental_impact": "Impact varies by material",
            "recycling_tips": "Consult local recycling center"
        })
    
    def _clean_text(self, text):
        """Clean and normalize text input"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def is_model_loaded(self):
        """Check if model components are properly loaded"""
        return all([self.vectorizer is not None, 
                   self.classifier is not None, 
                   self.material_types is not None])
    def get_sample_classifications(self):
        """
        Return sample distribution of common material types
        for visualization purposes (static or mock data).
        """
        return {
            "Plastic": 40,
            "Metal": 25,
            "Paper": 15,
            "Glass": 10,
            "Electronics": 5,
            "Textile": 5
        }

    

# Alternative: Create a model retrainer class
class ModelRetrainer:
    """Utility class to retrain the model with current scikit-learn version"""
    
    def __init__(self):
        self.sample_data = [
            ("plastic bottle", "plastic"),
            ("aluminum can", "metal"),
            ("newspaper", "paper"),
            ("glass jar", "glass"),
            ("banana peel", "organic"),
            ("old phone", "electronic"),
            ("cotton shirt", "textile"),
            ("cardboard box", "paper"),
            ("steel can", "metal"),
            ("food scraps", "organic"),
        ]
    
    def retrain_model(self, save_path="material_classifier_new.pkl"):
        """Retrain model with current scikit-learn version"""
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        descriptions = [item[0] for item in self.sample_data]
        labels = [item[1] for item in self.sample_data]
        
        # Create and train components
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(descriptions)
        
        # Use RandomForest which is more stable across versions
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X, labels)
        
        # Get material types
        material_types = list(set(labels))
        
        # Save model
        model_data = {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "material_types": material_types
        }
        
        joblib.dump(model_data, save_path)
        print(f"✅ New model trained and saved to {save_path}")
        
        return save_path


# Usage example and test function
def test_classifier():
    """Test the classifier with fallback training if needed"""
    
    model_path = "C:/Users/shree/Downloads/EcoAdvisor/EcoAdvisor/models/material_classifier.pkl"
    
    try:
        classifier = MaterialClassifier(model_path)
        if classifier.is_model_loaded():
            print("✅ Model loaded successfully")
            # Test classification
            test_items = ["plastic water bottle", "aluminum soda can", "old newspaper"]
            for item in test_items:
                result = classifier.classify_material(item)
                confidence = classifier.get_confidence_score(item)
                print(f"'{item}' -> {result} (confidence: {confidence:.2f})")
        else:
            print("❌ Model not fully loaded, retraining...")
            raise Exception("Model incomplete")
            
    except Exception as e:
        print(f"Loading failed: {str(e)}")
        print("🔄 Attempting to retrain model...")
        
        # Retrain with current version
        retrainer = ModelRetrainer()
        new_model_path = retrainer.retrain_model()
        
        # Try loading the new model
        classifier = MaterialClassifier(new_model_path)
        print("✅ New model created and loaded successfully")

if __name__ == "__main__":
    test_classifier()