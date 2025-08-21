import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re

class MaterialClassifier:
    """AI model for classifying waste materials based on text descriptions"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.material_types = ['plastic', 'metal', 'paper', 'glass', 'electronics', 'textile']
        self._train_model()
    
    def _create_training_data(self):
        """Create synthetic training data based on material descriptions"""
        training_data = {
            'plastic': [
                'clear plastic bottle water container',
                'plastic bag shopping grocery',
                'food container tupperware plastic',
                'plastic wrap cling film',
                'styrofoam takeout container',
                'plastic cups disposable',
                'pet bottle soda drink',
                'plastic packaging bubble wrap',
                'plastic toys children items',
                'vinyl plastic sheets',
                'polyethylene plastic film',
                'polystyrene foam packaging',
                'pvc plastic pipes',
                'plastic cutlery disposable',
                'plastic milk jug container'
            ],
            'metal': [
                'aluminum can soda beer',
                'steel can food container',
                'copper wire electrical',
                'iron metal scrap pieces',
                'brass fittings hardware',
                'tin can food storage',
                'aluminum foil kitchen',
                'metal bottle caps',
                'stainless steel cookware',
                'zinc coated metal',
                'lead pipes plumbing',
                'nickel metal parts',
                'chrome plated items',
                'cast iron cookware',
                'sheet metal scraps'
            ],
            'paper': [
                'newspaper print paper',
                'cardboard box packaging',
                'office paper white sheets',
                'magazine glossy paper',
                'paper bags brown kraft',
                'tissue paper soft',
                'paperboard packaging',
                'notebook paper lined',
                'copy paper white office',
                'corrugated cardboard',
                'paper plates disposable',
                'greeting cards paper',
                'paper cups coffee',
                'paper towels kitchen',
                'book pages printed'
            ],
            'glass': [
                'glass bottle wine beer',
                'window glass sheets',
                'glass jar food container',
                'drinking glass tumbler',
                'glass plate dish',
                'mirror glass reflective',
                'tempered glass safety',
                'glass light bulb',
                'glass vase decorative',
                'frosted glass opaque',
                'colored glass tinted',
                'glass fiber insulation',
                'laboratory glass beaker',
                'automotive glass windshield',
                'glass door panels'
            ],
            'electronics': [
                'computer monitor screen',
                'mobile phone smartphone',
                'television tv electronic',
                'circuit board pcb',
                'electronic cables wires',
                'battery rechargeable',
                'laptop computer notebook',
                'printer electronic device',
                'radio electronic music',
                'electronic components',
                'led lights electronic',
                'electronic appliance',
                'tablet electronic device',
                'electronic gaming console',
                'electronic sensors'
            ],
            'textile': [
                'cotton fabric cloth',
                'wool clothing sweater',
                'polyester fabric synthetic',
                'denim jeans clothing',
                'silk fabric luxury',
                'linen fabric natural',
                'clothing garments shirts',
                'fabric scraps textile',
                'carpet textile flooring',
                'curtain fabric window',
                'upholstery fabric furniture',
                'canvas fabric heavy',
                'textile waste clothing',
                'fabric remnants pieces',
                'bedding textile sheets'
            ]
        }
        
        descriptions = []
        labels = []
        
        for material_type, examples in training_data.items():
            for example in examples:
                descriptions.append(example)
                labels.append(material_type)
        
        return descriptions, labels
    
    def _train_model(self):
        """Train the classification model"""
        descriptions, labels = self._create_training_data()
        
        # Vectorize the text data
        X = self.vectorizer.fit_transform(descriptions)
        y = np.array(labels)
        
        # Train the classifier
        self.classifier.fit(X, y)
    
    def classify_material(self, description):
        """Classify a material based on its description"""
        if not description or not description.strip():
            return "unknown"
        
        # Clean and prepare the description
        cleaned_description = self._clean_text(description)
        
        # Vectorize the input
        X = self.vectorizer.transform([cleaned_description])
        
        # Make prediction
        prediction = self.classifier.predict(X)[0]
        
        return prediction
    
    def get_confidence_score(self, description):
        """Get confidence score for the classification"""
        if not description or not description.strip():
            return 0.0
        
        cleaned_description = self._clean_text(description)
        X = self.vectorizer.transform([cleaned_description])
        
        # Get probability scores
        probabilities = self.classifier.predict_proba(X)[0]
        max_probability = np.max(probabilities)
        
        return max_probability
    
    def get_top_predictions(self, description, n=3):
        """Get top N predictions with confidence scores"""
        if not description or not description.strip():
            return []
        
        cleaned_description = self._clean_text(description)
        X = self.vectorizer.transform([cleaned_description])
        
        probabilities = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1][:n]
        
        results = []
        for idx in sorted_indices:
            results.append({
                'material': classes[idx],
                'confidence': probabilities[idx]
            })
        
        return results
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_classification_details(self, material_type):
        """Get detailed information about a material classification"""
        details = {
            'plastic': {
                'common_types': ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS'],
                'identification': 'Look for recycling codes, flexibility, transparency',
                'recycling_difficulty': 'Medium',
                'main_sources': 'Packaging, containers, bottles'
            },
            'metal': {
                'common_types': ['Aluminum', 'Steel', 'Copper', 'Brass', 'Iron'],
                'identification': 'Magnetic properties, weight, color',
                'recycling_difficulty': 'Easy',
                'main_sources': 'Cans, construction, electronics'
            },
            'paper': {
                'common_types': ['Newsprint', 'Cardboard', 'Office paper', 'Mixed paper'],
                'identification': 'Fiber composition, coating, thickness',
                'recycling_difficulty': 'Easy',
                'main_sources': 'Packaging, printing, office waste'
            },
            'glass': {
                'common_types': ['Clear', 'Brown', 'Green', 'Mixed color'],
                'identification': 'Color, thickness, tempering',
                'recycling_difficulty': 'Easy',
                'main_sources': 'Bottles, containers, windows'
            },
            'electronics': {
                'common_types': ['Circuit boards', 'Cables', 'Batteries', 'Displays'],
                'identification': 'Components, hazardous materials',
                'recycling_difficulty': 'Hard',
                'main_sources': 'Consumer electronics, appliances'
            },
            'textile': {
                'common_types': ['Cotton', 'Polyester', 'Wool', 'Mixed fibers'],
                'identification': 'Fiber content, weave, condition',
                'recycling_difficulty': 'Medium',
                'main_sources': 'Clothing, household fabrics'
            }
        }
        
        return details.get(material_type, {})
    
    def get_sample_classifications(self):
        """Get sample data for visualization"""
        return {
            'Plastic': 35,
            'Metal': 20,
            'Paper': 25,
            'Glass': 10,
            'Electronics': 5,
            'Textile': 5
        }
