import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class ImprovedModelRetrainer:
    """Improved material classifier retrainer with better validation and overfitting prevention"""
    
    def __init__(self, csv_path="synthetic_waste_dataset.csv"):
        self.csv_path = csv_path
        self.vectorizer = None
        self.classifier = None
        self.material_types = None
        self.data_stats = {}
        
    def load_data(self):
        """Load and preprocess data from CSV file"""
        try:
            print(f"📁 Loading data from {self.csv_path}...")
            df = pd.read_csv(self.csv_path)
            
            # Check columns
            if 'description' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must contain 'description' and 'label' columns")
            
            # Clean data
            df = df.dropna(subset=['description', 'label'])
            df['description'] = df['description'].apply(self._clean_text)
            df['label'] = df['label'].str.strip().str.lower()
            df = df[df['description'].str.len() > 0]
            
            # Store data statistics
            self.data_stats = {
                'total_samples': len(df),
                'unique_labels': df['label'].nunique(),
                'class_distribution': df['label'].value_counts().to_dict(),
                'avg_description_length': df['description'].str.len().mean()
            }
            
            print(f"✅ Loaded {len(df)} samples")
            print(f"📊 Material types: {sorted(df['label'].unique())}")
            print(f"📊 Class distribution:")
            for label, count in df['label'].value_counts().items():
                print(f"   {label}: {count} samples")
            print(f"📏 Average description length: {self.data_stats['avg_description_length']:.1f} characters")
            
            return df['description'].tolist(), df['label'].tolist()
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _clean_text(self, text):
        """Clean and normalize text input"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Keep more information - only remove truly problematic characters
        text = re.sub(r'[^\w\s-]', ' ', text)  # Keep hyphens and underscores
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train_model(self, test_size=0.2, random_state=42, prevent_overfitting=True):
        """Train the material classification model with overfitting prevention"""
        
        descriptions, labels = self.load_data()
        
        if len(descriptions) == 0:
            raise ValueError("No valid training data found")
        
        self.material_types = sorted(list(set(labels)))
        print(f"🎯 Training model for {len(self.material_types)} material types")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            descriptions, labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        print(f"📈 Training set: {len(X_train)} samples")
        print(f"📉 Test set: {len(X_test)} samples")
        
        # Create vectorizer with parameters to prevent overfitting
        if prevent_overfitting:
            print("🛡️ Using overfitting prevention parameters...")
            vectorizer_params = {
                'max_features': min(2000, len(X_train)),  # Limit features
                'stop_words': 'english',
                'ngram_range': (1, 2),
                'min_df': max(2, len(X_train) // 500),  # Require words to appear in multiple docs
                'max_df': 0.8,  # Ignore very common words
                'sublinear_tf': True  # Use sublinear term frequency scaling
            }
            
            classifier_params = {
                'n_estimators': 50,  # Fewer trees
                'max_depth': 10,     # Limit tree depth
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',  # Use sqrt of features at each split
                'random_state': random_state,
                'n_jobs': -1
            }
        else:
            print("🚀 Using high-performance parameters...")
            vectorizer_params = {
                'max_features': 5000,
                'stop_words': 'english',
                'ngram_range': (1, 3),
                'min_df': 1,
                'max_df': 0.95
            }
            
            classifier_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': random_state,
                'n_jobs': -1
            }
        
        # Train vectorizer
        print("🔤 Creating text vectorizer...")
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"📊 Feature matrix shape: {X_train_vec.shape}")
        
        # Train classifier
        print("🤖 Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(**classifier_params)
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate model
        self._evaluate_model_comprehensive(X_train_vec, X_test_vec, y_train, y_test)
        
        return self.classifier, self.vectorizer
    
    def _evaluate_model_comprehensive(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        
        # Training accuracy
        y_train_pred = self.classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Test accuracy
        y_test_pred = self.classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Check for overfitting
        if train_accuracy - test_accuracy > 0.05:
            print("   ⚠️  Potential overfitting detected!")
            print("   💡 Consider using prevent_overfitting=True")
        else:
            print("   ✅ Good generalization (no significant overfitting)")
        
        # Cross-validation with more folds
        print("\n🔄 Cross-validation analysis:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=cv)
        print(f"   CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
        print(f"   CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Per-class performance
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Confusion matrix analysis
        self._analyze_confusion_matrix(y_test, y_test_pred)
        
        # Feature importance
        self._show_feature_importance()
        
        # Model complexity metrics
        self._show_model_complexity()
    
    def _analyze_confusion_matrix(self, y_true, y_pred):
        """Analyze confusion matrix to identify problem areas"""
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true)))
        
        print(f"\n🔍 Confusion Matrix Analysis:")
        
        # Find most confused pairs
        confused_pairs = []
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((true_label, pred_label, cm[i][j]))
        
        if confused_pairs:
            confused_pairs.sort(key=lambda x: x[2], reverse=True)
            print("   Most confused pairs:")
            for true_label, pred_label, count in confused_pairs[:5]:
                print(f"   • '{true_label}' → '{pred_label}': {count} errors")
        else:
            print("   ✅ No classification errors found!")
    
    def _show_feature_importance(self, top_n=15):
        """Show most important features for classification"""
        
        if hasattr(self.classifier, 'feature_importances_'):
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.classifier.feature_importances_
            
            # Get top features
            indices = np.argsort(importances)[::-1][:top_n]
            
            print(f"\n🔍 Top {top_n} Most Important Features:")
            for i, idx in enumerate(indices):
                print(f"   {i+1:2d}. '{feature_names[idx]}' ({importances[idx]:.4f})")
    
    def _show_model_complexity(self):
        """Show model complexity metrics"""
        print(f"\n📏 Model Complexity:")
        print(f"   Number of features: {len(self.vectorizer.get_feature_names_out())}")
        print(f"   Number of trees: {self.classifier.n_estimators}")
        print(f"   Max tree depth: {self.classifier.max_depth}")
        print(f"   Total tree nodes: {sum(tree.tree_.node_count for tree in self.classifier.estimators_)}")
    
    def save_model(self, save_path="models/material_classifier.pkl"):
        """Save the trained model with metadata"""
        
        if not self.classifier or not self.vectorizer:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "material_types": self.material_types,
            "model_info": {
                "sklearn_version": self._get_sklearn_version(),
                "feature_count": len(self.vectorizer.get_feature_names_out()),
                "classes": list(self.classifier.classes_),
                "data_stats": self.data_stats,
                "vectorizer_params": self.vectorizer.get_params(),
                "classifier_params": self.classifier.get_params()
            }
        }
        
        joblib.dump(model_data, save_path)
        file_size = os.path.getsize(save_path) / 1024
        print(f"\n✅ Model saved successfully!")
        print(f"   📁 Path: {save_path}")
        print(f"   📦 Size: {file_size:.1f} KB")
        
        return save_path
    
    def _get_sklearn_version(self):
        """Get current scikit-learn version"""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
    
    def test_model_extensively(self):
        """Comprehensive model testing"""
        
        if not self.classifier or not self.vectorizer:
            raise ValueError("Model must be trained before testing")
        
        # Test samples for each material type
        test_cases = {
            "plastic": [
                "clear plastic water bottle",
                "styrofoam food container cup",
                "plastic bag shopping grocery",
                "tupperware food storage container"
            ],
            "metal": [
                "aluminum soda can drink",
                "steel tin food can",
                "copper electrical wire",
                "iron metal scrap pieces"
            ],
            "paper": [
                "newspaper daily paper",
                "cardboard shipping box",
                "office printer paper white",
                "magazine glossy pages"
            ],
            "glass": [
                "glass wine bottle green",
                "broken window glass pieces",
                "mason jar glass container",
                "mirror reflective glass"
            ],
            "electronics": [
                "old smartphone mobile phone",
                "computer laptop device",
                "battery electronic waste",
                "television electronic screen"
            ],
            "textile": [
                "cotton t-shirt clothing",
                "wool winter sweater",
                "denim blue jeans pants",
                "silk fabric material"
            ]
        }
        
        print(f"\n🧪 Comprehensive Model Testing:")
        print("=" * 80)
        
        correct_predictions = 0
        total_predictions = 0
        
        for expected_class, descriptions in test_cases.items():
            print(f"\n🏷️  Testing '{expected_class}' samples:")
            print("-" * 40)
            
            for desc in descriptions:
                cleaned_desc = self._clean_text(desc)
                X = self.vectorizer.transform([cleaned_desc])
                
                prediction = self.classifier.predict(X)[0]
                
                if hasattr(self.classifier, 'predict_proba'):
                    probabilities = self.classifier.predict_proba(X)[0]
                    confidence = probabilities.max()
                    
                    # Get top 3 predictions
                    top_indices = probabilities.argsort()[::-1][:3]
                    top_predictions = [(self.classifier.classes_[i], probabilities[i]) 
                                     for i in top_indices]
                else:
                    confidence = 1.0
                    top_predictions = [(prediction, 1.0)]
                
                status = "✅" if prediction == expected_class else "❌"
                print(f"   {status} '{desc}'")
                print(f"      → Predicted: {prediction} ({confidence:.3f})")
                
                if prediction != expected_class:
                    print(f"      → Expected: {expected_class}")
                    print(f"      → Top 3: {[(p, f'{c:.3f}') for p, c in top_predictions]}")
                
                if prediction == expected_class:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        print(f"\n📊 Test Results Summary:")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")


def main():
    """Main function with improved error handling"""
    
    csv_file = "C:/Users/shree/Downloads/EcoAdvisor/EcoAdvisor/synthetic_waste_dataset.csv"
    model_save_path = "C:/Users/shree/Downloads/EcoAdvisor/EcoAdvisor/models/material_classifier.pkl"
    
    print("🚀 Starting Improved Material Classifier Training")
    print("=" * 60)
    
    try:
        # Initialize retrainer
        retrainer = ImprovedModelRetrainer(csv_file)
        
        # Train model with overfitting prevention
        print("\n🛡️ Training with overfitting prevention...")
        classifier, vectorizer = retrainer.train_model(
            test_size=0.2, 
            prevent_overfitting=True  # Set to False for maximum performance
        )
        
        # Save model
        saved_path = retrainer.save_model(model_save_path)
        
        # Comprehensive testing
        retrainer.test_model_extensively()
        
        print("\n" + "=" * 60)
        print("✅ Model training completed successfully!")
        print(f"📁 Model saved at: {saved_path}")
        
        # Integration example
        print(f"\n💡 Integration with your MaterialClassifier:")
        print("```python")
        print("from material_classifier import MaterialClassifier")
        print(f'classifier = MaterialClassifier("{model_save_path}")')
        print('result = classifier.classify_material("plastic bottle")')
        print("print(result)  # Should output: 'plastic'")
        print("```")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()