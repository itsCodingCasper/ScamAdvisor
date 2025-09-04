"""
Securities Market Fraud Detection System - Model Training
SEBI Safe Space Initiative

This module trains machine learning models for fraud detection.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """
    Model trainer class for fraud detection models
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
    
    def load_data(self):
        """Load training data from CSV files"""
        
        print("📂 Loading training data...")
        
        # Load social media data
        if os.path.exists('data/sample_data/social_media_posts.csv'):
            self.social_media_df = pd.read_csv('data/sample_data/social_media_posts.csv')
            print(f"✅ Loaded {len(self.social_media_df)} social media posts")
        else:
            print("❌ Social media data not found. Run data_generator.py first.")
            return False
        
        # Load corporate announcements
        if os.path.exists('data/sample_data/corporate_announcements.csv'):
            self.announcements_df = pd.read_csv('data/sample_data/corporate_announcements.csv')
            print(f"✅ Loaded {len(self.announcements_df)} corporate announcements")
        else:
            print("❌ Corporate announcements data not found.")
            return False
        
        return True
    
    def prepare_text_features(self, texts, vectorizer_type='tfidf', max_features=1000):
        """Prepare text features using TF-IDF vectorization"""
        
        if vectorizer_type == 'tfidf' or 'tfidf' in vectorizer_type:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='ascii'
            )
        else:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='ascii'
            )
        
        # Fit and transform texts
        text_features = vectorizer.fit_transform(texts)
        
        # Store vectorizer
        self.vectorizers[vectorizer_type] = vectorizer
        
        return text_features.toarray()
    
    def train_text_classifier(self, texts, labels, model_name='random_forest'):
        """Train text classification model"""
        
        print(f"\n🤖 Training {model_name} text classifier...")
        
        # Prepare features
        X = self.prepare_text_features(texts)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.encoders['text_labels'] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif model_name == 'logistic_regression':
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        print(f"   Training Accuracy: {train_score:.3f}")
        print(f"   Testing Accuracy: {test_score:.3f}")
        print(f"   CV Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Classification report
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Store model
        self.models[f'{model_name}_text'] = {
            'model': model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return model, test_score
    
    def train_anomaly_detector(self, features, contamination=0.1):
        """Train anomaly detection model"""
        
        print(f"\n🔍 Training anomaly detector with contamination={contamination}...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        self.scalers['anomaly_features'] = scaler
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        model.fit(X_scaled)
        
        # Predict anomalies
        anomaly_pred = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        anomalies_detected = (anomaly_pred == -1).sum()
        anomaly_rate = anomalies_detected / len(features)
        
        print(f"   Anomalies detected: {anomalies_detected} ({anomaly_rate:.1%})")
        print(f"   Mean anomaly score: {anomaly_scores.mean():.3f}")
        
        # Store model
        self.models['anomaly_detector'] = {
            'model': model,
            'contamination': contamination,
            'anomalies_detected': anomalies_detected,
            'anomaly_rate': anomaly_rate
        }
        
        return model, anomaly_pred, anomaly_scores
    
    def train_all_models(self):
        """Train all fraud detection models"""
        
        print("🚀 Starting comprehensive model training...")
        
        # Train text classification models for social media
        social_texts = self.social_media_df['text'].tolist()
        social_labels = self.social_media_df['label'].tolist()
        
        models_to_train = ['random_forest', 'logistic_regression', 'xgboost']
        
        for model_name in models_to_train:
            model, accuracy = self.train_text_classifier(social_texts, social_labels, model_name)
        
        # Train text classification for corporate announcements
        announcement_texts = self.announcements_df['text'].tolist()
        announcement_labels = self.announcements_df['label'].tolist()
        
        print(f"\n🏢 Training corporate announcement classifier...")
        
        # Prepare features for announcements
        X_ann = self.prepare_text_features(announcement_texts, 'tfidf_announcements')
        le_ann = LabelEncoder()
        y_ann = le_ann.fit_transform(announcement_labels)
        self.encoders['announcement_labels'] = le_ann
        
        # Train announcement classifier
        ann_model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(
            X_ann, y_ann, test_size=0.2, random_state=42, stratify=y_ann
        )
        
        ann_model.fit(X_ann_train, y_ann_train)
        ann_accuracy = ann_model.score(X_ann_test, y_ann_test)
        
        print(f"   Announcement Classifier Accuracy: {ann_accuracy:.3f}")
        
        self.models['announcement_classifier'] = {
            'model': ann_model,
            'test_accuracy': ann_accuracy
        }
        
        # Train anomaly detection model
        # Prepare numerical features
        numerical_features = []
        feature_names = []
        
        if 'engagement' in self.social_media_df.columns:
            numerical_features.append(self.social_media_df['engagement'].values)
            feature_names.append('engagement')
        
        if 'follower_count' in self.social_media_df.columns:
            numerical_features.append(self.social_media_df['follower_count'].values)
            feature_names.append('follower_count')
        
        if 'account_age_days' in self.social_media_df.columns:
            numerical_features.append(self.social_media_df['account_age_days'].values)
            feature_names.append('account_age_days')
        
        if numerical_features:
            feature_matrix = np.column_stack(numerical_features)
            anomaly_model, anomaly_pred, anomaly_scores = self.train_anomaly_detector(feature_matrix)
    
    def save_models(self):
        """Save all trained models"""
        
        print("\n💾 Saving trained models...")
        
        # Create models directory
        os.makedirs('data/models', exist_ok=True)
        
        # Save models
        for model_name, model_data in self.models.items():
            model_path = f'data/models/{model_name}.pkl'
            joblib.dump(model_data, model_path)
            print(f"   ✅ Saved {model_name}")
        
        # Save vectorizers
        for vec_name, vectorizer in self.vectorizers.items():
            vec_path = f'data/models/{vec_name}_vectorizer.pkl'
            joblib.dump(vectorizer, vec_path)
            print(f"   ✅ Saved {vec_name} vectorizer")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = f'data/models/{scaler_name}_scaler.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"   ✅ Saved {scaler_name} scaler")
        
        # Save encoders
        for encoder_name, encoder in self.encoders.items():
            encoder_path = f'data/models/{encoder_name}_encoder.pkl'
            joblib.dump(encoder, encoder_path)
            print(f"   ✅ Saved {encoder_name} encoder")
    
    def generate_model_summary(self):
        """Generate summary of trained models"""
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': len(self.models),
            'model_performance': {}
        }
        
        for model_name, model_data in self.models.items():
            if 'test_accuracy' in model_data:
                summary['model_performance'][model_name] = {
                    'accuracy': model_data['test_accuracy'],
                    'type': 'classification' if 'text' in model_name or 'announcement' in model_name else 'anomaly_detection'
                }
        
        # Save summary
        import json
        with open('data/models/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n📊 Model Training Summary:")
        print("=" * 50)
        for model_name, perf in summary['model_performance'].items():
            if perf['type'] == 'classification':
                print(f"{model_name}: {perf['accuracy']:.3f} accuracy")
            else:
                print(f"{model_name}: Anomaly detection model")
        
        return summary
    
    def visualize_performance(self):
        """Create visualizations of model performance"""
        
        print("\n📈 Generating performance visualizations...")
        
        # Extract performance metrics
        model_names = []
        accuracies = []
        
        for model_name, model_data in self.models.items():
            if 'test_accuracy' in model_data:
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(model_data['test_accuracy'])
        
        if model_names:
            # Create performance chart
            plt.figure(figsize=(12, 6))
            
            # Bar chart of accuracies
            bars = plt.bar(model_names, accuracies, color=['#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
            plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
            plt.ylabel('Accuracy Score')
            plt.xlabel('Models')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, accuracy in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('data/models/model_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("   ✅ Performance visualization saved to data/models/model_performance.png")

def main():
    """Main training function"""
    
    print("🎯 Securities Market Fraud Detection - Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    if not trainer.load_data():
        print("❌ Failed to load data. Please run data_generator.py first.")
        return
    
    # Train all models
    trainer.train_all_models()
    
    # Save models
    trainer.save_models()
    
    # Generate summary
    summary = trainer.generate_model_summary()
    
    # Create visualizations
    trainer.visualize_performance()
    
    print("\n🎉 Model training completed successfully!")
    print("📁 All models saved to 'data/models/' directory")
    print("📊 Training summary saved to 'data/models/training_summary.json'")

if __name__ == "__main__":
    main()