# Configuration settings for the Fraud Detection System

import os
from datetime import datetime

class Config:
    """Configuration class for the fraud detection system"""
    
    # Project Information
    PROJECT_NAME = "Securities Market Fraud Detection System"
    VERSION = "1.0.0"
    DESCRIPTION = "SEBI Safe Space Initiative - Comprehensive Fraud Prevention Tool"
    
    # Data Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample_data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    REGULATORY_DB_DIR = os.path.join(DATA_DIR, 'regulatory_db')
    
    # Model Configuration
    MODEL_CONFIGS = {
        'text_classifier': {
            'max_features': 1000,
            'ngram_range': (1, 2),
            'test_size': 0.2,
            'random_state': 42
        },
        'anomaly_detector': {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42
        },
        'risk_scorer': {
            'fraud_threshold': 0.7,
            'medium_risk_threshold': 0.4,
            'weights': {
                'text_fraud_probability': 0.4,
                'extreme_sentiment': 0.1,
                'content_fraud_score': 0.3,
                'anomaly_score': 0.2
            }
        }
    }
    
    # Fraud Detection Parameters
    FRAUD_KEYWORDS = [
        'guaranteed', 'secret', 'exclusive', 'urgent', 'limited time',
        'insider info', 'guaranteed returns', 'risk-free', 'double your money',
        'get rich quick', 'easy money', 'no risk', 'high returns',
        'instant profit', 'sure shot', 'hot tip', 'insider trading',
        'pump', 'dump', 'moon', 'rocket', 'diamond hands', 'to the moon'
    ]
    
    MANIPULATIVE_PHRASES = [
        'act now', 'don\'t miss out', 'once in a lifetime', 'limited spots',
        'exclusive offer', 'invite only', 'secret method', 'insider knowledge',
        'celebrity endorsed', 'government approved', 'tax free',
        'guaranteed profit', 'no loss', 'risk free'
    ]
    
    # Regulatory Database Configuration
    SEBI_REGISTERED_BROKERS = [
        "HDFC Securities", "ICICI Direct", "Zerodha", "Angel Broking",
        "Upstox", "Groww", "5paisa", "Sharekhan", "Motilal Oswal"
    ]
    
    # Social Media Platforms
    MONITORED_PLATFORMS = [
        'telegram', 'whatsapp', 'twitter', 'facebook', 'instagram', 'youtube'
    ]
    
    # Alert Configuration
    ALERT_SETTINGS = {
        'high_risk_threshold': 0.8,
        'medium_risk_threshold': 0.5,
        'notification_channels': ['email', 'dashboard', 'api'],
        'alert_retention_days': 30
    }
    
    # API Configuration
    API_SETTINGS = {
        'rate_limit': 1000,  # requests per hour
        'timeout': 30,  # seconds
        'max_retries': 3
    }
    
    # Dashboard Configuration
    DASHBOARD_CONFIG = {
        'refresh_interval': 60,  # seconds
        'max_alerts_display': 50,
        'chart_colors': ['#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff6b6b'],
        'themes': ['light', 'dark']
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': os.path.join(DATA_DIR, 'logs', 'fraud_detection.log'),
        'max_file_size': 10 * 1024 * 1024,  # 10 MB
        'backup_count': 5
    }
    
    # Performance Metrics
    PERFORMANCE_TARGETS = {
        'text_classification_accuracy': 0.90,
        'advisor_verification_accuracy': 0.85,
        'document_authenticity_accuracy': 0.88,
        'false_positive_rate': 0.05,
        'processing_speed_texts_per_hour': 10000
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.SAMPLE_DATA_DIR,
            cls.MODELS_DIR,
            cls.REGULATORY_DB_DIR,
            os.path.join(cls.DATA_DIR, 'logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get the full path for a model file"""
        return os.path.join(cls.MODELS_DIR, f"{model_name}.pkl")
    
    @classmethod
    def get_data_path(cls, filename):
        """Get the full path for a data file"""
        return os.path.join(cls.SAMPLE_DATA_DIR, filename)

# Development Configuration
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'

# Production Configuration  
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'INFO'
    
    # Enhanced security for production
    API_SETTINGS = {
        'rate_limit': 500,  # More conservative rate limiting
        'timeout': 15,
        'max_retries': 2,
        'authentication_required': True
    }

# Testing Configuration
class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    
    # Smaller datasets for testing
    MODEL_CONFIGS = {
        'text_classifier': {
            'max_features': 100,
            'ngram_range': (1, 1),
            'test_size': 0.3,
            'random_state': 42
        }
    }
