from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np

class ManualCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, method='sigmoid', cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.calibrated_classifiers_ = []
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        if self.cv == 'prefit':
            # Prefit mode - just calibrate the existing model
            self._fit_prefit(X, y)
        else:
            # CV mode - train multiple models and calibrate each
            self._fit_cv(X, y)
        return self
    
    def _fit_prefit(self, X, y):
        """Just calibrate an already-trained model"""
        # Get predictions from the already-trained model
        uncal_probs = self.base_estimator.predict_proba(X)[:, 1]
        
        # Fit calibrator
        if self.method == 'sigmoid':
            calibrator = LogisticRegression(C=1e10, solver='lbfgs')
            calibrator.fit(uncal_probs.reshape(-1, 1), y)
        else:  # isotonic
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(uncal_probs, y)
        
        # Store the calibrated model
        self.calibrated_classifiers_ = [(self.base_estimator, calibrator)]
    
    def _fit_cv(self, X, y):
        """Train multiple models with CV and calibrate each"""
        from sklearn.base import clone
        
        # Convert to numpy if needed
        if hasattr(X, 'iloc'):  # It's a DataFrame
            X_np = X.values
        else:
            X_np = X
        
        if hasattr(y, 'values'):  # It's a Series
            y_np = y.values
        else:
            y_np = y
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        self.calibrated_classifiers_ = []
        
        for train_idx, cal_idx in skf.split(X_np, y_np):
            # Split data using numpy indexing
            X_train_fold, X_cal_fold = X_np[train_idx], X_np[cal_idx]
            y_train_fold, y_cal_fold = y_np[train_idx], y_np[cal_idx]
            
            # Clone and train a fresh model for this fold
            fold_model = clone(self.base_estimator)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Get calibration predictions
            cal_probs = fold_model.predict_proba(X_cal_fold)[:, 1]
            
            # Fit calibrator
            if self.method == 'sigmoid':
                calibrator = LogisticRegression(C=1e10, solver='lbfgs')
                calibrator.fit(cal_probs.reshape(-1, 1), y_cal_fold)
            else:  # isotonic
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(cal_probs, y_cal_fold)
            
            # Store this fold's model and calibrator
            self.calibrated_classifiers_.append((fold_model, calibrator))
    
    def predict_proba(self, X):
        """Average predictions from all calibrated models"""
        all_probs = []
        
        for model, calibrator in self.calibrated_classifiers_:
            # Get uncalibrated probabilities
            uncal_probs = model.predict_proba(X)[:, 1]
            
            # Apply calibration
            if self.method == 'sigmoid':
                cal_probs = calibrator.predict_proba(uncal_probs.reshape(-1, 1))[:, 1]
            else:  # isotonic
                cal_probs = calibrator.transform(uncal_probs)
            
            all_probs.append(cal_probs)
        
        # Average across all models (or just use one if prefit)
        avg_probs = np.mean(all_probs, axis=0)
        return np.column_stack([1 - avg_probs, avg_probs])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
