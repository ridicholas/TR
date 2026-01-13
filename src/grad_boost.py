import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from typing import Dict, Tuple, Optional

class BaseGradientBoostingModel(BaseEstimator, ClassifierMixin):
    """Standard Gradient Boosting model for basic prediction"""
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6, **gb_params):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.gb_params = gb_params
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the standard Gradient Boosting model"""
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            **self.gb_params
        )
        
        # Simple validation-based early stopping
        if X_val is not None and y_val is not None:
            # Train with staged prediction for early stopping
            self.model.fit(X, y)
            
            # Find best n_estimators based on validation
            val_scores = []
            for pred in self.model.staged_predict_proba(X_val):
                val_loss = -np.mean(y_val * np.log(pred[:, 1] + 1e-15) + (1-y_val) * np.log(pred[:, 0] + 1e-15))
                val_scores.append(val_loss)
            
            best_n = np.argmin(val_scores) + 1
            print(f"Early stopping at {best_n} estimators")
            
            # Retrain with best number of estimators
            self.model = GradientBoostingClassifier(
                n_estimators=min(best_n + 10, self.n_estimators),  # Add small buffer
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                **self.gb_params
            )
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def get_confidence(self, X):
        """Get confidence scores (max probability)"""
        probas = self.predict_proba(X)
        return np.maximum(probas[:, 0], probas[:, 1])


class RevAILossFunction:
    """Custom loss function for ReV-AI optimization"""
    
    def __init__(self, human_decisions, human_confidence, adb_func, 
                 contradiction_reg=0.1, asym_loss=[1, 1], confidence_calibrator=None):
        self.human_decisions = human_decisions
        self.human_confidence = human_confidence
        self.adb_func = adb_func
        self.contradiction_reg = contradiction_reg
        self.asym_loss = asym_loss
        self.confidence_calibrator = confidence_calibrator
        
    def __call__(self, y_true, y_pred):
        """
        Custom loss function for ReV-AI objective
        y_pred: raw predictions (logits)
        y_true: true labels
        """
        # Convert logits to probabilities
        y_prob = expit(y_pred)
        y_decision = (y_prob > 0.5).astype(int)
        
        # Get calibrated confidence if available, otherwise use max probability
        if self.confidence_calibrator is not None:
            try:
                model_confidence = self.confidence_calibrator.predict(y_prob)
                model_confidence = np.clip(model_confidence, 0.51, 0.99)
            except:
                model_confidence = np.maximum(y_prob, 1 - y_prob)
        else:
            model_confidence = np.maximum(y_prob, 1 - y_prob)
        
        # Calculate agreement and p(accept)
        agreement = (y_decision == self.human_decisions).astype(int)
        
        try:
            p_accept = self.adb_func(self.human_confidence, model_confidence, agreement)
        except:
            # Fallback if ADB function fails
            p_accept = np.ones_like(agreement) * 0.5
        
        # Calculate ReV-AI loss for each instance
        losses = []
        for i in range(len(y_true)):
            # Loss from accepting advice
            if y_true[i] == 1:
                loss_accept = self.asym_loss[0] * (1 - y_prob[i])  # FN cost
            else:
                loss_accept = self.asym_loss[1] * y_prob[i]  # FP cost
            
            # Loss from rejecting advice (human decision)
            if y_true[i] == 1:
                loss_reject = self.asym_loss[0] * (1 - self.human_decisions[i])
            else:
                loss_reject = self.asym_loss[1] * self.human_decisions[i]
            
            # Contradiction cost
            contradiction_cost = self.contradiction_reg * (y_decision[i] != self.human_decisions[i])
            
            # Expected loss
            expected_loss = (p_accept[i] * loss_accept + 
                           (1 - p_accept[i]) * loss_reject + 
                           contradiction_cost)
            
            losses.append(expected_loss)
        
        return np.array(losses)


class IterativeRevAIGradientBoosting(BaseEstimator, ClassifierMixin):
    """
    ReV-AI Gradient Boosting with iterative training to handle circular dependency
    """
    
    def __init__(self, contradiction_reg=0.1, asym_loss=[1, 1], 
                 n_estimators=200, learning_rate=0.05, max_depth=4,
                 calibration_method='isotonic', n_iterations=3, **gb_params):
        self.contradiction_reg = contradiction_reg
        self.asym_loss = asym_loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.calibration_method = calibration_method
        self.n_iterations = n_iterations
        self.gb_params = gb_params
        self.decision_model = None
        self.confidence_calibrator = None
        
    def fit(self, X, y, human_decisions, human_confidence, adb_func, p_y_proba,
            X_val=None, y_val=None, human_decisions_val=None, 
            human_confidence_val=None, p_y_proba_val=None):
        
        # Store data
        self.adb_func = adb_func
        self.human_decisions_train = human_decisions
        self.human_confidence_train = human_confidence
        self.y_train = y
        self.X_train = X
        
        # Split for calibration
        train_idx, calib_idx = train_test_split(
            range(len(X)), test_size=0.3, stratify=y, random_state=42
        )
        
        X_train_main = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_calib = X.iloc[calib_idx] if hasattr(X, 'iloc') else X[calib_idx]
        y_train_main = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_calib = y.iloc[calib_idx] if hasattr(y, 'iloc') else y[calib_idx]
        
        human_decisions_main = human_decisions.iloc[train_idx] if hasattr(human_decisions, 'iloc') else human_decisions[train_idx]
        human_confidence_main = human_confidence.iloc[train_idx] if hasattr(human_confidence, 'iloc') else human_confidence[train_idx]
        
        self.train_idx = train_idx
        self.calib_idx = calib_idx
        
        # Initialize with standard model
        print("Initializing ReV-AI model...")
        self.decision_model = GradientBoostingClassifier(
            n_estimators=100,  # Start smaller
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            **self.gb_params
        )
        self.decision_model.fit(X_train_main, y_train_main)
        
        # Initialize calibrator
        self._update_confidence_calibrator(X_calib, y_calib)
        
        # Iterative training
        for iteration in range(self.n_iterations):
            print(f"ReV-AI Iteration {iteration + 1}/{self.n_iterations}")
            
            # Update calibrator
            self._update_confidence_calibrator(X_calib, y_calib)
            
            # Update decision model with current calibrator
            self._update_decision_model_approximation(X_train_main, y_train_main, 
                                                    human_decisions_main, human_confidence_main)
        
        # Final calibration update
        self._update_confidence_calibrator(X_calib, y_calib)
        return self
    
    def _update_confidence_calibrator(self, X_calib, y_calib):
        """Update confidence calibrator to match decision model's accuracy"""
        decision_probs = self.decision_model.predict_proba(X_calib)[:, 1]
        decision_binary = (decision_probs > 0.5).astype(int)
        y_correct = (decision_binary == y_calib).astype(int)
        
        if self.calibration_method == 'isotonic':
            self.confidence_calibrator = IsotonicRegression(out_of_bounds='clip')
            self.confidence_calibrator.fit(decision_probs, y_correct)
        else:
            self.confidence_calibrator = LogisticRegression()
            self.confidence_calibrator.fit(decision_probs.reshape(-1, 1), y_correct)
    
    def _update_decision_model_approximation(self, X_train, y_train, human_decisions, human_confidence):
        """
        Update decision model using sample reweighting to approximate ReV-AI objective
        This is a workaround since scikit-learn doesn't support fully custom objectives
        """
        
        # Get current predictions and confidences
        current_probs = self.decision_model.predict_proba(X_train)[:, 1]
        current_decisions = (current_probs > 0.5).astype(int)
        
        # Get calibrated confidence
        if self.calibration_method == 'isotonic':
            model_confidence = self.confidence_calibrator.predict(current_probs)
        else:
            model_confidence = self.confidence_calibrator.predict_proba(current_probs.reshape(-1, 1))[:, 1]
        
        model_confidence = np.clip(model_confidence, 0.51, 0.99)
        
        # Calculate p(accept) and expected losses
        agreement = (current_decisions == human_decisions).astype(int)
        p_accept = self.adb_func(human_confidence, model_confidence, agreement)
        
        # Calculate instance weights based on ReV-AI objective
        weights = []
        for i in range(len(y_train)):
            # Expected loss components
            if y_train.iloc[i] if hasattr(y_train, 'iloc') else y_train[i] == 1:
                loss_if_wrong = self.asym_loss[0]  # FN cost
            else:
                loss_if_wrong = self.asym_loss[1]  # FP cost
            
            # Weight by acceptance probability and contradiction cost
            contradiction_penalty = self.contradiction_reg * (current_decisions[i] != human_decisions.iloc[i] if hasattr(human_decisions, 'iloc') else human_decisions[i])
            
            # Higher weight for instances where advice matters more
            weight = p_accept[i] * loss_if_wrong + contradiction_penalty + 1.0  # Base weight of 1
            weights.append(max(weight, 0.1))  # Minimum weight to avoid zero
        
        weights = np.array(weights)
        weights = weights / np.mean(weights)  # Normalize
        
        # Retrain with weighted samples
        self.decision_model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            **self.gb_params
        )
        
        self.decision_model.fit(X_train, y_train, sample_weight=weights)
    
    def get_calibrated_confidence(self, X):
        """Get calibrated confidence scores"""
        if self.decision_model is None or self.confidence_calibrator is None:
            raise ValueError("Model not trained")
        
        decision_probs = self.decision_model.predict_proba(X)[:, 1]
        
        if self.calibration_method == 'isotonic':
            confidence = self.confidence_calibrator.predict(decision_probs)
        else:
            confidence = self.confidence_calibrator.predict_proba(decision_probs.reshape(-1, 1))[:, 1]
        
        return np.clip(confidence, 0.51, 0.99)
    
    def predict_proba(self, X):
        """Standard predict_proba interface"""
        if self.decision_model is None:
            raise ValueError("Model not trained")
        return self.decision_model.predict_proba(X)
    
    def predict(self, X):
        """Standard predict interface"""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    def get_confidence(self, X):
        return self.get_calibrated_confidence(X)
    
    def evaluate_calibration(self, X, y, n_bins=10):
        """Evaluate confidence calibration quality"""
        decision_probs = self.predict_proba(X)[:, 1]
        y_pred = (decision_probs > 0.5).astype(int)
        y_correct = (y_pred == y).astype(int)
        
        confidence_scores = self.get_calibrated_confidence(X)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_correct, confidence_scores, n_bins=n_bins
        )
        
        # Expected Calibration Error
        bin_sizes = []
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            bin_sizes.append(in_bin.sum())
        
        bin_sizes = np.array(bin_sizes)
        ece = np.sum(bin_sizes / len(X) * np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            'expected_calibration_error': ece,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'bin_sizes': bin_sizes
        }


# Shared helper functions (same as before)
def apply_expected_value_filter(model, X, human_decisions, human_confidence, 
                               adb_func, contradiction_reg, p_y_proba):
    """Apply expected value filter to any model's predictions"""
    model_probas = model.predict_proba(X)[:, 1]
    model_predictions = (model_probas > 0.5).astype(int)
    model_confidence = model.get_confidence(X)
    
    filtered_predictions = []
    advice_given = []
    
    for i in range(len(X)):
        if model_predictions[i] == human_decisions[i]:
            filtered_predictions.append(model_predictions[i])
            advice_given.append(True)
        else:
            # Disagreement - calculate expected values
            p_accept = adb_func(
                np.array([human_confidence[i]]),
                np.array([model_confidence[i]]),
                np.array([False])  # disagreement
            )[0]
            
            e_loss_accept = p_y_proba[i, 1] if model_predictions[i] == 0 else p_y_proba[i, 0]
            e_loss_reject = p_y_proba[i, 1] if human_decisions[i] == 0 else p_y_proba[i, 0]
            
            e_loss_advising = p_accept * e_loss_accept + (1 - p_accept) * e_loss_reject + contradiction_reg
            e_loss_withholding = p_y_proba[i, 1] if human_decisions[i] == 0 else p_y_proba[i, 0]
            
            should_advise = e_loss_advising < e_loss_withholding
            
            if should_advise:
                filtered_predictions.append(model_predictions[i])
                advice_given.append(True)
            else:
                filtered_predictions.append(human_decisions[i])
                advice_given.append(False)
    
    return np.array(filtered_predictions), np.array(advice_given)


def simulate_human_decisions(model, X, human_decisions, human_confidence, 
                           adb_func_ground_truth, use_filter=False, 
                           adb_func_estimated=None, contradiction_reg=0.1, 
                           p_y_proba=None):
    """Simulate final human decisions given model advice"""
    if use_filter and adb_func_estimated is not None:
        model_predictions, advice_given = apply_expected_value_filter(
            model, X, human_decisions, human_confidence, adb_func_estimated, 
            contradiction_reg, p_y_proba
        )
    else:
        model_predictions = model.predict(X)
        advice_given = np.ones(len(X), dtype=bool)
    
    model_confidence = model.get_confidence(X)
    final_decisions = []
    
    for i in range(len(X)):
        if not advice_given[i]:
            final_decisions.append(human_decisions[i])
        elif model_predictions[i] == human_decisions[i]:
            final_decisions.append(human_decisions[i])
        else:
            p_accept = adb_func_ground_truth(
                np.array([human_confidence[i]]),
                np.array([model_confidence[i]]),
                np.array([False])  # disagreement
            )[0]
            
            accepts = np.random.random() < p_accept
            if accepts:
                final_decisions.append(model_predictions[i])
            else:
                final_decisions.append(human_decisions[i])
    
    return np.array(final_decisions), advice_given, model_predictions


def comprehensive_evaluation(model, X_test, y_test, human_decisions, human_confidence,
                            adb_func_estimated, adb_func_ground_truth, 
                            contradiction_reg, p_y_proba):
    """Comprehensive evaluation for any sklearn model"""
    
    # Raw predictions
    raw_predictions = model.predict(X_test)
    raw_confidences = model.get_confidence(X_test)
    raw_advice_given = np.ones(len(X_test), dtype=bool)
    
    # Filtered predictions
    filtered_predictions, filtered_advice_given = apply_expected_value_filter(
        model, X_test, human_decisions, human_confidence, adb_func_estimated, 
        contradiction_reg, p_y_proba
    )
    
    # Final decisions
    final_raw, _, _ = simulate_human_decisions(
        model, X_test, human_decisions, human_confidence, adb_func_ground_truth, 
        use_filter=False
    )
    
    final_filtered, _, _ = simulate_human_decisions(
        model, X_test, human_decisions, human_confidence, adb_func_ground_truth, 
        use_filter=True, adb_func_estimated=adb_func_estimated, 
        contradiction_reg=contradiction_reg, p_y_proba=p_y_proba
    )
    
    results = {
        'raw_predictions': raw_predictions,
        'raw_confidences': raw_confidences,
        'raw_advice_given': raw_advice_given,
        'filtered_predictions': filtered_predictions,
        'filtered_confidences': raw_confidences,
        'filtered_advice_given': filtered_advice_given,
        'final_decisions_raw': final_raw,
        'final_decisions_filtered': final_filtered
    }
    
    return results