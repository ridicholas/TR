import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit
from typing import Dict, Tuple, Optional

class BaseXGBoostModel(BaseEstimator, ClassifierMixin):
    """Standard XGBoost model for basic prediction"""
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6, **xgb_params):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.xgb_params = xgb_params
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the standard XGBoost model"""
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set up parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'seed': 41,
            **self.xgb_params
        }
        
        # Set up validation
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train with early stopping only if validation data provided
        early_stopping = 20 if X_val is not None and y_val is not None else None
        
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        pred = self.model.predict(dtest)
        return np.column_stack([1 - pred, pred])
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
    
    def get_confidence(self, X):
        """Get confidence scores (max probability)"""
        probas = self.predict_proba(X)
        return np.maximum(probas[:, 0], probas[:, 1])
    
    # sklearn compatibility methods for hyperparameter tuning
    def score(self, X, y):
        """Default scoring method for sklearn compatibility (accuracy)"""
        from sklearn.metrics import roc_auc_score
        y_pred_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred_proba)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            **self.xgb_params
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if key in ['n_estimators', 'learning_rate', 'max_depth']:
                setattr(self, key, value)
            else:
                self.xgb_params[key] = value
        return self


class RevAIXGBoostModel(BaseEstimator, ClassifierMixin):
    """
    ReV-AI XGBoost model - Step 1: Custom objective equivalent to standard logistic
    """
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6, **xgb_params):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.xgb_params = xgb_params
        self.model = None
        
    def fit(self, X, y, human_decisions=None, human_confidence=None, adb_func=None, p_y_proba=None,
            X_val=None, y_val=None):
        """
        Fit the ReV-AI XGBoost model with custom objective
        
        Args:
            X: Training features
            y: Training labels  
            human_decisions: Human decisions on training data
            human_confidence: Human confidence on training data
            adb_func: Algorithm discretion behavior function
            p_y_proba: Ground truth probabilities for training data
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        
        # Store the ReV-AI specific inputs for future use
        self.human_decisions_train = human_decisions
        self.human_confidence_train = human_confidence
        self.adb_func = adb_func
        self.p_y_proba_train = p_y_proba
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set up parameters - using custom objective
        params = {
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'seed': 41,
            'disable_default_eval_metric': 1,  # Required when using custom objective
            **self.xgb_params
        }
        
        # Set up validation
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train with custom objective that mimics binary:logistic
        # For now, we ignore the ReV-AI inputs and use standard objective
        
        # Custom objectives have issues with early stopping, so disable it for now
        # Once we verify equivalence, we can add proper evaluation metrics
        self.model = xgb.train(
            params, dtrain,
            obj=self._revai_simple_objective,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=None,  # Disable early stopping for custom objective
            verbose_eval=False
        )
        
        return self
    
    def _standard_logistic_objective(self, y_pred, y_true):
        """
        Exact implementation of XGBoost's binary:logistic objective
        Based on: https://github.com/dmlc/xgboost/blob/master/src/objective/regression_obj.cu
        """
        labels = y_true.get_label()
        
        # Convert raw predictions to probabilities using sigmoid
        # XGBoost uses: p = 1 / (1 + exp(-pred))
        predt = np.copy(y_pred)  # Copy to avoid modifying original
        
        # Apply sigmoid transformation - this is exactly what XGBoost does internally
        # Numerically stable sigmoid implementation
        pos_mask = predt > 0
        neg_mask = ~pos_mask
        
        prob = np.zeros_like(predt)
        prob[pos_mask] = 1.0 / (1.0 + np.exp(-predt[pos_mask]))
        prob[neg_mask] = np.exp(predt[neg_mask]) / (1.0 + np.exp(predt[neg_mask]))
        
        # XGBoost binary:logistic gradient and hessian formulas:
        # grad = p - y  (where p is sigmoid(pred), y is label)
        # hess = p * (1 - p)
        grad = prob - labels
        hess = prob * (1.0 - prob)
        
        # Add small epsilon to prevent numerical issues (XGBoost does this)
        eps = 1e-16
        hess = np.maximum(hess, eps)
        
        return grad, hess
    
    def _revai_simple_objective(self, y_pred, y_true):
        """
        Simple ReV-AI objective that adds human decisions as input but keeps behavior identical to logistic.
        
        ReV-AI formula: p̂(a)𝒱(y, ŷ) + (1-p̂(a))𝒱(y, h) + α𝕀{ŷ ≠ h}
        
        For this version:
        - p̂(a) = 1.0 (human always accepts)
        - α = 0.0 (no contradiction cost)
        - 𝒱(y, d) = logistic loss
        
        This should produce identical results to standard logistic loss.
        """
        labels = y_true.get_label()
        
        # Get human decisions for this batch (stored during fit)
        if hasattr(self, 'human_decisions_train') and self.human_decisions_train is not None:
            # For simplicity, assume human_decisions are in the same order as training data
            # In practice, you'd need to track indices properly
            human_decisions = self.human_decisions_train
        else:
            # Fallback: assume humans always predict positive class
            human_decisions = np.ones_like(labels)
        
        # Convert raw predictions to probabilities using sigmoid
        predt = np.copy(y_pred)
        pos_mask = predt > 0
        neg_mask = ~pos_mask
        
        prob = np.zeros_like(predt)
        prob[pos_mask] = 1.0 / (1.0 + np.exp(-predt[pos_mask]))
        prob[neg_mask] = np.exp(predt[neg_mask]) / (1.0 + np.exp(predt[neg_mask]))
        
        # ReV-AI parameters (set to make this identical to standard logistic)
        p_accept = 1.0  # Human always accepts AI advice
        alpha = 0.0     # No contradiction cost
        
        # ReV-AI objective components:
        # 1. Loss if human accepts AI advice: p̂(a) * 𝒱(y, ŷ)
        # 2. Loss if human rejects AI advice: (1-p̂(a)) * 𝒱(y, h)  
        # 3. Contradiction cost: α * 𝕀{ŷ ≠ h}
        
        # For binary classification with logistic loss:
        # 𝒱(y, ŷ) = -y*log(σ(ŷ)) - (1-y)*log(1-σ(ŷ))
        # Gradient of 𝒱(y, ŷ) w.r.t. raw prediction = σ(ŷ) - y
        
        # Component 1: p_accept * gradient of 𝒱(y, ŷ)
        grad_ai_loss = prob - labels
        
        # Component 2: (1-p_accept) * gradient of 𝒱(y, h)
        # Since h is constant w.r.t. our prediction, gradient is 0
        grad_human_loss = np.zeros_like(prob)
        
        # Component 3: α * gradient of 𝕀{ŷ ≠ h}
        # This is complex since it involves the prediction, but for α=0, gradient is 0
        grad_contradiction = np.zeros_like(prob)
        
        # Total gradient
        grad = p_accept * grad_ai_loss + (1 - p_accept) * grad_human_loss + grad_contradiction
        
        # Hessian (second derivative)
        # For this simple case, same as standard logistic
        hess = prob * (1.0 - prob)
        
        # Add small epsilon to prevent numerical issues
        eps = 1e-16
        hess = np.maximum(hess, eps)

            # Just before return, compute and print loss occasionally
        if hasattr(self, '_iter_count'):
            self._iter_count += 1
        else:
            self._iter_count = 0
            
        if self._iter_count % 10 == 0:
            # Simple binary logistic loss: -y*log(p) - (1-y)*log(1-p)
            eps = 1e-15
            prob_clipped = np.clip(prob, eps, 1 - eps)
            loss = np.mean(-labels * np.log(prob_clipped) - (1 - labels) * np.log(1 - prob_clipped))
            print(f"Iter {self._iter_count}: Loss = {loss:.6f}")
        
        return grad, hess
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        raw_pred = self.model.predict(dtest)
        
        # When using custom objective, XGBoost returns raw logits
        # We need to manually apply sigmoid to get probabilities
        pred = 1.0 / (1.0 + np.exp(-raw_pred))
            
        return np.column_stack([1 - pred, pred])
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
    
    def get_confidence(self, X):
        """Get confidence scores (max probability)"""
        probas = self.predict_proba(X)
        return np.maximum(probas[:, 0], probas[:, 1])




def apply_expected_value_filter(model, X, human_decisions, human_confidence,
                               adb_func, contradiction_reg, p_y_proba,
                               baseline_model=None, X_base=None):
    """Apply expected value filter to any model's predictions
    
    Args:
        baseline_model: The baseline e_y model to use for computing confidence for ADB
        X_base: Non-augmented features to use with baseline_model
    """
    model_probas = model.predict_proba(X)[:, 1]
    model_predictions = (model_probas > 0.5).astype(int)
    
    # CRITICAL: Use baseline model for confidence calculation for ADB
    # This ensures consistency with training where ADB was trained on baseline confidence
    if baseline_model is not None and X_base is not None:
        baseline_probas = baseline_model.predict_proba(X_base)[:, 1]
        model_confidence = np.maximum(baseline_probas, 1-baseline_probas)
    else:
        model_confidence = np.maximum(model_probas, 1-model_probas)
    
    filtered_predictions = []
    advice_given = []
    
    # Convert to numpy arrays to avoid pandas indexing issues
    if hasattr(human_decisions, 'values'):
        human_decisions = human_decisions.values
    if hasattr(human_confidence, 'values'):
        human_confidence = human_confidence.values
    
    for i in range(len(X)):
        if model_predictions[i] == human_decisions[i]:
            filtered_predictions.append(model_predictions[i])
            advice_given.append(True)
        else:
            # Calculate expected values
            p_accept = adb_func(
                np.array([human_confidence[i]]),
                np.array([model_confidence[i]]),
                np.array([False])
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


def comprehensive_evaluation(model, X_test, y_test, human_decisions, human_confidence,
                            adb_func_estimated, adb_func_ground_truth,
                            contradiction_reg, p_y_proba, baseline_model=None, X_test_base=None):
    """Comprehensive evaluation for models
    
    Args:
        baseline_model: The baseline e_y model for computing confidence for ADB
        X_test_base: Non-augmented x_test_non_binarized
    """
    # Raw predictions
    raw_predictions = model.predict(X_test)
    
    # Get confidence from baseline model for ADB purposes
    if baseline_model is not None and X_test_base is not None:
        probas = baseline_model.predict_proba(X_test_base)
    else:
        probas = model.predict_proba(X_test)
    raw_confidences = np.maximum(probas[:, 0], probas[:, 1])
    
    raw_advice_given = np.ones(len(X_test), dtype=bool)
    
    # Filtered predictions
    filtered_predictions, filtered_advice_given = apply_expected_value_filter(
        model, X_test, human_decisions, human_confidence, adb_func_estimated,
        contradiction_reg, p_y_proba, baseline_model=baseline_model, X_base=X_test_base
    )
    
    # Final decisions
    final_raw, _, _ = simulate_human_decisions(
        model, X_test, human_decisions, human_confidence, adb_func_ground_truth,
        use_filter=False, baseline_model=baseline_model, X_base=X_test_base
    )
    
    final_filtered, _, _ = simulate_human_decisions(
        model, X_test, human_decisions, human_confidence, adb_func_ground_truth,
        use_filter=True, adb_func_estimated=adb_func_estimated,
        contradiction_reg=contradiction_reg, p_y_proba=p_y_proba,
        baseline_model=baseline_model, X_base=X_test_base
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


def simulate_human_decisions(model, X, human_decisions, human_confidence,
                            adb_func_ground_truth, use_filter=False,
                            adb_func_estimated=None, contradiction_reg=0.1,
                            p_y_proba=None, baseline_model=None, X_base=None):
    """Simulate final human decisions given model advice
    
    Args:
        baseline_model: The baseline e_y model to use for computing confidence for ADB
        X_base: Non-augmented features (x_test_non_binarized) to use with baseline_model
    """
    # Convert pandas Series to numpy arrays to avoid indexing issues
    if hasattr(human_decisions, 'values'):
        human_decisions = human_decisions.values
    if hasattr(human_confidence, 'values'):
        human_confidence = human_confidence.values
    
    if use_filter and adb_func_estimated is not None:
        model_predictions, advice_given = apply_expected_value_filter(
            model, X, human_decisions, human_confidence, adb_func_estimated,
            contradiction_reg, p_y_proba, baseline_model=baseline_model, X_base=X_base
        )
    else:
        model_predictions = model.predict(X)
        advice_given = np.ones(len(X), dtype=bool)
    
    # CRITICAL: Use baseline model for confidence calculation for ADB
    # This ensures consistency with training where ADB was trained on baseline confidence
    if baseline_model is not None and X_base is not None:
        probas = baseline_model.predict_proba(X_base)
    else:
        # Fallback: extract non-augmented features if they exist
        if hasattr(X, 'iloc'):
            # Check if X has augmented features (last 4 columns)
            if X.shape[1] > X_base.shape[1] if X_base is not None else False:
                X_for_conf = X.iloc[:, :-4]
            else:
                X_for_conf = X
        else:
            X_for_conf = X
        probas = model.predict_proba(X_for_conf)
    
    model_confidence = np.maximum(probas[:, 0], probas[:, 1])
    
    final_decisions = []
    for i in range(len(X)):
        if not advice_given[i]:
            # No advice given, human decides alone
            final_decisions.append(human_decisions[i])
        elif model_predictions[i] == human_decisions[i]:
            # Agreement, human follows their own decision
            final_decisions.append(human_decisions[i])
        else:
            # Disagreement, sample whether human accepts the advice
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





