import pandas as pd
import numpy as np
import pickle
import os
import yaml
import time
from human import Human
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import json
from typing import Dict, List, Any, Optional, Tuple
import anthropic  # Added for Claude API
from datetime import datetime

class LLMModel:
    """Standard LLM model for basic prediction with confidence"""
    
    def __init__(self, dataset_name: str, model_name: str = "claude-3-haiku-20240307"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.training_examples = []
        self.feature_names = []
        self.api_key = SECRET_KEY
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Claude client"""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # Store all training data for query-aware retrieval
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)
        
        # Normalize features for distance calculation
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)

    def _select_examples_for_instance(self, instance: Dict, k=8):
        """Select k most relevant examples for this specific instance"""
        # Convert instance to scaled features
        instance_array = np.array([instance[f] for f in self.feature_names]).reshape(1, -1)
        instance_scaled = self.scaler.transform(instance_array)
        
        # Find k nearest neighbors
        distances = np.linalg.norm(self.X_train_scaled - instance_scaled, axis=1)
        nearest_indices = np.argsort(distances)[:k*2]  # Get more candidates
        
        # Balance classes among nearest neighbors
        pos_indices = [i for i in nearest_indices if self.y_train.iloc[i] == 1][:k//2]
        neg_indices = [i for i in nearest_indices if self.y_train.iloc[i] == 0][:k//2]
        
        selected_indices = pos_indices + neg_indices
        
        return [{
            'features': self.X_train.iloc[idx].to_dict(),
            'label': int(self.y_train.iloc[idx])
        } for idx in selected_indices]
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for test instances"""
        probabilities = []
        
        for _, instance in X_test.iterrows():
            prob = self._predict_single_instance(instance.to_dict())
            probabilities.append([1-prob, prob])  # [prob_class_0, prob_class_1]
            
        return np.array(probabilities)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class labels for test instances"""
        probas = self.predict_proba(X_test)
        return (probas[:, 1] > 0.5).astype(int)
    
    def __getstate__(self):
        """Custom pickling to exclude the client"""
        state = self.__dict__.copy()
        # Remove the unpicklable client
        state['_client'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore state"""
        self.__dict__.update(state)
        self._client = None
    
    def comprehensive_evaluation(self, X_test: pd.DataFrame, human_decisions: pd.Series, 
                                human_confidences: pd.Series, adb_func_estimated, adb_func_ground_truth, 
                                contradiction_reg: float, e_y_model = None) -> Dict:
        """
        Comprehensive evaluation returning all four outcomes:
        1. Raw LLM advice (no filter)
        2. Filtered LLM advice (with expected value filter using estimated ADB)
        3. Final decision from raw advice (after human acceptance/rejection using ground truth ADB)
        4. Final decision from filtered advice (after human acceptance/rejection using ground truth ADB)
        """
        results = {
            'raw_predictions': [],
            'raw_confidences': [],
            'raw_advice_given': [],
            'filtered_predictions': [], 
            'filtered_confidences': [],
            'filtered_advice_given': [],
            'final_decisions_raw': [],
            'final_decisions_filtered': []
        }
        
        # Get ground truth probabilities if model provided
        p_y_probabilities = None
        if e_y_model is not None:
            p_y_probabilities = e_y_model.predict_proba(X_test)

        now = datetime.now()

        for i, (_, instance) in enumerate(X_test.iterrows()):
            human_decision = human_decisions.iloc[i]
            human_confidence = human_confidences[i]
            instance_p_y = p_y_probabilities[i] if p_y_probabilities is not None else None
            if i%10==0:
                print(f'processed{i}')
            
            # Get raw LLM prediction
            if i%18==0 and i != 0:
                print(f'checking time since last: {(datetime.now()-now).total_seconds()}')
                if (datetime.now()-now).total_seconds() < 60:
                    print(f'waiting: {61 - (datetime.now()-now).total_seconds()}')
                    time.sleep(61 - (datetime.now()-now).total_seconds())
                now = datetime.now()
            raw_confidence = self._predict_single_instance(instance.to_dict())
            raw_prediction = 1 if raw_confidence > 0.5 else 0
            raw_confidence = raw_confidence if raw_prediction == 1 else 1 - raw_confidence
            
            # Apply expected value filter using estimated ADB
            filter_result = self.apply_expected_value_filter(
                raw_confidence, raw_prediction, human_decision, 
                human_confidence, adb_func_estimated, contradiction_reg, instance_p_y
            )
            
            # Store raw and filtered advice
            results['raw_predictions'].append(raw_prediction)
            results['raw_confidences'].append(raw_confidence)
            results['raw_advice_given'].append(True)  # Standard LLM always gives advice
            
            results['filtered_predictions'].append(filter_result['final_advice'])
            results['filtered_confidences'].append(raw_confidence)  # Confidence unchanged
            results['filtered_advice_given'].append(filter_result['give_advice'])
            
            # Simulate human acceptance/rejection for final decisions using GROUND TRUTH ADB
            
            # Raw advice final decision
            if raw_prediction == human_decision:
                # Agreement case - human follows through regardless
                final_raw = human_decision
            else:
                # Disagreement case - use ground truth ADB model
                agreement = False
                p_accept = adb_func_ground_truth(
                    np.array([human_confidence]), 
                    np.array([raw_confidence]), 
                    np.array([agreement])
                )[0]
                accepts = np.random.random() < p_accept
                final_raw = raw_prediction if accepts else human_decision
            
            # Filtered advice final decision  
            if not filter_result['give_advice']:
                # No advice given - human decides alone
                final_filtered = human_decision
            elif filter_result['final_advice'] == human_decision:
                # Agreement case
                final_filtered = human_decision  
            else:
                # Filtered advice disagrees - use ground truth ADB model
                agreement = False
                p_accept = adb_func_ground_truth(
                    np.array([human_confidence]), 
                    np.array([raw_confidence]), 
                    np.array([agreement])
                )[0]
                accepts = np.random.random() < p_accept
                final_filtered = filter_result['final_advice'] if accepts else human_decision
            
            results['final_decisions_raw'].append(final_raw)
            results['final_decisions_filtered'].append(final_filtered)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
            
        return results

    def apply_expected_value_filter(self, llm_confidence: float, llm_prediction: int, 
                                   human_decision: int, human_confidence: float, 
                                   adb_func, contradiction_reg: float, p_y: np.ndarray = None) -> Dict:
        """
        Post-process LLM response using expected value calculation.
        Returns final advice decision based on economic analysis.
        
        Args:
            llm_confidence: LLM's confidence in its prediction (0-1)
            llm_prediction: LLM's predicted class (0 or 1) 
            human_decision: Human's original decision (0 or 1)
            human_confidence: Human's confidence level (0-1)
            adb_func: Function to calculate acceptance probability
            contradiction_reg: Cost of contradicting human
            p_y: Ground truth probabilities [p_class_0, p_class_1]
        """
        
        # If LLM agrees with human, no filtering needed
        if llm_prediction == human_decision:
            return {
                'give_advice': True,  # Could be True or False, outcome is same
                'final_advice': llm_prediction,
                'reasoning': 'Agreement case - outcome identical whether advice given or not'
            }
        
        # For disagreement cases, calculate expected values
        agreement = False  # LLM contradicts human
        
        # Get acceptance probability from ADB model
        p_accept = adb_func(
            np.array([human_confidence]), 
            np.array([llm_confidence]), 
            np.array([agreement])
        )[0]
        
        # Use LLM confidence as proxy for ground truth if p_y not available
        if p_y is None:
            if llm_prediction == 1:
                p_y = np.array([1 - llm_confidence, llm_confidence])
            else:
                p_y = np.array([llm_confidence, 1 - llm_confidence])
        
        # Expected loss from giving advice
        # If accepted: loss based on LLM prediction accuracy
        # If rejected: loss based on human decision accuracy  
        # Plus contradiction cost
        e_loss_accept = p_y[1] if llm_prediction == 0 else p_y[0]  # Error if LLM wrong
        e_loss_reject = p_y[1] if human_decision == 0 else p_y[0]  # Error if human wrong
        
        e_loss_advising = p_accept * e_loss_accept + (1 - p_accept) * e_loss_reject + contradiction_reg
        
        # Expected loss from withholding (human decides alone)
        e_loss_withholding = p_y[1] if human_decision == 0 else p_y[0]
        
        # Give advice only if it reduces expected loss
        should_advise = e_loss_advising < e_loss_withholding
        
        return {
            'give_advice': should_advise,
            'final_advice': llm_prediction if should_advise else human_decision,
            'reasoning': f'Expected loss: advising={e_loss_advising:.3f}, withholding={e_loss_withholding:.3f}, p_accept={p_accept:.3f}'
        }

    def _predict_single_instance(self, instance: Dict) -> float:
        """Predict probability for a single instance using Claude"""
        
        # Create few-shot examples
        examples_text = ""
        for i, ex in enumerate(self.training_examples):
            examples_text += f"Example {i+1}:\n"
            examples_text += f"Features: {ex['features']}\n"
            examples_text += f"Label: {ex['label']}\n\n"
        
        prompt = f"""Training Examples:
{examples_text}

New Instance:
Features: {instance}

Based on the training examples, predict the probability that this new instance has label 1.

***CRITICAL: PROVIDE YOUR ANSWER AS A SINGLE NUMBER BETWEEN 0 and 1 (e.g. 0.##). DO NOT PROVIDE ANY ADDITIONAL TEXT OR REASONING, ONLY RETURN THE PROBABILITY!"""
        
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=50,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract probability from response
            response_text = message.content[0].text.strip()
            probability = float(response_text)
            return np.clip(probability, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in LLM prediction: {e}")
            return 0.5  # Default to neutral probability


class RevLLMModel:
    """Augmented LLM model that considers human factors for advice generation"""
    
    def __init__(self, dataset_name: str, human_profile: str, contradiction_reg: float, 
                 model_name: str = "claude-3-haiku-20240307"):
        self.dataset_name = dataset_name
        self.human_profile = human_profile
        self.contradiction_reg = contradiction_reg
        self.model_name = model_name
        self.training_examples = []
        self.human_examples = []
        self.adb_model = None
        self.adb_examples = []
        self.api_key = SECRET_KEY
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Claude client"""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            human_decisions: pd.Series, human_confidences: pd.Series, adb_model):
        """Store training data and human behavioral examples"""
        self.adb_model = adb_model
        
        # Sample diverse examples for few-shot learning
        pos_indices = np.where(y_train == 1)[0]
        neg_indices = np.where(y_train == 0)[0]
        
        n_pos = min(4, len(pos_indices))
        n_neg = min(4, len(neg_indices))
        
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
        selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
        selected_indices = np.concatenate([selected_pos, selected_neg])
        
        for idx in selected_indices:
            self.training_examples.append({
                'features': X_train.iloc[idx].to_dict(),
                'true_label': int(y_train.iloc[idx]),
                'human_decision': int(human_decisions.iloc[idx]),
                'human_confidence': float(human_confidences[idx]),
                'human_was_correct': int(y_train.iloc[idx]) == int(human_decisions.iloc[idx])
            })
        
        # Generate ADB model examples for the LLM to understand acceptance behavior
        self.adb_examples = self._generate_adb_examples(human_confidences, adb_model)
    
    def _generate_adb_examples(self, human_confidences: pd.Series, adb_model) -> List[Dict]:
        """Generate examples showing how humans accept/reject advice based on confidences"""
        examples = []
        
        # Create a grid of confidence combinations to show the ADB model behavior
        human_conf_levels = [0.2, 0.5, 0.8, 0.95]  # Low to high human confidence
        model_conf_levels = [0.3, 0.6, 0.9]         # Low to high model confidence  
        
        # Agreement cases - human always "accepts" because AI validates their choice
        for h_conf in human_conf_levels:
            for m_conf in model_conf_levels:
                examples.append({
                    'human_confidence': h_conf,
                    'model_confidence': m_conf,
                    'agreement': True,
                    'acceptance_probability': 1.0,  # Always 1.0 for agreement
                    'likely_outcome': 'accept'
                })
        
        # Disagreement cases - use ADB model to predict acceptance
        for h_conf in human_conf_levels:
            for m_conf in model_conf_levels:
                # Get acceptance probability from ADB model for disagreement cases
                try:
                    # Call it exactly like tr.py does: self.fA(self.conf_human, conf_model, agreement, self.asym_scaler, self.asym_scaling)
                    # adb_model here is the ADB_model_wrapper function
                    accept_prob = adb_model(
                        np.array([h_conf]),     # human_conf
                        np.array([m_conf]),     # model_conf  
                        np.array([False]),      # agreement=False for disagreement
                        0,                      # asym_scaling=0 (standard case)
                        0                       # asym_scaler=0 (standard case)
                    )[0]
                except Exception as e:
                    print(f"ADB model error: {e}")
                    accept_prob = 0.5  # Default if ADB model fails
                
                examples.append({
                    'human_confidence': h_conf,
                    'model_confidence': m_conf,
                    'agreement': False,
                    'acceptance_probability': float(accept_prob),
                    'likely_outcome': 'accept' if accept_prob > 0.5 else 'reject'
                })
        
        return examples
    
    def predict(self, X_test: pd.DataFrame, human_decisions: pd.Series, 
                with_reset: bool = False, conf_human: Optional[pd.Series] = None, 
                p_y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[int], pd.Series]:
        """Generate advice for test instances"""
        
        advice_predictions = []
        covered_indices = []
        
        for i, (_, instance) in enumerate(X_test.iterrows()):
            human_decision = human_decisions.iloc[i]
            human_conf = conf_human[i] if conf_human is not None else 0.5
            
            advice_result = self._generate_advice_for_instance(
                instance.to_dict(), human_decision, human_conf, p_y[i] if p_y is not None else None
            )
            
            if advice_result['give_advice']:
                advice_predictions.append(advice_result['advice'])
                covered_indices.append(i)
            else:
                advice_predictions.append(human_decision)
        
        return np.array(advice_predictions), covered_indices, human_decisions
    
    def predictHumanInLoop(self, X_test: pd.DataFrame, human_decisions: pd.Series,
                          conf_human: pd.Series, fA_func, with_reset: bool = False,
                          p_y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Predict final decisions considering human acceptance behavior"""
        
        final_decisions = []
        coverage = []
        
        for i, (_, instance) in enumerate(X_test.iterrows()):
            human_decision = human_decisions.iloc[i]
            human_conf = conf_human.iloc[i]
            
            advice_result = self._generate_advice_for_instance(
                instance.to_dict(), human_decision, human_conf, p_y[i] if p_y is not None else None
            )
            
            if advice_result['give_advice']:
                # Simulate human acceptance using their ADB model
                agreement = advice_result['advice'] == human_decision
                p_accept = fA_func(
                    np.array([human_conf]), 
                    np.array([advice_result['confidence']]), 
                    np.array([agreement])
                )[0]
                
                # Simulate acceptance decision
                accepts = np.random.random() < p_accept
                
                if accepts:
                    final_decisions.append(advice_result['advice'])
                    coverage.append(1)
                else:
                    final_decisions.append(human_decision)
                    coverage.append(0)
            else:
                final_decisions.append(human_decision)
                coverage.append(-1)
        
        return np.array(final_decisions), np.array(coverage), human_decisions
    
    def comprehensive_evaluation(self, X_test: pd.DataFrame, human_decisions: pd.Series, 
                                human_confidences: pd.Series, adb_func_estimated, adb_func_ground_truth, 
                                contradiction_reg: float, e_y_model = None) -> Dict:
        """
        Comprehensive evaluation returning all four outcomes:
        1. Raw RevLLM advice (qualitative strategic decision)
        2. Filtered RevLLM advice (with expected value filter override using estimated ADB)
        3. Final decision from raw advice (after human acceptance/rejection using ground truth ADB)
        4. Final decision from filtered advice (after human acceptance/rejection using ground truth ADB)
        """
        results = {
            'raw_predictions': [],
            'raw_confidences': [],
            'raw_advice_given': [],
            'filtered_predictions': [], 
            'filtered_confidences': [],
            'filtered_advice_given': [],
            'final_decisions_raw': [],
            'final_decisions_filtered': []
        }
        
        # Get ground truth probabilities if model provided
        p_y_probabilities = None
        if e_y_model is not None:
            p_y_probabilities = e_y_model.predict_proba(X_test)
        
        now = datetime.now()
        for i, (_, instance) in enumerate(X_test.iterrows()):
            human_decision = human_decisions.iloc[i]
            human_confidence = human_confidences[i]
            instance_p_y = p_y_probabilities[i] if p_y_probabilities is not None else None
            
            # Get raw RevLLM strategic decision
            if i%10==0:
                print(f'processed{i}')
            
            # Get raw LLM prediction
            if i%18==0 and i != 0:
                print(f'checking time since last: {(datetime.now()-now).total_seconds()}')
                if (datetime.now()-now).total_seconds() < 60:
                    print(f'waiting: {61 - (datetime.now()-now).total_seconds()}')
                    time.sleep(61 - (datetime.now()-now).total_seconds())
                now = datetime.now()
            raw_result = self._generate_advice_for_instance(
                instance.to_dict(), human_decision, human_confidence, instance_p_y
            )
            
            if raw_result['give_advice']:
                raw_prediction = raw_result['advice'] 
                raw_confidence = raw_result['confidence']
                raw_advice_given = True
            else:
                # RevLLM chose not to give advice - need to get its prediction anyway
                # Make a separate prediction call
                raw_confidence = self._predict_single_instance(instance.to_dict())
                raw_prediction = 1 if raw_confidence > 0.5 else 0
                raw_confidence = raw_confidence if raw_prediction == 1 else 1 - raw_confidence
                raw_advice_given = False
            
            # Apply expected value filter to override RevLLM's decision using estimated ADB
            filter_result = self.apply_expected_value_filter(
                raw_confidence, raw_prediction, human_decision, 
                human_confidence, adb_func_estimated, contradiction_reg, instance_p_y
            )
            
            # Store raw and filtered advice
            results['raw_predictions'].append(raw_prediction)
            results['raw_confidences'].append(raw_confidence)
            results['raw_advice_given'].append(raw_advice_given)
            
            results['filtered_predictions'].append(filter_result['final_advice'])
            results['filtered_confidences'].append(raw_confidence)
            results['filtered_advice_given'].append(filter_result['give_advice'])
            
            # Simulate human acceptance/rejection for final decisions using GROUND TRUTH ADB
            
            # Raw advice final decision
            if not raw_advice_given:
                # RevLLM chose not to advise
                final_raw = human_decision
            elif raw_prediction == human_decision:
                # Agreement case
                final_raw = human_decision
            else:
                # RevLLM disagreement case - use ground truth ADB model
                agreement = False
                p_accept = adb_func_ground_truth(
                    np.array([human_confidence]), 
                    np.array([raw_confidence]), 
                    np.array([agreement])
                )[0]
                accepts = np.random.random() < p_accept
                final_raw = raw_prediction if accepts else human_decision
            
            # Filtered advice final decision
            if not filter_result['give_advice']:
                # Filter decided not to advise
                final_filtered = human_decision
            elif filter_result['final_advice'] == human_decision:
                # Agreement case
                final_filtered = human_decision
            else:
                # Filtered advice disagrees - use ground truth ADB model
                agreement = False
                p_accept = adb_func_ground_truth(
                    np.array([human_confidence]), 
                    np.array([raw_confidence]), 
                    np.array([agreement])
                )[0]
                accepts = np.random.random() < p_accept
                final_filtered = filter_result['final_advice'] if accepts else human_decision
            
            results['final_decisions_raw'].append(final_raw)
            results['final_decisions_filtered'].append(final_filtered)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
            
        return results

    def _predict_single_instance(self, instance: Dict) -> float:
        """Get RevLLM's raw prediction confidence without strategic considerations"""
        # Simplified prompt just for prediction
        prompt = f"""Based on these features: {instance}
        
What's the probability this instance has label 1?

***CRITICAL: PROVIDE YOUR ANSWER AS A SINGLE NUMBER BETWEEN 0 and 1 (e.g. 0.##). DO NOT PROVIDE ANY ADDITIONAL TEXT OR REASONING, ONLY RETURN THE PROBABILITY!"""
        
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=50,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text.strip()
            probability = float(response_text)
            return np.clip(probability, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in RevLLM prediction: {e}")
            return 0.5

    def _generate_advice_for_instance(self, instance: Dict, human_decision: int, 
                                    human_confidence: float, p_y: Optional[np.ndarray] = None) -> Dict:
        """Generate advice for a single instance using Claude reasoning"""
        
        # Create few-shot training examples
        training_examples_text = ""
        if hasattr(self, 'training_examples') and len(self.training_examples) > 0:
            training_examples_text = "Training Examples:\n"
            for i, ex in enumerate(self.training_examples[:8]):  # Limit to 8 examples
                accuracy_note = "✓ CORRECT" if ex['human_was_correct'] else "✗ INCORRECT" 
                training_examples_text += f"Example {i+1}:\n"
                training_examples_text += f"Features: {ex['features']}\n"
                training_examples_text += f"Ground Truth: {ex['true_label']}\n"
                training_examples_text += f"Human decided: {ex['human_decision']} (confidence: {ex['human_confidence']:.2f}) - {accuracy_note}\n\n"
            training_examples_text += "\n"
        
        # Create ADB model examples
        adb_behavior_text = ""
        if hasattr(self, 'adb_examples') and len(self.adb_examples) > 0:
            adb_behavior_text = "Human Acceptance Behavior Patterns:\n"
            adb_behavior_text += "When your advice MATCHES what the human was going to do:\n"
            adb_behavior_text += "• Human always follows through (no decision needed)\n\n"
            
            adb_behavior_text += "When your advice CONTRADICTS what the human was going to do:\n"
            
            # Show disagreement examples in order of human confidence
            disagreement_examples = [ex for ex in self.adb_examples if not ex['agreement']]
            disagreement_examples.sort(key=lambda x: x['human_confidence'])
            
            for ex in disagreement_examples[:6]:  # Show representative examples
                adb_behavior_text += f"• Human conf: {ex['human_confidence']:.1f}, Your conf: {ex['model_confidence']:.1f} → accept prob: {ex['acceptance_probability']:.2f}\n"
            
            adb_behavior_text += f"\nGeneral patterns: High contradiction cost ({self.contradiction_reg}) means be selective.\n\n"

        prompt = f"""You are an AI advisor working with a {self.human_profile} human decision-maker.
    Dataset: {self.dataset_name}
    Contradiction cost: {self.contradiction_reg}

    Your goal: Maximize team accuracy while minimizing unnecessary contradictions.

    {training_examples_text}{adb_behavior_text}

    Current Case:
    Features: {instance}
    Human will decide: {human_decision} (confidence: {human_confidence:.2f})

    Based on the training examples and acceptance patterns above, decide whether to give advice:

    1. What's your TRUE confidence about the correct answer? (0-1)
    2. What do you think the correct answer is? (0 or 1)  
    3. Should you give advice to this human?

    Strategic considerations:
    - If you agree with the human: giving advice has no downside but also no benefit
    - If you disagree: only give advice if you're confident enough that the benefit outweighs the contradiction cost
    - Consider the human's confidence level and the acceptance patterns shown above
    - Higher contradiction costs mean you should be more selective about contradicting humans
    - Be truthful about your confidence - don't manipulate it strategically
    - Look at the training examples to understand when this human tends to be right or wrong

    **CRITICAL: Respond in JSON format:
    {{
        "confidence": 0.0-1.0 (your TRUE confidence about the correct answer),
        "predicted_answer": 0 or 1,
        "give_advice": true/false,
        "advice": 0 or 1 (if giving advice, otherwise same as human_decision),
        "reasoning": "brief explanation of why you chose to give/withhold advice based on patterns above"
    }}"""
        
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=300,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text.strip()
            result = json.loads(response_text)
            
            # Validate and clean result
            if not result.get('give_advice', False):
                return {'give_advice': False}
            
            return {
                'give_advice': True,
                'advice': int(result.get('advice', human_decision)),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            print(f"Error in RevLLM advice generation: {e}")
            # Default to not giving advice on error
            return {'give_advice': False}
    
    def apply_expected_value_filter(self, llm_confidence: float, llm_prediction: int, 
                                   human_decision: int, human_confidence: float, 
                                   adb_func, contradiction_reg: float, p_y: np.ndarray = None) -> Dict:
        """
        Post-process LLM response using expected value calculation.
        Returns final advice decision based on economic analysis.
        
        Args:
            llm_confidence: LLM's confidence in its prediction (0-1)
            llm_prediction: LLM's predicted class (0 or 1) 
            human_decision: Human's original decision (0 or 1)
            human_confidence: Human's confidence level (0-1)
            adb_func: Function to calculate acceptance probability
            contradiction_reg: Cost of contradicting human
            p_y: Ground truth probabilities [p_class_0, p_class_1]
        """
        
        # If LLM agrees with human, no filtering needed
        if llm_prediction == human_decision:
            return {
                'give_advice': True,  # Could be True or False, outcome is same
                'final_advice': llm_prediction,
                'reasoning': 'Agreement case - outcome identical whether advice given or not'
            }
        
        # For disagreement cases, calculate expected values
        agreement = False  # LLM contradicts human
        
        # Get acceptance probability from ADB model
        p_accept = adb_func(
            np.array([human_confidence]), 
            np.array([llm_confidence]), 
            np.array([agreement])
        )[0]
        
        # Use LLM confidence as proxy for ground truth if p_y not available
        if p_y is None:
            if llm_prediction == 1:
                p_y = np.array([1 - llm_confidence, llm_confidence])
            else:
                p_y = np.array([llm_confidence, 1 - llm_confidence])
        
        # Expected loss from giving advice
        # If accepted: loss based on LLM prediction accuracy
        # If rejected: loss based on human decision accuracy  
        # Plus contradiction cost
        e_loss_accept = p_y[1] if llm_prediction == 0 else p_y[0]  # Error if LLM wrong
        e_loss_reject = p_y[1] if human_decision == 0 else p_y[0]  # Error if human wrong
        
        e_loss_advising = p_accept * e_loss_accept + (1 - p_accept) * e_loss_reject + contradiction_reg
        
        # Expected loss from withholding (human decides alone)
        e_loss_withholding = p_y[1] if human_decision == 0 else p_y[0]
        
        # Give advice only if it reduces expected loss
        should_advise = e_loss_advising < e_loss_withholding
        
        return {
            'give_advice': should_advise,
            'final_advice': llm_prediction if should_advise else human_decision,
            'reasoning': f'Expected loss: advising={e_loss_advising:.3f}, withholding={e_loss_withholding:.3f}, p_accept={p_accept:.3f}'
        }
    
    def __getstate__(self):
        """Custom pickling to exclude the client"""
        state = self.__dict__.copy()
        # Remove the unpicklable client
        state['_client'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore state"""
        self.__dict__.update(state)
        self._client = None