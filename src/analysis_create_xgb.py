import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rev_xg import BaseXGBoostModel, comprehensive_evaluation
import pickle
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error
from numpy import mean 
import progressbar
from run import ADB
import xgboost as xgb
from run import evaluate_adb_model
from copy import deepcopy
import os
import inspect
#token = tabpfn_client.get_access_token()
#from tabpfn_client import init, TabPFNClassifier
#import tabpfn_client
#token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiN2QxYzk5YWEtOWU1Zi00ZmRhLTk4ZGItMTQ2YzkwYmVjYzY0IiwiZXhwIjoxNzkyNTI5NDg0fQ.v67xBYN3xgSnH8K4mPx7xPzHpmSVN8qNGpr26emgly4'
#tabpfn_client.set_access_token(token)

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def noADB(human_conf, model_conf, agreement):
    return np.ones(len(human_conf))

def load_datasets(dataset, run_num):
    x_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtrain.csv', index_col=0).reset_index(drop=True)
    y_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/ytrain.csv', index_col=0).iloc[:, 0].reset_index(drop=True)
    x_train_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtrain_non_binarized.csv', index_col=0).reset_index(drop=True)
    x_learning_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xlearning_non_binarized.csv', index_col=0).reset_index(drop=True)
    x_learning = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xlearning.csv', index_col=0).reset_index(drop=True)
    y_learning = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/ylearning.csv', index_col=0).iloc[:, 0].reset_index(drop=True)
    x_human_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xhumantrain.csv', index_col=0).reset_index(drop=True)
    y_human_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/yhumantrain.csv', index_col=0).iloc[:, 0].reset_index(drop=True)

    x_val = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xval.csv', index_col=0).reset_index(drop=True)
    y_val = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/yval.csv', index_col=0).iloc[:, 0].reset_index(drop=True)
    x_test = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtest.csv', index_col=0).reset_index(drop=True)
    y_test = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/ytest.csv', index_col=0).iloc[:, 0].reset_index(drop=True)
    x_val_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xval_non_binarized.csv', index_col=0).reset_index(drop=True)
    x_test_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtest_non_binarized.csv', index_col=0).reset_index(drop=True)

    return x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized

def load_results(dataset, setting, run_num, cost, model):
    with open(f'results/{dataset}/run{run_num}/cost{float(cost)}/{model}_model{setting}.pkl', 'rb') as f:
        result = pickle.load(f)
        return result
    
def load_humans(dataset, setting, run_num):
    with open(f'results/{dataset}/run{run_num}/{setting}.pkl', 'rb') as f:
        human = pickle.load(f)
    with open(f'results/{dataset}/run{run_num}/adb_model_{setting}.pkl', 'rb') as f:
        adb_model = pickle.load(f)
    with open(f'results/{dataset}/run{run_num}/conf_model_{setting}.pkl', 'rb') as f:
        conf_model = pickle.load(f)
    return human, adb_model, conf_model

def make_model_results(dataset, whichtype, num_runs, costs, validation=False, which_to_do=['base_xgb', 'synth_xgb', 'base_tabPFN', 'synth_tabPFN']):

    # Create dataframe with comprehensive evaluation results for all models
    results = pd.DataFrame(data={
        # Base XGBoost Raw (unfiltered) results
        'base_xgb_raw_decision_loss': [[]],
        'base_xgb_raw_final_decision_loss': [[]],
        'base_xgb_raw_contradictions': [[]],
        'base_xgb_raw_advice_given_rate': [[]],
        'base_xgb_raw_objective': [[]],
        'base_xgb_raw_final_objective': [[]],
        
        # Base XGBoost Filtered results  
        'base_xgb_filtered_decision_loss': [[]],
        'base_xgb_filtered_final_decision_loss': [[]],
        'base_xgb_filtered_contradictions': [[]],
        'base_xgb_filtered_advice_given_rate': [[]],
        'base_xgb_filtered_objective': [[]],
        'base_xgb_filtered_final_objective': [[]],
        
        # Synthetic XGBoost (RevAI) Raw (unfiltered) results
        'synth_xgb_raw_decision_loss': [[]],
        'synth_xgb_raw_final_decision_loss': [[]],
        'synth_xgb_raw_contradictions': [[]],
        'synth_xgb_raw_advice_given_rate': [[]],
        'synth_xgb_raw_objective': [[]],
        'synth_xgb_raw_final_objective': [[]],
        
        # Synthetic XGBoost Filtered results
        'synth_xgb_filtered_decision_loss': [[]],
        'synth_xgb_filtered_final_decision_loss': [[]],
        'synth_xgb_filtered_contradictions': [[]],
        'synth_xgb_filtered_advice_given_rate': [[]],
        'synth_xgb_filtered_objective': [[]],
        'synth_xgb_filtered_final_objective': [[]],
        
        # Base TabPFN Raw (unfiltered) results
        'base_tabPFN_raw_decision_loss': [[]],
        'base_tabPFN_raw_final_decision_loss': [[]],
        'base_tabPFN_raw_contradictions': [[]],
        'base_tabPFN_raw_advice_given_rate': [[]],
        'base_tabPFN_raw_objective': [[]],
        'base_tabPFN_raw_final_objective': [[]],
        
        # Base TabPFN Filtered results  
        'base_tabPFN_filtered_decision_loss': [[]],
        'base_tabPFN_filtered_final_decision_loss': [[]],
        'base_tabPFN_filtered_contradictions': [[]],
        'base_tabPFN_filtered_advice_given_rate': [[]],
        'base_tabPFN_filtered_objective': [[]],
        'base_tabPFN_filtered_final_objective': [[]],
        
        # Synthetic TabPFN Raw (unfiltered) results
        'synth_tabPFN_raw_decision_loss': [[]],
        'synth_tabPFN_raw_final_decision_loss': [[]],
        'synth_tabPFN_raw_contradictions': [[]],
        'synth_tabPFN_raw_advice_given_rate': [[]],
        'synth_tabPFN_raw_objective': [[]],
        'synth_tabPFN_raw_final_objective': [[]],
        
        # Synthetic TabPFN Filtered results
        'synth_tabPFN_filtered_decision_loss': [[]],
        'synth_tabPFN_filtered_final_decision_loss': [[]],
        'synth_tabPFN_filtered_contradictions': [[]],
        'synth_tabPFN_filtered_advice_given_rate': [[]],
        'synth_tabPFN_filtered_objective': [[]],
        'synth_tabPFN_filtered_final_objective': [[]],
        
        # Human baseline
        'human_decision_loss': [[]]
    }, index=[costs[0]])

    for cost in costs[1:]:
        results.loc[cost] = [[] for i in range(len(results.columns))]

    bar = progressbar.ProgressBar()
    whichtype = whichtype
    
    for run in bar(range(num_runs)):
        
        bar = progressbar.ProgressBar()
        x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets(dataset, run)

        if validation == True:
            x_test = x_val
            y_test = y_val
            x_test_non_binarized = x_val_non_binarized
        
        dataset = dataset
        human, adb_mod, conf_mod = load_humans(dataset, whichtype, run)

        for cost in costs:
            print(f'producing for cost {cost} run {run}.....')
            
            # Load models
            if 'base_xgb' in which_to_do:
                base_xgb_mod = load_results(dataset, f'_{whichtype}', run, cost, 'base_xgb')
            else:
                base_xgb_mod = None
                    
            if 'synth_xgb' in which_to_do:
                synth_xgb_mod = load_results(dataset, f'_{whichtype}', run, cost, 'synth_xgb')
            else:
                synth_xgb_mod = None
            
            if 'base_tabPFN' in which_to_do:
                base_tabPFN_mod = load_results(dataset, f'_{whichtype}', run, cost, 'base_tabPFN')
            else:
                base_tabPFN_mod = None
                    
            if 'synth_tabPFN' in which_to_do:
                synth_tabPFN_mod = load_results(dataset, f'_{whichtype}', run, cost, 'synth_tabPFN')
            else:
                synth_tabPFN_mod = None
            
            # Load the final e_y model (for use in comprehensive_evaluation)
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/ey_model_{whichtype}.pkl', 'rb') as f:
                e_y_mod = pickle.load(f)
            
            # Create simple baseline model for generating confidence scores (same as in run.py training)
            baseline_model = e_y_mod

            # Initialize result storage for this cost/run combination
            base_xgb_raw_decision_loss = []
            base_xgb_raw_final_decision_loss = []
            base_xgb_raw_contradictions = []
            base_xgb_raw_advice_given_rate = []
            base_xgb_raw_objective = []
            base_xgb_raw_final_objective = []
            
            base_xgb_filtered_decision_loss = []
            base_xgb_filtered_final_decision_loss = []
            base_xgb_filtered_contradictions = []
            base_xgb_filtered_advice_given_rate = []
            base_xgb_filtered_objective = []
            base_xgb_filtered_final_objective = []
            
            synth_xgb_raw_decision_loss = []
            synth_xgb_raw_final_decision_loss = []
            synth_xgb_raw_contradictions = []
            synth_xgb_raw_advice_given_rate = []
            synth_xgb_raw_objective = []
            synth_xgb_raw_final_objective = []
            
            synth_xgb_filtered_decision_loss = []
            synth_xgb_filtered_final_decision_loss = []
            synth_xgb_filtered_contradictions = []
            synth_xgb_filtered_advice_given_rate = []
            synth_xgb_filtered_objective = []
            synth_xgb_filtered_final_objective = []
            
            base_tabPFN_raw_decision_loss = []
            base_tabPFN_raw_final_decision_loss = []
            base_tabPFN_raw_contradictions = []
            base_tabPFN_raw_advice_given_rate = []
            base_tabPFN_raw_objective = []
            base_tabPFN_raw_final_objective = []
            
            base_tabPFN_filtered_decision_loss = []
            base_tabPFN_filtered_final_decision_loss = []
            base_tabPFN_filtered_contradictions = []
            base_tabPFN_filtered_advice_given_rate = []
            base_tabPFN_filtered_objective = []
            base_tabPFN_filtered_final_objective = []
            
            synth_tabPFN_raw_decision_loss = []
            synth_tabPFN_raw_final_decision_loss = []
            synth_tabPFN_raw_contradictions = []
            synth_tabPFN_raw_advice_given_rate = []
            synth_tabPFN_raw_objective = []
            synth_tabPFN_raw_final_objective = []
            
            synth_tabPFN_filtered_decision_loss = []
            synth_tabPFN_filtered_final_decision_loss = []
            synth_tabPFN_filtered_contradictions = []
            synth_tabPFN_filtered_advice_given_rate = []
            synth_tabPFN_filtered_objective = []
            synth_tabPFN_filtered_final_objective = []
            
            human_decision_loss = []
            
            # Create ADB wrapper for evaluation
            learned_adb = ADB(adb_mod)
            
            for i in range(2):
                # Generate fresh human decisions and confidence for this iteration
                human_decisions = human.get_decisions(x_test, y_test)
                human_conf = human.get_confidence(x_test)
                
                # Get baseline model confidence (same as in run.py training)
                baseline_model_conf = np.maximum(baseline_model.predict_proba(x_test_non_binarized)[:, 0], 
                                                baseline_model.predict_proba(x_test_non_binarized)[:, 1])
                
                # Calculate acceptance probabilities using the ADB model
                p_accept_agree = learned_adb.ADB_model_wrapper(human_conf, baseline_model_conf, 
                                                            np.ones(len(human_decisions)))
                p_accept_disagree = learned_adb.ADB_model_wrapper(human_conf, baseline_model_conf, 
                                                                np.zeros(len(human_decisions)))
                
                # Create augmented features with the current iteration's human decisions
                x_test_augmented = x_test_non_binarized.copy()
                x_test_augmented['human_decision'] = human_decisions
                x_test_augmented['human_confidence'] = human_conf
                x_test_augmented['p_accept_agree'] = p_accept_agree
                x_test_augmented['p_accept_disagree'] = p_accept_disagree

                # Evaluate Base XGBoost using comprehensive_evaluation
                if base_xgb_mod is not None:
                    base_xgb_results = comprehensive_evaluation(
                        base_xgb_mod,
                        x_test_non_binarized, 
                        y_test,
                        human_decisions, 
                        human_conf, 
                        learned_adb.ADB_model_wrapper, 
                        human.ADB, 
                        cost, 
                        e_y_mod.predict_proba(x_test_non_binarized),
                        baseline_model=e_y_mod,  # ADD THIS
                        X_test_base=x_test_non_binarized  # ADD THIS
                    )
                    
                    # Raw Base XGBoost results
                    base_xgb_raw_decision_loss.append(1 - accuracy_score(base_xgb_results['raw_predictions'], y_test))
                    base_xgb_raw_final_decision_loss.append(1 - accuracy_score(base_xgb_results['final_decisions_raw'], y_test))
                    base_xgb_raw_contradictions.append((base_xgb_results['raw_predictions'] != human_decisions).sum())
                    base_xgb_raw_advice_given_rate.append(base_xgb_results['raw_advice_given'].mean())
                    base_xgb_raw_objective.append(base_xgb_raw_decision_loss[-1] + cost * base_xgb_raw_contradictions[-1] / len(y_test))
                    base_xgb_raw_final_objective.append(base_xgb_raw_final_decision_loss[-1] + cost * base_xgb_raw_contradictions[-1] / len(y_test))
                    
                    # Filtered Base XGBoost results
                    base_xgb_filtered_decision_loss.append(1 - accuracy_score(base_xgb_results['filtered_predictions'], y_test))
                    base_xgb_filtered_final_decision_loss.append(1 - accuracy_score(base_xgb_results['final_decisions_filtered'], y_test))
                    base_xgb_filtered_contradictions.append((base_xgb_results['filtered_predictions'] != human_decisions).sum())
                    base_xgb_filtered_advice_given_rate.append(base_xgb_results['filtered_advice_given'].mean())
                    base_xgb_filtered_objective.append(base_xgb_filtered_decision_loss[-1] + cost * base_xgb_filtered_contradictions[-1] / len(y_test))
                    base_xgb_filtered_final_objective.append(base_xgb_filtered_final_decision_loss[-1] + cost * base_xgb_filtered_contradictions[-1] / len(y_test))
                else:
                    # Default values when model not available
                    for lst in [base_xgb_raw_decision_loss, base_xgb_raw_final_decision_loss, base_xgb_filtered_decision_loss, base_xgb_filtered_final_decision_loss]:
                        lst.append(1.0)
                    for lst in [base_xgb_raw_contradictions, base_xgb_filtered_contradictions]:
                        lst.append(0)
                    for lst in [base_xgb_raw_advice_given_rate, base_xgb_filtered_advice_given_rate]:
                        lst.append(0.0)
                    for lst in [base_xgb_raw_objective, base_xgb_raw_final_objective, base_xgb_filtered_objective, base_xgb_filtered_final_objective]:
                        lst.append(1.0)

                # Evaluate Synthetic XGBoost using comprehensive_evaluation
                if synth_xgb_mod is not None:
                    synth_xgb_results = comprehensive_evaluation(
                        synth_xgb_mod,
                        x_test_augmented, 
                        y_test,
                        human_decisions, 
                        human_conf, 
                        learned_adb.ADB_model_wrapper, 
                        human.ADB, 
                        cost, 
                        e_y_mod.predict_proba(x_test_non_binarized),
                        baseline_model=e_y_mod,  # ADD THIS
                        X_test_base=x_test_non_binarized  # ADD THIS
                    )
                    
                    # Raw Synthetic XGBoost results
                    synth_xgb_raw_decision_loss.append(1 - accuracy_score(synth_xgb_results['raw_predictions'], y_test))
                    synth_xgb_raw_final_decision_loss.append(1 - accuracy_score(synth_xgb_results['final_decisions_raw'], y_test))
                    synth_xgb_raw_contradictions.append((synth_xgb_results['raw_predictions'] != human_decisions).sum())
                    synth_xgb_raw_advice_given_rate.append(synth_xgb_results['raw_advice_given'].mean())
                    synth_xgb_raw_objective.append(synth_xgb_raw_decision_loss[-1] + cost * synth_xgb_raw_contradictions[-1] / len(y_test))
                    synth_xgb_raw_final_objective.append(synth_xgb_raw_final_decision_loss[-1] + cost * synth_xgb_raw_contradictions[-1] / len(y_test))
                    
                    # Filtered Synthetic XGBoost results
                    synth_xgb_filtered_decision_loss.append(1 - accuracy_score(synth_xgb_results['filtered_predictions'], y_test))
                    synth_xgb_filtered_final_decision_loss.append(1 - accuracy_score(synth_xgb_results['final_decisions_filtered'], y_test))
                    synth_xgb_filtered_contradictions.append((synth_xgb_results['filtered_predictions'] != human_decisions).sum())
                    synth_xgb_filtered_advice_given_rate.append(synth_xgb_results['filtered_advice_given'].mean())
                    synth_xgb_filtered_objective.append(synth_xgb_filtered_decision_loss[-1] + cost * synth_xgb_filtered_contradictions[-1] / len(y_test))
                    synth_xgb_filtered_final_objective.append(synth_xgb_filtered_final_decision_loss[-1] + cost * synth_xgb_filtered_contradictions[-1] / len(y_test))
                else:
                    # Default values when model not available
                    for lst in [synth_xgb_raw_decision_loss, synth_xgb_raw_final_decision_loss, synth_xgb_filtered_decision_loss, synth_xgb_filtered_final_decision_loss]:
                        lst.append(1.0)
                    for lst in [synth_xgb_raw_contradictions, synth_xgb_filtered_contradictions]:
                        lst.append(0)
                    for lst in [synth_xgb_raw_advice_given_rate, synth_xgb_filtered_advice_given_rate]:
                        lst.append(0.0)
                    for lst in [synth_xgb_raw_objective, synth_xgb_raw_final_objective, synth_xgb_filtered_objective, synth_xgb_filtered_final_objective]:
                        lst.append(1.0)

                # Evaluate Base TabPFN using comprehensive_evaluation
                if base_tabPFN_mod is not None:
                    base_tabPFN_results = comprehensive_evaluation(
                        base_tabPFN_mod,
                        x_test_non_binarized, 
                        y_test,
                        human_decisions, 
                        human_conf, 
                        learned_adb.ADB_model_wrapper, 
                        human.ADB, 
                        cost, 
                        e_y_mod.predict_proba(x_test_non_binarized)
                    )
                    
                    # Raw Base TabPFN results
                    base_tabPFN_raw_decision_loss.append(1 - accuracy_score(base_tabPFN_results['raw_predictions'], y_test))
                    base_tabPFN_raw_final_decision_loss.append(1 - accuracy_score(base_tabPFN_results['final_decisions_raw'], y_test))
                    base_tabPFN_raw_contradictions.append((base_tabPFN_results['raw_predictions'] != human_decisions).sum())
                    base_tabPFN_raw_advice_given_rate.append(base_tabPFN_results['raw_advice_given'].mean())
                    base_tabPFN_raw_objective.append(base_tabPFN_raw_decision_loss[-1] + cost * base_tabPFN_raw_contradictions[-1] / len(y_test))
                    base_tabPFN_raw_final_objective.append(base_tabPFN_raw_final_decision_loss[-1] + cost * base_tabPFN_raw_contradictions[-1] / len(y_test))
                    
                    # Filtered Base TabPFN results
                    base_tabPFN_filtered_decision_loss.append(1 - accuracy_score(base_tabPFN_results['filtered_predictions'], y_test))
                    base_tabPFN_filtered_final_decision_loss.append(1 - accuracy_score(base_tabPFN_results['final_decisions_filtered'], y_test))
                    base_tabPFN_filtered_contradictions.append((base_tabPFN_results['filtered_predictions'] != human_decisions).sum())
                    base_tabPFN_filtered_advice_given_rate.append(base_tabPFN_results['filtered_advice_given'].mean())
                    base_tabPFN_filtered_objective.append(base_tabPFN_filtered_decision_loss[-1] + cost * base_tabPFN_filtered_contradictions[-1] / len(y_test))
                    base_tabPFN_filtered_final_objective.append(base_tabPFN_filtered_final_decision_loss[-1] + cost * base_tabPFN_filtered_contradictions[-1] / len(y_test))
                else:
                    # Default values when model not available
                    for lst in [base_tabPFN_raw_decision_loss, base_tabPFN_raw_final_decision_loss, base_tabPFN_filtered_decision_loss, base_tabPFN_filtered_final_decision_loss]:
                        lst.append(1.0)
                    for lst in [base_tabPFN_raw_contradictions, base_tabPFN_filtered_contradictions]:
                        lst.append(0)
                    for lst in [base_tabPFN_raw_advice_given_rate, base_tabPFN_filtered_advice_given_rate]:
                        lst.append(0.0)
                    for lst in [base_tabPFN_raw_objective, base_tabPFN_raw_final_objective, base_tabPFN_filtered_objective, base_tabPFN_filtered_final_objective]:
                        lst.append(1.0)

                # Evaluate Synthetic TabPFN using comprehensive_evaluation
                if synth_tabPFN_mod is not None:
                    synth_tabPFN_results = comprehensive_evaluation(
                        synth_tabPFN_mod,
                        x_test_augmented, 
                        y_test,
                        human_decisions, 
                        human_conf, 
                        learned_adb.ADB_model_wrapper, 
                        human.ADB, 
                        cost, 
                        e_y_mod.predict_proba(x_test_non_binarized)
                    )
                    
                    # Raw Synthetic TabPFN results
                    synth_tabPFN_raw_decision_loss.append(1 - accuracy_score(synth_tabPFN_results['raw_predictions'], y_test))
                    synth_tabPFN_raw_final_decision_loss.append(1 - accuracy_score(synth_tabPFN_results['final_decisions_raw'], y_test))
                    synth_tabPFN_raw_contradictions.append((synth_tabPFN_results['raw_predictions'] != human_decisions).sum())
                    synth_tabPFN_raw_advice_given_rate.append(synth_tabPFN_results['raw_advice_given'].mean())
                    synth_tabPFN_raw_objective.append(synth_tabPFN_raw_decision_loss[-1] + cost * synth_tabPFN_raw_contradictions[-1] / len(y_test))
                    synth_tabPFN_raw_final_objective.append(synth_tabPFN_raw_final_decision_loss[-1] + cost * synth_tabPFN_raw_contradictions[-1] / len(y_test))
                    
                    # Filtered Synthetic TabPFN results
                    synth_tabPFN_filtered_decision_loss.append(1 - accuracy_score(synth_tabPFN_results['filtered_predictions'], y_test))
                    synth_tabPFN_filtered_final_decision_loss.append(1 - accuracy_score(synth_tabPFN_results['final_decisions_filtered'], y_test))
                    synth_tabPFN_filtered_contradictions.append((synth_tabPFN_results['filtered_predictions'] != human_decisions).sum())
                    synth_tabPFN_filtered_advice_given_rate.append(synth_tabPFN_results['filtered_advice_given'].mean())
                    synth_tabPFN_filtered_objective.append(synth_tabPFN_filtered_decision_loss[-1] + cost * synth_tabPFN_filtered_contradictions[-1] / len(y_test))
                    synth_tabPFN_filtered_final_objective.append(synth_tabPFN_filtered_final_decision_loss[-1] + cost * synth_tabPFN_filtered_contradictions[-1] / len(y_test))
                else:
                    # Default values when model not available
                    for lst in [synth_tabPFN_raw_decision_loss, synth_tabPFN_raw_final_decision_loss, synth_tabPFN_filtered_decision_loss, synth_tabPFN_filtered_final_decision_loss]:
                        lst.append(1.0)
                    for lst in [synth_tabPFN_raw_contradictions, synth_tabPFN_filtered_contradictions]:
                        lst.append(0)
                    for lst in [synth_tabPFN_raw_advice_given_rate, synth_tabPFN_filtered_advice_given_rate]:
                        lst.append(0.0)
                    for lst in [synth_tabPFN_raw_objective, synth_tabPFN_raw_final_objective, synth_tabPFN_filtered_objective, synth_tabPFN_filtered_final_objective]:
                        lst.append(1.0)

                human_decision_loss.append(1 - accuracy_score(human_decisions, y_test))
                
                print(i)
            
            # Store averages for this run in results dataframe
            results.loc[cost, 'base_xgb_raw_decision_loss'].append(mean(base_xgb_raw_decision_loss))
            results.loc[cost, 'base_xgb_raw_final_decision_loss'].append(mean(base_xgb_raw_final_decision_loss))
            results.loc[cost, 'base_xgb_raw_contradictions'].append(mean(base_xgb_raw_contradictions))
            results.loc[cost, 'base_xgb_raw_advice_given_rate'].append(mean(base_xgb_raw_advice_given_rate))
            results.loc[cost, 'base_xgb_raw_objective'].append(mean(base_xgb_raw_objective))
            results.loc[cost, 'base_xgb_raw_final_objective'].append(mean(base_xgb_raw_final_objective))
            
            results.loc[cost, 'base_xgb_filtered_decision_loss'].append(mean(base_xgb_filtered_decision_loss))
            results.loc[cost, 'base_xgb_filtered_final_decision_loss'].append(mean(base_xgb_filtered_final_decision_loss))
            results.loc[cost, 'base_xgb_filtered_contradictions'].append(mean(base_xgb_filtered_contradictions))
            results.loc[cost, 'base_xgb_filtered_advice_given_rate'].append(mean(base_xgb_filtered_advice_given_rate))
            results.loc[cost, 'base_xgb_filtered_objective'].append(mean(base_xgb_filtered_objective))
            results.loc[cost, 'base_xgb_filtered_final_objective'].append(mean(base_xgb_filtered_final_objective))
            
            results.loc[cost, 'synth_xgb_raw_decision_loss'].append(mean(synth_xgb_raw_decision_loss))
            results.loc[cost, 'synth_xgb_raw_final_decision_loss'].append(mean(synth_xgb_raw_final_decision_loss))
            results.loc[cost, 'synth_xgb_raw_contradictions'].append(mean(synth_xgb_raw_contradictions))
            results.loc[cost, 'synth_xgb_raw_advice_given_rate'].append(mean(synth_xgb_raw_advice_given_rate))
            results.loc[cost, 'synth_xgb_raw_objective'].append(mean(synth_xgb_raw_objective))
            results.loc[cost, 'synth_xgb_raw_final_objective'].append(mean(synth_xgb_raw_final_objective))
            
            results.loc[cost, 'synth_xgb_filtered_decision_loss'].append(mean(synth_xgb_filtered_decision_loss))
            results.loc[cost, 'synth_xgb_filtered_final_decision_loss'].append(mean(synth_xgb_filtered_final_decision_loss))
            results.loc[cost, 'synth_xgb_filtered_contradictions'].append(mean(synth_xgb_filtered_contradictions))
            results.loc[cost, 'synth_xgb_filtered_advice_given_rate'].append(mean(synth_xgb_filtered_advice_given_rate))
            results.loc[cost, 'synth_xgb_filtered_objective'].append(mean(synth_xgb_filtered_objective))
            results.loc[cost, 'synth_xgb_filtered_final_objective'].append(mean(synth_xgb_filtered_final_objective))
            
            results.loc[cost, 'base_tabPFN_raw_decision_loss'].append(mean(base_tabPFN_raw_decision_loss))
            results.loc[cost, 'base_tabPFN_raw_final_decision_loss'].append(mean(base_tabPFN_raw_final_decision_loss))
            results.loc[cost, 'base_tabPFN_raw_contradictions'].append(mean(base_tabPFN_raw_contradictions))
            results.loc[cost, 'base_tabPFN_raw_advice_given_rate'].append(mean(base_tabPFN_raw_advice_given_rate))
            results.loc[cost, 'base_tabPFN_raw_objective'].append(mean(base_tabPFN_raw_objective))
            results.loc[cost, 'base_tabPFN_raw_final_objective'].append(mean(base_tabPFN_raw_final_objective))
            
            results.loc[cost, 'base_tabPFN_filtered_decision_loss'].append(mean(base_tabPFN_filtered_decision_loss))
            results.loc[cost, 'base_tabPFN_filtered_final_decision_loss'].append(mean(base_tabPFN_filtered_final_decision_loss))
            results.loc[cost, 'base_tabPFN_filtered_contradictions'].append(mean(base_tabPFN_filtered_contradictions))
            results.loc[cost, 'base_tabPFN_filtered_advice_given_rate'].append(mean(base_tabPFN_filtered_advice_given_rate))
            results.loc[cost, 'base_tabPFN_filtered_objective'].append(mean(base_tabPFN_filtered_objective))
            results.loc[cost, 'base_tabPFN_filtered_final_objective'].append(mean(base_tabPFN_filtered_final_objective))
            
            results.loc[cost, 'synth_tabPFN_raw_decision_loss'].append(mean(synth_tabPFN_raw_decision_loss))
            results.loc[cost, 'synth_tabPFN_raw_final_decision_loss'].append(mean(synth_tabPFN_raw_final_decision_loss))
            results.loc[cost, 'synth_tabPFN_raw_contradictions'].append(mean(synth_tabPFN_raw_contradictions))
            results.loc[cost, 'synth_tabPFN_raw_advice_given_rate'].append(mean(synth_tabPFN_raw_advice_given_rate))
            results.loc[cost, 'synth_tabPFN_raw_objective'].append(mean(synth_tabPFN_raw_objective))
            results.loc[cost, 'synth_tabPFN_raw_final_objective'].append(mean(synth_tabPFN_raw_final_objective))
            
            results.loc[cost, 'synth_tabPFN_filtered_decision_loss'].append(mean(synth_tabPFN_filtered_decision_loss))
            results.loc[cost, 'synth_tabPFN_filtered_final_decision_loss'].append(mean(synth_tabPFN_filtered_final_decision_loss))
            results.loc[cost, 'synth_tabPFN_filtered_contradictions'].append(mean(synth_tabPFN_filtered_contradictions))
            results.loc[cost, 'synth_tabPFN_filtered_advice_given_rate'].append(mean(synth_tabPFN_filtered_advice_given_rate))
            results.loc[cost, 'synth_tabPFN_filtered_objective'].append(mean(synth_tabPFN_filtered_objective))
            results.loc[cost, 'synth_tabPFN_filtered_final_objective'].append(mean(synth_tabPFN_filtered_final_objective))
            
            results.loc[cost, 'human_decision_loss'].append(mean(human_decision_loss))
            
    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))
    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results

def make_model_TL_v_cost_plot(results_means, results_stderrs, name, include_tabpfn=True):
    fig = plt.figure(figsize=(4, 3), dpi=200)
    color_dict = {
        'Base_XGB_Raw': '#FF6B6B', 
        'Base_XGB_Filtered': '#FF9999',
        'Synth_XGB_Raw': '#4ECDC4', 
        'Synth_XGB_Filtered': '#7DDDD7',
        'Base_TabPFN_Raw': '#9B59B6',
        'Base_TabPFN_Filtered': '#BB79D6',
        'Synth_TabPFN_Raw': '#F39C12',
        'Synth_TabPFN_Filtered': '#F5B041',
        'Human': 'darkgray'
    }
    
    # Normalize all metrics relative to human decision loss (value added)
    normalized_means = results_means.copy()
    normalized_stderrs = results_stderrs.copy()
    
    # Subtract human_decision_loss from all metrics to show improvement relative to human alone
    for col in results_means.columns:
        normalized_means[col] = results_means['human_decision_loss'] - results_means[col]
        # Standard errors remain the same since we're just shifting by a constant
    
    # Plot normalized final objectives (positive values = improvement over human)
    plt.plot(normalized_means.index[0:6], normalized_means['base_xgb_raw_final_objective'].iloc[0:6], 
             marker='o', c=color_dict['Base_XGB_Raw'], label='Base XGB Raw', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['base_xgb_filtered_final_objective'].iloc[0:6], 
             marker='s', c=color_dict['Base_XGB_Filtered'], label='Base XGB Filtered', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['synth_xgb_raw_final_objective'].iloc[0:6], 
             marker='^', c=color_dict['Synth_XGB_Raw'], label='Synth XGB Raw', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['synth_xgb_filtered_final_objective'].iloc[0:6], 
             marker='d', c=color_dict['Synth_XGB_Filtered'], label='Synth XGB Filtered', markersize=1.8, linewidth=0.9)
    
    if include_tabpfn:
        plt.plot(normalized_means.index[0:6], normalized_means['base_tabPFN_raw_final_objective'].iloc[0:6], 
                 marker='p', c=color_dict['Base_TabPFN_Raw'], label='Base TabPFN Raw', markersize=1.8, linewidth=0.9)
        plt.plot(normalized_means.index[0:6], normalized_means['base_tabPFN_filtered_final_objective'].iloc[0:6], 
                 marker='h', c=color_dict['Base_TabPFN_Filtered'], label='Base TabPFN Filtered', markersize=1.8, linewidth=0.9)
        plt.plot(normalized_means.index[0:6], normalized_means['synth_tabPFN_raw_final_objective'].iloc[0:6], 
                 marker='*', c=color_dict['Synth_TabPFN_Raw'], label='Synth TabPFN Raw', markersize=1.8, linewidth=0.9)
        plt.plot(normalized_means.index[0:6], normalized_means['synth_tabPFN_filtered_final_objective'].iloc[0:6], 
                 marker='X', c=color_dict['Synth_TabPFN_Filtered'], label='Synth TabPFN Filtered', markersize=1.8, linewidth=0.9)
    
    plt.plot(normalized_means.index[0:6], normalized_means['human_decision_loss'].iloc[0:6], 
             c=color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    
    # Add error bands
    error_bands = [
        ('human_decision_loss', 'Human'), 
        ('base_xgb_raw_final_objective', 'Base_XGB_Raw'), 
        ('base_xgb_filtered_final_objective', 'Base_XGB_Filtered'),
        ('synth_xgb_raw_final_objective', 'Synth_XGB_Raw'),
        ('synth_xgb_filtered_final_objective', 'Synth_XGB_Filtered')
    ]
    
    if include_tabpfn:
        error_bands.extend([
            ('base_tabPFN_raw_final_objective', 'Base_TabPFN_Raw'),
            ('base_tabPFN_filtered_final_objective', 'Base_TabPFN_Filtered'),
            ('synth_tabPFN_raw_final_objective', 'Synth_TabPFN_Raw'),
            ('synth_tabPFN_filtered_final_objective', 'Synth_TabPFN_Filtered')
        ])
    
    for key, color_key in error_bands:
        plt.fill_between(normalized_means.index[0:6], 
                    normalized_means[key].iloc[0:6] - normalized_stderrs[key].iloc[0:6],
                    normalized_means[key].iloc[0:6] + normalized_stderrs[key].iloc[0:6],
                    color=color_dict[color_key], alpha=0.2)
   
    plt.xlabel('Reconciliation Cost', fontsize=12)
    plt.ylabel('Value Added vs Human Alone', fontsize=12)
    plt.tick_params(labelrotation=45, labelsize=10)
    plt.legend(prop={'size': 5})
    plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')
    
    # Add horizontal line at y=0 to show human baseline
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    fig.savefig(f'results/{dataset}/plots/TL_Models_{dataset}_{name}.png', bbox_inches='tight')

# Main execution
costs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
num_runs = 10
datasets = ['heart_disease']
names = ['offset_01']
which_to_do =  ['base_tabPFN', 'synth_tabPFN'] #['base_xgb', 'synth_xgb']

for dataset in datasets:
    for name in names:
        if os.path.isfile(f'results/{dataset}/{name}_models_comprehensive_rs.pkl') and False:
            with open(f'results/{dataset}/{name}_models_comprehensive_rs.pkl', 'rb') as f:
                rs = pickle.load(f)
            with open(f'results/{dataset}/{name}_models_comprehensive_means.pkl', 'rb') as f:
                means = pickle.load(f)
            with open(f'results/{dataset}/{name}_models_comprehensive_std.pkl', 'rb') as f:
                std = pickle.load(f)

            with open(f'results/{dataset}/val_{name}_models_comprehensive_rs.pkl', 'rb') as f:
                val_rs = pickle.load(f)
            with open(f'results/{dataset}/val_{name}_models_comprehensive_means.pkl', 'rb') as f:
                val_means = pickle.load(f)
            with open(f'results/{dataset}/val_{name}_models_comprehensive_std.pkl', 'rb') as f:
                val_std = pickle.load(f)

        else:
            means, std, rs = make_model_results(dataset, name, num_runs, costs, validation=False, which_to_do=which_to_do)
            # Pickle and write means, std, and rs to file
            with open(f'results/{dataset}/{name}_models_tabPFN_means.pkl', 'wb') as f:
                pickle.dump(means, f)
            with open(f'results/{dataset}/{name}_models_tabPFN_std.pkl', 'wb') as f:
                pickle.dump(std, f)
            with open(f'results/{dataset}/{name}_models_tabPFN_rs.pkl', 'wb') as f:
                pickle.dump(rs, f)
        
            print(f'running for val {dataset} {name}')
            #val_means, val_std, val_rs = make_model_results(dataset, name, num_runs, costs, validation=True, which_to_do=which_to_do)
            # Pickle and write validation results
            #with open(f'results/{dataset}/val_{name}_models_comprehensive_means.pkl', 'wb') as f:
            #    pickle.dump(val_means, f)
            #with open(f'results/{dataset}/val_{name}_models_comprehensive_std.pkl', 'wb') as f:
            #    pickle.dump(val_std, f)
            #with open(f'results/{dataset}/val_{name}_models_comprehensive_rs.pkl', 'wb') as f:
            #    pickle.dump(val_rs, f)

        # Create plot - check if TabPFN models are included
        include_tabpfn = any('tabPFN' in model for model in which_to_do)
        make_model_TL_v_cost_plot(means, std, name, include_tabpfn=include_tabpfn)

print('Model analysis complete!')
