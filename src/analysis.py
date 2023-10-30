import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tr import *
from hyrs import *
from brs import *
import pickle
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error
from util_BOA import *
from statistics import mean 
import progressbar

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
    if model == 'brs':
        setting = ''
    else:
        setting = '_' + setting
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

costs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
costs = [0]
num_runs = 6
bar=progressbar.ProgressBar()
for run in bar(range(num_runs)):
    if run == 4:
        continue

    
    bar=progressbar.ProgressBar()
    x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets('heart_disease', run)

    human, adb_mod, conf_mod = load_humans('heart_disease', 'calibrated', run)

    brs_mod = load_results('heart_disease', 'calibrated' , run, 0.0, 'brs')


    for cost in costs:
        print(f'producing for cost {cost} run {run}.....')
        tr_mod = load_results('heart_disease', 'calibrated' , run, cost, 'tr')
        hyrs_mod = load_results('heart_disease', 'calibrated' , run, cost, 'hyrs')
        e_y_mod = xgb.XGBClassifier().fit(x_train_non_binarized, y_train)
        e_yb_mod = xgb.XGBClassifier().fit(x_train_non_binarized, tr_mod.Yb)

        
        tr_team_w_reset_decision_loss = []
        tr_team_wo_reset_decision_loss = []
        tr_model_w_reset_decision_loss = []
        tr_model_wo_reset_decision_loss = []
        hyrs_model_decision_loss = []
        hyrs_team_decision_loss = []
        brs_model_decision_loss = []
        brs_team_decision_loss = []


        tr_model_w_reset_contradictions = []
        tr_model_wo_reset_contradictions = []
        hyrs_model_contradictions = []
        brs_model_contradictions = []
        
        if cost == 0.0:
            brs_model_preds = brs_predict(brs_mod.rules, x_test)
            brs_conf = brs_predict_conf(brs_mod.rules, x_test, brs_mod)

        for i in range(15):
            tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human.get_decisions(x_test, y_test), human.get_confidence(x_test), human.ADB, with_reset=True, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
            tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human.get_decisions(x_test, y_test), human.get_confidence(x_test), human.ADB, with_reset=False, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

            tr_model_preds_with_reset = tr_mod.predict(x_test, human.get_decisions(x_test, y_test), with_reset=True, conf_human=human.get_confidence(x_test), p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
            tr_model_preds_no_reset = tr_mod.predict(x_test, human.get_decisions(x_test, y_test), with_reset=False, conf_human=human.get_confidence(x_test), p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

            hyrs_model_preds = hyrs_mod.predict(x_test, human.get_decisions(x_test, y_test))[0]
            hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human.get_decisions(x_test, y_test), human.get_confidence(x_test), human.ADB, x_test)

            
            brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human.get_decisions(x_test, y_test), human.get_confidence(x_test), human.ADB)

            tr_team_w_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_with_reset, y_test))
            tr_team_wo_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_no_reset, y_test))
            tr_model_w_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_with_reset, y_test))
            tr_model_wo_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_no_reset, y_test))
            hyrs_model_decision_loss.append(1 - accuracy_score(hyrs_model_preds, y_test))
            hyrs_team_decision_loss.append(1 - accuracy_score(hyrs_team_preds, y_test))
            brs_model_decision_loss.append(1 - accuracy_score(brs_model_preds, y_test))
            brs_team_decision_loss.append(1 - accuracy_score(brs_team_preds, y_test))

            tr_model_w_reset_contradictions.append((tr_model_preds_with_reset != human.get_decisions(x_test,y_test)).sum())
            tr_model_wo_reset_contradictions.append((tr_model_preds_no_reset != human.get_decisions(x_test,y_test)).sum())
            hyrs_model_contradictions.append((hyrs_model_preds != human.get_decisions(x_test,y_test)).sum())
            brs_model_contradictions.append((brs_model_preds != human.get_decisions(x_test,y_test)).sum())

            print(i)

        if run == 0:
            if cost == 0.0:
                results = pd.DataFrame({'tr_team_w_reset_decision_loss': mean(tr_team_w_reset_decision_loss),
                                        'tr_team_wo_reset_decision_loss': mean(tr_team_wo_reset_decision_loss),
                                        'tr_model_w_reset_decision_loss': mean(tr_model_w_reset_decision_loss),
                                        'tr_model_wo_reset_decision_loss': mean(tr_model_wo_reset_decision_loss),
                                        'hyrs_model_decision_loss': mean(hyrs_model_decision_loss),
                                        'hyrs_team_decision_loss': mean(hyrs_team_decision_loss),
                                        'brs_model_decision_loss': mean(brs_model_decision_loss),
                                        'brs_team_decision_loss': mean(brs_team_decision_loss),
                                        'tr_model_w_reset_contradictions': mean(tr_model_w_reset_contradictions),
                                        'tr_model_wo_reset_contradictions': mean(tr_model_wo_reset_contradictions),
                                        'hyrs_model_contradictions': mean(hyrs_model_contradictions),
                                        'brs_model_contradictions': mean(brs_model_contradictions)}, index=[cost])
            else:
                results.loc[cost] = [mean(tr_team_w_reset_decision_loss),
                                    mean(tr_team_wo_reset_decision_loss),
                                    mean(tr_model_w_reset_decision_loss),
                                    mean(tr_model_wo_reset_decision_loss),
                                    mean(hyrs_model_decision_loss),
                                    mean(hyrs_team_decision_loss),
                                    mean(brs_model_decision_loss),
                                    mean(brs_team_decision_loss),
                                    mean(tr_model_w_reset_contradictions),
                                    mean(tr_model_wo_reset_contradictions),
                                    mean(hyrs_model_contradictions),
                                    mean(brs_model_contradictions)]
                
        else:
            results.loc[cost] += [mean(tr_team_w_reset_decision_loss),
                                    mean(tr_team_wo_reset_decision_loss),
                                    mean(tr_model_w_reset_decision_loss),
                                    mean(tr_model_wo_reset_decision_loss),
                                    mean(hyrs_model_decision_loss),
                                    mean(hyrs_team_decision_loss),
                                    mean(brs_model_decision_loss),
                                    mean(brs_team_decision_loss),
                                    mean(tr_model_w_reset_contradictions),
                                    mean(tr_model_wo_reset_contradictions),
                                    mean(hyrs_model_contradictions),
                                    mean(brs_model_contradictions)]
    
results = results/num_runs
    
                            
                            
    



    

print('pause')

