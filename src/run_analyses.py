import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tr import *
from hyrs import *
from brs import *
import pickle
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error
from util_BOA import *
from numpy import mean 
import progressbar
from run import ADB
from run import evaluate_adb_model
from copy import deepcopy
import os

#making sure wd is file directory so hardcoded paths work
os.chdir("..")

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
        try:
            setting = '_' + setting
        except:
            setting = '_' + setting
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



def make_results(dataset, whichtype, num_runs, costs, validation=False):

    #create dataframe of empty lists with column headers below


    
    results = pd.DataFrame(data={'tr_team_w_reset_decision_loss': [[]],
                                'tr_team_wo_reset_decision_loss': [[]],
                                'tr_model_w_reset_decision_loss': [[]],
                                'tr_model_wo_reset_decision_loss': [[]],
                                'hyrs_model_decision_loss': [[]],
                                'hyrs_team_decision_loss': [[]],
                                'brs_model_decision_loss': [[]],
                                'brs_team_decision_loss': [[]],
                                'brs_team_w_reset_decision_loss': [[]],
                                'brs_model_w_reset_decision_loss': [[]],
                                'brs_team_w_reset_objective': [[]],
                                'brs_model_w_reset_objective': [[]],
                                'brs_model_w_reset_contradictions': [[]],
                                'tr_model_w_reset_contradictions': [[]],
                                'tr_model_wo_reset_contradictions': [[]],
                                'hyrs_model_contradictions': [[]],
                                'brs_model_contradictions':[[]],
                                'tr_team_w_reset_objective': [[]],
                                'tr_team_wo_reset_objective': [[]],
                                'tr_model_w_reset_objective': [[]],
                                'tr_model_wo_reset_objective': [[]],
                                'hyrs_model_objective': [[]],
                                'hyrs_team_objective':[[]],
                                'brs_model_objective': [[]],
                                'brs_team_objective': [[]],
                                'human_decision_loss': [[]],
                                'hyrs_norecon_objective': [[]],
                                'hyrs_norecon_model_decision_loss': [[]],
                                'hyrs_norecon_team_decision_loss': [[]], 
                                'hyrs_norecon_model_contradictions': [[]]}, index=[costs[0]]
                            )

    for cost in costs[1:]:
        results.loc[cost] = [[] for i in range(len(results.columns))]

    bar=progressbar.ProgressBar()
    whichtype = whichtype
    for run in bar(range(num_runs)):

        
        bar=progressbar.ProgressBar()
        x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets(dataset, run)

        if validation==True:
            x_test = x_val
            y_test = y_val
            x_test_non_binarized = x_val_non_binarized
        
        
        dataset = dataset
        human, adb_mod, conf_mod = load_humans(dataset, whichtype, run)

        brs_mod = load_results(dataset, whichtype , run, 0.0, 'brs')


        for cost in costs:
            print(f'producing for cost {cost} run {run}.....')
            tr_mod = load_results(dataset, whichtype, run, cost, 'tr')
            hyrs_mod = load_results(dataset, whichtype, run, cost, 'hyrs')
            #load e_y and e_yb mods
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/eyb_model_{whichtype}.pkl', 'rb') as f:
                e_yb_mod = pickle.load(f)
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/ey_model_{whichtype}.pkl', 'rb') as f:
                e_y_mod = pickle.load(f)

            tr_mod.df = x_train
            tr_mod.Y = y_train
            hyrs_mod.df = x_train
            hyrs_mod.Y = y_train

            
            tr_team_w_reset_decision_loss = []
            tr_team_wo_reset_decision_loss = []
            tr_model_w_reset_decision_loss = []
            tr_model_wo_reset_decision_loss = []
            hyrs_model_decision_loss = []
            hyrs_team_decision_loss = []
            brs_model_decision_loss = []
            brs_model_w_reset_decision_loss = []
            brs_team_w_reset_decision_loss = []
            brs_team_decision_loss = []


            tr_model_w_reset_contradictions = []
            tr_model_wo_reset_contradictions = []
            hyrs_model_contradictions = []
            brs_model_contradictions = []
            brs_model_w_reset_contradictions = []

            tr_team_w_reset_objective = []
            tr_team_wo_reset_objective = []
            tr_model_w_reset_objective = []
            tr_model_wo_reset_objective = []
            hyrs_model_objective = []
            hyrs_team_objective = []
            brs_model_objective = []
            brs_model_w_reset_objective = []
            brs_team_objective = []
            brs_team_w_reset_objective = []

            human_decision_loss = []
            hyrs_norecon_objective = []
            hyrs_norecon_model_decision_loss = []
            hyrs_norecon_team_decision_loss = []
            hyrs_norecon_model_contradictions = []
            
            if cost == 0.0:
                brs_mod.df = x_train
                brs_mod.Y = y_train
                brs_model_preds = brs_predict(brs_mod.opt_rules, x_test)
                brs_conf = brs_predict_conf(brs_mod.opt_rules, x_test, brs_mod)
                hyrs_norecon_mod = deepcopy(hyrs_mod)
            
            for i in range(50):
                
                

                if validation: 

                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    conf_mod_preds = conf_mod.predict(x_test_non_binarized)

                    learned_adb = ADB(adb_mod)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=False, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, learned_adb.ADB_model_wrapper, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, learned_adb.ADB_model_wrapper)

                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, learned_adb.ADB_model_wrapper, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, learned_adb.ADB_model_wrapper)
                    brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized), conf_model=brs_conf, fA=learned_adb.ADB_model_wrapper, asym_loss=[1,1], contradiction_reg=cost)
                    brs_team_preds_w_reset = brs_team_preds.copy()
                    brs_team_preds_w_reset[brs_reset] = human_decisions[brs_reset]
                    brs_model_preds_w_reset = brs_model_preds.copy()
                    brs_model_preds_w_reset[brs_reset] = human_decisions[brs_reset]

                    c_model = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]



                        
                else:
                    learned_adb = ADB(adb_mod)
                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions,human_conf, human.ADB, with_reset=False, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset, tr_mod_covered_w_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr_model_preds_no_reset, tr_mod_covered_no_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr_mod_confs = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]
                

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)
                    brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized), conf_model=brs_conf, fA=learned_adb.ADB_model_wrapper, asym_loss=[1,1], contradiction_reg=cost)
                    brs_team_preds_w_reset = brs_team_preds.copy()
                    brs_team_preds_w_reset[brs_reset] = human_decisions[brs_reset]
                    brs_model_preds_w_reset = brs_model_preds.copy()
                    brs_model_preds_w_reset[brs_reset] = human_decisions[brs_reset]
                    

                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)
                        

                tr_team_w_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_with_reset, y_test))
                tr_team_wo_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_no_reset, y_test))
                tr_model_w_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_with_reset, y_test))
                tr_model_wo_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_no_reset, y_test))
                hyrs_model_decision_loss.append(1 - accuracy_score(hyrs_model_preds, y_test))
                hyrs_team_decision_loss.append(1 - accuracy_score(hyrs_team_preds, y_test))
                brs_model_decision_loss.append(1 - accuracy_score(brs_model_preds, y_test))
                brs_team_decision_loss.append(1 - accuracy_score(brs_team_preds, y_test))
                brs_model_w_reset_decision_loss.append(1 - accuracy_score(brs_model_preds_w_reset, y_test))
                brs_team_w_reset_decision_loss.append(1 - accuracy_score(brs_team_preds_w_reset, y_test))
                

                tr_model_w_reset_contradictions.append((tr_model_preds_with_reset != human_decisions).sum())
                tr_model_wo_reset_contradictions.append((tr_model_preds_no_reset != human_decisions).sum())
                hyrs_model_contradictions.append((hyrs_model_preds != human_decisions).sum())
                brs_model_contradictions.append((brs_model_preds != human_decisions).sum())
                brs_model_w_reset_contradictions.append((brs_model_preds_w_reset != human_decisions).sum())

                tr_team_w_reset_objective.append(tr_team_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_team_wo_reset_objective.append(tr_team_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                tr_model_w_reset_objective.append(tr_model_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_model_wo_reset_objective.append(tr_model_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                hyrs_model_objective.append(hyrs_model_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                hyrs_team_objective.append(hyrs_team_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                brs_model_objective.append(brs_model_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_team_objective.append(brs_team_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_team_w_reset_objective.append(brs_team_w_reset_decision_loss[-1] + cost*(brs_model_w_reset_contradictions[-1])/len(y_test))
                brs_model_w_reset_objective.append(brs_model_w_reset_decision_loss[-1] + cost*(brs_model_w_reset_contradictions[-1])/len(y_test))
                                                  

                human_decision_loss.append(1 - accuracy_score(human_decisions, y_test))

                hyrs_norecon_model_decision_loss.append(1 - accuracy_score(hyrs_norecon_model_preds, y_test))
                hyrs_norecon_team_decision_loss.append(1 - accuracy_score(hyrs_norecon_team_preds, y_test))
                hyrs_norecon_model_contradictions.append((hyrs_norecon_model_preds != human_decisions).sum())
                hyrs_norecon_objective.append(hyrs_norecon_team_decision_loss[-1] + cost*(hyrs_norecon_model_contradictions[-1])/len(y_test))

                


                print(i)
            
            

            
            #append values to appropriate row in results
            results.loc[cost, 'tr_team_w_reset_decision_loss'].append(mean(tr_team_w_reset_decision_loss))
            results.loc[cost, 'tr_team_wo_reset_decision_loss'].append(mean(tr_team_wo_reset_decision_loss))
            results.loc[cost, 'tr_model_w_reset_decision_loss'].append(mean(tr_model_w_reset_decision_loss))
            results.loc[cost, 'tr_model_wo_reset_decision_loss'].append(mean(tr_model_wo_reset_decision_loss))
            results.loc[cost, 'hyrs_model_decision_loss'].append(mean(hyrs_model_decision_loss))
            results.loc[cost, 'hyrs_team_decision_loss'].append(mean(hyrs_team_decision_loss))
            results.loc[cost, 'brs_model_decision_loss'].append(mean(brs_model_decision_loss))
            results.loc[cost, 'brs_team_decision_loss'].append(mean(brs_team_decision_loss))
            results.loc[cost, 'tr_model_w_reset_contradictions'].append(mean(tr_model_w_reset_contradictions))
            results.loc[cost, 'tr_model_wo_reset_contradictions'].append(mean(tr_model_wo_reset_contradictions))
            results.loc[cost, 'hyrs_model_contradictions'].append(mean(hyrs_model_contradictions))
            results.loc[cost, 'brs_model_contradictions'].append(mean(brs_model_contradictions))
            results.loc[cost, 'tr_team_w_reset_objective'].append(mean(tr_team_w_reset_objective))
            results.loc[cost, 'tr_team_wo_reset_objective'].append(mean(tr_team_wo_reset_objective))
            results.loc[cost, 'tr_model_w_reset_objective'].append(mean(tr_model_w_reset_objective))
            results.loc[cost, 'tr_model_wo_reset_objective'].append(mean(tr_model_wo_reset_objective))
            results.loc[cost, 'hyrs_model_objective'].append(mean(hyrs_model_objective))
            results.loc[cost, 'hyrs_team_objective'].append(mean(hyrs_team_objective))
            results.loc[cost, 'brs_model_objective'].append(mean(brs_model_objective))
            results.loc[cost, 'brs_team_objective'].append(mean(brs_team_objective))
            results.loc[cost, 'human_decision_loss'].append(mean(human_decision_loss))
            results.loc[cost, 'hyrs_norecon_objective'].append(mean(hyrs_norecon_objective))
            results.loc[cost, 'hyrs_norecon_model_decision_loss'].append(mean(hyrs_norecon_model_decision_loss))
            results.loc[cost, 'hyrs_norecon_team_decision_loss'].append(mean(hyrs_norecon_team_decision_loss))
            results.loc[cost, 'hyrs_norecon_model_contradictions'].append(mean(hyrs_norecon_model_contradictions))
            results.loc[cost, 'brs_team_w_reset_decision_loss'].append(mean(brs_team_w_reset_decision_loss))
            results.loc[cost, 'brs_model_w_reset_decision_loss'].append(mean(brs_model_w_reset_decision_loss))
            results.loc[cost, 'brs_team_w_reset_objective'].append(mean(brs_team_w_reset_objective))
            results.loc[cost, 'brs_model_w_reset_objective'].append(mean(brs_model_w_reset_objective))
            results.loc[cost, 'brs_model_w_reset_contradictions'].append(mean(brs_model_w_reset_contradictions))
            
            
    
    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))




    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results


costs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
num_runs = 5
datasets = ['heart_disease', 'fico', 'hr']
names = ['biased', 'biased_dec_bias', 'offset_01']

for dataset in datasets:
    for name in names:
        if (name == 'biased') and (datasets == 'heart_disease'):
            continue
        print(f'running for {dataset} {name}')
        means, std, rs = make_results(dataset, name, num_runs, costs, validation=False)
        #pickle and write means, std, and rs to file
        with open(f'results/{dataset}/{name}_means.pkl', 'wb') as f:
            pickle.dump(means, f)
        with open(f'results/{dataset}/{name}_std.pkl', 'wb') as f:
            pickle.dump(std, f)
        with open(f'results/{dataset}/{name}_rs.pkl', 'wb') as f:
            pickle.dump(rs, f)
        
        print(f'running for val {dataset} {name}')
        val_means, val_std, val_rs = make_results(dataset, name, num_runs, costs, validation=True)
        #pickle and write means, std, and rs to file
        with open(f'results/{dataset}/val_{name}_means.pkl', 'wb') as f:
            pickle.dump(val_means, f)
        with open(f'results/{dataset}/val_{name}_std.pkl', 'wb') as f:
            pickle.dump(val_std, f)
        with open(f'results/{dataset}/val_{name}_rs.pkl', 'wb') as f:
            pickle.dump(val_rs, f)
        

