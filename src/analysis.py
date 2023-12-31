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
from run import ADB
from run import evaluate_adb_model
from copy import deepcopy
import os

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
    for run in bar(range(num_runs)):

        
        bar=progressbar.ProgressBar()
        x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets(dataset, run)

        if validation==True:
            x_test = x_val
            y_test = y_val
            x_test_non_binarized = x_val_non_binarized
        
        whichtype = whichtype + ''
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
            brs_team_decision_loss = []


            tr_model_w_reset_contradictions = []
            tr_model_wo_reset_contradictions = []
            hyrs_model_contradictions = []
            brs_model_contradictions = []

            tr_team_w_reset_objective = []
            tr_team_wo_reset_objective = []
            tr_model_w_reset_objective = []
            tr_model_wo_reset_objective = []
            hyrs_model_objective = []
            hyrs_team_objective = []
            brs_model_objective = []
            brs_team_objective = []

            human_decision_loss = []
            hyrs_norecon_objective = []
            hyrs_norecon_model_decision_loss = []
            hyrs_norecon_team_decision_loss = []
            hyrs_norecon_model_contradictions = []
            
            if cost == 0:
                brs_mod.df = x_train
                brs_mod.Y = y_train
                brs_model_preds = brs_predict(brs_mod.opt_rules, x_test)
                brs_conf = brs_predict_conf(brs_mod.opt_rules, x_test, brs_mod)
                hyrs_norecon_mod = deepcopy(hyrs_mod)

            decs = {}
            decs['t']={'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['e'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['y'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['m'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['f'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['em'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['ef'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['ym'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            decs['yf'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}


            contras = {}
            contras['t']={'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['e'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['y'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['m'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['f'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['em'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['ef'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['ym'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            contras['yf'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}

            totals = {}
            for i in range(20):
                
                

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

                    c_model = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]



                        
                else:
                    learned_adb = ADB(adb_mod)
                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions,human_conf, human.ADB, with_reset=False, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)

                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)



                        

                    

                '''
                total = len(y_test)
                totals['t'] = len(y_test)
                totals['e'] = len(y_test[x_test['age54.0'] == 1])
                totals['y'] = len(y_test[x_test['age54.0'] == 0])
                totals['m'] = len(y_test[x_test['sex_Male'] == 1])
                totals['f'] = len(y_test[x_test['sex_Male'] == 0])
                totals['em'] = len(y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)])
                totals['ef'] = len(y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)])
                totals['ym'] = len(y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)])
                totals['yf'] = len(y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)])





                #find number of incorrect predictions confusion matrix split along sex_Male and age54.0 variables
                preds = tr_team_preds_with_reset.copy()
                decs['t']['tr'].append((preds != y_test).sum()/total)
                decs['e']['tr'].append((preds[x_test['age54.0'] == 1] != y_test[x_test['age54.0'] == 1]).sum()/total)
                decs['y']['tr'].append((preds[x_test['age54.0'] == 0] != y_test[x_test['age54.0'] == 0]).sum()/total)
                decs['m']['tr'].append((preds[x_test['sex_Male'] == 1] != y_test[x_test['sex_Male'] == 1]).sum()/total)
                decs['f']['tr'].append((preds[x_test['sex_Male'] == 0] != y_test[x_test['sex_Male'] == 0]).sum()/total)

                decs['em']['tr'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['ef']['tr'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                decs['ym']['tr'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['yf']['tr'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)

                contras['t']['tr'].append((preds != human_decisions).sum()/total)
                contras['e']['tr'].append((preds[x_test['age54.0'] == 1] != human_decisions[x_test['age54.0'] == 1]).sum()/total)
                contras['y']['tr'].append((preds[x_test['age54.0'] == 0] != human_decisions[x_test['age54.0'] == 0]).sum()/total)
                contras['m']['tr'].append((preds[x_test['sex_Male'] == 1] != human_decisions[x_test['sex_Male'] == 1]).sum()/total)
                contras['f']['tr'].append((preds[x_test['sex_Male'] == 0] != human_decisions[x_test['sex_Male'] == 0]).sum()/total)

                contras['em']['tr'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != human_decisions[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                contras['ef']['tr'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != human_decisions[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                contras['ym']['tr'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != human_decisions[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                contras['yf']['tr'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != human_decisions[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)

                preds = hyrs_team_preds.copy()
                decs['t']['hyrs'].append((preds != y_test).sum()/total)
                decs['e']['hyrs'].append((preds[x_test['age54.0'] == 1] != y_test[x_test['age54.0'] == 1]).sum()/total)
                decs['y']['hyrs'].append((preds[x_test['age54.0'] == 0] != y_test[x_test['age54.0'] == 0]).sum()/total)
                decs['m']['hyrs'].append((preds[x_test['sex_Male'] == 1] != y_test[x_test['sex_Male'] == 1]).sum()/total)
                decs['f']['hyrs'].append((preds[x_test['sex_Male'] == 0] != y_test[x_test['sex_Male'] == 0]).sum()/total)

                decs['em']['hyrs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['ef']['hyrs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                decs['ym']['hyrs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['yf']['hyrs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)

                contras['t']['hyrs'].append((preds != human_decisions).sum()/total)
                contras['e']['hyrs'].append((preds[x_test['age54.0'] == 1] != human_decisions[x_test['age54.0'] == 1]).sum()/total)
                contras['y']['hyrs'].append((preds[x_test['age54.0'] == 0] != human_decisions[x_test['age54.0'] == 0]).sum()/total)
                contras['m']['hyrs'].append((preds[x_test['sex_Male'] == 1] != human_decisions[x_test['sex_Male'] == 1]).sum()/total)
                contras['f']['hyrs'].append((preds[x_test['sex_Male'] == 0] != human_decisions[x_test['sex_Male'] == 0]).sum()/total)

                contras['em']['hyrs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != human_decisions[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                contras['ef']['hyrs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != human_decisions[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                contras['ym']['hyrs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != human_decisions[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                contras['yf']['hyrs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != human_decisions[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)

                preds = brs_team_preds.copy()
                decs['t']['brs'].append((preds != y_test).sum()/total)
                decs['e']['brs'].append((preds[x_test['age54.0'] == 1] != y_test[x_test['age54.0'] == 1]).sum()/total)
                decs['y']['brs'].append((preds[x_test['age54.0'] == 0] != y_test[x_test['age54.0'] == 0]).sum()/total)
                decs['m']['brs'].append((preds[x_test['sex_Male'] == 1] != y_test[x_test['sex_Male'] == 1]).sum()/total)
                decs['f']['brs'].append((preds[x_test['sex_Male'] == 0] != y_test[x_test['sex_Male'] == 0]).sum()/total)

                decs['em']['brs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['ef']['brs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                decs['ym']['brs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['yf']['brs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)


                contras['t']['brs'].append((preds != human_decisions).sum()/total)
                contras['e']['brs'].append((preds[x_test['age54.0'] == 1] != human_decisions[x_test['age54.0'] == 1]).sum()/total)
                contras['y']['brs'].append((preds[x_test['age54.0'] == 0] != human_decisions[x_test['age54.0'] == 0]).sum()/total)
                contras['m']['brs'].append((preds[x_test['sex_Male'] == 1] != human_decisions[x_test['sex_Male'] == 1]).sum()/total)
                contras['f']['brs'].append((preds[x_test['sex_Male'] == 0] != human_decisions[x_test['sex_Male'] == 0]).sum()/total)

                contras['em']['brs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != human_decisions[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                contras['ef']['brs'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != human_decisions[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                contras['ym']['brs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != human_decisions[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                contras['yf']['brs'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != human_decisions[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)
                
                preds = human_decisions.copy()
                decs['t']['human'].append((preds != y_test).sum()/total)
                decs['e']['human'].append((preds[x_test['age54.0'] == 1] != y_test[x_test['age54.0'] == 1]).sum()/total)
                decs['y']['human'].append((preds[x_test['age54.0'] == 0] != y_test[x_test['age54.0'] == 0]).sum()/total)
                decs['m']['human'].append((preds[x_test['sex_Male'] == 1] != y_test[x_test['sex_Male'] == 1]).sum()/total)
                decs['f']['human'].append((preds[x_test['sex_Male'] == 0] != y_test[x_test['sex_Male'] == 0]).sum()/total)

                decs['em']['human'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['ef']['human'].append((preds[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 1) & (x_test['sex_Male'] == 0)]).sum()/total)
                decs['ym']['human'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 1)]).sum()/total)
                decs['yf']['human'].append((preds[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)] != y_test[(x_test['age54.0'] == 0) & (x_test['sex_Male'] == 0)]).sum()/total)




                '''
                tr_team_w_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_with_reset, y_test))
                tr_team_wo_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_no_reset, y_test))
                tr_model_w_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_with_reset, y_test))
                tr_model_wo_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_no_reset, y_test))
                hyrs_model_decision_loss.append(1 - accuracy_score(hyrs_model_preds, y_test))
                hyrs_team_decision_loss.append(1 - accuracy_score(hyrs_team_preds, y_test))
                brs_model_decision_loss.append(1 - accuracy_score(brs_model_preds, y_test))
                brs_team_decision_loss.append(1 - accuracy_score(brs_team_preds, y_test))

                tr_model_w_reset_contradictions.append((tr_model_preds_with_reset != human_decisions).sum())
                tr_model_wo_reset_contradictions.append((tr_model_preds_no_reset != human_decisions).sum())
                hyrs_model_contradictions.append((hyrs_model_preds != human_decisions).sum())
                brs_model_contradictions.append((brs_model_preds != human_decisions).sum())

                tr_team_w_reset_objective.append(tr_team_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_team_wo_reset_objective.append(tr_team_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                tr_model_w_reset_objective.append(tr_model_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_model_wo_reset_objective.append(tr_model_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                hyrs_model_objective.append(hyrs_model_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                hyrs_team_objective.append(hyrs_team_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                brs_model_objective.append(brs_model_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_team_objective.append(brs_team_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))

                human_decision_loss.append(1 - accuracy_score(human_decisions, y_test))

                hyrs_norecon_model_decision_loss.append(1 - accuracy_score(hyrs_norecon_model_preds, y_test))
                hyrs_norecon_team_decision_loss.append(1 - accuracy_score(hyrs_norecon_team_preds, y_test))
                hyrs_norecon_model_contradictions.append((hyrs_norecon_model_preds != human_decisions).sum())
                hyrs_norecon_objective.append(hyrs_norecon_team_decision_loss[-1] + cost*(hyrs_norecon_model_contradictions[-1])/len(y_test))

                


                print(i)
            '''
            tr_confusion = pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                        data = [[mean(decs['em']['tr']), mean(decs['ym']['tr']), mean(decs['m']['tr'])], 
                                                [mean(decs['ef']['tr']), mean(decs['yf']['tr']), mean(decs['f']['tr'])], 
                                                [mean(decs['e']['tr']), mean(decs['y']['tr']), mean(decs['t']['tr'])]])
            
            brs_confusion = pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
            data = [[mean(decs['em']['brs']), mean(decs['ym']['brs']), mean(decs['m']['brs'])], 
                    [mean(decs['ef']['brs']), mean(decs['yf']['brs']), mean(decs['f']['brs'])], 
                    [mean(decs['e']['brs']), mean(decs['y']['brs']), mean(decs['t']['brs'])]])
            

            human_confusion = pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                                           data= [[mean(decs['em']['human']), mean(decs['ym']['human']), mean(decs['m']['human'])], 
                                                  [mean(decs['ef']['human']), mean(decs['yf']['human']), mean(decs['f']['human'])], 
                                                  [mean(decs['e']['human']), mean(decs['y']['human']), mean(decs['t']['human'])]])

            tr_confusion_contras = pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                        data = [[mean(contras['em']['tr']), mean(contras['ym']['tr']), mean(contras['m']['tr'])], 
                                                [mean(contras['ef']['tr']), mean(contras['yf']['tr']), mean(contras['f']['tr'])], 
                                                [mean(contras['e']['tr']), mean(contras['y']['tr']), mean(contras['t']['tr'])]])
            
            totals_confusion = pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
            data = [[totals['em'], totals['ym'], totals['m']], 
                    [totals['ef'], totals['yf'], totals['f']], 
                    [totals['e'], totals['y'], totals['t']]])
            
            brs_confusion_contras = pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
            data = [[mean(contras['em']['brs']), mean(contras['ym']['brs']), mean(contras['m']['brs'])], 
                    [mean(contras['ef']['brs']), mean(contras['yf']['brs']), mean(contras['f']['brs'])], 
                    [mean(contras['e']['brs']), mean(contras['y']['brs']), mean(contras['t']['brs'])]])
            

            '''
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
            
            



    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))




    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results



costs = [0, 0.2, 0.4, 0.6, 0.8, 1]
num_runs = 5
dataset = 'heart_disease'

name = 'offset_01'

if os.path.isfile(f'results/{dataset}/offset_01_rs.pkl'):
    with open(f'results/{dataset}/offset_01_rs.pkl', 'rb') as f:
        of1_rs = pickle.load(f)
    with open(f'results/{dataset}/offset_01_means.pkl', 'rb') as f:
        of1_means = pickle.load(f)
    with open(f'results/{dataset}/offset_01_std.pkl', 'rb') as f:
        of1_std = pickle.load(f)
else:
    of1_means, of1_std, of1_rs = make_results(dataset, name, num_runs, costs, False)
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/offset_01_means.pkl', 'wb') as f:
        pickle.dump(of1_means, f)
    with open(f'results/{dataset}/offset_01_std.pkl', 'wb') as f:
        pickle.dump(of1_std, f)
    with open(f'results/{dataset}/offset_01_rs.pkl', 'wb') as f:
        pickle.dump(of1_rs, f)

name = 'offset_02'
if os.path.isfile(f'results/{dataset}/offset_02_rs.pkl'):
    with open(f'results/{dataset}/offset_02_rs.pkl', 'rb') as f:
        of2_rs = pickle.load(f)
    with open(f'results/{dataset}/offset_02_means.pkl', 'rb') as f:
        of2_means = pickle.load(f)
    with open(f'results/{dataset}/offset_02_std.pkl', 'rb') as f:
        of2_std = pickle.load(f)
else:
    of2_means, of2_std, of2_rs = make_results(dataset, name, num_runs, costs, False)
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/offset_02_means.pkl', 'wb') as f:
        pickle.dump(of2_means, f)
    with open(f'results/{dataset}/offset_02_std.pkl', 'wb') as f:
        pickle.dump(of2_std, f)
    with open(f'results/{dataset}/offset_02_rs.pkl', 'wb') as f:
        pickle.dump(of2_rs, f)


#val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)
#misr_means, misr_stderrs, misrs = make_results(dataset, name, num_runs, costs, False)
#val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)



name = 'biased'
if os.path.isfile(f'results/{dataset}/biased_rs.pkl'):
    with open(f'results/{dataset}/biased_rs.pkl', 'rb') as f:
        bia_rs = pickle.load(f)
    with open(f'results/{dataset}/biased_means.pkl', 'rb') as f:
        bia_means = pickle.load(f)
    with open(f'results/{dataset}/biased_std.pkl', 'rb') as f:
        bia_std = pickle.load(f)
else:
    bia_means, bia_std, bia_rs = make_results(dataset, name, num_runs, costs, validation=False)
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/biased_means.pkl', 'wb') as f:
        pickle.dump(bia_means, f)
    with open(f'results/{dataset}/biased_std.pkl', 'wb') as f:
        pickle.dump(bia_std, f)
    with open(f'results/{dataset}/biased_rs.pkl', 'wb') as f:
        pickle.dump(bia_rs, f)

name = 'offset_01'

if os.path.isfile(f'results/{dataset}/val_offset_01_rs.pkl'):
    with open(f'results/{dataset}/val_offset_01_rs.pkl', 'rb') as f:
        val_of1_rs = pickle.load(f)
    with open(f'results/{dataset}/val_offset_01_means.pkl', 'rb') as f:
        val_of1_means = pickle.load(f)
    with open(f'results/{dataset}/val_offset_01_std.pkl', 'rb') as f:
        val_of1_std = pickle.load(f)
else:
    val_of1_means, val_of1_std, val_of1_rs = make_results(dataset, name, num_runs, costs, True)
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/val_offset_01_means.pkl', 'wb') as f:
        pickle.dump(val_of1_means, f)
    with open(f'results/{dataset}/val_offset_01_std.pkl', 'wb') as f:
        pickle.dump(val_of1_std, f)
    with open(f'results/{dataset}/val_offset_01_rs.pkl', 'wb') as f:
        pickle.dump(val_of1_rs, f)

name = 'offset_02'
if os.path.isfile(f'results/{dataset}/val_offset_02_rs.pkl'):
    with open(f'results/{dataset}/val_offset_02_rs.pkl', 'rb') as f:
        val_of2_rs = pickle.load(f)
    with open(f'results/{dataset}/val_offset_02_means.pkl', 'rb') as f:
        val_of2_means = pickle.load(f)
    with open(f'results/{dataset}/val_offset_02_std.pkl', 'rb') as f:
        val_of2_std = pickle.load(f)
else:
    val_of2_means, val_of2_std, val_of2_rs = make_results(dataset, name, num_runs, costs, True)
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/val_offset_02_means.pkl', 'wb') as f:
        pickle.dump(val_of2_means, f)
    with open(f'results/{dataset}/val_offset_02_std.pkl', 'wb') as f:
        pickle.dump(val_of2_std, f)
    with open(f'results/{dataset}/val_offset_02_rs.pkl', 'wb') as f:
        pickle.dump(val_of2_rs, f)


#val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)
#misr_means, misr_stderrs, misrs = make_results(dataset, name, num_runs, costs, False)
#val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)



name = 'biased'
if os.path.isfile(f'results/{dataset}/val_biased_rs.pkl'):
    with open(f'results/{dataset}/val_biased_rs.pkl', 'rb') as f:
        val_bia_rs = pickle.load(f)
    with open(f'results/{dataset}/val_biased_means.pkl', 'rb') as f:
        val_bia_means = pickle.load(f)
    with open(f'results/{dataset}/val_biased_std.pkl', 'rb') as f:
        val_bia_std = pickle.load(f)
else:
    val_bia_means, val_bia_std, val_bia_rs = make_results(dataset, name, num_runs, costs, validation=True)
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/val_biased_means.pkl', 'wb') as f:
        pickle.dump(val_bia_means, f)
    with open(f'results/{dataset}/val_biased_std.pkl', 'wb') as f:
        pickle.dump(val_bia_std, f)
    with open(f'results/{dataset}/val_biased_rs.pkl', 'wb') as f:
        pickle.dump(val_bia_rs, f)

#val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)
#calr_means, calr_stderrs, calrs = make_results(dataset, name, num_runs, costs, False)
#calval_r_means, calval_r_stderrs, calval_rs = make_results('heart_disease', name, num_runs, costs, True)


#offval_r_means, offval_r_stderrs, offval_rs = make_results('heart_disease', name, num_runs, costs, True)

#cost = 0.2
#robust_rs = rs.copy()
#for i in range(len(val_rs['tr_team_w_reset_objective'][cost])):
#    if val_rs['tr_team_w_reset_objective'][cost][i] > val_rs['hyrs_team_objective'][cost][i]:
#        if val_rs['hyrs_team_objective'][cost][i] > val_rs['brs_team_objective'][cost][i]:
#            robust_rs['tr_team_w_reset_objective'][cost][i] = rs['brs_team_objective'][cost][i]
#            robust_rs['tr_model_w_reset_contradictions'][cost][i] = rs['brs_model_contradictions'][cost][i]
#            robust_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['brs_team_decision_loss'][cost][i]#


#        else:
#            robust_rs['tr_team_w_reset_objective'][cost][i] = rs['hyrs_team_objective'][cost][i]
#            robust_rs['tr_model_w_reset_contradictions'][cost][i] = rs['hyrs_model_contradictions'][cost][i]
#            robust_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['hyrs_team_decision_loss'][cost][i]#
#
#robust_rs.apply(lambda x: x.apply(lambda y: mean(y)))[['tr_team_w_reset_objective', 'tr_team_wo_reset_objective', 'hyrs_team_objective', 'brs_team_objective', 'human_decision_loss']]

#mis_r_means, mis_r_stderrs, mis_rs = make_results('heart_disease', 'miscalibrated', num_runs, costs, False)




def make_TL_v_cost_plot(results_means, results_stderrs, name):
    fig = plt.figure(figsize=(3, 2), dpi=200)
    color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42'}
    plt.plot(results_means.index[0:6], results_means['hyrs_norecon_objective'].iloc[0:6], marker = 'v', c=color_dict['HYRS'], label = 'TR-No(ADB, OrgVal)', markersize=1.8, linewidth=0.9)
    plt.plot(results_means.index[0:6], results_means['hyrs_team_objective'].iloc[0:6], marker = 'x', c=color_dict['HYRSRecon'], label = 'TR-No(ADB)', markersize=1.8, linewidth=0.9)
    plt.plot(results_means.index[0:6], results_means['tr_team_w_reset_objective'].iloc[0:6], marker = '.', c=color_dict['TR'], label='TR', markersize=1.8, linewidth=0.9)
    plt.plot(results_means.index[0:6], results_means['brs_team_objective'].iloc[0:6], marker = 's', c=color_dict['BRS'], label='Task-Only (Current Practice)', markersize=1.8, linewidth=0.9)
    
    plt.plot(results_means.index[0:6], results_means['human_decision_loss'].iloc[0:6], c = color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    
    plt.fill_between(results_means.index[0:6], 
                results_means['human_decision_loss'].iloc[0:6]-(results_stderrs['human_decision_loss'].iloc[0:6]),
                results_means['human_decision_loss'].iloc[0:6]+(results_stderrs['human_decision_loss'].iloc[0:6]) ,
                color=color_dict['Human'], alpha=0.2)
    plt.fill_between(results_means.index[0:6], 
                results_means['hyrs_team_objective'].iloc[0:6]-(results_stderrs['hyrs_team_objective'].iloc[0:6]),
                results_means['hyrs_team_objective'].iloc[0:6]+(results_stderrs['hyrs_team_objective'].iloc[0:6]) ,
                color=color_dict['HYRSRecon'], alpha=0.2)
    
    plt.fill_between(results_means.index[0:6], 
                results_means['hyrs_norecon_objective'].iloc[0:6]-(results_stderrs['hyrs_norecon_objective'].iloc[0:6]),
                results_means['hyrs_norecon_objective'].iloc[0:6]+(results_stderrs['hyrs_norecon_objective'].iloc[0:6]) ,
                color=color_dict['HYRS'], alpha=0.2)
    plt.fill_between(results_means.index[0:6], 
                results_means['brs_team_objective'].iloc[0:6]-(results_stderrs['brs_team_objective'].iloc[0:6]),
                results_means['brs_team_objective'].iloc[0:6]+(results_stderrs['brs_team_objective'].iloc[0:6]) ,
                color=color_dict['BRS'], alpha=0.2)
    plt.fill_between(results_means.index[0:6], 
                results_means['tr_team_w_reset_objective'].iloc[0:6]-(results_stderrs['tr_team_w_reset_objective'].iloc[0:6]),
                results_means['tr_team_w_reset_objective'].iloc[0:6]+(results_stderrs['tr_team_w_reset_objective'].iloc[0:6]),
                color=color_dict['TR'], alpha=0.2)
   
    plt.xlabel('Reconciliation Cost', fontsize=12)
    plt.ylabel('Total Team Loss', fontsize=12)
    plt.tick_params(labelrotation=45, labelsize=10)
    #plt.title('{} Setting'.format(setting), fontsize=15)
    plt.legend(prop={'size': 5})
    plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

    fig.savefig(f'results/{dataset}/plots/TL_{dataset}_{name}.png', bbox_inches='tight')
    #plt.show()

    #plt.clf()


def make_contradictions_v_decisionloss_plot(results_means, results_stderrs, name):
    fig = plt.figure(figsize=(3, 2), dpi=200)
    color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42'}
    plt.plot(results_means['brs_model_contradictions'], results_means['brs_team_decision_loss'].iloc[0:6], marker = 'v', c=color_dict['BRS'], label = 'Task-Only (Current Practice)', markersize=1.8, linewidth=0.9)
    #plt.plot(results_means['hyrs_model_contradictions'], results_means['hyrs_team_objective'].iloc[0:6], marker = 'x', c=color_dict['HYRSRecon'], label = 'TR-No(ADB)', markersize=1.8, linewidth=0.9)
    plt.plot(results_means['tr_model_w_reset_contradictions'], results_means['tr_team_w_reset_decision_loss'].iloc[0:6], marker = '.', c=color_dict['TR'], label='TR', markersize=1.8, linewidth=0.9)
    
    
    
    plt.plot([0,0,0,0,0, 0], results_means['human_decision_loss'].iloc[0:6], c = color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    
    for cost in results_stderrs.index:
        plt.fill_between(np.linspace(results_means.loc[cost, 'brs_model_contradictions'] - results_stderrs.loc[cost, 'brs_model_contradictions'], 
                                     results_means.loc[cost, 'brs_model_contradictions'] + results_stderrs.loc[cost, 'brs_model_contradictions'], 50), 
                    results_means.loc[cost, 'brs_team_decision_loss']-(results_stderrs.loc[cost, 'brs_team_decision_loss']),
                    results_means.loc[cost, 'brs_team_decision_loss']+(results_stderrs.loc[cost, 'brs_team_decision_loss']),
                    color=color_dict['BRS'], alpha=0.2)
        
        #plt.text(results_means.loc[cost, 'tr_model_w_reset_contradictions'], results_means.loc[cost, 'tr_team_w_reset_decision_loss'], f'cost = {str(cost)}', fontsize=4)
        
        plt.fill_between(np.linspace(results_means.loc[cost, 'tr_model_w_reset_contradictions'] - results_stderrs.loc[cost, 'tr_model_w_reset_contradictions'], 
                                    results_means.loc[cost, 'tr_model_w_reset_contradictions'] + results_stderrs.loc[cost, 'tr_model_w_reset_contradictions'], 50), 
                    results_means.loc[cost, 'tr_team_w_reset_decision_loss']-(results_stderrs.loc[cost, 'tr_team_w_reset_decision_loss']),
                    results_means.loc[cost, 'tr_team_w_reset_decision_loss']+(results_stderrs.loc[cost, 'tr_team_w_reset_decision_loss']),
                    color=color_dict['TR'], alpha=0.2)
   
    plt.xlabel('Contradictions', fontsize=12)
    plt.ylabel('Team Decision Loss', fontsize=12)
    plt.tick_params(labelrotation=45, labelsize=10)
    #plt.title('{} Setting'.format(setting), fontsize=15)
    plt.legend(prop={'size': 5})
    plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

    fig.savefig(f'results/{dataset}/plots/TDL_{dataset}_{name}.png', bbox_inches='tight')
    #plt.show()

    #plt.clf()




    
def cost_validation(rs, val_rs):
    new_rs = deepcopy(rs)
    for cost in rs.index:
        if cost==0.8:
            print('pause')
        for column in rs.columns:
            new_rs.loc[cost, column] = deepcopy(rs.loc[cost, column])
        for i in range(len(val_rs['tr_team_w_reset_objective'][cost])):
            x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized= load_datasets(dataset, i)
            curr_val_objective = val_rs['tr_team_w_reset_objective'][cost][i]
            for alt_cost in rs.index:
                alt_val_objective = val_rs['tr_team_w_reset_decision_loss'][alt_cost][i] + cost*(val_rs['tr_model_w_reset_contradictions'][alt_cost][i])/len(y_val)
                if alt_val_objective < curr_val_objective:
                    new_rs['tr_model_w_reset_contradictions'][cost][i] = rs['tr_model_w_reset_contradictions'][alt_cost][i].copy()
                    new_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['tr_team_w_reset_decision_loss'][alt_cost][i].copy()
                    new_rs['tr_team_w_reset_objective'][cost][i] = new_rs['tr_team_w_reset_decision_loss'][alt_cost][i] + cost*new_rs['tr_model_w_reset_contradictions'][alt_cost][i]/len(y_test)
                    print(f"cost: {cost}, new cost: {alt_cost}, i: {i}, replacing actual of {rs['tr_team_w_reset_objective'][cost][i]} with new of {new_rs['tr_team_w_reset_objective'][cost][i]}")
                    curr_val_objective = alt_val_objective
    new_results_means = new_rs.apply(lambda x: x.apply(lambda y: mean(y)))
    new_results_stderrs = new_rs.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return new_results_means, new_results_stderrs, new_rs



    
cval_of1_means, cval_of1_stderss, cval_of1_rs = cost_validation(of1_rs, val_of1_rs)        

cval_of2_means, cval_of2_stderss, cval_of2_rs = cost_validation(of2_rs, val_of2_rs)   

cval_bia_means, cval_bia_stderss, cval_bia_rs = cost_validation(bia_rs, val_bia_rs)     
                            
    



    

print('pause')

