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



def make_results(dataset, whichtype, num_runs, costs, validation=False, asym_costs=[1,1]):

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
    whichtype = whichtype #+ 'quickTest' #+ "_dec_bias"
    r_mean = []
    hyrs_R = []
    tr_R = []
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

            model_decs = {}
            model_decs['t']={'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['e'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['y'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['m'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}    
            model_decs['f'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['em'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['ef'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['ym'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            model_decs['yf'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}


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

            correct_contras = {}
            correct_contras['t']={'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['e'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['y'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['m'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['f'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['em'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['ef'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['ym'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            correct_contras['yf'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}

            accepted_contras = {}
            accepted_contras['t']={'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['e'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['y'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['m'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['f'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['em'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['ef'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['ym'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            accepted_contras['yf'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}

            covereds = {}
            covereds['t']={'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['e'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['y'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['m'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['f'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['em'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['ef'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['ym'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}
            covereds['yf'] = {'tr': [], 'hyrs': [], 'brs': [], 'human': []}



            totals = {}
            for i in range(50):
                
                

                if validation: 

                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    conf_mod_preds = conf_mod.predict(x_test_non_binarized)

                    learned_adb = ADB(adb_mod)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

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
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions,human_conf, human.ADB, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset, tr_mod_covered_w_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr_model_preds_no_reset, tr_mod_covered_no_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr_mod_confs = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]
                

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)
                    #brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized), conf_model=brs_conf, fA=learned_adb.ADB_model_wrapper, asym_loss=[1,1], contradiction_reg=0.0)
                    #brs_team_preds[brs_reset] == human_decisions[brs_reset]

                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)



                        

                    

                
                total = len(y_test)
                totals['t'] = len(y_test)
                totals['e'] = len(y_test[x_test['age54.0'] == 1])
                totals['y'] = len(y_test[x_test['age54.0'] == 0])
                totals['m'] = len(y_test[x_test['NumSatisfactoryTrades24.0'] == 1])
                totals['f'] = len(y_test[x_test['NumSatisfactoryTrades24.0'] == 0])
                totals['em'] = len(y_test[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)])
                totals['ef'] = len(y_test[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)])
                totals['ym'] = len(y_test[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)])
                totals['yf'] = len(y_test[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)])




                which_ones = ['tr', 'hyrs', 'brs', 'human']

                for which in which_ones:
                #find number of incorrect predictions confusion matrix split along sex_Male and age54.0 variables
                    if which =='tr':
                        preds = tr_team_preds_with_reset.copy()
                        
                    elif which == 'hyrs':
                        preds = hyrs_team_preds.copy()
                        
                    elif which == 'brs':
                        preds = brs_team_preds.copy()
                        
                    elif which == 'human':
                        preds = human_decisions.copy()
                    
                    asymCosts = y_test.replace({0: asym_costs[1], 1: asym_costs[0]}) 
                    decs['t'][which].append(((preds != y_test)*asymCosts).sum())
                    decs['e'][which].append(((preds != y_test)*asymCosts)[x_test['age54.0'] == 1].sum())
                    decs['y'][which].append(((preds != y_test)*asymCosts)[x_test['age54.0'] == 0].sum())
                    decs['m'][which].append(((preds != y_test)*asymCosts)[x_test['NumSatisfactoryTrades24.0'] == 1].sum())
                    decs['f'][which].append(((preds != y_test)*asymCosts)[x_test['NumSatisfactoryTrades24.0'] == 0].sum())

                    decs['em'][which].append(((preds != y_test)*asymCosts)[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    decs['ef'][which].append(((preds != y_test)*asymCosts)[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                    decs['ym'][which].append(((preds != y_test)*asymCosts)[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    decs['yf'][which].append(((preds != y_test)*asymCosts)[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())







                    
                    if which =='tr':
                        model_preds = tr_model_preds_with_reset.copy()
                        model_covereds = np.zeros(len(model_preds), dtype=bool)
                        model_covereds[tr_mod_covered_w_reset] = True

                    elif which == 'hyrs':
                        model_preds = hyrs_model_preds.copy()
                    elif which == 'brs':
                        model_preds = brs_model_preds.copy()
                    elif which == 'human':
                        model_preds = human_decisions.copy()
                
                    contras['t'][which].append((model_preds != human_decisions).sum())
                    contras['e'][which].append((model_preds != human_decisions)[x_test['age54.0'] == 1].sum())
                    contras['y'][which].append((model_preds != human_decisions)[x_test['age54.0'] == 0].sum())
                    contras['m'][which].append((model_preds != human_decisions)[x_test['NumSatisfactoryTrades24.0'] == 1].sum())
                    contras['f'][which].append((model_preds != human_decisions)[x_test['NumSatisfactoryTrades24.0'] == 0].sum())

                    contras['em'][which].append((model_preds != human_decisions)[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    contras['ef'][which].append((model_preds != human_decisions)[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                    contras['ym'][which].append((model_preds != human_decisions)[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    contras['yf'][which].append((model_preds != human_decisions)[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())

                    model_decs['t'][which].append(((model_preds != y_test)*asymCosts).sum())
                    model_decs['e'][which].append(((model_preds != y_test)*asymCosts)[x_test['age54.0'] == 1].sum())
                    model_decs['y'][which].append(((model_preds != y_test)*asymCosts)[x_test['age54.0'] == 0].sum())
                    model_decs['m'][which].append(((model_preds != y_test)*asymCosts)[x_test['NumSatisfactoryTrades24.0'] == 1].sum())
                    model_decs['f'][which].append(((model_preds != y_test)*asymCosts)[x_test['NumSatisfactoryTrades24.0'] == 0].sum())

                    model_decs['em'][which].append(((model_preds != y_test)*asymCosts)[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    model_decs['ef'][which].append(((model_preds != y_test)*asymCosts)[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                    model_decs['ym'][which].append(((model_preds != y_test)*asymCosts)[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    model_decs['yf'][which].append(((model_preds != y_test)*asymCosts)[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())

                    correct_contras['t'][which].append(((model_preds != human_decisions) & (model_preds == y_test)).sum())
                    correct_contras['e'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[x_test['age54.0'] == 1].sum())
                    correct_contras['y'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[x_test['age54.0'] == 0].sum())
                    correct_contras['m'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[x_test['NumSatisfactoryTrades24.0'] == 1].sum())
                    correct_contras['f'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[x_test['NumSatisfactoryTrades24.0'] == 0].sum())

                    correct_contras['em'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    correct_contras['ef'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                    correct_contras['ym'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    correct_contras['yf'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())

                    accepted_condition = (model_preds != human_decisions) & (model_preds == preds)

                    accepted_contras['t'][which].append(accepted_condition.sum())
                    accepted_contras['e'][which].append(accepted_condition[x_test['age54.0'] == 1].sum())
                    accepted_contras['y'][which].append(accepted_condition[x_test['age54.0'] == 0].sum())
                    accepted_contras['m'][which].append(accepted_condition[x_test['NumSatisfactoryTrades24.0'] == 1].sum())
                    accepted_contras['f'][which].append(accepted_condition[x_test['NumSatisfactoryTrades24.0'] == 0].sum())
                    accepted_contras['em'][which].append(accepted_condition[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    accepted_contras['ef'][which].append(accepted_condition[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                    accepted_contras['ym'][which].append(accepted_condition[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                    accepted_contras['yf'][which].append(accepted_condition[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())

                    if which =='tr':
                        covereds['t'][which].append(model_covereds.sum())
                        covereds['e'][which].append(model_covereds[x_test['age54.0'] == 1].sum())
                        covereds['y'][which].append(model_covereds[x_test['age54.0'] == 0].sum())
                        covereds['m'][which].append(model_covereds[x_test['NumSatisfactoryTrades24.0'] == 1].sum())
                        covereds['f'][which].append(model_covereds[x_test['NumSatisfactoryTrades24.0'] == 0].sum())
                        covereds['em'][which].append(model_covereds[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                        covereds['ef'][which].append(model_covereds[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                        covereds['ym'][which].append(model_covereds[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1)].sum())
                        covereds['yf'][which].append(model_covereds[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0)].sum())
                        

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
            
            if run==0:

                tr_conf_confusion = [pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[tr_mod_confs[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['NumSatisfactoryTrades24.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['NumSatisfactoryTrades24.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(x_test['age54.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['age54.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(tr_model_preds_with_reset != human_decisions)].mean()]])]
                
                
                
                brs_conf_confusion = [pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                data = [[brs_conf[(x_test['age54.0'] == 1) &(x_test['NumSatisfactoryTrades24.0'] == 1) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['NumSatisfactoryTrades24.0'] == 1) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['NumSatisfactoryTrades24.0'] == 0) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(x_test['age54.0'] == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['age54.0'] == 0) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(brs_model_preds != human_decisions)].mean()]])]
                
                tr_covered_confusion = [pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                data = [[mean(covereds['em']['tr']), mean(covereds['ym']['tr']), mean(covereds['m']['tr'])], 
                                        [mean(covereds['ef']['tr']), mean(covereds['yf']['tr']), mean(covereds['f']['tr'])], 
                                        [mean(covereds['e']['tr']), mean(covereds['y']['tr']), mean(covereds['t']['tr'])]])]
                

                tr_confusion = [pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(decs['em']['tr']), mean(decs['ym']['tr']), mean(decs['m']['tr'])], 
                                                    [mean(decs['ef']['tr']), mean(decs['yf']['tr']), mean(decs['f']['tr'])], 
                                                    [mean(decs['e']['tr']), mean(decs['y']['tr']), mean(decs['t']['tr'])]])]
                
                tr_model_confusion = [pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(model_decs['em']['tr']), mean(model_decs['ym']['tr']), mean(model_decs['m']['tr'])], 
                                                    [mean(model_decs['ef']['tr']), mean(model_decs['yf']['tr']), mean(model_decs['f']['tr'])], 
                                                    [mean(model_decs['e']['tr']), mean(model_decs['y']['tr']), mean(model_decs['t']['tr'])]])]
                
                hyrs_confusion = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(decs['em']['hyrs']), mean(decs['ym']['hyrs']), mean(decs['m']['hyrs'])], 
                                                    [mean(decs['ef']['hyrs']), mean(decs['yf']['hyrs']), mean(decs['f']['hyrs'])], 
                                                    [mean(decs['e']['hyrs']), mean(decs['y']['hyrs']), mean(decs['t']['hyrs'])]])]
                
                hyrs_model_confusion = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(model_decs['em']['hyrs']), mean(model_decs['ym']['hyrs']), mean(model_decs['m']['hyrs'])], 
                                                    [mean(model_decs['ef']['hyrs']), mean(model_decs['yf']['hyrs']), mean(model_decs['f']['hyrs'])], 
                                                    [mean(model_decs['e']['hyrs']), mean(model_decs['y']['hyrs']), mean(model_decs['t']['hyrs'])]])]
                
                brs_confusion = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[mean(decs['em']['brs']), mean(decs['ym']['brs']), mean(decs['m']['brs'])], 
                        [mean(decs['ef']['brs']), mean(decs['yf']['brs']), mean(decs['f']['brs'])], 
                        [mean(decs['e']['brs']), mean(decs['y']['brs']), mean(decs['t']['brs'])]])]
                
                brs_model_confusion = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[mean(model_decs['em']['brs']), mean(model_decs['ym']['brs']), mean(model_decs['m']['brs'])], 
                        [mean(model_decs['ef']['brs']), mean(model_decs['yf']['brs']), mean(model_decs['f']['brs'])], 
                        [mean(model_decs['e']['brs']), mean(model_decs['y']['brs']), mean(model_decs['t']['brs'])]])]
                

                human_confusion = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                                            data= [[mean(decs['em']['human']), mean(decs['ym']['human']), mean(decs['m']['human'])], 
                                                    [mean(decs['ef']['human']), mean(decs['yf']['human']), mean(decs['f']['human'])], 
                                                    [mean(decs['e']['human']), mean(decs['y']['human']), mean(decs['t']['human'])]])]

                tr_confusion_contras = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(contras['em']['tr']), mean(contras['ym']['tr']), mean(contras['m']['tr'])], 
                                                    [mean(contras['ef']['tr']), mean(contras['yf']['tr']), mean(contras['f']['tr'])], 
                                                    [mean(contras['e']['tr']), mean(contras['y']['tr']), mean(contras['t']['tr'])]])]
                
                hyrs_confusion_contras = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(contras['em']['hyrs']), mean(contras['ym']['hyrs']), mean(contras['m']['hyrs'])], 
                                                    [mean(contras['ef']['hyrs']), mean(contras['yf']['hyrs']), mean(contras['f']['hyrs'])], 
                                                    [mean(contras['e']['hyrs']), mean(contras['y']['hyrs']), mean(contras['t']['hyrs'])]])]

                tr_confusion_contras_accepted = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(accepted_contras['em']['tr']), mean(accepted_contras['ym']['tr']), mean(accepted_contras['m']['tr'])], 
                                                    [mean(accepted_contras['ef']['tr']), mean(accepted_contras['yf']['tr']), mean(accepted_contras['f']['tr'])], 
                                                    [mean(accepted_contras['e']['tr']), mean(accepted_contras['y']['tr']), mean(accepted_contras['t']['tr'])]])]
                
                tr_confusion_contras_correct = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(correct_contras['em']['tr']), mean(correct_contras['ym']['tr']), mean(correct_contras['m']['tr'])], 
                                                    [mean(correct_contras['ef']['tr']), mean(correct_contras['yf']['tr']), mean(correct_contras['f']['tr'])], 
                                                    [mean(correct_contras['e']['tr']), mean(correct_contras['y']['tr']), mean(correct_contras['t']['tr'])]])]
                
                hyrs_confusion_contras_correct = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                            data = [[mean(correct_contras['em']['hyrs']), mean(correct_contras['ym']['hyrs']), mean(correct_contras['m']['hyrs'])], 
                                    [mean(correct_contras['ef']['hyrs']), mean(correct_contras['yf']['hyrs']), mean(correct_contras['f']['hyrs'])], 
                                    [mean(correct_contras['e']['hyrs']), mean(correct_contras['y']['hyrs']), mean(correct_contras['t']['hyrs'])]])]
                
                totals_confusion = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[totals['em'], totals['ym'], totals['m']], 
                        [totals['ef'], totals['yf'], totals['f']], 
                        [totals['e'], totals['y'], totals['t']]])]
                
                brs_confusion_contras = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[mean(contras['em']['brs']), mean(contras['ym']['brs']), mean(contras['m']['brs'])], 
                        [mean(contras['ef']['brs']), mean(contras['yf']['brs']), mean(contras['f']['brs'])], 
                        [mean(contras['e']['brs']), mean(contras['y']['brs']), mean(contras['t']['brs'])]])]
                
                brs_confusion_contras_correct = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(correct_contras['em']['brs']), mean(correct_contras['ym']['brs']), mean(correct_contras['m']['brs'])], 
                                                    [mean(correct_contras['ef']['brs']), mean(correct_contras['yf']['brs']), mean(correct_contras['f']['brs'])], 
                                                    [mean(correct_contras['e']['brs']), mean(correct_contras['y']['brs']), mean(correct_contras['t']['brs'])]])]
                
                brs_confusion_contras_accepted = [pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(accepted_contras['em']['brs']), mean(accepted_contras['ym']['brs']), mean(accepted_contras['m']['brs'])], 
                                                    [mean(accepted_contras['ef']['brs']), mean(accepted_contras['yf']['brs']), mean(accepted_contras['f']['brs'])], 
                                                    [mean(accepted_contras['e']['brs']), mean(accepted_contras['y']['brs']), mean(accepted_contras['t']['brs'])]])]
                
            else:
                tr_confusion.append(pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(decs['em']['tr']), mean(decs['ym']['tr']), mean(decs['m']['tr'])], 
                                                    [mean(decs['ef']['tr']), mean(decs['yf']['tr']), mean(decs['f']['tr'])], 
                                                    [mean(decs['e']['tr']), mean(decs['y']['tr']), mean(decs['t']['tr'])]]))
                
                tr_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[tr_mod_confs[(x_test['age54.0'] == 1) &(x_test['NumSatisfactoryTrades24.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['NumSatisfactoryTrades24.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['NumSatisfactoryTrades24.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(x_test['age54.0'] == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(x_test['age54.0'] == 0) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(tr_model_preds_with_reset != human_decisions)].mean()]]))
                
                tr_covered_confusion.append(pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(decs['em']['tr']), mean(decs['ym']['tr']), mean(decs['m']['tr'])], 
                                                    [mean(decs['ef']['tr']), mean(decs['yf']['tr']), mean(decs['f']['tr'])], 
                                                    [mean(decs['e']['tr']), mean(decs['y']['tr']), mean(decs['t']['tr'])]]))
                
                brs_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                data = [[brs_conf[(x_test['age54.0'] == 1) &(x_test['NumSatisfactoryTrades24.0'] == 1) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['NumSatisfactoryTrades24.0'] == 1) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(x_test['age54.0'] == 1) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['age54.0'] == 0) & (x_test['NumSatisfactoryTrades24.0'] == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['NumSatisfactoryTrades24.0'] == 0) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(x_test['age54.0'] == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(x_test['age54.0'] == 0) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(brs_model_preds != human_decisions)].mean()]]))
                
                brs_model_confusion.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[mean(model_decs['em']['brs']), mean(model_decs['ym']['brs']), mean(model_decs['m']['brs'])], 
                        [mean(model_decs['ef']['brs']), mean(model_decs['yf']['brs']), mean(model_decs['f']['brs'])], 
                        [mean(model_decs['e']['brs']), mean(model_decs['y']['brs']), mean(model_decs['t']['brs'])]]))
                
                tr_model_confusion.append(pd.DataFrame(dtype = 'float', index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(model_decs['em']['tr']), mean(model_decs['ym']['tr']), mean(model_decs['m']['tr'])], 
                                                    [mean(model_decs['ef']['tr']), mean(model_decs['yf']['tr']), mean(model_decs['f']['tr'])], 
                                                    [mean(model_decs['e']['tr']), mean(model_decs['y']['tr']), mean(model_decs['t']['tr'])]]))
                
                hyrs_confusion.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(decs['em']['hyrs']), mean(decs['ym']['hyrs']), mean(decs['m']['hyrs'])], 
                                                    [mean(decs['ef']['hyrs']), mean(decs['yf']['hyrs']), mean(decs['f']['hyrs'])], 
                                                    [mean(decs['e']['hyrs']), mean(decs['y']['hyrs']), mean(decs['t']['hyrs'])]]))
                
                hyrs_model_confusion.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(model_decs['em']['hyrs']), mean(model_decs['ym']['hyrs']), mean(model_decs['m']['hyrs'])], 
                                                    [mean(model_decs['ef']['hyrs']), mean(model_decs['yf']['hyrs']), mean(model_decs['f']['hyrs'])], 
                                                    [mean(model_decs['e']['hyrs']), mean(model_decs['y']['hyrs']), mean(model_decs['t']['hyrs'])]]))
                
                hyrs_confusion_contras.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                            data = [[mean(contras['em']['hyrs']), mean(contras['ym']['hyrs']), mean(contras['m']['hyrs'])], 
                                    [mean(contras['ef']['hyrs']), mean(contras['yf']['hyrs']), mean(contras['f']['hyrs'])], 
                                    [mean(contras['e']['hyrs']), mean(contras['y']['hyrs']), mean(contras['t']['hyrs'])]]))
                
                brs_confusion.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[mean(decs['em']['brs']), mean(decs['ym']['brs']), mean(decs['m']['brs'])], 
                        [mean(decs['ef']['brs']), mean(decs['yf']['brs']), mean(decs['f']['brs'])], 
                        [mean(decs['e']['brs']), mean(decs['y']['brs']), mean(decs['t']['brs'])]]))
                

                human_confusion.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                                            data= [[mean(decs['em']['human']), mean(decs['ym']['human']), mean(decs['m']['human'])], 
                                                    [mean(decs['ef']['human']), mean(decs['yf']['human']), mean(decs['f']['human'])], 
                                                    [mean(decs['e']['human']), mean(decs['y']['human']), mean(decs['t']['human'])]]))

                tr_confusion_contras.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(contras['em']['tr']), mean(contras['ym']['tr']), mean(contras['m']['tr'])], 
                                                    [mean(contras['ef']['tr']), mean(contras['yf']['tr']), mean(contras['f']['tr'])], 
                                                    [mean(contras['e']['tr']), mean(contras['y']['tr']), mean(contras['t']['tr'])]]))
                tr_confusion_contras_accepted.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(accepted_contras['em']['tr']), mean(accepted_contras['ym']['tr']), mean(accepted_contras['m']['tr'])], 
                                                    [mean(accepted_contras['ef']['tr']), mean(accepted_contras['yf']['tr']), mean(accepted_contras['f']['tr'])], 
                                                    [mean(accepted_contras['e']['tr']), mean(accepted_contras['y']['tr']), mean(accepted_contras['t']['tr'])]]))
                
                tr_confusion_contras_correct.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(correct_contras['em']['tr']), mean(correct_contras['ym']['tr']), mean(correct_contras['m']['tr'])], 
                                                    [mean(correct_contras['ef']['tr']), mean(correct_contras['yf']['tr']), mean(correct_contras['f']['tr'])], 
                                                    [mean(correct_contras['e']['tr']), mean(correct_contras['y']['tr']), mean(correct_contras['t']['tr'])]]))
                
                hyrs_confusion_contras_correct.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                            data = [[mean(correct_contras['em']['hyrs']), mean(correct_contras['ym']['hyrs']), mean(correct_contras['m']['hyrs'])], 
                                    [mean(correct_contras['ef']['hyrs']), mean(correct_contras['yf']['hyrs']), mean(correct_contras['f']['hyrs'])], 
                                    [mean(correct_contras['e']['hyrs']), mean(correct_contras['y']['hyrs']), mean(correct_contras['t']['hyrs'])]]))
                
                totals_confusion.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[totals['em'], totals['ym'], totals['m']], 
                        [totals['ef'], totals['yf'], totals['f']], 
                        [totals['e'], totals['y'], totals['t']]]))
                
                brs_confusion_contras.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'],
                data = [[mean(contras['em']['brs']), mean(contras['ym']['brs']), mean(contras['m']['brs'])], 
                        [mean(contras['ef']['brs']), mean(contras['yf']['brs']), mean(contras['f']['brs'])], 
                        [mean(contras['e']['brs']), mean(contras['y']['brs']), mean(contras['t']['brs'])]]))
                
                brs_confusion_contras_correct.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(correct_contras['em']['brs']), mean(correct_contras['ym']['brs']), mean(correct_contras['m']['brs'])], 
                                                    [mean(correct_contras['ef']['brs']), mean(correct_contras['yf']['brs']), mean(correct_contras['f']['brs'])], 
                                                    [mean(correct_contras['e']['brs']), mean(correct_contras['y']['brs']), mean(correct_contras['t']['brs'])]]))
                
                brs_confusion_contras_accepted.append(pd.DataFrame(index=['ManyTrades', 'FewTrades', 'Total'], columns=['HighRisk', 'LowRisk', 'Total'], 
                                            data = [[mean(accepted_contras['em']['brs']), mean(accepted_contras['ym']['brs']), mean(accepted_contras['m']['brs'])], 
                                                    [mean(accepted_contras['ef']['brs']), mean(accepted_contras['yf']['brs']), mean(accepted_contras['f']['brs'])], 
                                                    [mean(accepted_contras['e']['brs']), mean(accepted_contras['y']['brs']), mean(accepted_contras['t']['brs'])]]))
            

            
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
            
            
        
    
    tr_confusion = pd.concat(tr_confusion)
    tr_model_confusion = pd.concat(tr_model_confusion)
    hyrs_model_confusion = pd.concat(hyrs_model_confusion)
    brs_model_confusion = pd.concat(brs_model_confusion)
    
    tr_conf_confusion = pd.concat(tr_conf_confusion)
    tr_covered_confusion = pd.concat(tr_covered_confusion)
    brs_confusion = pd.concat(brs_confusion)
    brs_conf_confusion = pd.concat(brs_conf_confusion)
    hyrs_confusion = pd.concat(hyrs_confusion)
    human_confusion = pd.concat(human_confusion)
    tr_confusion_contras = pd.concat(tr_confusion_contras)
    tr_confusion_contras_accepted = pd.concat(tr_confusion_contras_accepted)
    tr_confusion_contras_correct = pd.concat(tr_confusion_contras_correct)
    totals_confusion = pd.concat(totals_confusion)
    brs_confusion_contras = pd.concat(brs_confusion_contras)
    brs_confusion_contras_correct = pd.concat(brs_confusion_contras_correct)
    brs_confusion_contras_accepted = pd.concat(brs_confusion_contras_accepted)
    hyrs_confusion_contras = pd.concat(hyrs_confusion_contras)
    hyrs_confusion_contras_correct = pd.concat(hyrs_confusion_contras_correct)
    

    
    
    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))




    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results



costs = [0.0]
num_runs = 5
dataset = 'heart_disease'
case1_means, case1_std, case1_rs = make_results(dataset, 'biased_dec_bias', num_runs, costs, False, asym_costs=[1,1])
   

print('pause')

