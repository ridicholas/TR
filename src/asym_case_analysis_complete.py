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

def quick_perc(contras):
    temp = contras.copy()
    for row in range(temp.shape[0]):
        if temp.index[row] == 'Male':
            temp.iloc[row, :] = temp.iloc[row+2, :].copy()
        if temp.index[row] == 'Female':
            temp.iloc[row, :] = temp.iloc[row+1, :].copy()
    return temp

def weighted_avg_and_std(values, weights=None, filter=['Female','Male','Total'], num_runs=20):
        """
        Return the weighted average and standard deviation.

        They weights are in effect first normalized so that they 
        sum to 1 (and so they must not all be 0).

        values, weights -- NumPy ndarrays with the same shape.
        """
        if weights is not None:
                new_weights = weights.copy()
                new_weights[values.isna()] = 0
        result = []
        result_mean = []
        result_std = []
        for pop in filter:
                v = values.loc[pop, 'Total'][values.loc[pop, 'Total'].notna()]
                if weights is None:
                    w = np.ones(len(v))
                    average = np.nanmean(v)
                    variance = np.nanstd(v)**2
                else:
                        
                        w = new_weights.loc[pop, 'Total'][values.loc[pop, 'Total'].notna()]
                        average = np.average(v, weights=w)
                
                        variance = np.average((v-average)**2, weights=w)
                result.append(str(round(average,3)) + ' \pm ' + str(round(math.sqrt(variance)/math.sqrt(num_runs), 3)))
                result_mean.append(str(round(average,3)))
                result_std.append(str(round(math.sqrt(variance)/math.sqrt(num_runs), 3)))
        return result, result_mean, result_std

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
def noADB(human_conf, model_conf, agreement, asym_scaling=0, asym_scaler=0):
    return np.ones(len(human_conf))
def load_results(dataset, setting, run_num, cost, model):
    setting = '_' + setting
    #if model == 'hyrs':
    #    model = 'tr-no(ADB)'

    if model == 'brs':
        try:
            setting = '_biased'
            with open(f'results/{dataset}/run{run_num}/cost{float(cost)}/{model}_model{setting}.pkl', 'rb') as f:
                result = pickle.load(f)
                return result
        except:
            pass
        try:
            setting = '_biased_dec_bias'
            with open(f'results/{dataset}/run{run_num}/cost{float(cost)}/{model}_model{setting}.pkl', 'rb') as f:
                result = pickle.load(f)
                return result
        except:
            pass
        try:
            setting = '_offset_01'
            with open(f'results/{dataset}/run{run_num}/cost{float(cost)}/{model}_model{setting}.pkl', 'rb') as f:
                result = pickle.load(f)
                return result
        except:
            pass
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
                                'brs_w_reset_model_decision_loss': [[]],
                                'brs_w_reset_team_decision_loss': [[]],
                                'tr_model_w_reset_contradictions': [[]],
                                'tr_model_wo_reset_contradictions': [[]],
                                'hyrs_model_contradictions': [[]],
                                'brs_model_contradictions':[[]],
                                'brs_w_reset_model_contradictions': [[]],
                                'tr_team_w_reset_objective': [[]],
                                'tr_team_wo_reset_objective': [[]],
                                'tr_model_w_reset_objective': [[]],
                                'tr_model_wo_reset_objective': [[]],
                                'hyrs_model_objective': [[]],
                                'hyrs_team_objective':[[]],
                                'brs_model_objective': [[]],
                                'brs_team_objective': [[]],
                                'brs_w_reset_model_objective': [[]],
                                'brs_w_reset_team_objective': [[]],
                                'human_decision_loss': [[]],
                                'hyrs_norecon_objective': [[]],
                                'hyrs_norecon_model_decision_loss': [[]],
                                'hyrs_norecon_team_decision_loss': [[]], 
                               'hyrs_norecon_model_contradictions': [[]],
                               'tr2s_team_w_reset_decision_loss': [[]],
                               'tr2s_team_wo_reset_decision_loss': [[]],
                                 'tr2s_model_w_reset_decision_loss': [[]],
                                    'tr2s_model_wo_reset_decision_loss': [[]],
                                    'tr2s_model_w_reset_contradictions': [[]],
                                    'tr2s_model_wo_reset_contradictions': [[]],
                                    'tr2s_team_w_reset_objective': [[]],
                                    'tr2s_team_wo_reset_objective': [[]]
                               }, index=[costs[0]]
                            )

    for cost in costs[1:]:
        results.loc[cost] = [[] for i in range(len(results.columns))]

    bar=progressbar.ProgressBar()
    whichtype = whichtype +  'asymFinal_newbehav' #'case2_cal' #"_dec_bias"
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

        #x_test = x_train
        #y_test = y_train
        #x_test_non_binarized = x_train_non_binarized
        
        
        dataset = dataset
        human, adb_mod, conf_mod = load_humans(dataset, whichtype, run)




        



        for cost in costs:
            print(f'producing for cost {cost} run {run}.....')
            try:
                brs_mod = load_results(dataset, whichtype + '_asym' , run, cost, 'brs')
            except: 
                brs_mod = load_results(dataset, whichtype +'_asym' , run, 0.0, 'brs')
            tr2s_mod = load_results(dataset, whichtype +'_asym' , run, cost, 'tr')
            hyrs_mod = load_results(dataset, whichtype +'_asym' , run, cost, 'tr-no(ADB)')
            tr_mod = load_results(dataset, whichtype +'_asym', run, cost, 'tr')
            

            #load e_y and e_yb mods
            #with open(f'results/{dataset}/run{run}/cost{float(cost)}/eyb_model_{whichtype}.pkl', 'rb') as f:
            #    e_yb_mod = pickle.load(f)
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/ey_model_{whichtype + "_asym"}.pkl', 'rb') as f:
                e_y_mod = pickle.load(f)

            tr_mod.df = x_train
            tr_mod.Y = y_train
            tr2s_mod.df = x_train
            tr2s_mod.Y = y_train
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
            brs_w_reset_model_decision_loss = []
            brs_w_reset_team_decision_loss = []


            tr_model_w_reset_contradictions = []
            tr_model_wo_reset_contradictions = []
            hyrs_model_contradictions = []
            brs_model_contradictions = []
            brs_w_reset_model_contradictions = []

            tr_team_w_reset_objective = []
            tr_team_wo_reset_objective = []
            tr_model_w_reset_objective = []
            tr_model_wo_reset_objective = []
            hyrs_model_objective = []
            hyrs_team_objective = []
            brs_model_objective = []
            brs_team_objective = []
            brs_w_reset_model_objective = []
            brs_w_reset_team_objective = []

            human_decision_loss = []
            hyrs_norecon_objective = []
            hyrs_norecon_model_decision_loss = []
            hyrs_norecon_team_decision_loss = []
            hyrs_norecon_model_contradictions = []

            tr2s_team_w_reset_decision_loss = []
            tr2s_team_wo_reset_decision_loss = []
            tr2s_model_w_reset_decision_loss = []
            tr2s_model_wo_reset_decision_loss = []
            tr2s_model_w_reset_contradictions = []
            tr2s_model_wo_reset_contradictions = []
            tr2s_team_w_reset_objective = []
            tr2s_team_wo_reset_objective = []

            
            
            brs_mod.df = x_train
            brs_mod.Y = y_train
            brs_model_preds = brs_predict(brs_mod.opt_rules, x_test)
            brs_conf = brs_predict_conf(brs_mod.opt_rules, x_test, brs_mod)
            hyrs_norecon_mod = deepcopy(hyrs_mod)

            
            
            decs = {}
            decs['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            decs['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}

            model_decs = {}
            model_decs['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}    
            model_decs['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            model_decs['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}


            contras = {}
            contras['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            contras['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}

            correct_contras = {}
            correct_contras['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            correct_contras['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}

            accepted_contras = {}
            accepted_contras['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            accepted_contras['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}

            covereds = {}
            covereds['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covereds['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}

            covered_corrects = {}
            covered_corrects['t']={'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['e'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['y'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['m'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['f'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['em'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['ef'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['ym'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}
            covered_corrects['yf'] = {'tr': [], 'tr2s': [], 'hyrs': [], 'brs': [], 'brs_w_reset': [], 'human': []}



            totals = {}
            for i in range(25):
                
                

                if validation: 

                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    conf_mod_preds = conf_mod.predict(x_test_non_binarized)
                    human_scaling = human_decisions

                    learned_adb = ADB(adb_mod)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]

                    tr2s_team_preds_with_reset = tr2s_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    tr2s_team_preds_no_reset = tr2s_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]

                    tr2s_model_preds_with_reset = tr2s_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    tr2s_model_preds_no_reset = tr2s_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]

                    #hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    #hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, learned_adb.ADB_model_wrapper, x_test)
                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    hyrs_team_preds = hyrs_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]



                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0] #.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, learned_adb.ADB_model_wrapper, x_test)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, learned_adb.ADB_model_wrapper)

                    c_model = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]



                        
                else:
                    learned_adb = ADB(adb_mod)
                    #human.ADB = learned_adb.ADB_model_wrapper
                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    human_scaling = human_decisions
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions,human_conf, human.ADB, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]

                    tr_model_preds_with_reset, tr_mod_covered_w_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)
                    tr_model_preds_no_reset, tr_mod_covered_no_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)
                    tr_mod_confs = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]

                    tr2s_team_preds_with_reset = tr_team_preds_with_reset #tr2s_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr2s_team_preds_no_reset = tr_team_preds_no_reset #tr2s_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr2s_model_preds_with_reset, tr2s_mod_covered_w_reset = tr_model_preds_with_reset, tr_mod_covered_w_reset#tr2s_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr2s_model_preds_no_reset, tr2s_mod_covered_no_reset = tr_model_preds_no_reset, tr_mod_covered_no_reset#tr2s_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr2s_mod_confs = tr_mod_confs#tr2s_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr2s_mod.prs_min, nrs_min=tr2s_mod.nrs_min)[0]
                

                    #hyrs_model_preds, hyrs_model_covered, _ = hyrs_mod.predict(x_test, human_decisions) 
                    hyrs_model_preds, hyrs_model_covered, _ = hyrs_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)
                    #hyrs_mod_confs = hyrs_mod.get_model_conf_agreement(x_test, human_decisions)[0] 
                    hyrs_mod_confs = hyrs_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=hyrs_mod.prs_min, nrs_min=hyrs_mod.nrs_min)[0]
                    
                    #hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, human.ADB, x_test) 
                    hyrs_team_preds = hyrs_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized), asym_scaler=human.asym_scaler, asym_scaling=human_scaling)[0]
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)
                    brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), e_human_responses=human_decisions, conf_model=brs_conf, fA=learned_adb.ADB_model_wrapper, asym_loss=[1,1], contradiction_reg=cost)
                    brs_w_reset_team_preds = brs_team_preds.copy()
                    brs_w_reset_model_preds = brs_model_preds.copy()
                    brs_w_reset_team_preds[brs_reset] = human_decisions[brs_reset]
                    brs_w_reset_model_preds[brs_reset] = human_decisions[brs_reset]
                    brs_w_reset_covereds = np.where(brs_reset == False)[0]


                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    #hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    hyrs_norecon_team_preds = hyrs_norecon_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0] #




                        

                    

                
                total = len(y_test)
                totals['t'] = len(y_test)
                totals['e'] = len(y_test[y_test == 1])
                totals['y'] = len(y_test[y_test == 0])
                totals['m'] = len(y_test[human_decisions == 1])
                totals['f'] = len(y_test[human_decisions == 0])
                totals['em'] = len(y_test[(y_test == 1) & (human_decisions == 1)])
                totals['ef'] = len(y_test[(y_test == 1) & (human_decisions == 0)])
                totals['ym'] = len(y_test[(y_test == 0) & (human_decisions == 1)])
                totals['yf'] = len(y_test[(y_test == 0) & (human_decisions == 0)])




                which_ones = ['tr', 'tr2s', 'hyrs', 'brs', 'brs_w_reset', 'human']

                for which in which_ones:
                #find number of incorrect predictions confusion matrix split along sex_Male and age54.0 variables
                    if which =='tr':
                        preds = tr_team_preds_with_reset.copy()
                        
                    elif which == 'hyrs':
                        preds = hyrs_team_preds.copy()
                        
                    elif which == 'brs':
                        preds = brs_team_preds.copy()
                    elif which == 'brs_w_reset':
                        preds = brs_w_reset_team_preds.copy()
                    elif which == 'human':
                        preds = human_decisions.copy()
                    
                    elif which == 'tr2s':
                        preds = tr2s_team_preds_with_reset.copy()
                    
                    asymCosts = y_test.replace({0: asym_costs[1], 1: asym_costs[0]}) 
                    decs['t'][which].append(((preds != y_test)*asymCosts).sum())
                    decs['e'][which].append(((preds != y_test)*asymCosts)[y_test == 1].sum())
                    decs['y'][which].append(((preds != y_test)*asymCosts)[y_test == 0].sum())
                    decs['m'][which].append(((preds != y_test)*asymCosts)[human_decisions == 1].sum())
                    decs['f'][which].append(((preds != y_test)*asymCosts)[human_decisions == 0].sum())

                    decs['em'][which].append(((preds != y_test)*asymCosts)[(y_test == 1) & (human_decisions == 1)].sum())
                    decs['ef'][which].append(((preds != y_test)*asymCosts)[(y_test == 1) & (human_decisions == 0)].sum())
                    decs['ym'][which].append(((preds != y_test)*asymCosts)[(y_test == 0) & (human_decisions == 1)].sum())
                    decs['yf'][which].append(((preds != y_test)*asymCosts)[(y_test == 0) & (human_decisions == 0)].sum())







                    
                    if which == 'tr':
                        model_preds = tr_model_preds_with_reset.copy()
                        model_covereds = np.zeros(len(model_preds), dtype=bool)
                        model_covereds[tr_mod_covered_w_reset] = True
                    

                    elif which == 'hyrs':
                        model_preds = hyrs_model_preds.copy()
                        model_covereds = np.zeros(len(model_preds), dtype=bool)
                        model_covereds[hyrs_model_covered] = True

                    elif which == 'brs':
                        model_preds = brs_model_preds.copy()
                        
                    elif which == 'brs_w_reset':
                        model_preds = brs_w_reset_model_preds.copy()
                        model_covereds = np.zeros(len(brs_model_preds), dtype=bool)
                        model_covereds[brs_w_reset_covereds] = True
                    elif which == 'human':
                        model_preds = human_decisions.copy()

                    elif which == 'tr2s':
                        model_preds = tr2s_model_preds_with_reset.copy()
                        model_covereds = np.zeros(len(model_preds), dtype=bool)
                        model_covereds[tr2s_mod_covered_w_reset] = True
                
                    contras['t'][which].append((model_preds != human_decisions).sum())
                    contras['e'][which].append((model_preds != human_decisions)[y_test == 1].sum())
                    contras['y'][which].append((model_preds != human_decisions)[y_test == 0].sum())
                    contras['m'][which].append((model_preds != human_decisions)[human_decisions == 1].sum())
                    contras['f'][which].append((model_preds != human_decisions)[human_decisions == 0].sum())

                    contras['em'][which].append((model_preds != human_decisions)[(y_test == 1) & (human_decisions == 1)].sum())
                    contras['ef'][which].append((model_preds != human_decisions)[(y_test == 1) & (human_decisions == 0)].sum())
                    contras['ym'][which].append((model_preds != human_decisions)[(y_test == 0) & (human_decisions == 1)].sum())
                    contras['yf'][which].append((model_preds != human_decisions)[(y_test == 0) & (human_decisions == 0)].sum())

                    model_decs['t'][which].append(((model_preds != y_test)*asymCosts).sum())
                    model_decs['e'][which].append(((model_preds != y_test)*asymCosts)[y_test == 1].sum())
                    model_decs['y'][which].append(((model_preds != y_test)*asymCosts)[y_test == 0].sum())
                    model_decs['m'][which].append(((model_preds != y_test)*asymCosts)[human_decisions == 1].sum())
                    model_decs['f'][which].append(((model_preds != y_test)*asymCosts)[human_decisions == 0].sum())

                    model_decs['em'][which].append(((model_preds != y_test)*asymCosts)[(y_test == 1) & (human_decisions == 1)].sum())
                    model_decs['ef'][which].append(((model_preds != y_test)*asymCosts)[(y_test == 1) & (human_decisions == 0)].sum())
                    model_decs['ym'][which].append(((model_preds != y_test)*asymCosts)[(y_test == 0) & (human_decisions == 1)].sum())
                    model_decs['yf'][which].append(((model_preds != y_test)*asymCosts)[(y_test == 0) & (human_decisions == 0)].sum())

                    correct_contras['t'][which].append(((model_preds != human_decisions) & (model_preds == y_test)).sum())
                    correct_contras['e'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[y_test == 1].sum())
                    correct_contras['y'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[y_test == 0].sum())
                    correct_contras['m'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[human_decisions == 1].sum())
                    correct_contras['f'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[human_decisions == 0].sum())

                    correct_contras['em'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(y_test == 1) & (human_decisions == 1)].sum())
                    correct_contras['ef'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(y_test == 1) & (human_decisions == 0)].sum())
                    correct_contras['ym'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(y_test == 0) & (human_decisions == 1)].sum())
                    correct_contras['yf'][which].append(((model_preds != human_decisions) & (model_preds == y_test))[(y_test == 0) & (human_decisions == 0)].sum())

                    accepted_condition = (model_preds != human_decisions) & (model_preds == preds)

                    accepted_contras['t'][which].append(accepted_condition.sum())
                    accepted_contras['e'][which].append(accepted_condition[y_test == 1].sum())
                    accepted_contras['y'][which].append(accepted_condition[y_test == 0].sum())
                    accepted_contras['m'][which].append(accepted_condition[human_decisions == 1].sum())
                    accepted_contras['f'][which].append(accepted_condition[human_decisions == 0].sum())
                    accepted_contras['em'][which].append(accepted_condition[(y_test == 1) & (human_decisions == 1)].sum())
                    accepted_contras['ef'][which].append(accepted_condition[(y_test == 1) & (human_decisions == 0)].sum())
                    accepted_contras['ym'][which].append(accepted_condition[(y_test == 0) & (human_decisions == 1)].sum())
                    accepted_contras['yf'][which].append(accepted_condition[(y_test == 0) & (human_decisions == 0)].sum())


                    if which in ['tr', 'tr2s', 'hyrs', 'brs_w_reset']:
                        covereds['t'][which].append(model_covereds.sum())
                        covereds['e'][which].append(model_covereds[y_test == 1].sum())
                        covereds['y'][which].append(model_covereds[y_test == 0].sum())
                        covereds['m'][which].append(model_covereds[human_decisions == 1].sum())
                        covereds['f'][which].append(model_covereds[human_decisions == 0].sum())
                        covereds['em'][which].append(model_covereds[(y_test == 1) & (human_decisions == 1)].sum())
                        covereds['ef'][which].append(model_covereds[(y_test == 1) & (human_decisions == 0)].sum())
                        covereds['ym'][which].append(model_covereds[(y_test == 0) & (human_decisions == 1)].sum())
                        covereds['yf'][which].append(model_covereds[(y_test == 0) & (human_decisions == 0)].sum())

                        covered_corrects['t'][which].append((model_covereds & (model_preds == y_test)).sum())
                        covered_corrects['e'][which].append((model_covereds & (model_preds == y_test))[y_test == 1].sum())
                        covered_corrects['y'][which].append((model_covereds & (model_preds == y_test))[y_test == 0].sum())
                        covered_corrects['m'][which].append((model_covereds & (model_preds == y_test))[human_decisions == 1].sum())
                        covered_corrects['f'][which].append((model_covereds & (model_preds == y_test))[human_decisions == 0].sum())
                        covered_corrects['em'][which].append((model_covereds & (model_preds == y_test))[(y_test == 1) & (human_decisions == 1)].sum())
                        covered_corrects['ef'][which].append((model_covereds & (model_preds == y_test))[(y_test == 1) & (human_decisions == 0)].sum())
                        covered_corrects['ym'][which].append((model_covereds & (model_preds == y_test))[(y_test == 0) & (human_decisions == 1)].sum())
                        covered_corrects['yf'][which].append((model_covereds & (model_preds == y_test))[(y_test == 0) & (human_decisions == 0)].sum())
                        

                tr_team_w_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_with_reset, y_test))
                tr_team_wo_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_no_reset, y_test))
                tr_model_w_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_with_reset, y_test))
                tr_model_wo_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_no_reset, y_test))
                hyrs_model_decision_loss.append(1 - accuracy_score(hyrs_model_preds, y_test))
                hyrs_team_decision_loss.append(1 - accuracy_score(hyrs_team_preds, y_test))
                brs_model_decision_loss.append(1 - accuracy_score(brs_model_preds, y_test))
                brs_team_decision_loss.append(1 - accuracy_score(brs_team_preds, y_test))
                brs_w_reset_model_decision_loss.append(1 - accuracy_score(brs_w_reset_model_preds, y_test))
                brs_w_reset_team_decision_loss.append(1 - accuracy_score(brs_w_reset_team_preds, y_test))


                tr_model_w_reset_contradictions.append((tr_model_preds_with_reset != human_decisions).sum())
                tr_model_wo_reset_contradictions.append((tr_model_preds_no_reset != human_decisions).sum())
                hyrs_model_contradictions.append((hyrs_model_preds != human_decisions).sum())
                brs_model_contradictions.append((brs_model_preds != human_decisions).sum())
                brs_w_reset_model_contradictions.append((brs_w_reset_model_preds != human_decisions).sum())

                tr_team_w_reset_objective.append(tr_team_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_team_wo_reset_objective.append(tr_team_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                tr_model_w_reset_objective.append(tr_model_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_model_wo_reset_objective.append(tr_model_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                hyrs_model_objective.append(hyrs_model_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                hyrs_team_objective.append(hyrs_team_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                brs_model_objective.append(brs_model_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_team_objective.append(brs_team_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_w_reset_model_objective.append(brs_w_reset_model_decision_loss[-1] + cost*(brs_w_reset_model_contradictions[-1])/len(y_test))
                brs_w_reset_team_objective.append(brs_w_reset_team_decision_loss[-1] + cost*(brs_w_reset_model_contradictions[-1])/len(y_test))

                tr2s_team_w_reset_decision_loss.append(1 - accuracy_score(tr2s_team_preds_with_reset, y_test))
                tr2s_team_wo_reset_decision_loss.append(1 - accuracy_score(tr2s_team_preds_no_reset, y_test))
                tr2s_model_w_reset_decision_loss.append(1 - accuracy_score(tr2s_model_preds_with_reset, y_test))
                tr2s_model_wo_reset_decision_loss.append(1 - accuracy_score(tr2s_model_preds_no_reset, y_test))
                tr2s_model_w_reset_contradictions.append((tr2s_model_preds_with_reset != human_decisions).sum())
                tr2s_model_wo_reset_contradictions.append((tr2s_model_preds_no_reset != human_decisions).sum())
                tr2s_team_w_reset_objective.append(tr2s_team_w_reset_decision_loss[-1] + cost*(tr2s_model_w_reset_contradictions[-1])/len(y_test))
                tr2s_team_wo_reset_objective.append(tr2s_team_wo_reset_decision_loss[-1] + cost*(tr2s_model_wo_reset_contradictions[-1])/len(y_test))

                human_decision_loss.append(1 - accuracy_score(human_decisions, y_test))

                hyrs_norecon_model_decision_loss.append(1 - accuracy_score(hyrs_norecon_model_preds, y_test))
                hyrs_norecon_team_decision_loss.append(1 - accuracy_score(hyrs_norecon_team_preds, y_test))
                hyrs_norecon_model_contradictions.append((hyrs_norecon_model_preds != human_decisions).sum())
                hyrs_norecon_objective.append(hyrs_norecon_team_decision_loss[-1] + cost*(hyrs_norecon_model_contradictions[-1])/len(y_test))

                


                print(i)
            
            if run==0:

                tr_conf_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[tr_mod_confs[(y_test == 1) & (human_decisions == 1) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(y_test == 0) & (human_decisions == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(human_decisions == 1) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(y_test == 1) & (human_decisions == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(y_test == 0) & (human_decisions == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(human_decisions == 0) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(y_test == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(y_test == 0) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(tr_model_preds_with_reset != human_decisions)].mean()]])]
                
                tr2s_conf_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[tr2s_mod_confs[(y_test == 1) & (human_decisions == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr2s_mod_confs[(y_test == 0) & (human_decisions == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(human_decisions == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr2s_mod_confs[(y_test == 1) & (human_decisions == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(y_test == 0) & (human_decisions == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(human_decisions == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr2s_mod_confs[(y_test == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(y_test == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr2s_mod_confs[(tr2s_model_preds_with_reset != human_decisions)].mean()]])]
                
                hyrs_conf_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[hyrs_mod_confs[(y_test == 1) & (human_decisions == 1) & (hyrs_model_preds != human_decisions)].mean(), 
                                                     hyrs_mod_confs[(y_test == 0) & (human_decisions == 1) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(human_decisions == 1) & (hyrs_model_preds != human_decisions)].mean()], 
                                                     [hyrs_mod_confs[(y_test == 1) & (human_decisions == 0) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(y_test == 0) & (human_decisions == 0) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(human_decisions == 0) & (hyrs_model_preds != human_decisions)].mean()], 
                                                     [hyrs_mod_confs[(y_test == 1) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(y_test == 0) & (hyrs_model_preds != human_decisions)].mean(), 
                                                     hyrs_mod_confs[(hyrs_model_preds != human_decisions)].mean()]])]
                
                
                
                brs_conf_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[brs_conf[(y_test == 1) &(human_decisions == 1) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(y_test == 0) & (human_decisions == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 1) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (human_decisions == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (human_decisions == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 0) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(brs_model_preds != human_decisions)].mean()]])]
                
                brs_w_reset_conf_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[brs_conf[(y_test == 1)  & (human_decisions == 1) & (brs_w_reset_model_preds != human_decisions)].mean(), 
                                         brs_conf[(y_test == 0) & (human_decisions == 1) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 1) & (brs_w_reset_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (human_decisions == 0) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (human_decisions == 0) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 0) & (brs_w_reset_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (brs_w_reset_model_preds != human_decisions)].mean(), 
                                         brs_conf[(brs_w_reset_model_preds != human_decisions)].mean()]])]
                
                tr_covered_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['tr']), mean(covereds['ym']['tr']), mean(covereds['m']['tr'])], 
                                        [mean(covereds['ef']['tr']), mean(covereds['yf']['tr']), mean(covereds['f']['tr'])], 
                                        [mean(covereds['e']['tr']), mean(covereds['y']['tr']), mean(covereds['t']['tr'])]])]
                
                brs_w_reset_covered_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                data = [[mean(covereds['em']['brs_w_reset']), mean(covereds['ym']['brs_w_reset']), mean(covereds['m']['brs_w_reset'])], 
                        [mean(covereds['ef']['brs_w_reset']), mean(covereds['yf']['brs_w_reset']), mean(covereds['f']['brs_w_reset'])], 
                        [mean(covereds['e']['brs_w_reset']), mean(covereds['y']['brs_w_reset']), mean(covereds['t']['brs_w_reset'])]])]
                
                tr2s_covered_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['tr2s']), mean(covereds['ym']['tr2s']), mean(covereds['m']['tr2s'])], 
                                        [mean(covereds['ef']['tr2s']), mean(covereds['yf']['tr2s']), mean(covereds['f']['tr2s'])], 
                                        [mean(covereds['e']['tr2s']), mean(covereds['y']['tr2s']), mean(covereds['t']['tr2s'])]])]
                

                
                tr_covered_correct_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['tr']), mean(covered_corrects['ym']['tr']), mean(covered_corrects['m']['tr'])], 
                                        [mean(covered_corrects['ef']['tr']), mean(covered_corrects['yf']['tr']), mean(covered_corrects['f']['tr'])], 
                                        [mean(covered_corrects['e']['tr']), mean(covered_corrects['y']['tr']), mean(covered_corrects['t']['tr'])]])]
                
                tr2s_covered_correct_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['tr2s']), mean(covered_corrects['ym']['tr2s']), mean(covered_corrects['m']['tr2s'])], 
                                        [mean(covered_corrects['ef']['tr2s']), mean(covered_corrects['yf']['tr2s']), mean(covered_corrects['f']['tr2s'])], 
                                        [mean(covered_corrects['e']['tr2s']), mean(covered_corrects['y']['tr2s']), mean(covered_corrects['t']['tr2s'])]])]
                
                hyrs_covered_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['hyrs']), mean(covereds['ym']['hyrs']), mean(covereds['m']['hyrs'])], 
                                        [mean(covereds['ef']['hyrs']), mean(covereds['yf']['hyrs']), mean(covereds['f']['hyrs'])], 
                                        [mean(covereds['e']['hyrs']), mean(covereds['y']['hyrs']), mean(covereds['t']['hyrs'])]])]
                
                hyrs_covered_correct_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['hyrs']), mean(covered_corrects['ym']['hyrs']), mean(covered_corrects['m']['hyrs'])], 
                                        [mean(covered_corrects['ef']['hyrs']), mean(covered_corrects['yf']['hyrs']), mean(covered_corrects['f']['hyrs'])], 
                                        [mean(covered_corrects['e']['hyrs']), mean(covered_corrects['y']['hyrs']), mean(covered_corrects['t']['hyrs'])]])]
                
                brs_w_reset_covered_correct_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['brs_w_reset']), mean(covered_corrects['ym']['brs_w_reset']), mean(covered_corrects['m']['brs_w_reset'])], 
                                        [mean(covered_corrects['ef']['brs_w_reset']), mean(covered_corrects['yf']['brs_w_reset']), mean(covered_corrects['f']['brs_w_reset'])], 
                                        [mean(covered_corrects['e']['brs_w_reset']), mean(covered_corrects['y']['brs_w_reset']), mean(covered_corrects['t']['brs_w_reset'])]])]
                

                tr_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(decs['em']['tr']), mean(decs['ym']['tr']), mean(decs['m']['tr'])], 
                                                    [mean(decs['ef']['tr']), mean(decs['yf']['tr']), mean(decs['f']['tr'])], 
                                                    [mean(decs['e']['tr']), mean(decs['y']['tr']), mean(decs['t']['tr'])]])]
                
                tr2s_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(decs['em']['tr2s']), mean(decs['ym']['tr2s']), mean(decs['m']['tr2s'])], 
                                                    [mean(decs['ef']['tr2s']), mean(decs['yf']['tr2s']), mean(decs['f']['tr2s'])], 
                                                    [mean(decs['e']['tr2s']), mean(decs['y']['tr2s']), mean(decs['t']['tr2s'])]])]
                
                tr_model_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(model_decs['em']['tr']), mean(model_decs['ym']['tr']), mean(model_decs['m']['tr'])], 
                                                    [mean(model_decs['ef']['tr']), mean(model_decs['yf']['tr']), mean(model_decs['f']['tr'])], 
                                                    [mean(model_decs['e']['tr']), mean(model_decs['y']['tr']), mean(model_decs['t']['tr'])]])]
                
                tr2s_model_confusion = [pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(model_decs['em']['tr2s']), mean(model_decs['ym']['tr2s']), mean(model_decs['m']['tr2s'])], 
                                                    [mean(model_decs['ef']['tr2s']), mean(model_decs['yf']['tr2s']), mean(model_decs['f']['tr2s'])], 
                                                    [mean(model_decs['e']['tr2s']), mean(model_decs['y']['tr2s']), mean(model_decs['t']['tr2s'])]])]
                
                hyrs_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(decs['em']['hyrs']), mean(decs['ym']['hyrs']), mean(decs['m']['hyrs'])], 
                                                    [mean(decs['ef']['hyrs']), mean(decs['yf']['hyrs']), mean(decs['f']['hyrs'])], 
                                                    [mean(decs['e']['hyrs']), mean(decs['y']['hyrs']), mean(decs['t']['hyrs'])]])]
                
                hyrs_model_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(model_decs['em']['hyrs']), mean(model_decs['ym']['hyrs']), mean(model_decs['m']['hyrs'])], 
                                                    [mean(model_decs['ef']['hyrs']), mean(model_decs['yf']['hyrs']), mean(model_decs['f']['hyrs'])], 
                                                    [mean(model_decs['e']['hyrs']), mean(model_decs['y']['hyrs']), mean(model_decs['t']['hyrs'])]])]
                
                brs_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(decs['em']['brs']), mean(decs['ym']['brs']), mean(decs['m']['brs'])], 
                        [mean(decs['ef']['brs']), mean(decs['yf']['brs']), mean(decs['f']['brs'])], 
                        [mean(decs['e']['brs']), mean(decs['y']['brs']), mean(decs['t']['brs'])]])]
                
                
                brs_model_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(model_decs['em']['brs']), mean(model_decs['ym']['brs']), mean(model_decs['m']['brs'])], 
                        [mean(model_decs['ef']['brs']), mean(model_decs['yf']['brs']), mean(model_decs['f']['brs'])], 
                        [mean(model_decs['e']['brs']), mean(model_decs['y']['brs']), mean(model_decs['t']['brs'])]])]
                

                

                human_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                                            data= [[mean(decs['em']['human']), mean(decs['ym']['human']), mean(decs['m']['human'])], 
                                                    [mean(decs['ef']['human']), mean(decs['yf']['human']), mean(decs['f']['human'])], 
                                                    [mean(decs['e']['human']), mean(decs['y']['human']), mean(decs['t']['human'])]])]

                tr_confusion_contras = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(contras['em']['tr']), mean(contras['ym']['tr']), mean(contras['m']['tr'])], 
                                                    [mean(contras['ef']['tr']), mean(contras['yf']['tr']), mean(contras['f']['tr'])], 
                                                    [mean(contras['e']['tr']), mean(contras['y']['tr']), mean(contras['t']['tr'])]])]
                
                tr2s_confusion_contras = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(contras['em']['tr2s']), mean(contras['ym']['tr2s']), mean(contras['m']['tr2s'])], 
                                                    [mean(contras['ef']['tr2s']), mean(contras['yf']['tr2s']), mean(contras['f']['tr2s'])], 
                                                    [mean(contras['e']['tr2s']), mean(contras['y']['tr2s']), mean(contras['t']['tr2s'])]])]
                
                hyrs_confusion_contras = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(contras['em']['hyrs']), mean(contras['ym']['hyrs']), mean(contras['m']['hyrs'])], 
                                                    [mean(contras['ef']['hyrs']), mean(contras['yf']['hyrs']), mean(contras['f']['hyrs'])], 
                                                    [mean(contras['e']['hyrs']), mean(contras['y']['hyrs']), mean(contras['t']['hyrs'])]])]

                tr_confusion_contras_accepted = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['tr']), mean(accepted_contras['ym']['tr']), mean(accepted_contras['m']['tr'])], 
                                                    [mean(accepted_contras['ef']['tr']), mean(accepted_contras['yf']['tr']), mean(accepted_contras['f']['tr'])], 
                                                    [mean(accepted_contras['e']['tr']), mean(accepted_contras['y']['tr']), mean(accepted_contras['t']['tr'])]])]
                

                tr2s_confusion_contras_accepted = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['tr2s']), mean(accepted_contras['ym']['tr2s']), mean(accepted_contras['m']['tr2s'])], 
                                                    [mean(accepted_contras['ef']['tr2s']), mean(accepted_contras['yf']['tr2s']), mean(accepted_contras['f']['tr2s'])], 
                                                    [mean(accepted_contras['e']['tr2s']), mean(accepted_contras['y']['tr2s']), mean(accepted_contras['t']['tr2s'])]])]
                
                hyrs_confusion_contras_accepted = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['hyrs']), mean(accepted_contras['ym']['hyrs']), mean(accepted_contras['m']['hyrs'])], 
                                                    [mean(accepted_contras['ef']['hyrs']), mean(accepted_contras['yf']['hyrs']), mean(accepted_contras['f']['hyrs'])], 
                                                    [mean(accepted_contras['e']['hyrs']), mean(accepted_contras['y']['hyrs']), mean(accepted_contras['t']['hyrs'])]])]
                
                tr_confusion_contras_correct = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['tr']), mean(correct_contras['ym']['tr']), mean(correct_contras['m']['tr'])], 
                                                    [mean(correct_contras['ef']['tr']), mean(correct_contras['yf']['tr']), mean(correct_contras['f']['tr'])], 
                                                    [mean(correct_contras['e']['tr']), mean(correct_contras['y']['tr']), mean(correct_contras['t']['tr'])]])]
                
                tr2s_confusion_contras_correct = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['tr2s']), mean(correct_contras['ym']['tr2s']), mean(correct_contras['m']['tr2s'])], 
                                                    [mean(correct_contras['ef']['tr2s']), mean(correct_contras['yf']['tr2s']), mean(correct_contras['f']['tr2s'])], 
                                                    [mean(correct_contras['e']['tr2s']), mean(correct_contras['y']['tr2s']), mean(correct_contras['t']['tr2s'])]])]
                
                hyrs_confusion_contras_correct = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                            data = [[mean(correct_contras['em']['hyrs']), mean(correct_contras['ym']['hyrs']), mean(correct_contras['m']['hyrs'])], 
                                    [mean(correct_contras['ef']['hyrs']), mean(correct_contras['yf']['hyrs']), mean(correct_contras['f']['hyrs'])], 
                                    [mean(correct_contras['e']['hyrs']), mean(correct_contras['y']['hyrs']), mean(correct_contras['t']['hyrs'])]])]
                
                totals_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[totals['em'], totals['ym'], totals['m']], 
                        [totals['ef'], totals['yf'], totals['f']], 
                        [totals['e'], totals['y'], totals['t']]])]
                
                brs_confusion_contras = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(contras['em']['brs']), mean(contras['ym']['brs']), mean(contras['m']['brs'])], 
                        [mean(contras['ef']['brs']), mean(contras['yf']['brs']), mean(contras['f']['brs'])], 
                        [mean(contras['e']['brs']), mean(contras['y']['brs']), mean(contras['t']['brs'])]])]
                

                
                brs_confusion_contras_correct = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['brs']), mean(correct_contras['ym']['brs']), mean(correct_contras['m']['brs'])], 
                                                    [mean(correct_contras['ef']['brs']), mean(correct_contras['yf']['brs']), mean(correct_contras['f']['brs'])], 
                                                    [mean(correct_contras['e']['brs']), mean(correct_contras['y']['brs']), mean(correct_contras['t']['brs'])]])]
                

                
                brs_confusion_contras_accepted = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['brs']), mean(accepted_contras['ym']['brs']), mean(accepted_contras['m']['brs'])], 
                                                    [mean(accepted_contras['ef']['brs']), mean(accepted_contras['yf']['brs']), mean(accepted_contras['f']['brs'])], 
                                                    [mean(accepted_contras['e']['brs']), mean(accepted_contras['y']['brs']), mean(accepted_contras['t']['brs'])]])]
                
                brs_w_reset_confusion_contras_accepted = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['brs_w_reset']), mean(accepted_contras['ym']['brs_w_reset']), mean(accepted_contras['m']['brs_w_reset'])], 
                                                    [mean(accepted_contras['ef']['brs_w_reset']), mean(accepted_contras['yf']['brs_w_reset']), mean(accepted_contras['f']['brs_w_reset'])], 
                                                    [mean(accepted_contras['e']['brs_w_reset']), mean(accepted_contras['y']['brs_w_reset']), mean(accepted_contras['t']['brs_w_reset'])]])]

                brs_w_reset_confusion_contras_correct = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['brs_w_reset']), mean(correct_contras['ym']['brs_w_reset']), mean(correct_contras['m']['brs_w_reset'])], 
                                                    [mean(correct_contras['ef']['brs_w_reset']), mean(correct_contras['yf']['brs_w_reset']), mean(correct_contras['f']['brs_w_reset'])], 
                                                    [mean(correct_contras['e']['brs_w_reset']), mean(correct_contras['y']['brs_w_reset']), mean(correct_contras['t']['brs_w_reset'])]])]
                
                brs_w_reset_confusion_contras = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(contras['em']['brs_w_reset']), mean(contras['ym']['brs_w_reset']), mean(contras['m']['brs_w_reset'])], 
                        [mean(contras['ef']['brs_w_reset']), mean(contras['yf']['brs_w_reset']), mean(contras['f']['brs_w_reset'])], 
                        [mean(contras['e']['brs_w_reset']), mean(contras['y']['brs_w_reset']), mean(contras['t']['brs_w_reset'])]])]
                
                brs_w_reset_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(decs['em']['brs_w_reset']), mean(decs['ym']['brs_w_reset']), mean(decs['m']['brs_w_reset'])], 
                        [mean(decs['ef']['brs_w_reset']), mean(decs['yf']['brs_w_reset']), mean(decs['f']['brs_w_reset'])], 
                        [mean(decs['e']['brs_w_reset']), mean(decs['y']['brs_w_reset']), mean(decs['t']['brs_w_reset'])]])]
                
                brs_w_reset_model_confusion = [pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(model_decs['em']['brs_w_reset']), mean(model_decs['ym']['brs_w_reset']), mean(model_decs['m']['brs_w_reset'])], 
                        [mean(model_decs['ef']['brs_w_reset']), mean(model_decs['yf']['brs_w_reset']), mean(model_decs['f']['brs_w_reset'])], 
                        [mean(model_decs['e']['brs_w_reset']), mean(model_decs['y']['brs_w_reset']), mean(model_decs['t']['brs_w_reset'])]])]
                
            else:
                tr_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(decs['em']['tr']), mean(decs['ym']['tr']), mean(decs['m']['tr'])], 
                                                    [mean(decs['ef']['tr']), mean(decs['yf']['tr']), mean(decs['f']['tr'])], 
                                                    [mean(decs['e']['tr']), mean(decs['y']['tr']), mean(decs['t']['tr'])]]))
                
                tr_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[tr_mod_confs[(y_test == 1) &(human_decisions == 1) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(y_test == 0) & (human_decisions == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(human_decisions == 1) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(y_test == 1) & (human_decisions == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(y_test == 0) & (human_decisions == 0) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(human_decisions == 0) & (tr_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr_mod_confs[(y_test == 1) & (tr_model_preds_with_reset != human_decisions)].mean(),
                                                     tr_mod_confs[(y_test == 0) & (tr_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr_mod_confs[(tr_model_preds_with_reset != human_decisions)].mean()]]))
                
                tr2s_covered_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['tr2s']), mean(covereds['ym']['tr2s']), mean(covereds['m']['tr2s'])], 
                                        [mean(covereds['ef']['tr2s']), mean(covereds['yf']['tr2s']), mean(covereds['f']['tr2s'])], 
                                        [mean(covereds['e']['tr2s']), mean(covereds['y']['tr2s']), mean(covereds['t']['tr2s'])]]))
                
                brs_w_reset_covered_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['brs_w_reset']), mean(covereds['ym']['brs_w_reset']), mean(covereds['m']['brs_w_reset'])], 
                                        [mean(covereds['ef']['brs_w_reset']), mean(covereds['yf']['brs_w_reset']), mean(covereds['f']['brs_w_reset'])], 
                                        [mean(covereds['e']['brs_w_reset']), mean(covereds['y']['brs_w_reset']), mean(covereds['t']['brs_w_reset'])]]))
                
                tr2s_covered_correct_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['tr2s']), mean(covered_corrects['ym']['tr2s']), mean(covered_corrects['m']['tr2s'])], 
                                        [mean(covered_corrects['ef']['tr2s']), mean(covered_corrects['yf']['tr2s']), mean(covered_corrects['f']['tr2s'])], 
                                        [mean(covered_corrects['e']['tr2s']), mean(covered_corrects['y']['tr2s']), mean(covered_corrects['t']['tr2s'])]]))
                
                brs_w_reset_covered_correct_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['brs_w_reset']), mean(covered_corrects['ym']['brs_w_reset']), mean(covered_corrects['m']['brs_w_reset'])], 
                                        [mean(covered_corrects['ef']['brs_w_reset']), mean(covered_corrects['yf']['brs_w_reset']), mean(covered_corrects['f']['brs_w_reset'])], 
                                        [mean(covered_corrects['e']['brs_w_reset']), mean(covered_corrects['y']['brs_w_reset']), mean(covered_corrects['t']['brs_w_reset'])]]))
                
                tr2s_confusion_contras_accepted.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['tr2s']), mean(accepted_contras['ym']['tr2s']), mean(accepted_contras['m']['tr2s'])], 
                                                    [mean(accepted_contras['ef']['tr2s']), mean(accepted_contras['yf']['tr2s']), mean(accepted_contras['f']['tr2s'])], 
                                                    [mean(accepted_contras['e']['tr2s']), mean(accepted_contras['y']['tr2s']), mean(accepted_contras['t']['tr2s'])]]))
                
                tr2s_confusion_contras_correct.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['tr2s']), mean(correct_contras['ym']['tr2s']), mean(correct_contras['m']['tr2s'])], 
                                                    [mean(correct_contras['ef']['tr2s']), mean(correct_contras['yf']['tr2s']), mean(correct_contras['f']['tr2s'])], 
                                                    [mean(correct_contras['e']['tr2s']), mean(correct_contras['y']['tr2s']), mean(correct_contras['t']['tr2s'])]]))
                
                tr2s_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[tr2s_mod_confs[(y_test == 1) & (human_decisions == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr2s_mod_confs[(y_test == 0) & (human_decisions == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(human_decisions == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr2s_mod_confs[(y_test == 1) & (human_decisions == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(y_test == 0) & (human_decisions == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(human_decisions == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean()], 
                                                     [tr2s_mod_confs[(y_test == 1) & (tr2s_model_preds_with_reset != human_decisions)].mean(),
                                                     tr2s_mod_confs[(y_test == 0) & (tr2s_model_preds_with_reset != human_decisions)].mean(), 
                                                     tr2s_mod_confs[(tr2s_model_preds_with_reset != human_decisions)].mean()]]))
                
                tr2s_model_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(model_decs['em']['tr2s']), mean(model_decs['ym']['tr2s']), mean(model_decs['m']['tr2s'])], 
                                                    [mean(model_decs['ef']['tr2s']), mean(model_decs['yf']['tr2s']), mean(model_decs['f']['tr2s'])], 
                                                    [mean(model_decs['e']['tr2s']), mean(model_decs['y']['tr2s']), mean(model_decs['t']['tr2s'])]]))
                
                tr2s_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(decs['em']['tr2s']), mean(decs['ym']['tr2s']), mean(decs['m']['tr2s'])], 
                                                    [mean(decs['ef']['tr2s']), mean(decs['yf']['tr2s']), mean(decs['f']['tr2s'])], 
                                                    [mean(decs['e']['tr2s']), mean(decs['y']['tr2s']), mean(decs['t']['tr2s'])]]))
                
                tr2s_confusion_contras.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(contras['em']['tr2s']), mean(contras['ym']['tr2s']), mean(contras['m']['tr2s'])], 
                                                    [mean(contras['ef']['tr2s']), mean(contras['yf']['tr2s']), mean(contras['f']['tr2s'])], 
                                                    [mean(contras['e']['tr2s']), mean(contras['y']['tr2s']), mean(contras['t']['tr2s'])]]))
                
                hyrs_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[hyrs_mod_confs[(y_test == 1) & (human_decisions == 1) & (hyrs_model_preds != human_decisions)].mean(), 
                                                     hyrs_mod_confs[(y_test == 0) & (human_decisions == 1) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(human_decisions == 1) & (hyrs_model_preds != human_decisions)].mean()], 
                                                     [hyrs_mod_confs[(y_test == 1) & (human_decisions == 0) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(y_test == 0) & (human_decisions == 0) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(human_decisions == 0) & (hyrs_model_preds != human_decisions)].mean()], 
                                                     [hyrs_mod_confs[(y_test == 1) & (hyrs_model_preds != human_decisions)].mean(),
                                                     hyrs_mod_confs[(y_test == 0) & (hyrs_model_preds != human_decisions)].mean(), 
                                                     hyrs_mod_confs[(hyrs_model_preds != human_decisions)].mean()]]))
                
                tr_covered_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['tr']), mean(covereds['ym']['tr']), mean(covereds['m']['tr'])], 
                                        [mean(covereds['ef']['tr']), mean(covereds['yf']['tr']), mean(covereds['f']['tr'])], 
                                        [mean(covereds['e']['tr']), mean(covereds['y']['tr']), mean(covereds['t']['tr'])]]))
                
                tr_covered_correct_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['tr']), mean(covered_corrects['ym']['tr']), mean(covered_corrects['m']['tr'])], 
                                        [mean(covered_corrects['ef']['tr']), mean(covered_corrects['yf']['tr']), mean(covered_corrects['f']['tr'])], 
                                        [mean(covered_corrects['e']['tr']), mean(covered_corrects['y']['tr']), mean(covered_corrects['t']['tr'])]]))
                
                hyrs_covered_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covereds['em']['hyrs']), mean(covereds['ym']['hyrs']), mean(covereds['m']['hyrs'])], 
                                        [mean(covereds['ef']['hyrs']), mean(covereds['yf']['hyrs']), mean(covereds['f']['hyrs'])], 
                                        [mean(covereds['e']['hyrs']), mean(covereds['y']['hyrs']), mean(covereds['t']['hyrs'])]]))
                
                hyrs_covered_correct_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[mean(covered_corrects['em']['hyrs']), mean(covered_corrects['ym']['hyrs']), mean(covered_corrects['m']['hyrs'])], 
                                        [mean(covered_corrects['ef']['hyrs']), mean(covered_corrects['yf']['hyrs']), mean(covered_corrects['f']['hyrs'])], 
                                        [mean(covered_corrects['e']['hyrs']), mean(covered_corrects['y']['hyrs']), mean(covered_corrects['t']['hyrs'])]]))
                
                brs_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[brs_conf[(y_test == 1) &(human_decisions == 1) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(y_test == 0) & (human_decisions == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 1) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (human_decisions == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (human_decisions == 0) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 0) & (brs_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (brs_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (brs_model_preds != human_decisions)].mean(), 
                                         brs_conf[(brs_model_preds != human_decisions)].mean()]]))
                
                brs_w_reset_conf_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                data = [[brs_conf[(y_test == 1) & (human_decisions == 1) & (brs_w_reset_model_preds != human_decisions)].mean(), 
                                         brs_conf[(y_test == 0) & (human_decisions == 1) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 1) & (brs_w_reset_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (human_decisions == 0) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (human_decisions == 0) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(human_decisions == 0) & (brs_w_reset_model_preds != human_decisions)].mean()], 
                                         [brs_conf[(y_test == 1) & (brs_w_reset_model_preds != human_decisions)].mean(),
                                         brs_conf[(y_test == 0) & (brs_w_reset_model_preds != human_decisions)].mean(), 
                                         brs_conf[(brs_w_reset_model_preds != human_decisions)].mean()]]))
                
                brs_model_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(model_decs['em']['brs']), mean(model_decs['ym']['brs']), mean(model_decs['m']['brs'])], 
                        [mean(model_decs['ef']['brs']), mean(model_decs['yf']['brs']), mean(model_decs['f']['brs'])], 
                        [mean(model_decs['e']['brs']), mean(model_decs['y']['brs']), mean(model_decs['t']['brs'])]]))
                
                tr_model_confusion.append(pd.DataFrame(dtype = 'float', index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(model_decs['em']['tr']), mean(model_decs['ym']['tr']), mean(model_decs['m']['tr'])], 
                                                    [mean(model_decs['ef']['tr']), mean(model_decs['yf']['tr']), mean(model_decs['f']['tr'])], 
                                                    [mean(model_decs['e']['tr']), mean(model_decs['y']['tr']), mean(model_decs['t']['tr'])]]))
                
                hyrs_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(decs['em']['hyrs']), mean(decs['ym']['hyrs']), mean(decs['m']['hyrs'])], 
                                                    [mean(decs['ef']['hyrs']), mean(decs['yf']['hyrs']), mean(decs['f']['hyrs'])], 
                                                    [mean(decs['e']['hyrs']), mean(decs['y']['hyrs']), mean(decs['t']['hyrs'])]]))
                
                hyrs_model_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(model_decs['em']['hyrs']), mean(model_decs['ym']['hyrs']), mean(model_decs['m']['hyrs'])], 
                                                    [mean(model_decs['ef']['hyrs']), mean(model_decs['yf']['hyrs']), mean(model_decs['f']['hyrs'])], 
                                                    [mean(model_decs['e']['hyrs']), mean(model_decs['y']['hyrs']), mean(model_decs['t']['hyrs'])]]))
                
                hyrs_confusion_contras.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                            data = [[mean(contras['em']['hyrs']), mean(contras['ym']['hyrs']), mean(contras['m']['hyrs'])], 
                                    [mean(contras['ef']['hyrs']), mean(contras['yf']['hyrs']), mean(contras['f']['hyrs'])], 
                                    [mean(contras['e']['hyrs']), mean(contras['y']['hyrs']), mean(contras['t']['hyrs'])]]))
                
                brs_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(decs['em']['brs']), mean(decs['ym']['brs']), mean(decs['m']['brs'])], 
                        [mean(decs['ef']['brs']), mean(decs['yf']['brs']), mean(decs['f']['brs'])], 
                        [mean(decs['e']['brs']), mean(decs['y']['brs']), mean(decs['t']['brs'])]]))
                

                human_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                                            data= [[mean(decs['em']['human']), mean(decs['ym']['human']), mean(decs['m']['human'])], 
                                                    [mean(decs['ef']['human']), mean(decs['yf']['human']), mean(decs['f']['human'])], 
                                                    [mean(decs['e']['human']), mean(decs['y']['human']), mean(decs['t']['human'])]]))

                tr_confusion_contras.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(contras['em']['tr']), mean(contras['ym']['tr']), mean(contras['m']['tr'])], 
                                                    [mean(contras['ef']['tr']), mean(contras['yf']['tr']), mean(contras['f']['tr'])], 
                                                    [mean(contras['e']['tr']), mean(contras['y']['tr']), mean(contras['t']['tr'])]]))
                tr_confusion_contras_accepted.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['tr']), mean(accepted_contras['ym']['tr']), mean(accepted_contras['m']['tr'])], 
                                                    [mean(accepted_contras['ef']['tr']), mean(accepted_contras['yf']['tr']), mean(accepted_contras['f']['tr'])], 
                                                    [mean(accepted_contras['e']['tr']), mean(accepted_contras['y']['tr']), mean(accepted_contras['t']['tr'])]]))
                
                hyrs_confusion_contras_accepted.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['hyrs']), mean(accepted_contras['ym']['hyrs']), mean(accepted_contras['m']['hyrs'])], 
                                                    [mean(accepted_contras['ef']['hyrs']), mean(accepted_contras['yf']['hyrs']), mean(accepted_contras['f']['hyrs'])], 
                                                    [mean(accepted_contras['e']['hyrs']), mean(accepted_contras['y']['hyrs']), mean(accepted_contras['t']['hyrs'])]]))
                
                tr_confusion_contras_correct.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['tr']), mean(correct_contras['ym']['tr']), mean(correct_contras['m']['tr'])], 
                                                    [mean(correct_contras['ef']['tr']), mean(correct_contras['yf']['tr']), mean(correct_contras['f']['tr'])], 
                                                    [mean(correct_contras['e']['tr']), mean(correct_contras['y']['tr']), mean(correct_contras['t']['tr'])]]))
                
                hyrs_confusion_contras_correct.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                            data = [[mean(correct_contras['em']['hyrs']), mean(correct_contras['ym']['hyrs']), mean(correct_contras['m']['hyrs'])], 
                                    [mean(correct_contras['ef']['hyrs']), mean(correct_contras['yf']['hyrs']), mean(correct_contras['f']['hyrs'])], 
                                    [mean(correct_contras['e']['hyrs']), mean(correct_contras['y']['hyrs']), mean(correct_contras['t']['hyrs'])]]))
                
                totals_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[totals['em'], totals['ym'], totals['m']], 
                        [totals['ef'], totals['yf'], totals['f']], 
                        [totals['e'], totals['y'], totals['t']]]))
                
                brs_confusion_contras.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(contras['em']['brs']), mean(contras['ym']['brs']), mean(contras['m']['brs'])], 
                        [mean(contras['ef']['brs']), mean(contras['yf']['brs']), mean(contras['f']['brs'])], 
                        [mean(contras['e']['brs']), mean(contras['y']['brs']), mean(contras['t']['brs'])]]))
                
                brs_confusion_contras_correct.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['brs']), mean(correct_contras['ym']['brs']), mean(correct_contras['m']['brs'])], 
                                                    [mean(correct_contras['ef']['brs']), mean(correct_contras['yf']['brs']), mean(correct_contras['f']['brs'])], 
                                                    [mean(correct_contras['e']['brs']), mean(correct_contras['y']['brs']), mean(correct_contras['t']['brs'])]]))
                
                brs_confusion_contras_accepted.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['brs']), mean(accepted_contras['ym']['brs']), mean(accepted_contras['m']['brs'])], 
                                                    [mean(accepted_contras['ef']['brs']), mean(accepted_contras['yf']['brs']), mean(accepted_contras['f']['brs'])], 
                                                    [mean(accepted_contras['e']['brs']), mean(accepted_contras['y']['brs']), mean(accepted_contras['t']['brs'])]]))
                
                brs_w_reset_confusion_contras_accepted.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(accepted_contras['em']['brs_w_reset']), mean(accepted_contras['ym']['brs_w_reset']), mean(accepted_contras['m']['brs_w_reset'])], 
                                                    [mean(accepted_contras['ef']['brs_w_reset']), mean(accepted_contras['yf']['brs_w_reset']), mean(accepted_contras['f']['brs_w_reset'])], 
                                                    [mean(accepted_contras['e']['brs_w_reset']), mean(accepted_contras['y']['brs_w_reset']), mean(accepted_contras['t']['brs_w_reset'])]]))

                brs_w_reset_confusion_contras_correct.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'], 
                                            data = [[mean(correct_contras['em']['brs_w_reset']), mean(correct_contras['ym']['brs_w_reset']), mean(correct_contras['m']['brs_w_reset'])], 
                                                    [mean(correct_contras['ef']['brs_w_reset']), mean(correct_contras['yf']['brs_w_reset']), mean(correct_contras['f']['brs_w_reset'])], 
                                                    [mean(correct_contras['e']['brs_w_reset']), mean(correct_contras['y']['brs_w_reset']), mean(correct_contras['t']['brs_w_reset'])]]))
                
                brs_w_reset_confusion_contras.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(contras['em']['brs_w_reset']), mean(contras['ym']['brs_w_reset']), mean(contras['m']['brs_w_reset'])], 
                        [mean(contras['ef']['brs_w_reset']), mean(contras['yf']['brs_w_reset']), mean(contras['f']['brs_w_reset'])], 
                        [mean(contras['e']['brs_w_reset']), mean(contras['y']['brs_w_reset']), mean(contras['t']['brs_w_reset'])]]))
                
                brs_w_reset_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(decs['em']['brs_w_reset']), mean(decs['ym']['brs_w_reset']), mean(decs['m']['brs_w_reset'])], 
                        [mean(decs['ef']['brs_w_reset']), mean(decs['yf']['brs_w_reset']), mean(decs['f']['brs_w_reset'])], 
                        [mean(decs['e']['brs_w_reset']), mean(decs['y']['brs_w_reset']), mean(decs['t']['brs_w_reset'])]]))
                
                brs_w_reset_model_confusion.append(pd.DataFrame(index=['Male', 'Female', 'Total'], columns=['Elderly', 'Young', 'Total'],
                data = [[mean(model_decs['em']['brs_w_reset']), mean(model_decs['ym']['brs_w_reset']), mean(model_decs['m']['brs_w_reset'])], 
                        [mean(model_decs['ef']['brs_w_reset']), mean(model_decs['yf']['brs_w_reset']), mean(model_decs['f']['brs_w_reset'])], 
                        [mean(model_decs['e']['brs_w_reset']), mean(model_decs['y']['brs_w_reset']), mean(model_decs['t']['brs_w_reset'])]]))
            

            
            #append values to appropriate row in results
            results.loc[cost, 'tr_team_w_reset_decision_loss'].append(mean(tr_team_w_reset_decision_loss))
            results.loc[cost, 'tr_team_wo_reset_decision_loss'].append(mean(tr_team_wo_reset_decision_loss))
            results.loc[cost, 'tr_model_w_reset_decision_loss'].append(mean(tr_model_w_reset_decision_loss))
            results.loc[cost, 'tr_model_wo_reset_decision_loss'].append(mean(tr_model_wo_reset_decision_loss))
            results.loc[cost, 'hyrs_model_decision_loss'].append(mean(hyrs_model_decision_loss))
            results.loc[cost, 'hyrs_team_decision_loss'].append(mean(hyrs_team_decision_loss))
            results.loc[cost, 'brs_model_decision_loss'].append(mean(brs_model_decision_loss))
            results.loc[cost, 'brs_team_decision_loss'].append(mean(brs_team_decision_loss))
            results.loc[cost, 'brs_w_reset_team_decision_loss'].append(mean(brs_w_reset_team_decision_loss))
            results.loc[cost, 'brs_w_reset_model_decision_loss'].append(mean(brs_w_reset_model_decision_loss))
            results.loc[cost, 'tr_model_w_reset_contradictions'].append(mean(tr_model_w_reset_contradictions))
            results.loc[cost, 'tr_model_wo_reset_contradictions'].append(mean(tr_model_wo_reset_contradictions))
            results.loc[cost, 'hyrs_model_contradictions'].append(mean(hyrs_model_contradictions))
            results.loc[cost, 'brs_model_contradictions'].append(mean(brs_model_contradictions))
            results.loc[cost, 'brs_w_reset_model_contradictions'].append(mean(brs_w_reset_model_contradictions))
            results.loc[cost, 'tr_team_w_reset_objective'].append(mean(tr_team_w_reset_objective))
            results.loc[cost, 'tr_team_wo_reset_objective'].append(mean(tr_team_wo_reset_objective))
            results.loc[cost, 'tr_model_w_reset_objective'].append(mean(tr_model_w_reset_objective))
            results.loc[cost, 'tr_model_wo_reset_objective'].append(mean(tr_model_wo_reset_objective))
            results.loc[cost, 'hyrs_model_objective'].append(mean(hyrs_model_objective))
            results.loc[cost, 'hyrs_team_objective'].append(mean(hyrs_team_objective))
            results.loc[cost, 'brs_model_objective'].append(mean(brs_model_objective))
            results.loc[cost, 'brs_team_objective'].append(mean(brs_team_objective))
            results.loc[cost, 'brs_w_reset_model_objective'].append(mean(brs_w_reset_model_objective))
            results.loc[cost, 'brs_w_reset_team_objective'].append(mean(brs_w_reset_team_objective))
            results.loc[cost, 'human_decision_loss'].append(mean(human_decision_loss))
            results.loc[cost, 'hyrs_norecon_objective'].append(mean(hyrs_norecon_objective))
            results.loc[cost, 'hyrs_norecon_model_decision_loss'].append(mean(hyrs_norecon_model_decision_loss))
            results.loc[cost, 'hyrs_norecon_team_decision_loss'].append(mean(hyrs_norecon_team_decision_loss))
            results.loc[cost, 'hyrs_norecon_model_contradictions'].append(mean(hyrs_norecon_model_contradictions))
            results.loc[cost, 'tr2s_team_w_reset_decision_loss'].append(mean(tr2s_team_w_reset_decision_loss))
            results.loc[cost, 'tr2s_team_wo_reset_decision_loss'].append(mean(tr2s_team_wo_reset_decision_loss))
            results.loc[cost, 'tr2s_model_w_reset_decision_loss'].append(mean(tr2s_model_w_reset_decision_loss))
            results.loc[cost, 'tr2s_model_w_reset_contradictions'].append(mean(tr2s_model_w_reset_contradictions))
            results.loc[cost, 'tr2s_model_wo_reset_contradictions'].append(mean(tr2s_model_wo_reset_contradictions))
            results.loc[cost, 'tr2s_model_wo_reset_decision_loss'].append(mean(tr2s_model_wo_reset_decision_loss))
            results.loc[cost, 'tr2s_team_w_reset_objective'].append(mean(tr2s_team_w_reset_objective))
            results.loc[cost, 'tr2s_team_wo_reset_objective'].append(mean(tr2s_team_wo_reset_objective))
            
            
            
        
    
    tr_confusion = pd.concat(tr_confusion)
    tr_model_confusion = pd.concat(tr_model_confusion)
    hyrs_model_confusion = pd.concat(hyrs_model_confusion)
    brs_model_confusion = pd.concat(brs_model_confusion)
    brs_w_reset_model_confusion = pd.concat(brs_w_reset_model_confusion)
    brs_w_reset_covered_confusion = pd.concat(brs_w_reset_covered_confusion)
    
    tr_conf_confusion = pd.concat(tr_conf_confusion)
    hyrs_conf_confusion = pd.concat(hyrs_conf_confusion)
    tr_covered_confusion = pd.concat(tr_covered_confusion)
    hyrs_covered_confusion = pd.concat(hyrs_covered_confusion)
    brs_confusion = pd.concat(brs_confusion)
    brs_conf_confusion = pd.concat(brs_conf_confusion)
    brs_w_reset_confusion = pd.concat(brs_w_reset_confusion)
    brs_w_reset_conf_confusion = pd.concat(brs_w_reset_conf_confusion)
    hyrs_confusion = pd.concat(hyrs_confusion)
    human_confusion = pd.concat(human_confusion)
    tr_confusion_contras = pd.concat(tr_confusion_contras)
    tr_confusion_contras_accepted = pd.concat(tr_confusion_contras_accepted)
    tr_confusion_contras_correct = pd.concat(tr_confusion_contras_correct)
    totals_confusion = pd.concat(totals_confusion)
    brs_confusion_contras = pd.concat(brs_confusion_contras)
    brs_confusion_contras_correct = pd.concat(brs_confusion_contras_correct)
    brs_confusion_contras_accepted = pd.concat(brs_confusion_contras_accepted)
    brs_w_reset_confusion_contras = pd.concat(brs_w_reset_confusion_contras)
    brs_w_reset_confusion_contras_correct = pd.concat(brs_w_reset_confusion_contras_correct)
    brs_w_reset_confusion_contras_accepted = pd.concat(brs_w_reset_confusion_contras_accepted)
    hyrs_confusion_contras = pd.concat(hyrs_confusion_contras)
    hyrs_confusion_contras_correct = pd.concat(hyrs_confusion_contras_correct)
    hyrs_confusion_contras_accepted = pd.concat(hyrs_confusion_contras_accepted)
    brs_w_reset_covered_correct_confusion = pd.concat(brs_w_reset_covered_correct_confusion)

    tr_covered_correct_confusion = pd.concat(tr_covered_correct_confusion)
    hyrs_covered_correct_confusion = pd.concat(hyrs_covered_correct_confusion)

    tr2s_covered_confusion = pd.concat(tr2s_covered_confusion)
    tr2s_covered_correct_confusion = pd.concat(tr2s_covered_correct_confusion)
    tr2s_confusion_contras_accepted = pd.concat(tr2s_confusion_contras_accepted)
    tr2s_confusion_contras_correct = pd.concat(tr2s_confusion_contras_correct)
    tr2s_conf_confusion = pd.concat(tr2s_conf_confusion)
    tr2s_model_confusion = pd.concat(tr2s_model_confusion)
    tr2s_confusion = pd.concat(tr2s_confusion)
    tr2s_confusion_contras = pd.concat(tr2s_confusion_contras)

     

    
    
    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))

    '''
    #multiply all female rows by 3 in human_confusion
    human_confusion2 = human_confusion.copy()
    human_confusion2[human_confusion2.index=='Female'] = 3*human_confusion2[human_confusion2.index=='Female']
    #make 'Total' rows be sum of 'female' and 'male' row directly above
    human_confusion2.loc['Total'] = human_confusion2.loc

    tr_confusion2 = tr_confusion.copy()
    tr_confusion2[tr_confusion2.index=='Female'] = 3*tr_confusion2[tr_confusion2.index=='Female']

    hyrs_confusion2 = hyrs_confusion.copy()
    hyrs_confusion2[hyrs_confusion2.index=='Female'] = 3*hyrs_confusion2[hyrs_confusion.index=='Female']

    brs_confusion2 = brs_confusion.copy()
    brs_confusion2[brs_confusion2.index=='Female'] = 3*brs_confusion2[brs_confusion2.index=='Female']
    '''


    case_results = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 
                                         'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])
    case_means = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 
                                         'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])
    
    case_std = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 
                                         'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])

    case_results.loc['Advising Rate', :] = weighted_avg_and_std((tr_covered_confusion/totals_confusion), totals_confusion)[0]
    
    case_results.loc['Contradiction Rate', :]  = weighted_avg_and_std((tr_confusion_contras/totals_confusion), totals_confusion)[0]
    case_results.loc['Advising Confidence', :] = weighted_avg_and_std((tr_conf_confusion), weights=totals_confusion)[0]
    case_results.loc['Contradiction Acceptance Rate', :] = weighted_avg_and_std((tr_confusion_contras_accepted/tr_confusion_contras), tr_confusion_contras)[0]
    case_results.loc['Improvement w.r.t. TDL', :] = weighted_avg_and_std(((human_confusion-tr_confusion)/total), totals_confusion)[0]
    case_results.loc['Reconciliation Costs Incurred', :] = weighted_avg_and_std(-(cost * tr_confusion_contras)/total, totals_confusion)[0]
    case_results.loc['Improved in TTL w.r.t. Human', :] = weighted_avg_and_std(((human_confusion-tr_confusion)-(cost * tr_confusion_contras))/total, totals_confusion)[0]
    case_results.loc['Advising Accuracy', :] = weighted_avg_and_std(tr_confusion_contras_correct/tr_confusion_contras, tr_confusion_contras)[0]
    case_results.loc['Self Advising Rate', :] = weighted_avg_and_std((tr_confusion_contras/quick_perc(tr_confusion_contras)), quick_perc(tr_confusion_contras))[0]

    tr2s_case_results = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence',
                                            'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])
    
    tr2s_case_results.loc['Advising Rate', :] = weighted_avg_and_std((tr2s_covered_confusion/totals_confusion), totals_confusion)[0]
    tr2s_case_results.loc['Contradiction Rate', :] = weighted_avg_and_std((tr2s_confusion_contras/totals_confusion), totals_confusion)[0]
    tr2s_case_results.loc['Advising Confidence', :] = weighted_avg_and_std((tr2s_conf_confusion), weights=totals_confusion)[0]
    tr2s_case_results.loc['Contradiction Acceptance Rate', :] = weighted_avg_and_std((tr2s_confusion_contras_accepted/tr2s_confusion_contras), tr2s_confusion_contras)[0]
    tr2s_case_results.loc['Improvement w.r.t. TDL', :] = weighted_avg_and_std(((human_confusion-tr2s_confusion)/total), totals_confusion)[0]
    tr2s_case_results.loc['Reconciliation Costs Incurred', :] = weighted_avg_and_std(-(cost * tr2s_confusion_contras)/total, totals_confusion)[0]
    tr2s_case_results.loc['Improved in TTL w.r.t. Human', :] = weighted_avg_and_std(((human_confusion-tr2s_confusion)-(cost * tr2s_confusion_contras))/total, totals_confusion)[0]
    tr2s_case_results.loc['Advising Accuracy', :] = weighted_avg_and_std(tr2s_confusion_contras_correct/tr2s_confusion_contras, tr2s_confusion_contras)[0]
    tr2s_case_results.loc['Self Advising Rate', :] = weighted_avg_and_std((tr2s_confusion_contras/quick_perc(tr2s_confusion_contras)), quick_perc(tr2s_confusion_contras))[0]

    hyrs_case_results = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 
                                         'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])
    hyrs_case_results.loc['Advising Rate', :] = weighted_avg_and_std((hyrs_covered_confusion/totals_confusion), totals_confusion)[0]
    hyrs_case_results.loc['Contradiction Rate', :] = weighted_avg_and_std((hyrs_confusion_contras/totals_confusion), totals_confusion)[0]
    hyrs_case_results.loc['Advising Confidence', :] = weighted_avg_and_std((hyrs_conf_confusion), weights=totals_confusion)[0]
    hyrs_case_results.loc['Contradiction Acceptance Rate', :] = weighted_avg_and_std((hyrs_confusion_contras_accepted/hyrs_confusion_contras), hyrs_confusion_contras)[0]
    hyrs_case_results.loc['Improvement w.r.t. TDL', :] = weighted_avg_and_std(((human_confusion-hyrs_confusion)/total), totals_confusion)[0]
    hyrs_case_results.loc['Reconciliation Costs Incurred', :] = weighted_avg_and_std(-(cost * hyrs_confusion_contras)/total, totals_confusion)[0]
    hyrs_case_results.loc['Improved in TTL w.r.t. Human', :] = weighted_avg_and_std(((human_confusion-hyrs_confusion)-(cost * hyrs_confusion_contras))/total, totals_confusion)[0]
    hyrs_case_results.loc['Advising Accuracy', :] = weighted_avg_and_std(hyrs_confusion_contras_correct/hyrs_confusion_contras, hyrs_confusion_contras)[0]
    hyrs_case_results.loc['Self Advising Rate', :] = weighted_avg_and_std((hyrs_confusion_contras/quick_perc(hyrs_confusion_contras)), quick_perc(hyrs_confusion_contras))[0]

    brs_case_results = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 
                                         'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])
    
    brs_case_results.loc['Advising Rate', :] = '1.000 \pm 0.000'
    brs_case_results.loc['Contradiction Rate', :] = weighted_avg_and_std((brs_confusion_contras/totals_confusion), totals_confusion)[0]
    brs_case_results.loc['Advising Confidence', :] = weighted_avg_and_std((brs_conf_confusion), weights=totals_confusion)[0]
    brs_case_results.loc['Contradiction Acceptance Rate', :] = weighted_avg_and_std((brs_confusion_contras_accepted/brs_confusion_contras), brs_confusion_contras)[0]
    brs_case_results.loc['Improvement w.r.t. TDL', :] = weighted_avg_and_std(((human_confusion-brs_confusion)/total), totals_confusion)[0]
    brs_case_results.loc['Reconciliation Costs Incurred', :] = weighted_avg_and_std(-(cost * brs_confusion_contras)/total, totals_confusion)[0]
    brs_case_results.loc['Improved in TTL w.r.t. Human', :] = weighted_avg_and_std(((human_confusion-brs_confusion)-(cost * brs_confusion_contras))/total, totals_confusion)[0]
    brs_case_results.loc['Advising Accuracy', :] = weighted_avg_and_std(brs_confusion_contras_correct/brs_confusion_contras, brs_confusion_contras)[0]
    brs_case_results.loc['Self Advising Rate', :] = weighted_avg_and_std((brs_confusion_contras/quick_perc(brs_confusion_contras)), quick_perc(brs_confusion_contras))[0]

    brs_w_reset_case_results = pd.DataFrame(index = ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 
                                         'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human'], columns = ['Female', 'Male', 'Total'])
    
    brs_w_reset_case_results.loc['Advising Rate', :] = weighted_avg_and_std((brs_w_reset_covered_confusion/totals_confusion), totals_confusion)[0]
    brs_w_reset_case_results.loc['Contradiction Rate', :] = weighted_avg_and_std((brs_w_reset_confusion_contras/totals_confusion), totals_confusion)[0]
    brs_w_reset_case_results.loc['Advising Confidence', :] = weighted_avg_and_std((brs_w_reset_conf_confusion), weights=totals_confusion)[0]
    brs_w_reset_case_results.loc['Contradiction Acceptance Rate', :] = weighted_avg_and_std((brs_w_reset_confusion_contras_accepted/brs_w_reset_confusion_contras), brs_w_reset_confusion_contras)[0]
    brs_w_reset_case_results.loc['Improvement w.r.t. TDL', :] = weighted_avg_and_std(((human_confusion-brs_w_reset_confusion)/total), totals_confusion)[0]
    brs_w_reset_case_results.loc['Reconciliation Costs Incurred', :] = weighted_avg_and_std(-(cost * brs_w_reset_confusion_contras)/total, totals_confusion)[0]
    brs_w_reset_case_results.loc['Improved in TTL w.r.t. Human', :] = weighted_avg_and_std(((human_confusion-brs_w_reset_confusion)-(cost * brs_w_reset_confusion_contras))/total, totals_confusion)[0]
    brs_w_reset_case_results.loc['Advising Accuracy', :] = weighted_avg_and_std(brs_w_reset_confusion_contras_correct/brs_w_reset_confusion_contras, brs_w_reset_confusion_contras)[0]
    brs_w_reset_case_results.loc['Self Advising Rate', :] = weighted_avg_and_std((brs_w_reset_model_confusion/quick_perc(brs_w_reset_model_confusion)), quick_perc(brs_w_reset_model_confusion))[0]

    bars = pd.DataFrame(columns=['Method', 'Group', 'Mean', 'Std', 'Metric'])
    boxes = pd.DataFrame(columns=['Method', 'Group', 'Value', 'Metric'])
    method_dict = {'TR': case_results, 'TR-no(ADB)': hyrs_case_results, 'Task-Only (Current Practice)': brs_case_results, 'TR-SelectiveOnly': brs_w_reset_case_results}
    for method in ['TR','TR-no(ADB)', 'Task-Only (Current Practice)', 'TR-SelectiveOnly']:
        for group in ['Female', 'Male', 'Total']:
            for metric in ['Advising Accuracy', 'Advising Rate', 'Contradiction Rate', 'Advising Confidence', 'Contradiction Acceptance Rate', 'Improvement w.r.t. TDL', 'Reconciliation Costs Incurred', 'Improved in TTL w.r.t. Human', 'Self Advising Rate']:
                bars = bars.append({'Method': method, 'Group': group, 'Mean': float(method_dict[method].loc[metric, group].split(' \pm ')[0]), 'Std': float(method_dict[method].loc[metric, group].split(' \pm ')[1]), 'Metric': metric}, ignore_index=True)
    bars['Doctor'] = 'Doctor B'
    bars.to_pickle('results/bars_docB.pkl')


    






    '''
    import seaborn as sns
    d_tr = ((human_confusion-tr_confusion)-(cost * tr_confusion_contras))/totals_confusion
    d_hyrs = ((human_confusion-hyrs_confusion)-(cost * hyrs_confusion_contras))/totals_confusion
    d_brs = ((human_confusion-brs_confusion)-(cost * brs_confusion_contras))/totals_confusion
    d_brs_reset = ((human_confusion-brs_confusion)-(cost * brs_w_reset_confusion_contras))/totals_confusion
    '''
    d_tr = tr_conf_confusion
    d_hyrs = hyrs_conf_confusion
    d_brs = brs_conf_confusion
    d_brs_reset = brs_w_reset_conf_confusion
    d_tr_contras = (tr_confusion_contras/totals_confusion)['Total']
    d_hyrs_contras = (hyrs_confusion_contras/totals_confusion)['Total']
    d_brs_contras = (brs_confusion_contras/totals_confusion)['Total']
    d_brs_reset_contras = (brs_w_reset_conf_confusion/totals_confusion)['Total']
    d_tr_accept = (tr_confusion_contras_accepted/tr_confusion_contras)['Total']
    d_hyrs_accept = (hyrs_confusion_contras_accepted/hyrs_confusion_contras)['Total']
    d_brs_accept = (brs_confusion_contras_accepted/brs_confusion_contras)['Total']
    d_brs_reset_accept = (brs_w_reset_confusion_contras_accepted/brs_w_reset_confusion_contras)['Total']
    d_tr_advAcc = (tr_confusion_contras_correct/tr_confusion_contras)['Total']
    d_hyrs_advAcc = (hyrs_confusion_contras_correct/hyrs_confusion_contras)['Total']
    d_brs_advAcc = (brs_confusion_contras_correct/brs_confusion_contras)['Total']
    d_brs_reset_advAcc = (brs_w_reset_confusion_contras_correct/brs_w_reset_confusion_contras)['Total']
    d_tr = d_tr['Total']
    d_hyrs = d_hyrs['Total']
    d_brs = d_brs['Total']
    d_brs_reset = d_brs_reset['Total']
    d_tr = pd.DataFrame(d_tr)
    d_hyrs = pd.DataFrame(d_hyrs)
    d_brs = pd.DataFrame(d_brs)
    d_brs_reset = pd.DataFrame(d_brs_reset)
    d_tr.columns = ['Confidence']
    d_hyrs.columns = ['Confidence']
    d_brs.columns = ['Confidence']
    d_brs_reset.columns = ['Confidence']
    d_tr['ContraRate'] = d_tr_contras
    d_hyrs['ContraRate'] = d_hyrs_contras
    d_brs['ContraRate'] = d_brs_contras
    d_brs_reset['ContraRate'] = d_brs_reset_contras
    d_tr['AcceptRate'] = d_tr_accept
    d_hyrs['AcceptRate'] = d_hyrs_accept
    d_brs['AcceptRate'] = d_brs_accept
    d_tr['AdviceAcc'] = d_tr_advAcc
    d_hyrs['AdviceAcc'] = d_hyrs_advAcc
    d_brs['AdviceAcc'] = d_brs_advAcc
    d_brs_reset['AdviceAcc'] = d_brs_reset_advAcc
    d_brs_reset['AcceptRate'] = d_brs_reset_accept
    d_tr["Method"] = 'TR'
    d_hyrs["Method"] = 'TR-no(ADB)'
    d_brs["Method"] = 'Task-Only (Current Practice)'
    d_brs_reset["Method"] = 'TR-SelectiveOnly'
    bplot_data = pd.concat([d_tr, d_brs, d_brs_reset, d_hyrs])
    bplot_data['Group'] = bplot_data.index
    
    
    bplot_data['Doctor'] = 'Doctor B'
    bplot_data.to_pickle('results/bplotB.pkl')
    
    color_dict = {'TR': '#348ABD', 'tr': '#348ABD', 'TR-no(ADB)': '#8EBA42', 'tr-no(ADB)': '#8EBA42', 'Task-Only (Current Practice)':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42', 'BRSselect': '#FF7F00'}
    '''
color_dict = {'TR': '#348ABD', 'tr': '#348ABD', 'TR-no(ADB)': '#8EBA42', 
              'tr-no(ADB)': '#8EBA42', 'Task-Only (Current Practice)': '#988ED5', 
              'Human': 'darkgray', 'HYRSRecon': '#8EBA42', 'BRSselect': '#FF7F00'}
    
# Methods ordered from top to bottom: 'tr', 'Task-Only (Current Practice)', 'tr-no(ADB)'
methods = ['TR', 'Task-Only (Current Practice)', 'TR-no(ADB)']
docs = ['Doctor B']

# Initialize structures to store male and female proportions for each method
bar_means_female = {}
bar_means_male = {}

# Extract the percentage values for female and male groups separately
for method in methods:
    # Female group
    bar_means_female[method] = list(bars[(bars.Group == 'Female') & 
                                         (bars.Method != 'TR-SelectiveOnly') & 
                                         (bars.Metric == 'Self Advising Rate') & 
                                         (bars.Method == method)].sort_values(by='Doctor').Mean)
    
    # Male group
    bar_means_male[method] = list(bars[(bars.Group == 'Male') & 
                                       (bars.Method != 'TR-SelectiveOnly') & 
                                       (bars.Metric == 'Self Advising Rate') & 
                                       (bars.Method == method)].sort_values(by='Doctor').Mean)

# Set up figure and axes for two side-by-side horizontal bar plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Data for Doctor B (first entry) and Doctor B (second entry)
doc_a_female = [bar_means_female[method][0] for method in methods]  # Female percentages for Doctor B
doc_a_male = [bar_means_male[method][0] for method in methods]      # Male percentages for Doctor B

#doc_b_female = [bar_means_female[method][0] for method in methods]  # Female percentages for Doctor B
#doc_b_male = [bar_means_male[method][0] for method in methods]      # Male percentages for Doctor B

# X-axis positions for the methods (reverse the order to display them top to bottom correctly)
x_pos = np.arange(len(methods))[::-1]

# Function to add percentage labels along with "Male" and "Female"
def add_percentage_labels(ax, male_values, female_values, x_pos):
    for i, (male, female) in enumerate(zip(male_values, female_values)):
        # Adding Male label
        ax.text(male / 2, x_pos[i], f'Male: {male:.2%}', va='center', ha='center', color='black', fontweight='bold')
        # Adding Female label
        ax.text(male + female / 2, x_pos[i], f'Female: {female:.2%}', va='center', ha='center', color='black', fontweight='bold')

# Doctor B plot
for i, method in enumerate(methods):
    # Male bar (no hatch)
    ax.barh(x_pos[i], doc_a_male[i], color=color_dict[method], label='Male', edgecolor='black')
    # Female bar (dash hatch, alpha=0.7)
    ax.barh(x_pos[i], doc_a_female[i], left=doc_a_male[i], color=color_dict[method], edgecolor='black',  alpha=0.7)

ax.set_yticks(x_pos)
ax.set_yticklabels(methods, rotation=45)
ax.set_title('Doctor B')
ax.set_xlim(0, 1)
ax.set_xlabel('Proportion of Contradicting Advice Given')

# Add percentage labels for Doctor B
add_percentage_labels(ax, doc_a_male, doc_a_female, x_pos)

# Doctor B plot
for i, method in enumerate(methods):
    # Male bar (no hatch)
    ax2.barh(x_pos[i], doc_b_male[i], color=color_dict[method], label='Male', edgecolor='black')
    # Female bar (dash hatch, alpha=0.7)
    ax2.barh(x_pos[i], doc_b_female[i], left=doc_b_male[i], color=color_dict[method], edgecolor='black', alpha=0.7)

ax2.set_yticks(x_pos)
ax2.set_yticklabels(methods)
ax2.set_title('Doctor B')
ax2.set_xlim(0, 1)
ax2.set_xlabel('Proportion of Contradicting Advice Given')

# Add percentage labels for Doctor B
add_percentage_labels(ax2, doc_b_male, doc_b_female, x_pos)

# Add a common legend (no hatch for Male, dash hatch for Female)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='grey', edgecolor='black', label='Male'),
                   Patch(facecolor='grey', edgecolor='black', hatch='-', label='Female', alpha=0.7)]

#fig.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()


    
# Define groups and metrics
groups = ['Female', 'Male', 'Total']
metrics1 = ['Confidence', 'AdviceAcc', 'ContraRate', 'AcceptRate']
metrics2 = ['ContraRate', 'Confidence', 'AcceptRate', 'Vadded']

# Cleaned names for the metrics to be used in titles and y-axis labels
cleaned_metric_names = {
    'ContraRate': 'Advising rate',
    'Confidence': 'Advising confidence \n',
    'AcceptRate': '   Advice  \n acceptance rate',
    'AdviceAcc': 'Advising accuracy \n',
    'Vadded': 'Value added'
}

# Create subplots: 5 rows, 3 columns
fig, axes = plt.subplots(3, 3, figsize=(5.5, 4.5))

# Adjust layout
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for col, group in enumerate(groups):
    ax = axes[0, col]
    ax2 = axes[1, col]
    ax3 = axes[2, col]
    hatch_dict = {
        'TR': '/',  # Single diagonal line
        'TR-no(ADB)': '.',  # Dotted pattern
        'Task-Only (Current Practice)': '-'  # Horizontal line
    }
    docs = ['Doctor B']
    methods = ['TR', 'Task-Only (Current Practice)', 'TR-no(ADB)']
    cleaned_methods = ['TR', 'Task-\nOnly', 'TR-\nno(ADB)']
    
    bar_means = {}
    bar_se = {}
    tdl_bar_means = {}
    tdl_bar_se = {}
    cost_bar_means = {}
    cost_bar_se = {}
    
    for method in methods: 
        bar_means[method] = list(bars[
            (bars.Group == group) & 
            (bars.Method != 'TR-SelectiveOnly') & 
            (bars.Metric == 'Improved in TTL w.r.t. Human') & 
            (bars.Method == method)
        ].sort_values(by='Doctor').Mean)
        
        bar_se[method] = list(bars[
            (bars.Group == group) & 
            (bars.Method != 'TR-SelectiveOnly') & 
            (bars.Metric == 'Improved in TTL w.r.t. Human') & 
            (bars.Method == method)
        ].sort_values(by='Doctor').Std)

        tdl_bar_means[method] = list(bars[
            (bars.Group == group) & 
            (bars.Method != 'TR-SelectiveOnly') & 
            (bars.Metric == 'Improvement w.r.t. TDL') & 
            (bars.Method == method)
        ].sort_values(by='Doctor').Mean)
        
        tdl_bar_se[method] = list(bars[
            (bars.Group == group) & 
            (bars.Method != 'TR-SelectiveOnly') & 
            (bars.Metric == 'Improvement w.r.t. TDL') & 
            (bars.Method == method)
        ].sort_values(by='Doctor').Std)

        cost_bar_means[method] = -1*np.array(list(bars[
            (bars.Group == group) & 
            (bars.Method != 'TR-SelectiveOnly') & 
            (bars.Metric == 'Reconciliation Costs Incurred') & 
            (bars.Method == method)
        ].sort_values(by='Doctor').Mean))
        
        cost_bar_se[method] = list(bars[
            (bars.Group == group) & 
            (bars.Method != 'TR-SelectiveOnly') & 
            (bars.Metric == 'Reconciliation Costs Incurred') & 
            (bars.Method == method)
        ].sort_values(by='Doctor').Std)

    x = np.arange(len(methods)) * 0.7 # Create distinct x positions for each method
    width = 0.5  # Width of the bars
    
    # Now plot each method's bar at the corresponding x position
    for i, method in enumerate(methods):
        rects_tdl = ax.bar(x[i], bar_means[method], width, color=color_dict[method], 
                           edgecolor='black', yerr=bar_se[method])

        rects_tdl2 = ax2.bar(x[i], tdl_bar_means[method], width, color=color_dict[method], 
                           edgecolor='black', yerr=tdl_bar_se[method])
        rects_tdl3 = ax3.bar(x[i], cost_bar_means[method], width, color=color_dict[method], 
                           edgecolor='black', yerr=cost_bar_se[method])
        

    if group == 'Female':
        ax.set_ylabel('Value added', size=8)
        ax2.set_ylabel('Accuracy improvement \n w.r.t. human', size=8)
        ax3.set_ylabel('Advising costs \n incurred', size=8)
    
    for col, group in enumerate(groups):
        axes[0, col].set_title(f"{group}", fontsize=10)

    # Set the x-tick labels to the cleaned method names
    ax.set_xticks(x)
    ax.set_ylim(-0.065, 0.05)
    ax.set_xticklabels(cleaned_methods, size=6)
    ax2.set_xticks(x)
    ax2.set_ylim(-0.035, 0.05)
    ax2.set_xticklabels(cleaned_methods, size=6)
    ax3.set_xticks(x)
    ax3.set_ylim(0.0, 0.035)
    ax3.set_xticklabels(cleaned_methods, size=6)

    # Optional: add a legend for clarity
    #ax.legend(prop={'size': 4.5})

    subs = ['a','b','c','d','e','f','g','h','i']
    pad = 5
    plt.tight_layout()

    for ax, sub in zip(axes.flatten(), subs):
        ax.annotate(sub, xy=(-0.4, 1.05), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=10, ha='center', va='baseline', weight='bold')

    

# Create subplots: 5 rows, 3 columns
fig, axes = plt.subplots(4, 3, figsize=(5.5, 6))

# Adjust layout
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Function to generate each plot
def create_boxplot(ax, group, metric):
    sns.boxplot(
        data=bplot_data[(bplot_data['Method'] != 'TR-SelectiveOnly') & (bplot_data['Group'] == group)],
        y=metric, palette=color_dict, x='Method',
        hue_order=['TR', 'Task-Only (Current Practice)', 'TR-no(ADB)'],
        showfliers=False, ax=ax
    )
    # Reduce legend size
    #ax.legend(prop={'size': 4.5})
    
    # Remove individual titles, x-axis, and y-axis labels
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

# Loop over each row and column to create the subplots
for row, metric in enumerate(metrics1):
    for col, group in enumerate(groups):
        ax = axes[row, col]  # Get the axis for the current subplot
        create_boxplot(ax, group, metric)
        if group == 'Female':
           ax.set_ylabel(cleaned_metric_names[metric], size=8)
        ax.set_xticklabels(cleaned_methods, size=6)
        if metric == 'AdviceAcc':
            ax.set_ylim(0.0, 1.05)
        elif metric == 'ContraRate':
            ax.set_ylim(-.05, 0.55)
        elif metric == 'Confidence':
            ax.set_ylim(0.6, 1.05)
        elif metric == 'AcceptRate':
            ax.set_ylim(-0.05, 1.05)

# Set shared column titles (Female, Male, Total)
for col, group in enumerate(groups):
    axes[0, col].set_title(f"{group}", fontsize=12)

# Set shared row titles (Contradiction Rate, Advising Confidence, Contradiction Acceptance Rate)
#for row, metric in enumerate(metrics2):
#    fig.text(0.03, 0.9 - row * 0.215, cleaned_metric_names[metric], va='center', rotation='vertical', fontsize=12)


    
# Show the full plot
subs = ['a','b','c','d','e','f','g','h','i','j','k','l']
pad = 5
plt.tight_layout()

for ax, sub in zip(axes.flatten(), subs):
    ax.annotate(sub, xy=(-0.3, 1.05), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=10, ha='center', va='baseline', weight='bold')
plt.show()
plt.savefig('boxes.png', dpi=400)
    '''
    return results_means, results_stderrs, results



costs = [0.2]
num_runs = 5
dataset = 'heart_disease'
case1_means, case1_std, case1_rs = make_results(dataset, 'biased', num_runs, costs, False, asym_costs=[3,1])
   

print('pause')

