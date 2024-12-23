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
import inspect
from scipy.stats import bernoulli, uniform

#making sure wd is file directory so hardcoded paths work
os.chdir("..")
def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def noADB(human_conf, model_conf, agreement):
    return np.ones(len(human_conf))

def randADB(human_conf, model_conf, agreement):
    return np.random.rand(len(human_conf))

def halfADB(human_conf, model_conf, agreement):
    return np.ones(len(human_conf))*0.5

def advADB(c_human, c_model, agreement, delta=5, beta=0.5, k=0.63, gamma=0.95):
    # from will you accept the AI recommendation
    

    

    def w(p, k):
        return (p**k)/((p**k)+(1-p)**k)

    c_human_new = c_human.copy()
    c_human_new[c_human_new <= 0] = 0.0000001
    c_human_new[c_human_new >= 1] = 0.9999999

    # transform human confidence back to probability of human's estimate of their choice being correct
    c_human_new = (c_human_new/2)+0.5

    c_model_new = c_model.copy()
    c_model_new[c_model_new <= 0] = 0.0000001
    c_model_new[c_model_new >= 1] = 0.9999999

    c_human_new[~agreement] = 1-c_human_new[~agreement]
    a = (c_model_new**gamma)/((c_model_new**gamma)+((1-c_model_new)**gamma))
    b = (c_human_new**gamma)/((c_human_new**gamma)+((1-c_human_new)**gamma))

    conf = 1/(1+(((1-a)*(1-b))/(a*b)))

    util_accept = (1+beta)*w(conf, k)-beta
    util_reject = 1-(1+beta)*w(conf, k)

    prob = np.exp(delta*util_accept) / \
        (np.exp(delta*util_accept)+np.exp(delta*util_reject))
    df = pd.DataFrame({'c_human': c_human, 'c_human_new': c_human_new,
                    'conf': conf, 'c_model': c_model, 'agreement': agreement, 'prob': prob})

    return 1-prob


def parADB(c_human, c_model, agreement, delta=5, beta=0.5, k=0.63, gamma=0.95):
    # from will you accept the AI recommendation
    

    

    def w(p, k):
        return (p**k)/((p**k)+(1-p)**k)

    c_human_new = c_human.copy()
    c_human_new[c_human_new <= 0] = 0.0000001
    c_human_new[c_human_new >= 1] = 0.9999999

    # transform human confidence back to probability of human's estimate of their choice being correct
    c_human_new = (c_human_new/2)+0.5

    c_model_new = c_model.copy()
    c_model_new[c_model_new <= 0] = 0.0000001
    c_model_new[c_model_new >= 1] = 0.9999999

    c_human_new[~agreement] = 1-c_human_new[~agreement]
    a = (c_model_new**gamma)/((c_model_new**gamma)+((1-c_model_new)**gamma))
    b = (c_human_new**gamma)/((c_human_new**gamma)+((1-c_human_new)**gamma))

    conf = 1/(1+(((1-a)*(1-b))/(a*b)))

    util_accept = (1+beta)*w(conf, k)-beta
    util_reject = 1-(1+beta)*w(conf, k)

    prob = np.exp(delta*util_accept) / \
        (np.exp(delta*util_accept)+np.exp(delta*util_reject))
    df = pd.DataFrame({'c_human': c_human, 'c_human_new': c_human_new,
                    'conf': conf, 'c_model': c_model, 'agreement': agreement, 'prob': prob})
    
    prob = np.where(random(len(prob)) > 0.9, 1-prob, prob)


    return prob

def par30ADB(c_human, c_model, agreement, delta=5, beta=0.5, k=0.63, gamma=0.95):
    # from will you accept the AI recommendation
    

    

    def w(p, k):
        return (p**k)/((p**k)+(1-p)**k)

    c_human_new = c_human.copy()
    c_human_new[c_human_new <= 0] = 0.0000001
    c_human_new[c_human_new >= 1] = 0.9999999

    # transform human confidence back to probability of human's estimate of their choice being correct
    c_human_new = (c_human_new/2)+0.5

    c_model_new = c_model.copy()
    c_model_new[c_model_new <= 0] = 0.0000001
    c_model_new[c_model_new >= 1] = 0.9999999

    c_human_new[~agreement] = 1-c_human_new[~agreement]
    a = (c_model_new**gamma)/((c_model_new**gamma)+((1-c_model_new)**gamma))
    b = (c_human_new**gamma)/((c_human_new**gamma)+((1-c_human_new)**gamma))

    conf = 1/(1+(((1-a)*(1-b))/(a*b)))

    util_accept = (1+beta)*w(conf, k)-beta
    util_reject = 1-(1+beta)*w(conf, k)

    prob = np.exp(delta*util_accept) / \
        (np.exp(delta*util_accept)+np.exp(delta*util_reject))
    df = pd.DataFrame({'c_human': c_human, 'c_human_new': c_human_new,
                    'conf': conf, 'c_model': c_model, 'agreement': agreement, 'prob': prob})
    
    prob = np.where(random(len(prob)) > 0.7, 1-prob, prob)


    return prob


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

#x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets('hr', 0)

#with open(f'results/hr/run0/biased.pkl', 'rb') as f:
#    human = pickle.load(f)

def load_results(dataset, setting, run_num, cost, model):
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



def make_results(dataset, whichtype, num_runs, costs, validation=False, which_to_do=['tr', 'tr-no(adb)', 'brs']):

    #create dataframe of empty lists with column headers below


    
    results = pd.DataFrame(data={'tr_team_w_reset_decision_loss': [[]],
                                 'tr2s_team_w_reset_decision_loss': [[]],
                                 'trnoADB_team_w_reset_decision_loss': [[]],
                                'tr_team_wo_reset_decision_loss': [[]],
                                'tr_model_w_reset_decision_loss': [[]],
                                'tr2s_model_w_reset_decision_loss': [[]],
                                'trnoadb_model_w_reset_decision_loss': [[]],
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
                                'hyrs_team_w_reset_decision_loss': [[]],
                                'hyrs_model_w_reset_decision_loss': [[]],
                                'hyrs_team_w_reset_objective': [[]],
                                'hyrs_model_w_reset_objective': [[]],
                                'hyrs_model_w_reset_contradictions': [[]],
                                'tr_model_w_reset_contradictions': [[]],
                                'tr_model_wo_reset_contradictions': [[]],
                                'tr2s_model_w_reset_contradictions': [[]],
                                'trnoadb_model_w_reset_contradictions': [[]],
                                'hyrs_model_contradictions': [[]],
                                'brs_model_contradictions':[[]],
                                'tr_team_w_reset_objective': [[]],
                                'tr_team_wo_reset_objective': [[]],
                                'tr_model_w_reset_objective': [[]],
                                'tr_model_wo_reset_objective': [[]],
                                'tr2s_team_w_reset_objective': [[]],
                                'tr2s_model_w_reset_objective': [[]],
                                'trnoadb_team_w_reset_objective': [[]],
                                'trnoadb_model_w_reset_objective': [[]],
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
    r_adb = []
    r_par30 = []
    r_par = []

    for run in bar(range(num_runs)):
        

        
        bar=progressbar.ProgressBar()
        x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets(dataset, run)

        if validation==True:
            x_test = x_val
            y_test = y_val
            x_test_non_binarized = x_val_non_binarized
        
        
        dataset = dataset
        human, adb_mod, conf_mod = load_humans(dataset, whichtype, run)
        


        for cost in costs:
            print(f'producing for cost {cost} run {run}.....')
            tr_mod = load_results(dataset, f'_{whichtype}', run, cost, 'tr')
            hyrs_mod = load_results(dataset, f'_{whichtype}', run, cost, 'tr')
            trnoadb_mod = load_results(dataset, f'_{whichtype}', run, cost, 'tr-no(ADB)')
            tr2s_mod = load_results(dataset, f'_{whichtype}', run, cost, 'tr')
            brs_mod = load_results(dataset, f'_{whichtype}' , run, 0.0, 'brs')
            #load e_y and e_yb mods
            #with open(f'results/{dataset}/run{run}/cost{float(cost)}/eyb_model_{whichtype}.pkl', 'rb') as f:
            #    e_yb_mod = pickle.load(f)
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/ey_model_{whichtype}.pkl', 'rb') as f:
                e_y_mod = pickle.load(f)

            tr_mod.df = x_train
            tr_mod.Y = y_train
            tr2s_mod.df = x_train
            tr2s_mod.Y = y_train
            trnoadb_mod.df = x_train
            trnoadb_mod.Y = y_train
            hyrs_mod.df = x_train
            hyrs_mod.Y = y_train

            
            tr_team_w_reset_decision_loss = []
            tr_team_wo_reset_decision_loss = []
            tr_model_w_reset_decision_loss = []
            tr_model_wo_reset_decision_loss = []
            tr2s_model_w_reset_decision_loss = []
            tr2s_team_w_reset_decision_loss = []
            trnoadb_model_w_reset_decision_loss = []
            trnoadb_team_w_reset_decision_loss = []
            hyrs_model_decision_loss = []
            hyrs_team_decision_loss = []
            brs_model_decision_loss = []
            brs_model_w_reset_decision_loss = []
            brs_team_w_reset_decision_loss = []
            brs_team_decision_loss = []


            tr_model_w_reset_contradictions = []
            tr_model_wo_reset_contradictions = []
            tr2s_model_w_reset_contradictions = []
            trnoadb_model_w_reset_contradictions = []
            hyrs_model_contradictions = []
            brs_model_contradictions = []
            brs_model_w_reset_contradictions = []

            tr_team_w_reset_objective = []
            tr2s_team_w_reset_objective = []
            trnoadb_team_w_reset_objective = []
            tr_team_wo_reset_objective = []
            tr_model_w_reset_objective = []
            tr2s_model_w_reset_objective = []
            trnoadb_model_w_reset_objective = []
            tr_model_wo_reset_objective = []
            hyrs_model_objective = []
            hyrs_team_objective = []
            brs_model_objective = []
            brs_model_w_reset_objective = []
            brs_team_objective = []
            brs_team_w_reset_objective = []
            hyrs_model_w_reset_decision_loss = []
            hyrs_team_w_reset_decision_loss = []
            hyrs_model_w_reset_contradictions = []
            hyrs_model_w_reset_objective = []
            hyrs_team_w_reset_objective = []


            human_decision_loss = []
            hyrs_norecon_objective = []
            hyrs_norecon_model_decision_loss = []
            hyrs_norecon_team_decision_loss = []
            hyrs_norecon_model_contradictions = []
            


            hyrs_norecon_mod = deepcopy(tr_mod)

            
            brs_mod.df = x_train
            brs_mod.Y = y_train



            #rocs = []
            #for i in range(1000):
            #    rocs.append(roc_auc_score(y_train, e_y_mod.predict_proba(x_train_non_binarized)[:,1]))


            
            for i in range(50):
                
                

                


                human_decisions = human.get_decisions(x_test, y_test)
                human_conf = human.get_confidence(x_test)
                #conf_mod_preds = conf_mod.predict(x_test_non_binarized)

                learned_adb = tr_mod.fA
                tr_mod_confs, agreement = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min) 
                if i == 0:
                    rocs_adb = []
                    rocs_no = []
                    rocs_rand = []
                    rocs_half = []
                    rocs_adv = []
                    rocs_par = []
                    rocs_par30 = []
                    rocs_learned = []
                    rocs_human = []

                    for i in range(1000):
                        p = learned_adb(human_conf, tr_mod_confs, agreement)[agreement==False]
                        p_adb = human.ADB(human_conf, tr_mod_confs, agreement)[agreement==False]
                        p_parADB = parADB(human_conf, tr_mod_confs, agreement)[agreement==False]
                        p_par30ADB = par30ADB(human_conf, tr_mod_confs, agreement)[agreement==False]
                        try:
                            rocs_adb.append(roc_auc_score(bernoulli.rvs(p=p_adb, size=(agreement==False).sum()).astype(bool), p))
                        except: 
                            pass
                        try:   
                            rocs_par.append(roc_auc_score(bernoulli.rvs(p=p_parADB, size=(agreement==False).sum()).astype(bool), p))
                        except:
                            pass
                        try:
                            rocs_par30.append(roc_auc_score(bernoulli.rvs(p=p_par30ADB, size=(agreement==False).sum()).astype(bool), p)) 
                        except:
                            pass

                if 'tr' in which_to_do:
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, parADB, with_reset=True,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                else:
                    tr_model_preds_with_reset = np.ones(len(y_test))
                    tr_team_preds_with_reset = np.ones(len(y_test))
                    tr_model_preds_no_reset = np.ones(len(y_test))
                    tr_team_preds_no_reset = np.ones(len(y_test))

                hyrs_team_preds_w_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, par30ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                hyrs_team_preds = trnoadb_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                hyrs_model_preds_w_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                hyrs_model_preds = trnoadb_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                if 'tr2s' in which_to_do:
                    tr2s_team_preds_with_reset = trnoadb_mod.predictHumanInLoop(x_test, human_decisions, human_conf, par30ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr2s_model_preds_with_reset = trnoadb_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                else:
                    tr2s_team_preds_with_reset = np.ones(len(y_test))
                    tr2s_model_preds_with_reset = np.ones(len(y_test))
                

                

                
                if 'tr-no(adb)' in which_to_do:
                    trnoadb_model_preds_w_reset = trnoadb_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf,  p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    trnoadb_team_preds_w_reset = trnoadb_mod.predictHumanInLoop(x_test, human_decisions, human_conf, parADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                else:
                    trnoadb_model_preds_w_reset = np.ones(len(y_test))
                    trnoadb_team_preds_w_reset = np.ones(len(y_test))
                    hyrs_norecon_model_preds = np.ones(len(y_test))
                    hyrs_norecon_team_preds = np.ones(len(y_test))

                if 'brs' in which_to_do:
                    brs_model_preds = brs_predict(brs_mod.opt_rules, x_test)
                    brs_conf = brs_predict_conf(brs_mod.opt_rules, x_test, brs_mod)
                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, parADB)
                    brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), e_human_responses=human_decisions, conf_model=brs_conf, fA=parADB, asym_loss=[1,1], contradiction_reg=cost)
                    brs_team_preds_w_reset = brs_team_preds.copy()
                    brs_team_preds_w_reset[brs_reset] = human_decisions[brs_reset]
                    brs_model_preds_w_reset = brs_model_preds.copy()
                    brs_model_preds_w_reset[brs_reset] = human_decisions[brs_reset]

                    brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, par30ADB)
                    brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), e_human_responses=human_decisions, conf_model=brs_conf, fA=par30ADB, asym_loss=[1,1], contradiction_reg=cost)
                    brs_team_preds[brs_reset] = human_decisions[brs_reset]
                    brs_model_preds = brs_model_preds.copy()
                    brs_model_preds[brs_reset] = human_decisions[brs_reset]

                    hyrs_norecon_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)
                    brs_reset = brs_expected_loss_filter(brs_mod, x_test, brs_model_preds, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), e_human_responses=human_decisions, conf_model=brs_conf, fA=human.ADB, asym_loss=[1,1], contradiction_reg=cost)
                    hyrs_norecon_team_preds[brs_reset] = human_decisions[brs_reset]
                    hyrs_norecon_model_preds = brs_model_preds.copy()
                    hyrs_norecon_model_preds[brs_reset] = human_decisions[brs_reset]


                else:
                    brs_team_preds = np.ones(len(y_test))
                    brs_model_preds = np.ones(len(y_test))
                    brs_conf = np.ones(len(y_test))
                    brs_reset = np.zeros(len(y_test))
                    brs_team_preds_w_reset = np.ones(len(y_test))
                    brs_model_preds_w_reset = np.ones(len(y_test))





                   

                
                        

                tr_team_w_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_with_reset, y_test))
                tr2s_team_w_reset_decision_loss.append(1 - accuracy_score(tr2s_team_preds_with_reset, y_test))
                trnoadb_team_w_reset_decision_loss.append(1 - accuracy_score(trnoadb_team_preds_w_reset, y_test))
                tr_team_wo_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_no_reset, y_test))
                tr_model_w_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_with_reset, y_test))
                tr2s_model_w_reset_decision_loss.append(1 - accuracy_score(tr2s_model_preds_with_reset, y_test))
                trnoadb_model_w_reset_decision_loss.append(1 - accuracy_score(trnoadb_model_preds_w_reset, y_test))
                tr_model_wo_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_no_reset, y_test))
                hyrs_model_decision_loss.append(1 - accuracy_score(hyrs_model_preds, y_test))
                hyrs_team_decision_loss.append(1 - accuracy_score(hyrs_team_preds, y_test))
                brs_model_decision_loss.append(1 - accuracy_score(brs_model_preds, y_test))
                brs_team_decision_loss.append(1 - accuracy_score(brs_team_preds, y_test))
                brs_model_w_reset_decision_loss.append(1 - accuracy_score(brs_model_preds_w_reset, y_test))
                brs_team_w_reset_decision_loss.append(1 - accuracy_score(brs_team_preds_w_reset, y_test))
                hyrs_model_w_reset_decision_loss.append(1 - accuracy_score(hyrs_model_preds_w_reset, y_test))
                hyrs_team_w_reset_decision_loss.append(1 - accuracy_score(hyrs_team_preds_w_reset, y_test))
                

                tr_model_w_reset_contradictions.append((tr_model_preds_with_reset != human_decisions).sum())
                tr2s_model_w_reset_contradictions.append((tr2s_model_preds_with_reset != human_decisions).sum())
                trnoadb_model_w_reset_contradictions.append((trnoadb_model_preds_w_reset != human_decisions).sum())                                   
                tr_model_wo_reset_contradictions.append((tr_model_preds_no_reset != human_decisions).sum())
                hyrs_model_contradictions.append((hyrs_model_preds != human_decisions).sum())
                brs_model_contradictions.append((brs_model_preds != human_decisions).sum())
                brs_model_w_reset_contradictions.append((brs_model_preds_w_reset != human_decisions).sum())
                hyrs_model_w_reset_contradictions.append((hyrs_model_preds_w_reset != human_decisions).sum())

                tr_team_w_reset_objective.append(tr_team_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr2s_team_w_reset_objective.append(tr2s_team_w_reset_decision_loss[-1] + cost*(tr2s_model_w_reset_contradictions[-1])/len(y_test))
                trnoadb_team_w_reset_objective.append(trnoadb_team_w_reset_decision_loss[-1] + cost*(trnoadb_model_w_reset_contradictions[-1])/len(y_test))
                tr_team_wo_reset_objective.append(tr_team_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                tr_model_w_reset_objective.append(tr_model_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr2s_model_w_reset_objective.append(tr2s_model_w_reset_decision_loss[-1] + cost*(tr2s_model_w_reset_contradictions[-1])/len(y_test))
                trnoadb_model_w_reset_objective.append(trnoadb_model_w_reset_decision_loss[-1] + cost*(trnoadb_model_w_reset_contradictions[-1])/len(y_test))
                tr_model_wo_reset_objective.append(tr_model_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                hyrs_model_objective.append(hyrs_model_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                hyrs_team_objective.append(hyrs_team_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                brs_model_objective.append(brs_model_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_team_objective.append(brs_team_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                brs_team_w_reset_objective.append(brs_team_w_reset_decision_loss[-1] + cost*(brs_model_w_reset_contradictions[-1])/len(y_test))
                brs_model_w_reset_objective.append(brs_model_w_reset_decision_loss[-1] + cost*(brs_model_w_reset_contradictions[-1])/len(y_test))
                hyrs_team_w_reset_objective.append(hyrs_team_w_reset_decision_loss[-1] + cost*(hyrs_model_w_reset_contradictions[-1])/len(y_test))
                hyrs_model_w_reset_objective.append(hyrs_model_w_reset_decision_loss[-1] + cost*(hyrs_model_w_reset_contradictions[-1])/len(y_test))
                                                  

                human_decision_loss.append(1 - accuracy_score(human_decisions, y_test))

                hyrs_norecon_model_decision_loss.append(1 - accuracy_score(hyrs_norecon_model_preds, y_test))
                hyrs_norecon_team_decision_loss.append(1 - accuracy_score(hyrs_norecon_team_preds, y_test))
                hyrs_norecon_model_contradictions.append((hyrs_norecon_model_preds != human_decisions).sum())
                hyrs_norecon_objective.append(hyrs_norecon_team_decision_loss[-1] + cost*(hyrs_norecon_model_contradictions[-1])/len(y_test))

                


                print(i)
            
            

            
            #append values to appropriate row in results
            results.loc[cost, 'tr_team_w_reset_decision_loss'].append(mean(tr_team_w_reset_decision_loss))
            results.loc[cost, 'tr2s_team_w_reset_decision_loss'].append(mean(tr2s_team_w_reset_decision_loss))
            results.loc[cost, 'trnoADB_team_w_reset_decision_loss'].append(mean(trnoadb_team_w_reset_decision_loss))
            results.loc[cost, 'tr_team_wo_reset_decision_loss'].append(mean(tr_team_wo_reset_decision_loss))
            results.loc[cost, 'tr_model_w_reset_decision_loss'].append(mean(tr_model_w_reset_decision_loss))
            results.loc[cost, 'tr2s_model_w_reset_decision_loss'].append(mean(tr2s_model_w_reset_decision_loss))
            results.loc[cost, 'trnoadb_model_w_reset_decision_loss'].append(mean(trnoadb_model_w_reset_decision_loss))
            results.loc[cost, 'tr_model_wo_reset_decision_loss'].append(mean(tr_model_wo_reset_decision_loss))
            results.loc[cost, 'hyrs_model_decision_loss'].append(mean(hyrs_model_decision_loss))
            results.loc[cost, 'hyrs_team_decision_loss'].append(mean(hyrs_team_decision_loss))
            results.loc[cost, 'brs_model_decision_loss'].append(mean(brs_model_decision_loss))
            results.loc[cost, 'brs_team_decision_loss'].append(mean(brs_team_decision_loss))
            results.loc[cost, 'tr_model_w_reset_contradictions'].append(mean(tr_model_w_reset_contradictions))
            results.loc[cost, 'tr2s_model_w_reset_contradictions'].append(mean(tr2s_model_w_reset_contradictions))
            results.loc[cost, 'trnoadb_model_w_reset_contradictions'].append(mean(trnoadb_model_w_reset_contradictions))
            results.loc[cost, 'tr_model_wo_reset_contradictions'].append(mean(tr_model_wo_reset_contradictions))
            results.loc[cost, 'hyrs_model_contradictions'].append(mean(hyrs_model_contradictions))
            results.loc[cost, 'brs_model_contradictions'].append(mean(brs_model_contradictions))
            results.loc[cost, 'tr_team_w_reset_objective'].append(mean(tr_team_w_reset_objective))
            results.loc[cost, 'tr2s_team_w_reset_objective'].append(mean(tr2s_team_w_reset_objective))
            results.loc[cost, 'trnoadb_team_w_reset_objective'].append(mean(trnoadb_team_w_reset_objective))
            results.loc[cost, 'tr_team_wo_reset_objective'].append(mean(tr_team_wo_reset_objective))
            results.loc[cost, 'tr_model_w_reset_objective'].append(mean(tr_model_w_reset_objective))
            results.loc[cost, 'tr2s_model_w_reset_objective'].append(mean(tr2s_model_w_reset_objective))
            results.loc[cost, 'trnoadb_model_w_reset_objective'].append(mean(trnoadb_model_w_reset_objective))
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
            results.loc[cost, 'hyrs_team_w_reset_decision_loss'].append(mean(hyrs_team_w_reset_decision_loss))
            results.loc[cost, 'hyrs_model_w_reset_decision_loss'].append(mean(hyrs_model_w_reset_decision_loss))
            results.loc[cost, 'hyrs_team_w_reset_objective'].append(mean(hyrs_team_w_reset_objective))
            results.loc[cost, 'hyrs_model_w_reset_objective'].append(mean(hyrs_model_w_reset_objective))
            results.loc[cost, 'hyrs_model_w_reset_contradictions'].append(mean(hyrs_model_w_reset_contradictions))
            if validation == False:
                r_par = mean(rocs_par)
                r_par30 = mean(rocs_par30)
                r_adb = mean(rocs_adb)



            
            
    
    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))
    if validation == False:
        roc_dict = pd.DataFrame({'r_par': r_par, 'r_par30': r_par30, 'r_adb': r_adb}, index=[f'{dataset}_{whichtype}'])
        roc_dict.to_pickle(f'results/{dataset}/roc_dict_{whichtype}.pkl')

    




    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results





def make_TL_v_cost_plot(results_means, results_stderrs, name):
    fig = plt.figure(figsize=(3, 2), dpi=200)
    color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42', 'BRSselect': '#FF7F00'}  #75c361
    #plt.plot(results_means.index[0:10], results_means['hyrs_norecon_objective'].iloc[0:10], marker = 'v', c=color_dict['HYRS'], label = 'TR-No(ADB, OrgVal)', markersize=1.8, linewidth=0.9)
    #plt.plot(results_means.index[0:10], results_means['hyrs_team_objective'].iloc[0:10], marker = 'x', c=color_dict['HYRSRecon'], label = 'TR-No(ADB)', markersize=1.8, linewidth=0.9)
    #plt.plot(results_means.index[0:10], results_means['tr_team_w_reset_objective'].iloc[0:10], marker = '.', c=color_dict['TR'], label='TR', markersize=1.8, linewidth=0.9)
    plt.plot(results_means.index[0:10], results_means['brs_team_objective'].iloc[0:10], marker = 's', c=color_dict['BRS'], label='Task-Only (Current Practice)', markersize=1.8, linewidth=0.9)
   #plt.plot(results_means.index[0:10], results_means['brs_team_w_reset_objective'].iloc[0:10], marker = 'v', c=color_dict['BRSselect'], label='TR-SelectiveOnly', markersize=1.8, linewidth=0.9)
    
    plt.plot(results_means.index[0:10], results_means['human_decision_loss'].iloc[0:10], c = color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    
    plt.fill_between(results_means.index[0:10], 
                results_means['human_decision_loss'].iloc[0:10]-(results_stderrs['human_decision_loss'].iloc[0:10]),
                results_means['human_decision_loss'].iloc[0:10]+(results_stderrs['human_decision_loss'].iloc[0:10]),
                color=color_dict['Human'], alpha=0.2)
    '''
    plt.fill_between(results_means.index[0:10], 
                results_means['hyrs_team_objective'].iloc[0:10]-(results_stderrs['hyrs_team_objective'].iloc[0:10]),
                results_means['hyrs_team_objective'].iloc[0:10]+(results_stderrs['hyrs_team_objective'].iloc[0:10]) ,
                color=color_dict['HYRSRecon'], alpha=0.2)'''
    '''
    plt.fill_between(results_means.index[0:10], 
                results_means['hyrs_norecon_objective'].iloc[0:10]-(results_stderrs['hyrs_norecon_objective'].iloc[0:10]),
                results_means['hyrs_norecon_objective'].iloc[0:10]+(results_stderrs['hyrs_norecon_objective'].iloc[0:10]) ,
                color=color_dict['HYRS'], alpha=0.2)
                '''
    plt.fill_between(results_means.index[0:10], 
                results_means['brs_team_objective'].iloc[0:10]-(results_stderrs['brs_team_objective'].iloc[0:10]),
                results_means['brs_team_objective'].iloc[0:10]+(results_stderrs['brs_team_objective'].iloc[0:10]) ,
                color=color_dict['BRS'], alpha=0.2)
    '''
    plt.fill_between(results_means.index[0:10], 
                results_means['tr_team_w_reset_objective'].iloc[0:10]-(results_stderrs['tr_team_w_reset_objective'].iloc[0:10]),
                results_means['tr_team_w_reset_objective'].iloc[0:10]+(results_stderrs['tr_team_w_reset_objective'].iloc[0:10]),
                color=color_dict['TR'], alpha=0.2)'''
    '''
    plt.fill_between(results_means.iloc[0:10].index,
                     results_means['brs_team_w_reset_objective'].iloc[0:10]-(results_stderrs['brs_team_w_reset_objective'].iloc[0:10]),
                results_means['brs_team_w_reset_objective'].iloc[0:10]+(results_stderrs['brs_team_w_reset_objective'].iloc[0:10]),
                color=color_dict['BRSselect'], alpha=0.2)'''
   
    plt.xlabel('Reconciliation Cost', fontsize=12)
    plt.ylabel('Total Team Loss', fontsize=12)
    plt.tick_params(labelrotation=45, labelsize=10)
    #plt.title('{} Setting'.format(setting), fontsize=15)
    #plt.title('Income Prediction (Adult Dataset)', fontsize=15)
    plt.legend(prop={'size': 5})
    plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

    fig.savefig(f'results/{dataset}/plots/TL_{dataset}_{name}_discretion.png', bbox_inches='tight')
    #plt.show()

    #plt.clf()




costs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

num_runs = 10
datasets = ['heart_disease']
names = ['biased_dec_bias']
which_to_do = ['tr', 'tr2s', 'tr-no(adb)', 'brs']

for dataset in datasets:
    for name in names:
        #if (name == 'biased') and (datasets == 'heart_disease'):
        #    continue

        if 'tr' not in which_to_do or 'tr2s' not in which_to_do or 'tr-no(adb)' not in which_to_do or 'brs' not in which_to_do:
            name = f'{name}_discretion_{which_to_do}'
        if os.path.isfile(f'results/{dataset}/{name}_discretion_rs.pkl') and False:
            with open(f'results/{dataset}/{name}_discretion_rs.pkl', 'rb') as f:
                rs = pickle.load(f)
            with open(f'results/{dataset}/{name}_discretion_means.pkl', 'rb') as f:
                means = pickle.load(f)
            with open(f'results/{dataset}/{name}_discretion_std.pkl', 'rb') as f:
                std = pickle.load(f)

            with open(f'results/{dataset}/val_{name}_discretion_rs.pkl', 'rb') as f:
                val_rs = pickle.load(f)
            with open(f'results/{dataset}/val_{name}_discretion_means.pkl', 'rb') as f:
                val_means = pickle.load(f)
            with open(f'results/{dataset}/val_{name}_discretion_std.pkl', 'rb') as f:
                val_std = pickle.load(f)


            #cval_means, cval_stderss, cval_rs, cost_tracker, final_cost = cost_validation(rs, val_rs)
            #cval_means, cval_stderss, cval_rs = cost_plus(rs)
            #cval_val_means, cval_val_stderrs, cval_val_rs = cost_plus(val_rs)
            #rval_means, rval_stderss, rval_rs, rules_tracker = robust_rules(rs, val_rs)
            #ccval_means, ccval_stderss, ccval_rs, _, _ = cost_validation(val_rs, val_rs) 
            #rcval_means, rcval_stderss, rcval_rs, both_tracker = robust_rules(cval_rs, ccval_rs)   
            #make_multi_TL_v_cost_plot(rval_means, rval_stderss, name, axs[datarow, behaviorrow])
            #make_multi_TL_v_cost_plot(means, std, name, axs[datarow, behaviorrow])
            


        else:
            means, std, rs = make_results(dataset, name, num_runs, costs, validation=False, which_to_do=which_to_do)
            #pickle and write means, std, and rs to file
            with open(f'results/{dataset}/{name}_discretion_means.pkl', 'wb') as f:
                pickle.dump(means, f)
            with open(f'results/{dataset}/{name}_discretion_std.pkl', 'wb') as f:
                pickle.dump(std, f)
            with open(f'results/{dataset}/{name}_discretion_rs.pkl', 'wb') as f:
                pickle.dump(rs, f)
        
            print(f'running for val {dataset} {name}_discretion')
            val_means, val_std, val_rs = make_results(dataset, name, num_runs, costs, validation=True, which_to_do=which_to_do)
            #pickle and write means, std, and rs to file
            with open(f'results/{dataset}/val_{name}_discretion_means.pkl', 'wb') as f:
                pickle.dump(val_means, f)
            with open(f'results/{dataset}/val_{name}_discretion_std.pkl', 'wb') as f:
                pickle.dump(val_std, f)
            with open(f'results/{dataset}/val_{name}_discretion_rs.pkl', 'wb') as f:
                pickle.dump(val_rs, f)
    







print('pause')

