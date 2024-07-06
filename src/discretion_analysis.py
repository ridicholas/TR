import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tr import *
from hyrs import *
from brs import *
import pickle
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error
from util_BOA import *
from numpy import mean 
import progressbar
from run import ADB
from run import evaluate_adb_model
from copy import deepcopy
import os

#os.chdir("..")

def load_datasets(dataset, run_num):
    x_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtrain.csv', index_col=0)
    y_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/ytrain.csv', index_col=0).iloc[:, 0]
    x_train_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtrain_non_binarized.csv', index_col=0)
    x_learning_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{0}/xlearning_non_binarized.csv', index_col=0)
    x_learning = pd.read_csv(f'datasets/{dataset}/processed/run{0}/xlearning.csv', index_col=0)
    y_learning = pd.read_csv(f'datasets/{dataset}/processed/run{0}/ylearning.csv', index_col=0).iloc[:, 0]
    x_human_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xhumantrain.csv', index_col=0)
    y_human_train = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/yhumantrain.csv', index_col=0).iloc[:, 0]

    x_val = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xval.csv', index_col=0)
    y_val = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/yval.csv', index_col=0).iloc[:, 0]
    x_test = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtest.csv', index_col=0)
    y_test = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/ytest.csv', index_col=0).iloc[:, 0]
    x_val_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xval_non_binarized.csv', index_col=0)
    x_test_non_binarized = pd.read_csv(f'datasets/{dataset}/processed/run{run_num}/xtest_non_binarized.csv', index_col=0)

    return x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized

def load_discretion_results(dataset, setting, run_num, cost, model, size):
    if model == 'brs':
        setting = ''
    else:
        setting = '_' + setting + f'_discretion{size}'
    with open(f'results/{dataset}/run{run_num}/cost{float(cost)}/{model}_model{setting}.pkl', 'rb') as f:
        result = pickle.load(f)
        return result
    
def load_discretion_humans(dataset, setting, run_num, size):
    setting = setting + f'_discretion{size}'
    with open(f'results/{dataset}/run{0}/{setting}.pkl', 'rb') as f:
        human = pickle.load(f)
    with open(f'results/{dataset}/run{0}/adb_model_{setting}.pkl', 'rb') as f:
        adb_model = pickle.load(f)
    with open(f'results/{dataset}/run{0}/conf_model_{setting}.pkl', 'rb') as f:
        conf_model = pickle.load(f)
    return human, adb_model, conf_model

def make_discretion_results(dataset, whichtype, num_runs, costs, validation=False, size='True'):

    #create dataframe of empty lists with column headers below


    
    results = pd.DataFrame(data={'tr_team_w_reset_decision_loss': [[]],
                                'tr_team_wo_reset_decision_loss': [[]],
                                'tr_model_w_reset_decision_loss': [[]],
                                'tr_model_wo_reset_decision_loss': [[]],
                                'hyrs_model_decision_loss': [[]],
                                'hyrs_team_decision_loss': [[]],
                                'hyrs_model_w_reset_decision_loss': [[]],
                                'hyrs_team_w_reset_decision_loss': [[]],
                                'brs_model_decision_loss': [[]],
                                'brs_team_decision_loss': [[]],
                                'tr_model_w_reset_contradictions': [[]],
                                'tr_model_wo_reset_contradictions': [[]],
                                'hyrs_model_contradictions': [[]],
                                'hyrs_model_w_reset_contradictions': [[]],
                                'brs_model_contradictions':[[]],
                                'tr_team_w_reset_objective': [[]],
                                'tr_team_wo_reset_objective': [[]],
                                'tr_model_w_reset_objective': [[]],
                                'tr_model_wo_reset_objective': [[]],
                                'hyrs_model_objective': [[]],
                                'hyrs_team_objective':[[]],
                                'hyrs_model_w_reset_objective': [[]],
                                'hyrs_team_w_reset_objective': [[]],
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
        
        x_test = x_test[~x_test.index.isin(x_learning.index)].reset_index(drop=True)
        y_test = y_test[~y_test.index.isin(y_learning.index)].reset_index(drop=True)
        x_test_non_binarized = x_test_non_binarized[~x_test_non_binarized.index.isin(x_learning_non_binarized.index)].reset_index(drop=True)

        dataset = dataset
        human, adb_mod, conf_mod = load_discretion_humans(dataset, whichtype, run, size)


        brs_mod = load_discretion_results(dataset, whichtype , run, 0.0, 'brs', size)


        for cost in costs:
            print(f'producing for cost {cost} run {run}.....')
            tr_mod = load_discretion_results(dataset, whichtype, run, cost, 'tr', size)
            hyrs_mod = load_discretion_results(dataset, whichtype, run, cost, 'hyrs', size)
            #load e_y and e_yb mods
            #with open(f'results/{dataset}/run{run}/cost{float(cost)}/eyb_model_{whichtype}_discretion{size}.pkl', 'rb') as f:
            #    e_yb_mod = pickle.load(f)
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/ey_model_{whichtype}_discretion{size}.pkl', 'rb') as f:
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
            hyrs_model_w_reset_decision_loss = []
            hyrs_team_w_reset_decision_loss = []
            brs_model_decision_loss = []
            brs_team_decision_loss = []


            tr_model_w_reset_contradictions = []
            tr_model_wo_reset_contradictions = []
            hyrs_model_contradictions = []
            hyrs_model_w_reset_contradictions = []
            brs_model_contradictions = []

            tr_team_w_reset_objective = []
            tr_team_wo_reset_objective = []
            tr_model_w_reset_objective = []
            tr_model_wo_reset_objective = []
            hyrs_model_objective = []
            hyrs_team_objective = []
            hyrs_model_w_reset_objective = []
            hyrs_team_w_reset_objective = []
            brs_model_objective = []
            brs_team_objective = []

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
            auc_scores = []
            mae_scores = []
            for i in range(50):
                
                

                if validation: 

                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    

                    learned_adb = ADB(adb_mod)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, learned_adb.ADB_model_wrapper, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_reset = hyrs_mod.expected_loss_filter(x_test, hyrs_model_preds, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), e_human_responses=human_decisions, conf_model=None, fA=learned_adb.ADB_model_wrapper, asym_loss=[1,1], contradiction_reg=cost)
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, learned_adb.ADB_model_wrapper, x_test)
                    hyrs_team_preds_w_reset = hyrs_team_preds.copy()
                    hyrs_team_preds_w_reset[hyrs_reset] = human_decisions[hyrs_reset]
                    hyrs_model_preds_w_reset = hyrs_model_preds.copy()
                    hyrs_model_preds_w_reset[hyrs_reset] = human_decisions[hyrs_reset]
                    #brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, learned_adb.ADB_model_wrapper)

                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, learned_adb.ADB_model_wrapper, x_test)
                    #brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, learned_adb.ADB_model_wrapper)

                    c_model = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)[0]



                        
                else:
                    learned_adb = ADB(adb_mod)
                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions,human_conf, human.ADB, with_reset=False, p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset, tr_mod_covered_w_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr_model_preds_no_reset, tr_mod_covered_no_reset, _ = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized))
                    tr_mod_confs, tr_agreement = tr_mod.get_model_conf_agreement(x_test, human_decisions, prs_min=tr_mod.prs_min, nrs_min=tr_mod.nrs_min)

                    human_adb = human.ADB(human_conf, tr_mod_confs, tr_agreement)
                    if cost==0.0:
                        mae_scores.append(mean_absolute_error(human_adb[~tr_agreement], learned_adb.ADB_model_wrapper(human_conf, tr_mod_confs, tr_agreement)[~tr_agreement]))
                    
                    
                        realized_accepts = np.random.binomial(1, human_adb)
                        auc_scores.append(roc_auc_score(realized_accepts[~tr_agreement], learned_adb.ADB_model_wrapper(human_conf, tr_mod_confs, tr_agreement)[~tr_agreement]))
                    

                

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_reset = hyrs_mod.expected_loss_filter(x_test, hyrs_model_preds, conf_human=human_conf, p_y=e_y_mod.predict_proba(x_test_non_binarized), e_human_responses=human_decisions, conf_model=None, fA= human.ADB, asym_loss=[1,1], contradiction_reg=cost)
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    hyrs_team_preds_w_reset = hyrs_team_preds.copy()
                    hyrs_team_preds_w_reset[hyrs_reset] = human_decisions[hyrs_reset]
                    hyrs_model_preds_w_reset = hyrs_model_preds.copy()
                    hyrs_model_preds_w_reset[hyrs_reset] = human_decisions[hyrs_reset]
                    #brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)

                    hyrs_norecon_model_preds = hyrs_norecon_mod.predict(x_test, human_decisions)[0]
                    hyrs_norecon_team_preds = hyrs_norecon_mod.humanifyPreds(hyrs_norecon_model_preds, human_decisions, human_conf, human.ADB, x_test)
                    #brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)
                        

                tr_team_w_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_with_reset, y_test))
                tr_team_wo_reset_decision_loss.append(1 - accuracy_score(tr_team_preds_no_reset, y_test))
                tr_model_w_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_with_reset, y_test))
                tr_model_wo_reset_decision_loss.append(1 - accuracy_score(tr_model_preds_no_reset, y_test))
                hyrs_model_decision_loss.append(1 - accuracy_score(hyrs_model_preds, y_test))
                hyrs_team_decision_loss.append(1 - accuracy_score(hyrs_team_preds, y_test))
                hyrs_model_w_reset_decision_loss.append(1 - accuracy_score(hyrs_model_preds_w_reset, y_test))
                hyrs_team_w_reset_decision_loss.append(1 - accuracy_score(hyrs_team_preds_w_reset, y_test))

                #brs_model_decision_loss.append(1 - accuracy_score(brs_model_preds, y_test))
                #brs_team_decision_loss.append(1 - accuracy_score(brs_team_preds, y_test))

                tr_model_w_reset_contradictions.append((tr_model_preds_with_reset != human_decisions).sum())
                tr_model_wo_reset_contradictions.append((tr_model_preds_no_reset != human_decisions).sum())
                hyrs_model_contradictions.append((hyrs_model_preds != human_decisions).sum())
                hyrs_model_w_reset_contradictions.append((hyrs_model_preds_w_reset != human_decisions).sum())
                #brs_model_contradictions.append((brs_model_preds != human_decisions).sum())

                tr_team_w_reset_objective.append(tr_team_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_team_wo_reset_objective.append(tr_team_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                tr_model_w_reset_objective.append(tr_model_w_reset_decision_loss[-1] + cost*(tr_model_w_reset_contradictions[-1])/len(y_test))
                tr_model_wo_reset_objective.append(tr_model_wo_reset_decision_loss[-1] + cost*(tr_model_wo_reset_contradictions[-1])/len(y_test))
                hyrs_model_objective.append(hyrs_model_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                hyrs_team_objective.append(hyrs_team_decision_loss[-1] + cost*(hyrs_model_contradictions[-1])/len(y_test))
                hyrs_model_w_reset_objective.append(hyrs_model_w_reset_decision_loss[-1] + cost*(hyrs_model_w_reset_contradictions[-1])/len(y_test))
                hyrs_team_w_reset_objective.append(hyrs_team_w_reset_decision_loss[-1] + cost*(hyrs_model_w_reset_contradictions[-1])/len(y_test))
                #brs_model_objective.append(brs_model_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))
                #brs_team_objective.append(brs_team_decision_loss[-1] + cost*(brs_model_contradictions[-1])/len(y_test))

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
            results.loc[cost, 'hyrs_model_w_reset_decision_loss'].append(mean(hyrs_model_w_reset_decision_loss))
            results.loc[cost, 'hyrs_team_w_reset_decision_loss'].append(mean(hyrs_team_w_reset_decision_loss))
            #results.loc[cost, 'brs_model_decision_loss'].append(mean(brs_model_decision_loss))
            #results.loc[cost, 'brs_team_decision_loss'].append(mean(brs_team_decision_loss))
            results.loc[cost, 'tr_model_w_reset_contradictions'].append(mean(tr_model_w_reset_contradictions))
            results.loc[cost, 'tr_model_wo_reset_contradictions'].append(mean(tr_model_wo_reset_contradictions))
            results.loc[cost, 'hyrs_model_contradictions'].append(mean(hyrs_model_contradictions))
            results.loc[cost, 'hyrs_model_w_reset_contradictions'].append(mean(hyrs_model_w_reset_contradictions))
            #results.loc[cost, 'brs_model_contradictions'].append(mean(brs_model_contradictions))
            results.loc[cost, 'tr_team_w_reset_objective'].append(mean(tr_team_w_reset_objective))
            results.loc[cost, 'tr_team_wo_reset_objective'].append(mean(tr_team_wo_reset_objective))
            results.loc[cost, 'tr_model_w_reset_objective'].append(mean(tr_model_w_reset_objective))
            results.loc[cost, 'tr_model_wo_reset_objective'].append(mean(tr_model_wo_reset_objective))
            results.loc[cost, 'hyrs_model_objective'].append(mean(hyrs_model_objective))
            results.loc[cost, 'hyrs_team_objective'].append(mean(hyrs_team_objective))
            results.loc[cost, 'hyrs_model_w_reset_objective'].append(mean(hyrs_model_w_reset_objective))
            results.loc[cost, 'hyrs_team_w_reset_objective'].append(mean(hyrs_team_w_reset_objective))
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



costs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ADB_sizes = ['True', '1', '02']
num_runs = 10
dataset = 'heart_disease'
name = 'biased'

#bia_meansTrue, bia_stdTrue, bia_rsTrue = make_discretion_results(dataset, name, num_runs, costs, validation=False, size='True')
#val_bia_meansTrue, val_bia_stdTrue, val_bia_rsTrue = make_discretion_results(dataset, name, num_runs, costs, validation=True, size='True')

#bia_means1, bia_std1, bia_rs1 = make_discretion_results(dataset, name, num_runs, costs, validation=False, size='1')
#val_bia_means1, val_bia_std1, val_bia_rs1 = make_discretion_results(dataset, name, num_runs, costs, validation=True, size='1')

#bia_means02, bia_std02, bia_rs02 = make_discretion_results(dataset, name, num_runs, costs, validation=False, size='02')
#val_bia_means02, val_bia_std02, val_bia_rs02 = make_discretion_results(dataset, name, num_runs, costs, validation=True, size='02')

sizes = ['True', '1', '02']


#########TRUE_RESULTS###############

if os.path.isfile(f'results/{dataset}/biased_rs_discretionTrue.pkl'):
    with open(f'results/{dataset}/biased_rs_discretionTrue.pkl', 'rb') as f:
        bia_rsTrue = pickle.load(f)
    with open(f'results/{dataset}/biased_means_discretionTrue.pkl', 'rb') as f:
        bia_meansTrue = pickle.load(f)
    with open(f'results/{dataset}/biased_std_discretionTrue.pkl', 'rb') as f:
        bia_stdTrue = pickle.load(f)
else:
    bia_meansTrue, bia_stdTrue, bia_rsTrue = make_discretion_results(dataset, name, num_runs, costs, validation=False, size='True')
    with open(f'results/{dataset}/biased_means_discretionTrue.pkl', 'wb') as f:
        pickle.dump(bia_meansTrue, f)
    with open(f'results/{dataset}/biased_std_discretionTrue.pkl', 'wb') as f:
        pickle.dump(bia_stdTrue, f)
    with open(f'results/{dataset}/biased_rs_discretionTrue.pkl', 'wb') as f:
        pickle.dump(bia_rsTrue, f)


if os.path.isfile(f'results/{dataset}/val_biased_rsTrue.pkl'):
    with open(f'results/{dataset}/val_biased_rsTrue.pkl', 'rb') as f:
        val_bia_rsTrue = pickle.load(f)
    with open(f'results/{dataset}/val_biased_meansTrue.pkl', 'rb') as f:
        val_bia_meansTrue = pickle.load(f)
    with open(f'results/{dataset}/val_biased_stdTrue.pkl', 'rb') as f:
        val_bia_stdTrue = pickle.load(f)
else:
    val_bia_meansTrue, val_bia_stdTrue, val_bia_rsTrue = make_discretion_results(dataset, name, num_runs, costs, validation=True, size='True')
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/val_biased_meansTrue.pkl', 'wb') as f:
        pickle.dump(val_bia_meansTrue, f)
    with open(f'results/{dataset}/val_biased_stdTrue.pkl', 'wb') as f:
        pickle.dump(val_bia_stdTrue, f)
    with open(f'results/{dataset}/val_biased_rsTrue.pkl', 'wb') as f:
        pickle.dump(val_bia_rsTrue, f)

        #########1_RESULTS###############
if os.path.isfile(f'results/{dataset}/biased_rs_discretion1.pkl'):
    with open(f'results/{dataset}/biased_rs_discretion1.pkl', 'rb') as f:
        bia_rs1 = pickle.load(f)
    with open(f'results/{dataset}/biased_means_discretion1.pkl', 'rb') as f:
        bia_means1 = pickle.load(f)
    with open(f'results/{dataset}/biased_std_discretion1.pkl', 'rb') as f:
        bia_std1 = pickle.load(f)
else:
    bia_means1, bia_std1, bia_rs1 = make_discretion_results(dataset, name, num_runs, costs, validation=False, size='1')
    with open(f'results/{dataset}/biased_means_discretion1.pkl', 'wb') as f:
        pickle.dump(bia_means1, f)
    with open(f'results/{dataset}/biased_std_discretion1.pkl', 'wb') as f:
        pickle.dump(bia_std1, f)
    with open(f'results/{dataset}/biased_rs_discretion1.pkl', 'wb') as f:
        pickle.dump(bia_rs1, f)


if os.path.isfile(f'results/{dataset}/val_biased_rs1.pkl'):
    with open(f'results/{dataset}/val_biased_rs1.pkl', 'rb') as f:
        val_bia_rs1 = pickle.load(f)
    with open(f'results/{dataset}/val_biased_means1.pkl', 'rb') as f:
        val_bia_means1 = pickle.load(f)
    with open(f'results/{dataset}/val_biased_std1.pkl', 'rb') as f:
        val_bia_std1 = pickle.load(f)
else:
    val_bia_means1, val_bia_std1, val_bia_rs1 = make_discretion_results(dataset, name, num_runs, costs, validation=True, size='1')
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/val_biased_means1.pkl', 'wb') as f:
        pickle.dump(val_bia_means1, f)
    with open(f'results/{dataset}/val_biased_std1.pkl', 'wb') as f:
        pickle.dump(val_bia_std1, f)
    with open(f'results/{dataset}/val_biased_rs1.pkl', 'wb') as f:
        pickle.dump(val_bia_rs1, f)


#########02_RESULTS###############
if os.path.isfile(f'results/{dataset}/biased_rs_discretion02.pkl'):
    with open(f'results/{dataset}/biased_rs_discretion02.pkl', 'rb') as f:
        bia_rs02 = pickle.load(f)
    with open(f'results/{dataset}/biased_means_discretion02.pkl', 'rb') as f:
        bia_means02 = pickle.load(f)
    with open(f'results/{dataset}/biased_std_discretion02.pkl', 'rb') as f:
        bia_std02 = pickle.load(f)
else:
    bia_means02, bia_std02, bia_rs02 = make_discretion_results(dataset, name, num_runs, costs, validation=False, size='02')
    with open(f'results/{dataset}/biased_means_discretion02.pkl', 'wb') as f:
        pickle.dump(bia_means02, f)
    with open(f'results/{dataset}/biased_std_discretion02.pkl', 'wb') as f:
        pickle.dump(bia_std02, f)
    with open(f'results/{dataset}/biased_rs_discretion02.pkl', 'wb') as f:
        pickle.dump(bia_rs02, f)


if os.path.isfile(f'results/{dataset}/val_biased_rs02.pkl'):
    with open(f'results/{dataset}/val_biased_rs02.pkl', 'rb') as f:
        val_bia_rs02 = pickle.load(f)
    with open(f'results/{dataset}/val_biased_means02.pkl', 'rb') as f:
        val_bia_means02 = pickle.load(f)
    with open(f'results/{dataset}/val_biased_std02.pkl', 'rb') as f:
        val_bia_std02 = pickle.load(f)
else:
    val_bia_means02, val_bia_std02, val_bia_rs02 = make_discretion_results(dataset, name, num_runs, costs, validation=True, size='02')
    #pickle and write means, std, and rs to file
    with open(f'results/{dataset}/val_biased_means02.pkl', 'wb') as f:
        pickle.dump(val_bia_means02, f)
    with open(f'results/{dataset}/val_biased_std02.pkl', 'wb') as f:
        pickle.dump(val_bia_std02, f)
    with open(f'results/{dataset}/val_biased_rs02.pkl', 'wb') as f:
        pickle.dump(val_bia_rs02, f)





def make_TL_v_cost_plot(results_means, results_stderrs, name, sizes = ['True', '1', '02']):
    fig = plt.figure(figsize=(3, 2), dpi=400)
    color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42'}
    
    #plt.plot(results_means[0].index[0:6], results_means[0]['hyrs_norecon_objective'].iloc[0:6], marker = 'v', c=color_dict['HYRS'], label = 'TR-No(ADB, OrgVal)', markersize=1.8, linewidth=0.9)
    plt.plot(results_means[0].index[0:6], results_means[0]['hyrs_team_objective'].iloc[0:6], marker = 'x', c=color_dict['HYRSRecon'], label = 'TR-No(ADB)', markersize=1.8, linewidth=0.9)
    #plt.plot(results_means[0].index[0:6], results_means[0]['brs_team_objective'].iloc[0:6], marker = 's', c=color_dict['BRS'], label='Task-Only (Current Practice)', markersize=1.8, linewidth=0.9)
    plt.plot(results_means[0].index[0:6], results_means[0]['human_decision_loss'].iloc[0:6], c = color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    
    plt.fill_between(results_means[0].index[0:6], 
                results_means[0]['human_decision_loss'].iloc[0:6]-(results_stderrs[0]['human_decision_loss'].iloc[0:6]),
                results_means[0]['human_decision_loss'].iloc[0:6]+(results_stderrs[0]['human_decision_loss'].iloc[0:6]) ,
                color=color_dict['Human'], alpha=0.2)
    plt.fill_between(results_means[0].index[0:6], 
                results_means[0]['hyrs_team_objective'].iloc[0:6]-(results_stderrs[0]['hyrs_team_objective'].iloc[0:6]),
                results_means[0]['hyrs_team_objective'].iloc[0:6]+(results_stderrs[0]['hyrs_team_objective'].iloc[0:6]) ,
                color=color_dict['HYRSRecon'], alpha=0.2)
    
    #plt.fill_between(results_means[0].index[0:6], 
    #            results_means[0]['hyrs_norecon_objective'].iloc[0:6]-(results_stderrs[0]['hyrs_norecon_objective'].iloc[0:6]),
    #            results_means[0]['hyrs_norecon_objective'].iloc[0:6]+(results_stderrs[0]['hyrs_norecon_objective'].iloc[0:6]) ,
    #            color=color_dict['HYRS'], alpha=0.2)
    #plt.fill_between(results_means[0].index[0:6], 
    #            results_means[0]['brs_team_objective'].iloc[0:6]-(results_stderrs[0]['brs_team_objective'].iloc[0:6]),
    #            results_means[0]['brs_team_objective'].iloc[0:6]+(results_stderrs[0]['brs_team_objective'].iloc[0:6]) ,
    #            color=color_dict['BRS'], alpha=0.2)

   
    sizes_marker_dict = {'True': 'o', '1': 'x', '02': 's'}
    sizes_lines_dict = {'True': '-', '1': '--', '02': '-.'}
    auc_dict = {'True': 0.931, '1': 0.868, '02': 0.882}
    mae_dict = {'True': 0.0, '1': 0.12, '02': 0.085}

    sizes = ['True','1', '02']
    for i in range(len(sizes)):
        c=color_dict['TR']
        plt.plot(results_means[i].index[0:6], results_means[i]['tr_team_w_reset_objective'].iloc[0:6], marker = sizes_marker_dict[sizes[i]], label=f'TR, AUC: {auc_dict[sizes[i]]}, MAE: {mae_dict[sizes[i]]}', markersize=1.8, linewidth=0.9, linestyle=sizes_lines_dict[sizes[i]])
        plt.fill_between(results_means[i].index[0:6], 
                    results_means[i]['tr_team_w_reset_objective'].iloc[0:6]-(results_stderrs[i]['tr_team_w_reset_objective'].iloc[0:6]),
                    results_means[i]['tr_team_w_reset_objective'].iloc[0:6]+(results_stderrs[i]['tr_team_w_reset_objective'].iloc[0:6]), alpha=0.2)
   
    plt.xlabel('Reconciliation Cost', fontsize=12)
    plt.ylabel('Total Team Loss', fontsize=12)
    plt.tick_params(labelrotation=45, labelsize=10)
    #plt.title('{} Setting'.format(setting), fontsize=15)
    plt.legend(prop={'size': 5})
    plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

    fig.savefig(f'results/{dataset}/plots/discretion_TL_{dataset}_{name}.png', bbox_inches='tight')
    #plt.show()

    #plt.clf()






def robust_rules(rs, val_rs):
    new_rs = deepcopy(rs)
    for cost in rs.index:
        for column in rs.columns:
            new_rs.loc[cost, column] = deepcopy(rs.loc[cost, column])
        for i in range(len(val_rs['tr_team_w_reset_objective'][cost])):
            x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized= load_datasets(dataset, i)
            curr_val_objective = val_rs['tr_team_w_reset_objective'][cost][i]
            if val_rs['hyrs_team_objective'][cost][i] < curr_val_objective:
                new_rs['tr_model_w_reset_contradictions'][cost][i] = rs['hyrs_model_contradictions'][cost][i].copy()
                new_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['hyrs_team_decision_loss'][cost][i].copy()
                new_rs['tr_team_w_reset_objective'][cost][i] = new_rs['tr_team_w_reset_decision_loss'][cost][i] + 0*new_rs['tr_model_w_reset_contradictions'][cost][i]/len(y_test)
                print(f"cost: {cost}, i: {i}, replacing actual of {rs['tr_team_w_reset_objective'][cost][i]} with new of {new_rs['tr_team_w_reset_objective'][cost][i]}")
                curr_val_objective = val_rs['hyrs_team_objective'][cost][i]
            if val_rs['brs_team_objective'][cost][i] < curr_val_objective:
                new_rs['tr_model_w_reset_contradictions'][cost][i] = rs['brs_model_contradictions'][cost][i].copy()
                new_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['brs_team_decision_loss'][cost][i].copy()
                new_rs['tr_team_w_reset_objective'][cost][i] = new_rs['tr_team_w_reset_decision_loss'][cost][i] + 0*new_rs['tr_model_w_reset_contradictions'][cost][i]/len(y_test)
                print(f"cost: {cost}, i: {i}, replacing actual of {rs['tr_team_w_reset_objective'][cost][i]} with new of {new_rs['tr_team_w_reset_objective'][cost][i]}")
                curr_val_objective = val_rs['brs_team_objective'][cost][i]
                
    new_results_means = new_rs.apply(lambda x: x.apply(lambda y: mean(y)))
    new_results_stderrs = new_rs.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))
    return new_results_means, new_results_stderrs, new_rs

    
def cost_validation(rs, val_rs):
    new_rs = deepcopy(rs)
    for cost in rs.index:
        if cost==1:
            print('pause')
        for column in rs.columns:
            new_rs.loc[cost, column] = deepcopy(rs.loc[cost, column])
        for i in range(len(val_rs['tr_team_w_reset_objective'][cost])):
            x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized= load_datasets(dataset, i)
            y_val = y_val[~y_val.index.isin(y_learning.index)]
            curr_val_objective = val_rs['tr_team_w_reset_objective'][cost][i]
            for alt_cost in rs.index:
                if alt_cost == cost:
                    continue
                alt_val_objective = val_rs['tr_team_w_reset_decision_loss'][alt_cost][i] + cost*(val_rs['tr_model_w_reset_contradictions'][alt_cost][i])/len(y_val)
                if alt_val_objective < curr_val_objective:
                    new_rs['tr_model_w_reset_contradictions'][cost][i] = rs['tr_model_w_reset_contradictions'][alt_cost][i].copy()
                    new_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['tr_team_w_reset_decision_loss'][alt_cost][i].copy()
                    new_rs['tr_team_w_reset_objective'][cost][i] = new_rs['tr_team_w_reset_decision_loss'][alt_cost][i] + cost*new_rs['tr_model_w_reset_contradictions'][alt_cost][i]/len(y_test)
                    print(f"cost: {cost}, new cost: {alt_cost}, i: {i}, replacing actual of {rs['tr_team_w_reset_objective'][cost][i]} with new of {new_rs['tr_team_w_reset_objective'][cost][i]}")
                    print(f"cost: {cost}, new cost: {alt_cost}, i: {i}, replacing val of {curr_val_objective} with new of {alt_val_objective}")
                    curr_val_objective = alt_val_objective
    new_results_means = new_rs.apply(lambda x: x.apply(lambda y: mean(y)))
    new_results_stderrs = new_rs.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return new_results_means, new_results_stderrs, new_rs



    

cval_bia_meansTrue, cval_bia_stdTrue, cval_bia_rsTrue = cost_validation(bia_rsTrue, val_bia_rsTrue)      
rval_bia_meansTrue, rval_bia_stdTrue, rval_bia_rsTrue = robust_rules(bia_rsTrue, val_bia_rsTrue)    
ccval_bia_meansTrue, ccval_bia_stdTrue, ccval_bia_rsTrue = cost_validation(val_bia_rsTrue, val_bia_rsTrue)     
rcval_bia_meansTrue, rcval_bia_stdTrue, rcval_bia_rsTrue = robust_rules(cval_bia_rsTrue, ccval_bia_rsTrue)     

cval_bia_means1, cval_bia_std1, cval_bia_rs1 = cost_validation(bia_rs1, val_bia_rs1)      
rval_bia_means1, rval_bia_std1, rval_bia_rs1 = robust_rules(bia_rs1, val_bia_rs1)    
ccval_bia_means1, ccval_bia_std1, ccval_bia_rs1 = cost_validation(val_bia_rs1, val_bia_rs1)     
rcval_bia_means1, rcval_bia_std1, rcval_bia_rs1 = robust_rules(cval_bia_rs1, ccval_bia_rs1)   

cval_bia_means02, cval_bia_std02, cval_bia_rs02 = cost_validation(bia_rs02, val_bia_rs02)      
rval_bia_means02, rval_bia_std02, rval_bia_rs02 = robust_rules(bia_rs02, val_bia_rs02)    
ccval_bia_means02, ccval_bia_std02, ccval_bia_rs02 = cost_validation(val_bia_rs02, val_bia_rs02)     
rcval_bia_means02, rcval_bia_std02, rcval_bia_rs02 = robust_rules(cval_bia_rs02, ccval_bia_rs02)   

    



    

print('pause')
