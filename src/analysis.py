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
                                'human_decision_loss': [[]]}, index=[costs[0]]
                            )

    for cost in costs[1:]:
        results.loc[cost] = [[] for i in range(len(results.columns))]

    bar=progressbar.ProgressBar()
    for run in bar(range(num_runs)):

        
        bar=progressbar.ProgressBar()
        x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets('heart_disease', run)

        if validation==True:
            x_test = x_val
            y_test = y_val
            x_test_non_binarized = x_val_non_binarized

        human, adb_mod, conf_mod = load_humans(dataset, whichtype, run)
        human.dataset = dataset

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
            
            if cost == 0.0:
                brs_mod.df = x_train
                brs_mod.Y = y_train
                brs_model_preds = brs_predict(brs_mod.opt_rules, x_test)
                brs_conf = brs_predict_conf(brs_mod.opt_rules, x_test, brs_mod)

            for i in range(25):
                
                

                if validation: 

                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    conf_mod_preds = conf_mod.predict(x_test_non_binarized)

                    learned_adb = ADB(adb_mod)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, conf_mod_preds, learned_adb.ADB_model_wrapper, with_reset=True, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, conf_mod_preds, learned_adb.ADB_model_wrapper, with_reset=False, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=conf_mod_preds, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=conf_mod_preds, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, conf_mod_preds, learned_adb.ADB_model_wrapper, x_test)

                    if cost == 0.0:
                        brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, conf_mod_preds, learned_adb.ADB_model_wrapper)
                else:
                    human_decisions = human.get_decisions(x_test, y_test)
                    human_conf = human.get_confidence(x_test)
                    tr_team_preds_with_reset = tr_mod.predictHumanInLoop(x_test, human_decisions, human_conf, human.ADB, with_reset=True, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_team_preds_no_reset = tr_mod.predictHumanInLoop(x_test, human_decisions,human_conf, human.ADB, with_reset=False, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    tr_model_preds_with_reset = tr_mod.predict(x_test, human_decisions, with_reset=True, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]
                    tr_model_preds_no_reset = tr_mod.predict(x_test, human_decisions, with_reset=False, conf_human=human_conf, p_yb=e_yb_mod.predict_proba(x_test_non_binarized), p_y=e_y_mod.predict_proba(x_test_non_binarized))[0]

                    hyrs_model_preds = hyrs_mod.predict(x_test, human_decisions)[0]
                    hyrs_team_preds = hyrs_mod.humanifyPreds(hyrs_model_preds, human_decisions, human_conf, human.ADB, x_test)

                    if cost == 0.0:
                        brs_team_preds = brs_humanifyPreds(brs_model_preds, brs_conf, human_decisions, human_conf, human.ADB)


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


                print(i)
            '''
            if run == 0:
                if cost == 0.0:
                    results = pd.DataFrame({'tr_team_w_reset_decision_loss': np.array([mean(tr_team_w_reset_decision_loss)]),
                                            'tr_team_wo_reset_decision_loss': np.array([mean(tr_team_wo_reset_decision_loss)]),
                                            'tr_model_w_reset_decision_loss': np.array([mean(tr_model_w_reset_decision_loss)]),
                                            'tr_model_wo_reset_decision_loss': np.array([mean(tr_model_wo_reset_decision_loss)]),
                                            'hyrs_model_decision_loss': np.array([mean(hyrs_model_decision_loss)]),
                                            'hyrs_team_decision_loss': np.array([mean(hyrs_team_decision_loss)]),
                                            'brs_model_decision_loss': np.array([mean(brs_model_decision_loss)]),
                                            'brs_team_decision_loss': np.array([mean(brs_team_decision_loss)]),
                                            'tr_model_w_reset_contradictions': np.array([mean(tr_model_w_reset_contradictions)]),
                                            'tr_model_wo_reset_contradictions': np.array([mean(tr_model_wo_reset_contradictions)]),
                                            'hyrs_model_contradictions': np.array([mean(hyrs_model_contradictions)]),
                                            'brs_model_contradictions': np.array([mean(brs_model_contradictions)]), 
                                            'tr_team_w_reset_objective': np.array([mean(tr_team_w_reset_objective)]),
                                            'tr_team_wo_reset_objective': np.array([mean(tr_team_wo_reset_objective)]),
                                            'tr_model_w_reset_objective': np.array([mean(tr_model_w_reset_objective)]),
                                            'tr_model_wo_reset_objective': np.array([mean(tr_model_wo_reset_objective)]),
                                            'hyrs_model_objective': np.array([mean(hyrs_model_objective)]),
                                            'hyrs_team_objective': np.array([mean(hyrs_team_objective)]),
                                            'brs_model_objective': np.array([mean(brs_model_objective)]),
                                            'brs_team_objective': np.array([mean(brs_team_objective)]),
                                            'human_decision_loss': np.array([mean(human_decision_loss)])}, index=[cost])
                    

                else:
                    results.loc[cost] = [np.array([mean(tr_team_w_reset_decision_loss)]),
                                        np.array([mean(tr_team_wo_reset_decision_loss)]),
                                        np.array([mean(tr_model_w_reset_decision_loss)]),
                                        np.array([mean(tr_model_wo_reset_decision_loss)]),
                                        np.array([mean(hyrs_model_decision_loss)]),
                                        np.array([mean(hyrs_team_decision_loss)]),
                                        np.array([mean(brs_model_decision_loss)]),
                                        np.array([mean(brs_team_decision_loss)]),
                                        np.array([mean(tr_model_w_reset_contradictions)]),
                                        np.array([mean(tr_model_wo_reset_contradictions)]),
                                        np.array([mean(hyrs_model_contradictions)]),
                                        np.array([mean(brs_model_contradictions)]), 
                                        np.array([mean(tr_team_w_reset_objective)]),
                                        np.array([mean(tr_team_wo_reset_objective)]),
                                        np.array([mean(tr_model_w_reset_objective)]),
                                        np.array([mean(tr_model_wo_reset_objective)]),
                                        np.array([mean(hyrs_model_objective)]),
                                        np.array([mean(hyrs_team_objective)]),
                                        np.array([mean(brs_model_objective)]),
                                        np.array([mean(brs_team_objective)]),
                                        np.array([mean(human_decision_loss)])
                                        ]
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



    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))




    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results



costs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
costs = [0.0]
num_runs = 5
name = 'miscalibrated'

#cal_r_means, cal_r_stderrs, cal_rs = make_results('heart_disease', name, num_runs, costs, False)
#val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)
r_means, r_stderrs, rs = make_results('heart_disease', name, num_runs, costs, False)
val_r_means, val_r_stderrs, val_rs = make_results('heart_disease', name, num_runs, costs, True)

cost = 0.0
robust_rs = rs.copy()
for i in range(len(val_rs['tr_team_w_reset_objective'][cost])):
    if val_rs['tr_team_w_reset_objective'][cost][i] > val_rs['hyrs_team_objective'][cost][i]:
        if val_rs['hyrs_team_objective'][cost][i] > val_rs['brs_team_objective'][cost][i]:
            robust_rs['tr_team_w_reset_objective'][cost][i] = rs['brs_team_objective'][cost][i]
            robust_rs['tr_model_w_reset_contradictions'][cost][i] = rs['brs_model_contradictions'][cost][i]
            robust_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['brs_team_decision_loss'][cost][i]


        else:
            robust_rs['tr_team_w_reset_objective'][cost][i] = rs['hyrs_team_objective'][cost][i]
            robust_rs['tr_model_w_reset_contradictions'][cost][i] = rs['hyrs_model_contradictions'][cost][i]
            robust_rs['tr_team_w_reset_decision_loss'][cost][i] = rs['hyrs_team_decision_loss'][cost][i]

robust_rs.apply(lambda x: x.apply(lambda y: mean(y)))[['tr_team_w_reset_objective', 'tr_team_wo_reset_objective', 'hyrs_team_objective', 'brs_team_objective', 'human_decision_loss']]

#mis_r_means, mis_r_stderrs, mis_rs = make_results('heart_disease', 'miscalibrated', num_runs, costs, False)



'''
def make_TL_v_cost_plot(results_means, results_stderrs, name):
    color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42'}
    plt.plot(r_means.index, r_means['hyrs_team_objective'], marker = 'v', c=color_dict['HYRS'], label = 'TR-No(ADB, OrgVal)', markersize=1.8, linewidth=0.9)
    plt.plot(costFrame['Costs'], costFrame['R_HyRS_Objective'], marker = 'x', c=color_dict['HYRSRecon'], label = 'TR-No(ADB)', markersize=1.8, linewidth=0.9)
    plt.plot(costFrame['Costs'], costFrame['TR_Objective'], marker = '.', c=color_dict['TR'], label='TR', markersize=1.8, linewidth=0.9)
    plt.plot(costFrame['Costs'], costFrame['BRS_Objective'], marker = 's', c=color_dict['BRS'], label='Task-Only (Current Practice)', markersize=1.8, linewidth=0.9)
    
    plt.axhline(costFrame['Human Only'][0], c = color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    plt.fill_between(costFrame['Costs'], 
                costFrame['Human Only']-(costFrame['Human Only SE']),
                costFrame['Human Only']+(costFrame['Human Only SE']) ,
                color=color_dict['Human'], alpha=0.2)
    plt.fill_between(costFrame['Costs'], 
                costFrame['HyRS_Objective']-(costFrame['HyRS_Objective_SE']),
                costFrame['HyRS_Objective']+(costFrame['HyRS_Objective_SE']) ,
                color=color_dict['HYRS'], alpha=0.2)
    
    plt.fill_between(costFrame['Costs'], 
                costFrame['R_HyRS_Objective']-(costFrame['R_HyRS_Objective_SE']),
                costFrame['R_HyRS_Objective']+(costFrame['R_HyRS_Objective_SE']) ,
                color=color_dict['HYRSRecon'], alpha=0.2)
    plt.fill_between(costFrame['Costs'], 
                costFrame['BRS_Objective']-(costFrame['BRS_Objective_SE']),
                costFrame['BRS_Objective']+(costFrame['BRS_Objective_SE']) ,
                color=color_dict['BRS'], alpha=0.2)
    plt.fill_between(costFrame['Costs'], 
                costFrame['TR_Objective']-(costFrame['TR_Objective_SE']),
                costFrame['TR_Objective']+(costFrame['TR_Objective_SE']),
                color=color_dict['TR'], alpha=0.2)
    plt.xlabel('Reconciliation Cost', fontsize=12)
    plt.ylabel('Total Team Loss', fontsize=12)
    plt.tick_params(labelrotation=45, labelsize=10)
    #plt.title('{} Setting'.format(setting), fontsize=15)
    plt.legend(prop={'size': 5})
    plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

    fig.savefig(f'Plots/TL_{setting_type}_len{rule_len}_{data}_{setting}.png', bbox_inches='tight')

    plt.clf()
'''



    

    
                            
                            
    



    

print('pause')

