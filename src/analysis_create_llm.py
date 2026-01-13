import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from numpy import mean 
import progressbar
from run import ADB
import xgboost as xgb
import os

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

def make_llm_results(dataset, whichtype, num_runs, costs, validation=False, which_to_do=['llm', 'synth_llm']):

    # Create dataframe with comprehensive evaluation results for LLM models
    results = pd.DataFrame(data={
        # Base LLM Raw (unfiltered) results
        'llm_raw_decision_loss': [[]],
        'llm_raw_final_decision_loss': [[]],
        'llm_raw_contradictions': [[]],
        'llm_raw_advice_given_rate': [[]],
        'llm_raw_objective': [[]],
        'llm_raw_final_objective': [[]],
        
        # Base LLM Filtered results  
        'llm_filtered_decision_loss': [[]],
        'llm_filtered_final_decision_loss': [[]],
        'llm_filtered_contradictions': [[]],
        'llm_filtered_advice_given_rate': [[]],
        'llm_filtered_objective': [[]],
        'llm_filtered_final_objective': [[]],
        
        # Synthetic LLM (RevAI) Raw (unfiltered) results
        'synth_llm_raw_decision_loss': [[]],
        'synth_llm_raw_final_decision_loss': [[]],
        'synth_llm_raw_contradictions': [[]],
        'synth_llm_raw_advice_given_rate': [[]],
        'synth_llm_raw_objective': [[]],
        'synth_llm_raw_final_objective': [[]],
        
        # Synthetic LLM Filtered results
        'synth_llm_filtered_decision_loss': [[]],
        'synth_llm_filtered_final_decision_loss': [[]],
        'synth_llm_filtered_contradictions': [[]],
        'synth_llm_filtered_advice_given_rate': [[]],
        'synth_llm_filtered_objective': [[]],
        'synth_llm_filtered_final_objective': [[]],
        
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
            if 'llm' in which_to_do:
                llm_mod = load_results(dataset, f'_{whichtype}', run, cost, 'llm')
            else:
                llm_mod = None
                    
            if 'synth_llm' in which_to_do:
                synth_llm_mod = load_results(dataset, f'_{whichtype}', run, cost, 'synth_llm')
            else:
                synth_llm_mod = None
            
            # Load the final e_y model (for use in comprehensive_evaluation)
            with open(f'results/{dataset}/run{run}/cost{float(cost)}/ey_model_{whichtype}.pkl', 'rb') as f:
                e_y_mod = pickle.load(f)

            # Initialize result storage for this cost/run combination
            llm_raw_decision_loss = []
            llm_raw_final_decision_loss = []
            llm_raw_contradictions = []
            llm_raw_advice_given_rate = []
            llm_raw_objective = []
            llm_raw_final_objective = []
            
            llm_filtered_decision_loss = []
            llm_filtered_final_decision_loss = []
            llm_filtered_contradictions = []
            llm_filtered_advice_given_rate = []
            llm_filtered_objective = []
            llm_filtered_final_objective = []
            
            synth_llm_raw_decision_loss = []
            synth_llm_raw_final_decision_loss = []
            synth_llm_raw_contradictions = []
            synth_llm_raw_advice_given_rate = []
            synth_llm_raw_objective = []
            synth_llm_raw_final_objective = []
            
            synth_llm_filtered_decision_loss = []
            synth_llm_filtered_final_decision_loss = []
            synth_llm_filtered_contradictions = []
            synth_llm_filtered_advice_given_rate = []
            synth_llm_filtered_objective = []
            synth_llm_filtered_final_objective = []
            
            human_decision_loss = []
            
            # Create ADB wrapper for evaluation
            learned_adb = ADB(adb_mod)
            
            for i in range(50):
                # Generate fresh human decisions and confidence for this iteration
                human_decisions = human.get_decisions(x_test, y_test)
                human_conf = human.get_confidence(x_test)

                # Evaluate Base LLM using comprehensive_evaluation
                if llm_mod is not None:
                    llm_results = llm_mod.comprehensive_evaluation(
                        x_test_non_binarized, 
                        human_decisions, 
                        human_conf, 
                        learned_adb.ADB_model_wrapper, 
                        human.ADB, 
                        cost, 
                        e_y_mod
                    )
                    
                    # Raw Base LLM results
                    llm_raw_decision_loss.append(1 - accuracy_score(llm_results['raw_predictions'], y_test))
                    llm_raw_final_decision_loss.append(1 - accuracy_score(llm_results['final_decisions_raw'], y_test))
                    llm_raw_contradictions.append((llm_results['raw_predictions'] != human_decisions).sum())
                    llm_raw_advice_given_rate.append(llm_results['raw_advice_given'].mean())
                    llm_raw_objective.append(llm_raw_decision_loss[-1] + cost * llm_raw_contradictions[-1] / len(y_test))
                    llm_raw_final_objective.append(llm_raw_final_decision_loss[-1] + cost * llm_raw_contradictions[-1] / len(y_test))
                    
                    # Filtered Base LLM results
                    llm_filtered_decision_loss.append(1 - accuracy_score(llm_results['filtered_predictions'], y_test))
                    llm_filtered_final_decision_loss.append(1 - accuracy_score(llm_results['final_decisions_filtered'], y_test))
                    llm_filtered_contradictions.append((llm_results['filtered_predictions'] != human_decisions).sum())
                    llm_filtered_advice_given_rate.append(llm_results['filtered_advice_given'].mean())
                    llm_filtered_objective.append(llm_filtered_decision_loss[-1] + cost * llm_filtered_contradictions[-1] / len(y_test))
                    llm_filtered_final_objective.append(llm_filtered_final_decision_loss[-1] + cost * llm_filtered_contradictions[-1] / len(y_test))
                else:
                    # Default values when model not available
                    for lst in [llm_raw_decision_loss, llm_raw_final_decision_loss, llm_filtered_decision_loss, llm_filtered_final_decision_loss]:
                        lst.append(1.0)
                    for lst in [llm_raw_contradictions, llm_filtered_contradictions]:
                        lst.append(0)
                    for lst in [llm_raw_advice_given_rate, llm_filtered_advice_given_rate]:
                        lst.append(0.0)
                    for lst in [llm_raw_objective, llm_raw_final_objective, llm_filtered_objective, llm_filtered_final_objective]:
                        lst.append(1.0)

                # Evaluate Synthetic LLM using comprehensive_evaluation
                if synth_llm_mod is not None:
                    synth_llm_results = synth_llm_mod.comprehensive_evaluation(
                        x_test_non_binarized, 
                        human_decisions, 
                        human_conf, 
                        learned_adb.ADB_model_wrapper, 
                        human.ADB, 
                        cost, 
                        e_y_mod
                    )
                    
                    # Raw Synthetic LLM results
                    synth_llm_raw_decision_loss.append(1 - accuracy_score(synth_llm_results['raw_predictions'], y_test))
                    synth_llm_raw_final_decision_loss.append(1 - accuracy_score(synth_llm_results['final_decisions_raw'], y_test))
                    synth_llm_raw_contradictions.append((synth_llm_results['raw_predictions'] != human_decisions).sum())
                    synth_llm_raw_advice_given_rate.append(synth_llm_results['raw_advice_given'].mean())
                    synth_llm_raw_objective.append(synth_llm_raw_decision_loss[-1] + cost * synth_llm_raw_contradictions[-1] / len(y_test))
                    synth_llm_raw_final_objective.append(synth_llm_raw_final_decision_loss[-1] + cost * synth_llm_raw_contradictions[-1] / len(y_test))
                    
                    # Filtered Synthetic LLM results
                    synth_llm_filtered_decision_loss.append(1 - accuracy_score(synth_llm_results['filtered_predictions'], y_test))
                    synth_llm_filtered_final_decision_loss.append(1 - accuracy_score(synth_llm_results['final_decisions_filtered'], y_test))
                    synth_llm_filtered_contradictions.append((synth_llm_results['filtered_predictions'] != human_decisions).sum())
                    synth_llm_filtered_advice_given_rate.append(synth_llm_results['filtered_advice_given'].mean())
                    synth_llm_filtered_objective.append(synth_llm_filtered_decision_loss[-1] + cost * synth_llm_filtered_contradictions[-1] / len(y_test))
                    synth_llm_filtered_final_objective.append(synth_llm_filtered_final_decision_loss[-1] + cost * synth_llm_filtered_contradictions[-1] / len(y_test))
                else:
                    # Default values when model not available
                    for lst in [synth_llm_raw_decision_loss, synth_llm_raw_final_decision_loss, synth_llm_filtered_decision_loss, synth_llm_filtered_final_decision_loss]:
                        lst.append(1.0)
                    for lst in [synth_llm_raw_contradictions, synth_llm_filtered_contradictions]:
                        lst.append(0)
                    for lst in [synth_llm_raw_advice_given_rate, synth_llm_filtered_advice_given_rate]:
                        lst.append(0.0)
                    for lst in [synth_llm_raw_objective, synth_llm_raw_final_objective, synth_llm_filtered_objective, synth_llm_filtered_final_objective]:
                        lst.append(1.0)

                human_decision_loss.append(1 - accuracy_score(human_decisions, y_test))
                
                print(i)
            
            # Store averages for this run in results dataframe
            results.loc[cost, 'llm_raw_decision_loss'].append(mean(llm_raw_decision_loss))
            results.loc[cost, 'llm_raw_final_decision_loss'].append(mean(llm_raw_final_decision_loss))
            results.loc[cost, 'llm_raw_contradictions'].append(mean(llm_raw_contradictions))
            results.loc[cost, 'llm_raw_advice_given_rate'].append(mean(llm_raw_advice_given_rate))
            results.loc[cost, 'llm_raw_objective'].append(mean(llm_raw_objective))
            results.loc[cost, 'llm_raw_final_objective'].append(mean(llm_raw_final_objective))
            
            results.loc[cost, 'llm_filtered_decision_loss'].append(mean(llm_filtered_decision_loss))
            results.loc[cost, 'llm_filtered_final_decision_loss'].append(mean(llm_filtered_final_decision_loss))
            results.loc[cost, 'llm_filtered_contradictions'].append(mean(llm_filtered_contradictions))
            results.loc[cost, 'llm_filtered_advice_given_rate'].append(mean(llm_filtered_advice_given_rate))
            results.loc[cost, 'llm_filtered_objective'].append(mean(llm_filtered_objective))
            results.loc[cost, 'llm_filtered_final_objective'].append(mean(llm_filtered_final_objective))
            
            results.loc[cost, 'synth_llm_raw_decision_loss'].append(mean(synth_llm_raw_decision_loss))
            results.loc[cost, 'synth_llm_raw_final_decision_loss'].append(mean(synth_llm_raw_final_decision_loss))
            results.loc[cost, 'synth_llm_raw_contradictions'].append(mean(synth_llm_raw_contradictions))
            results.loc[cost, 'synth_llm_raw_advice_given_rate'].append(mean(synth_llm_raw_advice_given_rate))
            results.loc[cost, 'synth_llm_raw_objective'].append(mean(synth_llm_raw_objective))
            results.loc[cost, 'synth_llm_raw_final_objective'].append(mean(synth_llm_raw_final_objective))
            
            results.loc[cost, 'synth_llm_filtered_decision_loss'].append(mean(synth_llm_filtered_decision_loss))
            results.loc[cost, 'synth_llm_filtered_final_decision_loss'].append(mean(synth_llm_filtered_final_decision_loss))
            results.loc[cost, 'synth_llm_filtered_contradictions'].append(mean(synth_llm_filtered_contradictions))
            results.loc[cost, 'synth_llm_filtered_advice_given_rate'].append(mean(synth_llm_filtered_advice_given_rate))
            results.loc[cost, 'synth_llm_filtered_objective'].append(mean(synth_llm_filtered_objective))
            results.loc[cost, 'synth_llm_filtered_final_objective'].append(mean(synth_llm_filtered_final_objective))
            
            results.loc[cost, 'human_decision_loss'].append(mean(human_decision_loss))
            
    results_means = results.apply(lambda x: x.apply(lambda y: mean(y)))
    results_stderrs = results.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))

    return results_means, results_stderrs, results

def make_llm_TL_v_cost_plot(results_means, results_stderrs, name):
    fig = plt.figure(figsize=(4, 3), dpi=200)
    color_dict = {
        'LLM_Raw': '#FF6B6B', 
        'LLM_Filtered': '#FF9999',
        'Synth_LLM_Raw': '#4ECDC4', 
        'Synth_LLM_Filtered': '#7DDDD7',
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
    plt.plot(normalized_means.index[0:6], normalized_means['llm_raw_final_objective'].iloc[0:6], 
             marker='o', c=color_dict['LLM_Raw'], label='LLM Raw', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['llm_filtered_final_objective'].iloc[0:6], 
             marker='s', c=color_dict['LLM_Filtered'], label='LLM Filtered', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['synth_llm_raw_final_objective'].iloc[0:6], 
             marker='^', c=color_dict['Synth_LLM_Raw'], label='Synth LLM Raw', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['synth_llm_filtered_final_objective'].iloc[0:6], 
             marker='d', c=color_dict['Synth_LLM_Filtered'], label='Synth LLM Filtered', markersize=1.8, linewidth=0.9)
    plt.plot(normalized_means.index[0:6], normalized_means['human_decision_loss'].iloc[0:6], 
             c=color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
    
    # Add error bands
    for key, color_key in [('human_decision_loss', 'Human'), 
                          ('llm_raw_final_objective', 'LLM_Raw'), 
                          ('llm_filtered_final_objective', 'LLM_Filtered'),
                          ('synth_llm_raw_final_objective', 'Synth_LLM_Raw'),
                          ('synth_llm_filtered_final_objective', 'Synth_LLM_Filtered')]:
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

    fig.savefig(f'results/{dataset}/plots/TL_LLM_{dataset}_{name}.png', bbox_inches='tight')

# Main execution
costs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
num_runs = 10
datasets = ['hr']
names = ['biased_dec_bias']
which_to_do = ['llm', 'synth_llm']

for dataset in datasets:
    for name in names:
        if os.path.isfile(f'results/{dataset}/{name}_llm_comprehensive_rs.pkl') and False:
            with open(f'results/{dataset}/{name}_llm_comprehensive_rs.pkl', 'rb') as f:
                rs = pickle.load(f)
            with open(f'results/{dataset}/{name}_llm_comprehensive_means.pkl', 'rb') as f:
                means = pickle.load(f)
            with open(f'results/{dataset}/{name}_llm_comprehensive_std.pkl', 'rb') as f:
                std = pickle.load(f)

            with open(f'results/{dataset}/val_{name}_llm_comprehensive_rs.pkl', 'rb') as f:
                val_rs = pickle.load(f)
            with open(f'results/{dataset}/val_{name}_llm_comprehensive_means.pkl', 'rb') as f:
                val_means = pickle.load(f)
            with open(f'results/{dataset}/val_{name}_llm_comprehensive_std.pkl', 'rb') as f:
                val_std = pickle.load(f)

        else:
            means, std, rs = make_llm_results(dataset, name, num_runs, costs, validation=False, which_to_do=which_to_do)
            # Pickle and write means, std, and rs to file with llm prefix
            with open(f'results/{dataset}/{name}_llm_comprehensive_means.pkl', 'wb') as f:
                pickle.dump(means, f)
            with open(f'results/{dataset}/{name}_llm_comprehensive_std.pkl', 'wb') as f:
                pickle.dump(std, f)
            with open(f'results/{dataset}/{name}_llm_comprehensive_rs.pkl', 'wb') as f:
                pickle.dump(rs, f)
        
            print(f'running for val {dataset} {name}')
            val_means, val_std, val_rs = make_llm_results(dataset, name, num_runs, costs, validation=True, which_to_do=which_to_do)
            # Pickle and write validation results with llm prefix
            with open(f'results/{dataset}/val_{name}_llm_comprehensive_means.pkl', 'wb') as f:
                pickle.dump(val_means, f)
            with open(f'results/{dataset}/val_{name}_llm_comprehensive_std.pkl', 'wb') as f:
                pickle.dump(val_std, f)
            with open(f'results/{dataset}/val_{name}_llm_comprehensive_rs.pkl', 'wb') as f:
                pickle.dump(val_rs, f)

        # Create plot
        make_llm_TL_v_cost_plot(means, std, name)

print('LLM analysis complete!')