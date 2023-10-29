from human import Human
import pandas as pd
import yaml
from tr import *
from hyrs import *
from brs import *
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import os
import pickle
from scipy.stats import bernoulli, uniform
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error
import numpy as np

class ADB(object):
    def __init__(self, adb_model) -> None:
        self.adb_model = adb_model
    def ADB_model_wrapper(self, human_conf, model_conf, agreement):
        X = pd.DataFrame({'human_conf': human_conf, 'model_confs': model_conf, 'agreement':agreement})
        return self.adb_model.predict_proba(X)[:, 1]
    
#a run of the experiment consists of: a dataset, a human, a set of models, and a set of parameters

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





def evaluate_adb_model(adb_model, human, x_test, c_human_true, c_human_estimate, c_model, agreement):
    scores = []
    for i in range(100):
        p_accepts = human.ADB(c_human_true, c_model, agreement)
        realized_accepts = bernoulli.rvs(p=p_accepts, size=len(p_accepts))
        scores.append(roc_auc_score(realized_accepts, adb_model.predict_proba(pd.DataFrame({'human_conf': c_human_estimate, 'model_confs': c_model, 'agreement':agreement}))[:, 1]))
    return np.array(scores).mean()

def run(dataset, run_num, human_name, runtype='standard', which_models=['tr'], contradiction_reg=0, remake_humans=False):   
    # load data
    x_train, y_train, x_train_non_binarized, x_learning_non_binarized, x_learning, y_learning, x_human_train, y_human_train, x_val, y_val, x_test, y_test, x_val_non_binarized, x_test_non_binarized = load_datasets(dataset, run_num)

    if not os.path.exists(f'results/{dataset}/run{run_num}'):
        os.makedirs(f'results/{dataset}/run{run_num}')

    # make human
    if remake_humans or not os.path.exists(f'results/{dataset}/run{run_num}/{human_name}.pkl'):
        human = Human(human_name, x_human_train, y_human_train)
        with open(f'results/{dataset}/run{run_num}/{human_name}.pkl', 'wb') as f:
            pickle.dump(human, f)
    else:
        with open(f'results/{dataset}/run{run_num}/{human_name}.pkl', 'rb') as f:
            human = pickle.load(f)

    #train confidence model
    if remake_humans or not os.path.exists(f'results/{dataset}/run{run_num}/conf_model_{human_name}.pkl'):
        conf_model = xgb.XGBRegressor().fit(x_learning_non_binarized, human.get_confidence(x_learning))
        with open(f'results/{dataset}/run{run_num}/conf_model_{human_name}.pkl', 'wb') as f:
            pickle.dump(conf_model, f)
    else:
        with open(f'results/{dataset}/run{run_num}/conf_model_{human_name}.pkl', 'rb') as f:
            conf_model = pickle.load(f)
    #train ADB model
    initial_task_model = LogisticRegression().fit(x_train.iloc[0:5, :], y_train.iloc[0:5])

    if remake_humans or not os.path.exists(f'results/{dataset}/run{run_num}/adb_model_{human_name}.pkl'):

        adb_learning_data = pd.DataFrame({'human_conf': human.get_confidence(x_learning), 
                                        'model_confs': initial_task_model.predict_proba(x_learning).max(axis=1), 
                                        'agreement': (initial_task_model.predict(x_learning) == human.get_decisions(x_learning, y_learning))})
        
        p_accepts = human.ADB(adb_learning_data['human_conf'], adb_learning_data['model_confs'], adb_learning_data['agreement'])
        realized_accepts = bernoulli.rvs(p=p_accepts, size=len(p_accepts))
        adb_model = xgb.XGBClassifier().fit(adb_learning_data, realized_accepts)
        with open(f'results/{dataset}/run{run_num}/adb_model_{human_name}.pkl', 'wb') as f:
            pickle.dump(adb_model, f)
    else:
        with open(f'results/{dataset}/run{run_num}/adb_model_{human_name}.pkl', 'rb') as f:
            adb_model = pickle.load(f)
    
    adb = ADB(adb_model)


    
    




    
    # load params
    with open(f'src/{runtype}_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    Niteration = config['Niteration']
    Nchain = config['Nchain']
    Nlevel =  config['Nlevel']
    Nrules = config['Nrules']
    supp = config['supp']
    maxlen = config['maxlen']
    protected = config['protected']
    budget = config['budget']
    sample_ratio = config['sample_ratio']
    alpha = config['alpha']
    beta = config['beta']
    iters = Niteration
    fairness_reg = config['fairness_reg']
    asym_loss = config['asym_loss']

    #check if run/cost directory below exists
    if not os.path.exists(f'results/{dataset}/run{run_num}/cost{contradiction_reg}'):
        os.makedirs(f'results/{dataset}/run{run_num}/cost{contradiction_reg}')

    # train advising
    if 'hyrs' in which_models:
        hyrs_model = hyrs(x_train, y_train, human.get_decisions(x_train, y_train))

        hyrs_model.set_parameters(alpha = alpha, beta=beta, contradiction_reg=contradiction_reg, force_complete_coverage=False, asym_loss=asym_loss)
        hyrs_model.generate_rulespace(supp = supp, maxlen=maxlen, N=Nrules, need_negcode=True, method='randomforest',criteria='precision')
        _, _, _ = hyrs_model.train(Niteration=Niteration, T0=0.01, print_message=False)

        #write hyrs model
        with open(f'results/{dataset}/run{run_num}/cost{contradiction_reg}/hyrs_model_{human_name}.pkl', 'wb') as f:
            pickle.dump(hyrs_model, f)
    
    if 'brs' in which_models: 
        brs_model = brs(x_train, y_train)
        brs_model.generate_rules(supp = supp, maxlen=maxlen, N=Nrules,  method='randomforest')
        brs_model.set_parameters()

       
        _ = brs_model.fit(Niteration=Niteration, Nchain=1, print_message=False)

        #write brs model
        with open(f'results/{dataset}/run{run_num}/cost{contradiction_reg}/brs_model.pkl', 'wb') as f:
            pickle.dump(brs_model, f)

    if 'tr' in which_models:
        #train estimates
        e_y_mod = xgb.XGBClassifier().fit(x_train_non_binarized, y_train)
        e_yb_mod = xgb.XGBClassifier().fit(x_train_non_binarized, human.get_decisions(x_train, y_train))

        tr_model = tr(x_train, y_train,
                    human.get_decisions(x_train, y_train),
                    human.get_confidence(x_train), 
                    p_y=e_y_mod.predict_proba(x_train_non_binarized),
                    p_yb=e_yb_mod.predict_proba(x_train_non_binarized))

        tr_model.set_parameters(alpha = alpha, beta=beta, contradiction_reg=contradiction_reg, fairness_reg=fairness_reg, force_complete_coverage=False, asym_loss=asym_loss, fA=adb.ADB_model_wrapper)

        tr_model.generate_rulespace(supp = supp, maxlen=maxlen, N=Nrules, need_negcode=True, method='randomforest',criteria='precision')
        _, _, _ = tr_model.train(Niteration=Niteration, T0=0.01, print_message=False)

        
        #write tr model
        with open(f'results/{dataset}/run{run_num}/cost{contradiction_reg}/tr_model_{human_name}.pkl', 'wb') as f:
            pickle.dump(tr_model, f)

#run('heart_disease', 2, 'slightly_miscalibrated', runtype='standard', which_models=['hyrs','brs','tr'], contradiction_reg=0.4, remake_humans=False)



    

        

    
    
