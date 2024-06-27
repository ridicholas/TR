import numpy as np
from random import shuffle
import pandas as pd
import os
from sklearn.model_selection import train_test_split as split
from scipy.io import arff


def make_heart_data(numQs=5, num_runs=50):
    startDict = {}
    # train
    startDict['Xtrain'] = pd.read_csv(
        'datasets/heart_disease/heart_disease_uci.csv')
    startDict['Ytrain'] = startDict['Xtrain']['num']
    startDict['Ytrain'] = startDict['Ytrain'].replace({2: 1, 3: 1, 4: 1})

    startDict['Xtrain'].drop(columns=['num', 'id', 'dataset'], inplace=True)

    startDict['Xtrain_non_binarized'] = startDict['Xtrain'].copy()
    string_cols = []
    for col in startDict['Xtrain'].columns:
        if startDict['Xtrain'][col].dtype == float or startDict['Xtrain'][col].dtype == int:
            for q in range(1, numQs):
                quantile = round((1/(numQs+1))*q, 2)
                quantVal = round(np.quantile(
                    startDict['Xtrain'][col].dropna(), q=quantile), 2)
                startDict['Xtrain'][col+'{}'.format(quantVal)] = (
                    startDict['Xtrain'][col] > quantVal).astype(int)
            if startDict['Xtrain'][col].isna().sum() > 0:
                startDict['Xtrain'][col +
                                    '{}'.format('_nan')] = startDict['Xtrain'][col].isna().astype(int)

            startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])
        elif col != 'classification':
            string_cols.append(col)

    startDict['Xtrain'] = pd.get_dummies(
        startDict['Xtrain'], columns=string_cols, drop_first=True)
    startDict['Xtrain_non_binarized'] = pd.get_dummies(
        startDict['Xtrain_non_binarized'], columns=string_cols, drop_first=True)

    startDict['Xtrain_start'] = startDict['Xtrain'].copy()
    startDict['Ytrain_start'] = startDict['Ytrain'].copy()
    startDict['Xtrain_non_binarized_start'] = startDict['Xtrain_non_binarized'].copy()

    for i in range(num_runs):
        if i < 20:
            continue

        # make human_training
        startDict['Xtrain'], startDict['Xhuman_train'], startDict['Ytrain'], \
            startDict['Yhuman_train'], startDict['Xtrain_non_binarized'], startDict['Xhuman_train_non_binarized'] = split(startDict['Xtrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Xtrain_non_binarized_start'],
                                                                                                                          test_size=0.08,
                                                                                                                          stratify=startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          random_state=i)

        # make confidence and ADB training
        startDict['Xtrain'], startDict['Xlearning'], startDict['Ytrain'], startDict['Ylearning'], startDict['Xtrain_non_binarized'], startDict['Xlearning_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Ytrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Xtrain_non_binarized'],
                                                                                                                                                                                  test_size=0.15,
                                                                                                                                                                                  stratify=startDict[
            'Ytrain'],
            random_state=i)

        # make val
        startDict['Xtrain'], startDict['Xval'], startDict['Ytrain'], startDict['Yval'], startDict['Xtrain_non_binarized'], startDict['Xval_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Xtrain_non_binarized'],
                                                                                                                                                                   test_size=0.12,
                                                                                                                                                                   stratify=startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   random_state=i)

        # make test
        startDict['Xtrain'], startDict['Xtest'], startDict['Ytrain'], \
            startDict['Ytest'], startDict['Xtrain_non_binarized'], startDict['Xtest_non_binarized'] = split(startDict['Xtrain'],
                                                                                                            startDict['Ytrain'],
                                                                                                            startDict['Xtrain_non_binarized'],
                                                                                                            test_size=0.2,
                                                                                                            stratify=startDict[
                                                                                                                'Ytrain'],
                                                                                                            random_state=i)

        outdir = f'datasets/heart_disease/processed/run{i}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        startDict['Xtrain'].to_csv(f'{outdir}/xtrain.csv')
        startDict['Xhuman_train'].to_csv(f'{outdir}/xhumantrain.csv')
        startDict['Xval'].to_csv(f'{outdir}/xval.csv')
        startDict['Xtest'].to_csv(f'{outdir}/xtest.csv')

        startDict['Ytrain'].to_csv(f'{outdir}/ytrain.csv')
        startDict['Yhuman_train'].to_csv(f'{outdir}/yhumantrain.csv')
        startDict['Yval'].to_csv(f'{outdir}/yval.csv')
        startDict['Ytest'].to_csv(f'{outdir}/ytest.csv')

        startDict['Xtrain_non_binarized'].to_csv(
            f'{outdir}/xtrain_non_binarized.csv')
        startDict['Xhuman_train_non_binarized'].to_csv(
            f'{outdir}/xhumantrain_non_binarized.csv')
        startDict['Xval_non_binarized'].to_csv(
            f'{outdir}/xval_non_binarized.csv')
        startDict['Xtest_non_binarized'].to_csv(
            f'{outdir}/xtest_non_binarized.csv')

        startDict['Xlearning_non_binarized'].to_csv(f'{outdir}/xlearning_non_binarized.csv')
        startDict['Ylearning'].to_csv(f'{outdir}/ylearning.csv')
        startDict['Xlearning'].to_csv(f'{outdir}/xlearning.csv')

def make_fico_data(numQs=5, num_runs=50):
    startDict = {}
    # train
    startDict['Xtrain'] = pd.read_csv(
        'datasets/fico/FICO.csv')
    startDict['Ytrain'] = startDict['Xtrain']['RiskPerformance']
    startDict['Ytrain'] = startDict['Ytrain'].replace({"Bad": 1, "Good":0})

    startDict['Xtrain'].drop(columns=['RiskPerformance'], inplace=True)

    startDict['Xtrain_non_binarized'] = startDict['Xtrain'].copy()
    string_cols = []
    for col in startDict['Xtrain'].columns:
        if (startDict['Xtrain'][col].dtype == float or startDict['Xtrain'][col].dtype == int) and (col not in ['RiskPerformance', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver']):
            for q in range(1, numQs):
                quantile = round((1/(numQs+1))*q, 2)
                quantVal = round(np.quantile(
                    startDict['Xtrain'][col].dropna(), q=quantile), 2)
                startDict['Xtrain'][col+'{}'.format(quantVal)] = (
                    startDict['Xtrain'][col] > quantVal).astype(int)
            if startDict['Xtrain'][col].isna().sum() > 0:
                startDict['Xtrain'][col +
                                    '{}'.format('_nan')] = startDict['Xtrain'][col].isna().astype(int)

            startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])
        else:
            string_cols.append(col)

    startDict['Xtrain'] = pd.get_dummies(
        startDict['Xtrain'], columns=string_cols, drop_first=True)
    startDict['Xtrain_non_binarized'] = pd.get_dummies(
        startDict['Xtrain_non_binarized'], columns=string_cols, drop_first=True)

    startDict['Xtrain_start'] = startDict['Xtrain'].copy()
    startDict['Ytrain_start'] = startDict['Ytrain'].copy()
    startDict['Xtrain_non_binarized_start'] = startDict['Xtrain_non_binarized'].copy()

    for i in range(num_runs):
        if i < 20:
            continue

        # make human_training
        startDict['Xtrain'], startDict['Xhuman_train'], startDict['Ytrain'], \
            startDict['Yhuman_train'], startDict['Xtrain_non_binarized'], startDict['Xhuman_train_non_binarized'] = split(startDict['Xtrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Xtrain_non_binarized_start'],
                                                                                                                          test_size=0.1,
                                                                                                                          stratify=startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          random_state=i)

        # make confidence and ADB training
        startDict['Xtrain'], startDict['Xlearning'], startDict['Ytrain'], startDict['Ylearning'], startDict['Xtrain_non_binarized'], startDict['Xlearning_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Ytrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Xtrain_non_binarized'],
                                                                                                                                                                                  test_size=0.15,
                                                                                                                                                                                  stratify=startDict[
            'Ytrain'],
            random_state=i)

        # make val
        startDict['Xtrain'], startDict['Xval'], startDict['Ytrain'], startDict['Yval'], startDict['Xtrain_non_binarized'], startDict['Xval_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Xtrain_non_binarized'],
                                                                                                                                                                   test_size=0.1,
                                                                                                                                                                   stratify=startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   random_state=i)

        # make test
        startDict['Xtrain'], startDict['Xtest'], startDict['Ytrain'], \
            startDict['Ytest'], startDict['Xtrain_non_binarized'], startDict['Xtest_non_binarized'] = split(startDict['Xtrain'],
                                                                                                            startDict['Ytrain'],
                                                                                                            startDict['Xtrain_non_binarized'],
                                                                                                            test_size=0.15,
                                                                                                            stratify=startDict[
                                                                                                                'Ytrain'],
                                                                                                            random_state=i)

        outdir = f'datasets/fico/processed/run{i}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        startDict['Xtrain'].to_csv(f'{outdir}/xtrain.csv')
        startDict['Xhuman_train'].to_csv(f'{outdir}/xhumantrain.csv')
        startDict['Xval'].to_csv(f'{outdir}/xval.csv')
        startDict['Xtest'].to_csv(f'{outdir}/xtest.csv')

        startDict['Ytrain'].to_csv(f'{outdir}/ytrain.csv')
        startDict['Yhuman_train'].to_csv(f'{outdir}/yhumantrain.csv')
        startDict['Yval'].to_csv(f'{outdir}/yval.csv')
        startDict['Ytest'].to_csv(f'{outdir}/ytest.csv')

        startDict['Xtrain_non_binarized'].to_csv(
            f'{outdir}/xtrain_non_binarized.csv')
        startDict['Xhuman_train_non_binarized'].to_csv(
            f'{outdir}/xhumantrain_non_binarized.csv')
        startDict['Xval_non_binarized'].to_csv(
            f'{outdir}/xval_non_binarized.csv')
        startDict['Xtest_non_binarized'].to_csv(
            f'{outdir}/xtest_non_binarized.csv')

        startDict['Xlearning_non_binarized'].to_csv(f'{outdir}/xlearning_non_binarized.csv')
        startDict['Ylearning'].to_csv(f'{outdir}/ylearning.csv')
        startDict['Xlearning'].to_csv(f'{outdir}/xlearning.csv')


def make_hr_data(numQs=5, num_runs=50):
    startDict = {}
    # train
    data = arff.loadarff('datasets/hr/Train-Natural-HR_employee_attrition.arff')
    startDict['Xtrain'] = pd.DataFrame(data[0])
    startDict['Ytrain'] = startDict['Xtrain']['Attrition']
    startDict['Ytrain'] = startDict['Ytrain'].str.decode('UTF-8')
    startDict['Ytrain'] = startDict['Ytrain'].replace({"Yes": 1, "No": 0})
    startDict['Xtrain'].drop('Attrition', inplace=True, axis=1)

    

    
    str_df = startDict['Xtrain'].select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        startDict['Xtrain'][col] = str_df[col]
    startDict['Xtrain_non_binarized'] = startDict['Xtrain'].copy()
    categoricals = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'JobSatisfaction',
                    'MaritalStatus', 'Over18', 'OverTime']
    string_cols = []
    for col in startDict['Xtrain'].columns:
        if ((startDict['Xtrain'][col].dtype == float) or (startDict['Xtrain'][col].dtype == int)) and (col not in categoricals):
            for q in range(1, numQs):
                quantile = round((1/(numQs+1))*q, 2)
                quantVal = round(np.quantile(
                    startDict['Xtrain'][col].dropna(), q=quantile), 2)
                startDict['Xtrain'][col+'{}'.format(quantVal)] = (
                    startDict['Xtrain'][col] > quantVal).astype(int)
            if startDict['Xtrain'][col].isna().sum() > 0:
                startDict['Xtrain'][col +
                                    '{}'.format('_nan')] = startDict['Xtrain'][col].isna().astype(int)

            startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])
        elif col != 'classification':
            string_cols.append(col)

    startDict['Xtrain'] = pd.get_dummies(
        startDict['Xtrain'], columns=string_cols, drop_first=True)
    startDict['Xtrain_non_binarized'] = pd.get_dummies(
        startDict['Xtrain_non_binarized'], columns=string_cols, drop_first=True)

    startDict['Xtrain_start'] = startDict['Xtrain'].copy()
    startDict['Ytrain_start'] = startDict['Ytrain'].copy()
    startDict['Xtrain_non_binarized_start'] = startDict['Xtrain_non_binarized'].copy()

    for i in range(num_runs):
        if i < 20:
            continue

        # make human_training
        startDict['Xtrain'], startDict['Xhuman_train'], startDict['Ytrain'], \
            startDict['Yhuman_train'], startDict['Xtrain_non_binarized'], startDict['Xhuman_train_non_binarized'] = split(startDict['Xtrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Xtrain_non_binarized_start'],
                                                                                                                          test_size=0.1,
                                                                                                                          stratify=startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          random_state=i)

        # make confidence and ADB training
        startDict['Xtrain'], startDict['Xlearning'], startDict['Ytrain'], startDict['Ylearning'], startDict['Xtrain_non_binarized'], startDict['Xlearning_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Ytrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Xtrain_non_binarized'],
                                                                                                                                                                                  test_size=0.15,
                                                                                                                                                                                  stratify=startDict[
            'Ytrain'],
            random_state=i)

        # make val
        startDict['Xtrain'], startDict['Xval'], startDict['Ytrain'], startDict['Yval'], startDict['Xtrain_non_binarized'], startDict['Xval_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Xtrain_non_binarized'],
                                                                                                                                                                   test_size=0.05,
                                                                                                                                                                   stratify=startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   random_state=i)

        # make test
        startDict['Xtrain'], startDict['Xtest'], startDict['Ytrain'], \
            startDict['Ytest'], startDict['Xtrain_non_binarized'], startDict['Xtest_non_binarized'] = split(startDict['Xtrain'],
                                                                                                            startDict['Ytrain'],
                                                                                                            startDict['Xtrain_non_binarized'],
                                                                                                            test_size=0.2,
                                                                                                            stratify=startDict[
                                                                                                                'Ytrain'],
                                                                                                            random_state=i)

        outdir = f'datasets/hr/processed/run{i}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        startDict['Xtrain'].to_csv(f'{outdir}/xtrain.csv')
        startDict['Xhuman_train'].to_csv(f'{outdir}/xhumantrain.csv')
        startDict['Xval'].to_csv(f'{outdir}/xval.csv')
        startDict['Xtest'].to_csv(f'{outdir}/xtest.csv')

        startDict['Ytrain'].to_csv(f'{outdir}/ytrain.csv')
        startDict['Yhuman_train'].to_csv(f'{outdir}/yhumantrain.csv')
        startDict['Yval'].to_csv(f'{outdir}/yval.csv')
        startDict['Ytest'].to_csv(f'{outdir}/ytest.csv')

        startDict['Xtrain_non_binarized'].to_csv(
            f'{outdir}/xtrain_non_binarized.csv')
        startDict['Xhuman_train_non_binarized'].to_csv(
            f'{outdir}/xhumantrain_non_binarized.csv')
        startDict['Xval_non_binarized'].to_csv(
            f'{outdir}/xval_non_binarized.csv')
        startDict['Xtest_non_binarized'].to_csv(
            f'{outdir}/xtest_non_binarized.csv')

        startDict['Xlearning_non_binarized'].to_csv(f'{outdir}/xlearning_non_binarized.csv')
        startDict['Ylearning'].to_csv(f'{outdir}/ylearning.csv')
        startDict['Xlearning'].to_csv(f'{outdir}/xlearning.csv')

def make_adult_data(numQs=5, num_runs=50):
    startDict = {}
    # train
    startDict['Xtrain'] = pd.read_csv(
        'datasets/adult/adult_train1.csv')
    startDict['Ytrain'] = startDict['Xtrain']['Y']
  

    startDict['Xtrain'].drop(columns=['Y'], inplace=True)

    startDict['Xtrain_non_binarized'] = startDict['Xtrain'].copy()
    num_cols = ['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in num_cols:
        
        for q in range(1, numQs):
            quantile = round((1/(numQs+1))*q, 2)
            quantVal = round(np.quantile(
                startDict['Xtrain'][col].dropna(), q=quantile), 2)
            startDict['Xtrain'][col+'{}'.format(quantVal)] = (
                startDict['Xtrain'][col] > quantVal).astype(int)
        

        startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])


    

    startDict['Xtrain_start'] = startDict['Xtrain'].copy()
    startDict['Ytrain_start'] = startDict['Ytrain'].copy()
    startDict['Xtrain_non_binarized_start'] = startDict['Xtrain_non_binarized'].copy()

    for i in range(num_runs):

        # make human_training
        startDict['Xtrain'], startDict['Xhuman_train'], startDict['Ytrain'], \
            startDict['Yhuman_train'], startDict['Xtrain_non_binarized'], startDict['Xhuman_train_non_binarized'] = split(startDict['Xtrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          startDict[
                                                                                                                              'Xtrain_non_binarized_start'],
                                                                                                                          test_size=0.1,
                                                                                                                          stratify=startDict[
                                                                                                                              'Ytrain_start'],
                                                                                                                          random_state=i)

        # make confidence and ADB training
        startDict['Xtrain'], startDict['Xlearning'], startDict['Ytrain'], startDict['Ylearning'], startDict['Xtrain_non_binarized'], startDict['Xlearning_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Ytrain'],
                                                                                                                                                                                  startDict[
                                                                                                                                                                                      'Xtrain_non_binarized'],
                                                                                                                                                                                  test_size=0.1,
                                                                                                                                                                                  stratify=startDict[
            'Ytrain'],
            random_state=i)


        # make smaller training size 
        _, startDict['Xtrain'], _, startDict['Ytrain'], _, startDict['Xtrain_non_binarized'] = split(startDict['Xtrain'],
                                                                                                           startDict['Ytrain'],
                                                                                                           startDict['Xtrain_non_binarized'],
                                                                                                           test_size=0.4,
                                                                                                           stratify=startDict['Ytrain'], 
                                                                                                           random_state=i)


        # make val
        startDict['Xtrain'], startDict['Xval'], startDict['Ytrain'], startDict['Yval'], startDict['Xtrain_non_binarized'], startDict['Xval_non_binarized'] = split(startDict['Xtrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   startDict[
                                                                                                                                                                       'Xtrain_non_binarized'],
                                                                                                                                                                   test_size=0.1,
                                                                                                                                                                   stratify=startDict[
                                                                                                                                                                       'Ytrain'],
                                                                                                                                                                   random_state=i)

        # make test
        startDict['Xtrain'], startDict['Xtest'], startDict['Ytrain'], \
            startDict['Ytest'], startDict['Xtrain_non_binarized'], startDict['Xtest_non_binarized'] = split(startDict['Xtrain'],
                                                                                                            startDict['Ytrain'],
                                                                                                            startDict['Xtrain_non_binarized'],
                                                                                                            test_size=0.15,
                                                                                                            stratify=startDict[
                                                                                                                'Ytrain'],
                                                                                                            random_state=i)

        outdir = f'datasets/adult/processed/run{i}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        startDict['Xtrain'].to_csv(f'{outdir}/xtrain.csv')
        startDict['Xhuman_train'].to_csv(f'{outdir}/xhumantrain.csv')
        startDict['Xval'].to_csv(f'{outdir}/xval.csv')
        startDict['Xtest'].to_csv(f'{outdir}/xtest.csv')

        startDict['Ytrain'].to_csv(f'{outdir}/ytrain.csv')
        startDict['Yhuman_train'].to_csv(f'{outdir}/yhumantrain.csv')
        startDict['Yval'].to_csv(f'{outdir}/yval.csv')
        startDict['Ytest'].to_csv(f'{outdir}/ytest.csv')

        startDict['Xtrain_non_binarized'].to_csv(
            f'{outdir}/xtrain_non_binarized.csv')
        startDict['Xhuman_train_non_binarized'].to_csv(
            f'{outdir}/xhumantrain_non_binarized.csv')
        startDict['Xval_non_binarized'].to_csv(
            f'{outdir}/xval_non_binarized.csv')
        startDict['Xtest_non_binarized'].to_csv(
            f'{outdir}/xtest_non_binarized.csv')

        startDict['Xlearning_non_binarized'].to_csv(f'{outdir}/xlearning_non_binarized.csv')
        startDict['Ylearning'].to_csv(f'{outdir}/ylearning.csv')
        startDict['Xlearning'].to_csv(f'{outdir}/xlearning.csv')


#make_heart_data(numQs=5, num_runs=30)
#make_fico_data(numQs=5, num_runs=30)
#make_hr_data(numQs=5, num_runs=30)
#make_adult_data(numQs=5, num_runs=20)
