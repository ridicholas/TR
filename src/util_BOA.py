import pandas as pd 
import numpy as np
import math
from itertools import chain, combinations
import itertools
from numpy.random import random
from bisect import bisect_left
from random import sample
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
import time
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    print('in find_lt,{}'.format(a))
    raise ValueError


def log_gampoiss(k,alpha,beta):
    import math
    k = int(k)
    return math.lgamma(k+alpha)+alpha*np.log(beta)-math.lgamma(alpha)-math.lgamma(k+1)-(alpha+k)*np.log(1+beta)


def log_betabin(k,n,alpha,beta):
    import math
    try:
        Const =  math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print('alpha = {}, beta = {}'.format(alpha,beta))
    if isinstance(k,list) or isinstance(k,np.ndarray):
        if len(k)!=len(n):
            print('length of k is %d and length of n is %d'%(len(k),len(n)))
            raise ValueError
        lbeta = []
        for ki,ni in zip(k,n):
            # lbeta.append(math.lgamma(ni+1)- math.lgamma(ki+1) - math.lgamma(ni-ki+1) + math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
            lbeta.append(math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
        return np.array(lbeta)
    else:
        return math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const
        # return math.lgamma(n+1)- math.lgamma(k+1) - math.lgamma(n-k+1) + math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const

def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN

def brs_predict(rules,df):
    Z = [[] for rule in rules]
    dfn = 1-df #df has negative associations
    dfn.columns = [name.strip() + 'neg' for name in df.columns]
    df = pd.concat([df,dfn],axis = 1)
    for i,rule in enumerate(rules):
        Z[i] = (np.sum(df[list(rule)],axis=1)==len(rule)).astype(int)
    Yhat = (np.sum(Z,axis=0)>0).astype(int)
    return Yhat

def brs_predict_conf(rules, df, brs_model):
    Z = [[] for rule in rules]
    dfn = 1-df #df has negative associations
    dfn.columns = [name.strip() + 'neg' for name in df.columns]
    df = pd.concat([df,dfn],axis = 1)
    for i,rule in enumerate(rules):
        rule_index = brs_model.rules.index(rule)
        rule_conf = brs_model.p_precision_matrix[:, rule_index].max()
        Z[i] = (np.sum(df[list(rule)],axis=1)==len(rule)).astype(int) * rule_conf
    if (brs_predict(rules, brs_model.df) == 0).sum() == 0:
        neg_conf = 0
    else:
        neg_conf = (brs_predict(rules, brs_model.df)[brs_predict(rules, brs_model.df) == 0] == brs_model.Y[brs_predict(rules, brs_model.df) == 0]).sum()/(brs_predict(rules, brs_model.df) == 0).sum()
    Yhat_conf = (np.max(Z,axis=0))
    Yhat_conf[Yhat_conf == 0] = neg_conf
    return Yhat_conf

def brs_humanifyPreds(preds, conf_model, Yb, conf_human, fA):
        agreement = preds == Yb
        paccept = fA(conf_human, conf_model, agreement)
        accept = bernoulli.rvs(paccept, size=len(paccept)).astype(bool)    
        finalPreds = Yb.copy()
        finalPreds[accept] = preds[accept]
        return finalPreds



def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = 'neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)   
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules


def brs_expected_loss_filter(brs_mod, x, y_rules, conf_human, p_y=None, e_human_responses=None, conf_model=None, fA=None, asym_loss=[1,1], contradiction_reg=0.0):
   
    conf_model = brs_predict_conf(brs_mod.opt_rules, x, brs_mod)
    yb = e_human_responses
    
    agreement0 = (y_rules == 0) #hypothetical if human chooses 0
    agreement1 = (y_rules == 1) #hypothetical if human chooses 1
    conf_human0 = conf_human.copy()
    conf_human1 = conf_human.copy()
    conf_human0[e_human_responses == 1] = -conf_human0[e_human_responses == 1] #confidence inverted if human expected to choose 1 because situation is hypothetical human chooses 0
    conf_human1[e_human_responses == 0] = -conf_human1[e_human_responses == 0] #confidence inverted if human expected to choose 0 because situation is hypothetical human chooses 1

    p_a = ((yb==0).astype(int)*fA(conf_human0, conf_model, agreement0)) + ((yb==1).astype(int)*fA(conf_human1, conf_model, agreement1))


    e_loss_from_accept = p_a*((p_y[:, 0]*(y_rules == 1).astype(int)*asym_loss[1]) + p_y[:, 1]*(y_rules == 0).astype(int)*asym_loss[0])
    e_loss_from_reject = (1-p_a)*((p_y[:,0]*(yb==1).astype(int)*asym_loss[1]) + p_y[:, 1]*(yb==0).astype(int)*asym_loss[0])
    e_loss_from_contradict = contradiction_reg*(((yb==0).astype(int)*(y_rules==1).astype(int)) + (yb==1).astype(int)*(y_rules==0).astype(int))

    e_loss_from_advising = e_loss_from_accept + e_loss_from_reject + e_loss_from_contradict

    e_loss_from_withholding = p_y[:,0]*(yb==1).astype(int)*asym_loss[1] + p_y[:,1]*(yb==0).astype(int)*asym_loss[0]

    reset = e_loss_from_withholding < e_loss_from_advising

    return reset