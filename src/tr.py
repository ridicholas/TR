import pandas as pd 
#from fim import fpgrowth,fim
import numpy as np
import math
from itertools import chain, combinations
import itertools
from numpy.random import random
from scipy import sparse
from bisect import bisect_left
from random import sample
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
import time
import scipy
from sklearn.preprocessing import binarize
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from random import choices
import xgboost as xgb

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

class tr(object):
    def __init__(self, binary_data,Y,Yb, conf_human=None, p_yb=None, p_y=None):
        self.df = binary_data  
        self.Y = pd.Series(Y)
        self.N = float(len(Y))
        self.Yb = pd.Series(Yb)
        self.conf_human = pd.Series(conf_human)
        self.p_yb = p_yb.copy()
        self.p_y = p_y.copy()

    
    def set_parameters(self, alpha = 0, beta = 0, fairness_reg = 0, contradiction_reg = 0, fA=0.5, force_complete_coverage=False, asym_loss = [1,1], fair_feat=None):
        """
        asym_loss = [loss from False Negatives (ground truth positive), loss from False Positives (ground truth negative)]
        """
        # input al and bl are lists
        self.alpha = alpha
        self.beta = beta
        self.fairness_reg = fairness_reg
        self.fairness_feature = fair_feat
        self.contradiction_reg = contradiction_reg
        self.fA = fA
        self.force_complete_coverage = force_complete_coverage
        self.asym_loss = asym_loss


    def generate_rulespace(self,supp,maxlen,N, need_negcode = False,njobs = 5, method = 'fpgrowth',criteria = 'IG',add_rules = []):
        if method == 'fpgrowth':
            if need_negcode:
                df = 1-self.df 
                df.columns = [name.strip() + 'neg' for name in self.df.columns]
                df = pd.concat([self.df,df],axis = 1)
            else:
                df = 1 - self.df
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
            itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]  
            prules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            prules = [np.sort(x[0]).tolist() for x in prules]
            nrules= fpgrowth([itemMatrix[i] for i in nindex],supp = supp,zmin = 1,zmax = maxlen)
            nrules = [np.sort(x[0]).tolist() for x in nrules]
        else:
            print('Using random forest to generate rules ...')
            prules = []
            for length in range(max(min(2,maxlen), 1),maxlen+1,1):
                n_estimators = 250*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    prules.extend(extract_rules(clf.estimators_[n],self.df.columns))
            prules = [list(x) for x in set(tuple(np.sort(x)) for x in prules)] 
            
            if self.force_complete_coverage:
                nrules = [[self.df.columns[0]], [self.df.columns[0] + 'neg']]
            else:
                nrules = []
                for length in range(max(min(2,maxlen), 1),maxlen+1,1):
                    n_estimators = 250*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                    clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                    clf.fit(self.df,1-self.Y)
                    for n in range(n_estimators):
                        nrules.extend(extract_rules(clf.estimators_[n],self.df.columns))
                nrules = [list(x) for x in set(tuple(np.sort(x)) for x in nrules)]
              
            df = 1-self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df,df],axis = 1)
        self.prules, self.pRMatrix, self.psupp, self.pprecision, self.perror, self.p_precision_matrix = self.screen_rules(prules,df,self.Y,N,supp)
        if self.force_complete_coverage:
            self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror, self.n_precision_matrix = self.screen_rules(nrules,df,1-self.Y,N,0)
        else:
            self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror, self.n_precision_matrix = self.screen_rules(nrules,df,1-self.Y,N,supp)

        # print '\tTook %0.3fs to generate %d rules' % (self.screen_time, len(self.rules))

    def screen_rules(self,rules,df,y,N,supp,criteria = 'precision',njobs = 5, add_rules = [], no_select=False):
        N = int(N)
        # print 'screening rules'
        start_time = time.time()
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        # mat = sparse.csr_matrix.dot(df,ruleMatrix)
        mat = np.matrix(df)*ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)

        Zpos = [Z[i] for i in np.where(y>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        supp_select = np.where(TP>=supp*sum(y)/100)[0]
        FP = np.array(np.sum(Z,axis = 0))[0] - TP
        p1 = TP.astype(float)/(TP+FP)
        
        if self.force_complete_coverage or no_select==True:
            supp_select = np.array([i for i in supp_select])
        else:
            supp_select = np.array([i for i in supp_select if p1[i]>np.mean(y)])
        select = np.argsort(p1[supp_select])[::-1][:N].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        RMatrix = np.array(Z[:,ind])
        rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules]
        supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind]
        precision_matrix = np.array(np.multiply(Z[:,ind], p1[ind]))
        

        return rules, RMatrix, supp, p1[ind], FP[ind], precision_matrix
    

    def get_rule_training_precision(self, the_rule, pos, prules, nrules):
        dfn = 1-self.df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in self.df.columns]
        df_test = pd.concat([self.df,dfn],axis = 1)

        if len(prules):
            p = [[] for rule in prules]
            
            
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
                

            p = (np.sum(p,axis=0)>0).astype(int)   
        else:
            p = np.zeros(len(self.Yb))

        if len(nrules):
            n = [[] for rule in nrules]
            
            
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
                

            n = (np.sum(n,axis=0)>0).astype(int)   
        else:
            n = np.zeros(len(self.Yb))
        

        n = n.astype(bool)
        p = p.astype(bool)

        
        
        covered = (np.sum(df_test[list(the_rule)],axis=1)==len(the_rule)).astype(bool)
        
        if pos:
            precision = (self.Y[covered] == 1).sum()/sum(covered)
        else:
            if (sum(covered) == len(covered)) & len(nrules)==2: #means the negative ruleset is default coverage of everything not positive from brs
                precision = (self.Y[~p] == 0)/sum(~p)
            elif sum(covered & ~p) == 0 :
                precision = 0
            else:
                precision = (self.Y[covered & ~p] == 0).sum()/sum(covered & ~p)
        
        return precision

    def get_model_conf_agreement(self, df, Yb, prs_min=None, nrs_min=None):
        if prs_min == None:
            prs_min = self.prs_min
        if nrs_min == None: 
            nrs_min = self.nrs_min
        #get max confidence of each rule 
        prules = [self.prules[i] for i in prs_min]
        nrules = [self.nrules[i] for i in nrs_min]
        
        
            
        pprecisions = [self.get_rule_training_precision(i, pos=True, prules=prules, nrules=nrules) for i in prules]
        nprecisions = [self.get_rule_training_precision(i, pos=False, prules=prules, nrules=nrules) for i in nrules]
        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1-df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df,dfn],axis = 1)
        if len(prules):
            p = [[] for rule in prules]
            pconfs = [[] for rule in prules]
            
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
                pconfs[i] = p[i].replace(1, pprecisions[i])

            p = (np.sum(p,axis=0)>0).astype(int)
            pconfs = (np.max(pconfs,axis=0))
        else:
            p = np.zeros(len(Yb))
            pconfs = p.copy()
        if len(nrules):
            n = [[] for rule in nrules]
            nconfs = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
                nconfs[i] = n[i].replace(1, nprecisions[i])
            n = (np.sum(n,axis=0)>0).astype(int)
            nconfs = (np.max(nconfs,axis=0))
        else:
            n = np.zeros(len(Yb))
            nconfs = n.copy()
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        confs = nconfs.astype(float)
        confs[pind] = pconfs[pind]
        rulePreds = Yb.copy()
        rulePreds[nind] = 0
        rulePreds[pind] = 1
        agreement = (rulePreds == Yb)


        return confs, agreement
        








    def train(self, Niteration = 500, print_message=False, interpretability = 'size', T0 = 0.01, start_rules=None):

        p_yb = self.p_yb
        p_y = self.p_y
        e_human_responses = self.p_yb.argmax(axis=1)

        t = time.time()
        curr_max_iter = 0
        self.maps = []
        fA = self.fA
        use_paccept=True
        int_flag = int(interpretability =='size')
        T0 = T0
        #find passed in rules in self rules
        nprules = len(self.prules)
        pnrules = len(self.nrules)
        prs_curr = []
        if self.force_complete_coverage:
            nrs_curr = list(range(pnrules))
        else:
            nrs_curr = []
        
        if start_rules != None:
            prs_curr = start_rules['prs']
            nrs_curr = start_rules['nrs']
        
        obj_curr = 1000000000
        obj_min = obj_curr
        self.maps.append([-1,obj_curr,prs_curr,nrs_curr,[]])
        p = np.sum(self.pRMatrix[:,prs_curr],axis = 1)>0
        n = np.sum(self.nRMatrix[:,nrs_curr],axis = 1)>0
        rulePreds_curr = self.Yb.copy()
        rulePreds_curr[n.astype(bool)] = 0
        rulePreds_curr[p.astype(bool)] = 1
        model_conf_curr, _ = self.get_model_conf_agreement(self.df, self.Yb, prs_min=prs_curr, nrs_min=nrs_curr)

        reset responses using expection
        reset = self.expected_loss_filter(self.df, rulePreds_curr, conf_human=self.conf_human, 
                                          prs=prs_curr, nrs=nrs_curr, p_yb=p_yb, p_y=p_y, 
                                          e_human_responses=e_human_responses, conf_model=model_conf_curr)
        p[reset] = False
        n[reset] = False
        rulePreds_curr[reset] = self.Yb[reset]


        fairness_curr = 0
        fairness_new = 0

        
        overlap_curr = np.multiply(p,n)
        pcovered_curr = p
        ncovered_curr = n ^ overlap_curr

        covered_curr = np.logical_xor(p,n) + overlap_curr

    

        
        
        
        Yhat_curr,TP,FP,TN,FN, numRejects_curr, Yhat_soft_curr  = self.compute_obj(pcovered_curr,ncovered_curr, model_conf_curr, use_paccept=use_paccept)

        #print(Yhat_curr,TP,FP,TN,FN)
        nfeatures = len(np.unique([con.split('_')[0] for i in prs_curr for con in self.prules[i]])) + len(np.unique([con.split('_')[0] for i in nrs_curr for con in self.nrules[i]]))
        asymCosts = self.Y.replace({0: self.asym_loss[1], 1: self.asym_loss[0]}) 
        
        

        
       

        err_curr = (np.abs(self.Y - Yhat_soft_curr) * asymCosts).sum()
        err_curr = (np.abs(self.Y - Yhat_soft_curr)).sum()

        #post_hoc filtering optimization adjustment
        contras_curr = np.sum(self.Yb != rulePreds_curr)
        contradictions_curr = (self.Yb != rulePreds_curr)
        


        if self.fairness_reg > 0:
            sensitive = self.df[self.fairness_feature] == 1
            _,TP,FP,TN,FN, _, _ = self.compute_obj(pcovered_curr,ncovered_curr, model_conf_curr, sensitive, use_paccept=use_paccept)
            accuracy_sensitive = TP + TN / TP + FP + TN + FN

            not_sensitive = self.df[self.fairness_feature] == 0
            _,TP,FP,TN,FN, _, _ = self.compute_obj(pcovered_curr,ncovered_curr, model_conf_curr, not_sensitive, use_paccept=use_paccept)
            accuracy_not_sensitive = TP + TN / TP + FP + TN + FN

            fairness_curr = np.abs(accuracy_sensitive-accuracy_not_sensitive)
        else:
            fairness_curr = 0
            fairness_new = 0

        
        


        obj_curr = (err_curr)/self.N + (self.fairness_reg * (fairness_curr)) + (self.contradiction_reg*(contras_curr/self.N))+ self.alpha*(int_flag *(len(prs_curr) + len(nrs_curr))+(1-int_flag)*nfeatures)+ self.beta * sum(~covered_curr)/self.N
        
        
        self.actions = []
        for iter in range(Niteration):
            if iter >0.75 * Niteration:
                prs_curr,nrs_curr,pcovered_curr,ncovered_curr,overlap_curr,covered_curr, Yhat_curr = prs_opt[:],nrs_opt[:],pcovered_opt[:],ncovered_opt[:],overlap_opt[:],covered_opt[:], Yhat_opt[:]
            #print("currp: {}, currn: {}, curr_err: {}, curr_covered: {}, curr_contras: {}, curr_obj: {}".format(prs_curr, nrs_curr, err_curr, covered_curr.sum(), contras_curr, err_curr + (contras_curr * self.contradiction_reg)))
            if iter < -100:
                use_paccept = False
            elif iter == -100:
                print('switching objective to use paccept')
                use_paccept = True
                Yhat_new,TP,FP,TN,FN, numRejects_new, Yhat_soft_new = self.compute_obj(pcovered_new,ncovered_new, model_conf_new, use_paccept=use_paccept)
                err_new = (np.abs(self.Y - Yhat_soft_new) * asymCosts).sum()
                self.Yhat_new = Yhat_new
                rulePreds_new = self.Yb.copy()
                rulePreds_new[ncovered_new] = 0
                rulePreds_new[pcovered_new] = 1
                contras_new = np.sum(rulePreds_new != self.Yb)
                nfeatures = len(np.unique([con.split('_')[0] for i in prs_new for con in self.prules[i]])) + len(np.unique([con.split('_')[0] for i in nrs_new for con in self.nrules[i]]))
                obj_new = (err_new)/self.N + (self.fairness_reg * (fairness_new)) + (self.contradiction_reg*(contras_new/self.N))+self.alpha*(int_flag *(len(prs_new) + len(nrs_new))+(1-int_flag)*nfeatures)+ self.beta * sum(~covered_new)/self.N
                self.maps.append([iter,obj_new,prs_new,nrs_new])
            
            if (iter == 0) and (start_rules != None):
                prs_new = start_rules['prs']
                nrs_new = start_rules['nrs']
                p = np.sum(self.pRMatrix[:,prs_curr],axis = 1)>0
                n = np.sum(self.nRMatrix[:,nrs_curr],axis = 1)>0
                overlap_new = np.multiply(p,n)
                pcovered_new = p
                ncovered_new = n ^ overlap_new
                covered_new = np.logical_xor(p,n) + overlap_new
            else:
                prs_new,nrs_new , pcovered_new,ncovered_new,overlap_new,covered_new= self.propose_rs(prs_curr,nrs_curr,pcovered_curr,ncovered_curr,overlap_curr,covered_curr, Yhat_curr, Yhat_soft_curr, contras_curr, obj_min,print_message, use_paccept)
                rulePreds_new = self.Yb.copy()
                rulePreds_new[ncovered_new] = 0
                rulePreds_new[pcovered_new] = 1
                model_conf_new, _ = self.get_model_conf_agreement(self.df, self.Yb, prs_min=prs_new, nrs_min=nrs_new)
               reset = self.expected_loss_filter(self.df, rulePreds_new, conf_human=self.conf_human, 
                                                  prs=prs_new, nrs=nrs_new, p_yb=p_yb, p_y=p_y, 
                                                  e_human_responses=e_human_responses,conf_model=model_conf_new)
                rulePreds_new[reset] = self.Yb[reset]
                pcovered_new[reset] = 0
                ncovered_new[reset] = 0
                overlap_new[reset] = 0
                covered_new[reset] = 0

            if len(prs_new) > 0:
                p_model_conf_new = np.max(self.p_precision_matrix[:,prs_new],axis = 1)
            else: 
                p_model_conf_new = np.zeros(len(pcovered_new))
            if len(nrs_new) > 0:
                n_model_conf_new = np.max(self.n_precision_matrix[:,nrs_new],axis = 1)
            else:
                n_model_conf_new = np.zeros(len(ncovered_new)) 


            self.covered1 = covered_new[:]
            self.Yhat_curr = Yhat_curr
            #model_conf_new, _ = self.get_model_conf_agreement(self.df, self.Yb, prs_min=prs_new, nrs_min=nrs_new)
            Yhat_new,TP,FP,TN,FN, numRejects_new, Yhat_soft_new = self.compute_obj(pcovered_new,ncovered_new, model_conf_new, use_paccept=use_paccept)
            err_new = (np.abs(self.Y - Yhat_soft_new) * asymCosts).sum()
            
            if self.fairness_reg > 0:
                sensitive = self.df[self.fairness_feature] == 1
                _,TP,FP,TN,FN, _, _ = self.compute_obj(pcovered_new,ncovered_new, model_conf_new, sensitive, use_paccept=use_paccept)
                accuracy_sensitive = TP + TN / TP + FP + TN + FN

                not_sensitive = self.df[self.fairness_feature] == 0
                _,TP,FP,TN,FN, _, _ = self.compute_obj(pcovered_new,ncovered_new, model_conf_new, not_sensitive, use_paccept=use_paccept)
                accuracy_not_sensitive = TP + TN / TP + FP + TN + FN
        
                fairness_new = np.abs(accuracy_sensitive-accuracy_not_sensitive)

            self.Yhat_new = Yhat_new
            contras_new = np.sum(rulePreds_new != self.Yb)
            nfeatures = len(np.unique([con.split('_')[0] for i in prs_new for con in self.prules[i]])) + len(np.unique([con.split('_')[0] for i in nrs_new for con in self.nrules[i]]))
            obj_new = (err_new)/self.N + (self.fairness_reg * (fairness_new)) + (self.contradiction_reg*(contras_new/self.N))+self.alpha*(int_flag *(len(prs_new) + len(nrs_new))+(1-int_flag)*nfeatures)+ self.beta * sum(~covered_new)/self.N
            T = T0**(iter/Niteration)
            alpha = np.exp(float(-obj_new +obj_curr)/T) # minimize
            

            if (obj_new < self.maps[-1][1]):
                prs_opt,nrs_opt,obj_opt,pcovered_opt,ncovered_opt,overlap_opt,covered_opt, Yhat_opt, numRejects_opt, err_opt, Yhat_soft_opt, contras_opt = prs_new[:],nrs_new[:],obj_new,pcovered_new[:],ncovered_new[:],overlap_new[:],covered_new[:], Yhat_new[:], numRejects_new, err_new, Yhat_soft_new.copy(), contras_new
                perror, nerror, oerror, berror = self.diagnose(pcovered_new,ncovered_new^overlap_new,overlap_new,covered_new,Yhat_new)
                accuracy_min = float(TP+TN)/self.N
                explainability_min = sum(covered_new)/self.N
                covered_min = covered_new
                
                print('\n**  max at iter = {} ** \n {}(obj) = {}(error) + {}(coverage) + {}(rejection)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(iter,round(obj_new,3), err_new/self.N, (self.fairness_reg * (covered_new.sum()/self.N)), (self.contradiction_reg*(contras_new/self.N)), (TP+TN+0.0)/self.N,sum(covered_new)/self.N,nfeatures,perror,nerror,oerror,berror ))
                print(f'time from last max: {time.time() - t}, seconds per iteration: {(time.time() - t)/(max(1,iter-curr_max_iter))}')
                t=time.time()
                curr_max_iter = iter

                
                self.maps.append([iter,obj_new,prs_new,nrs_new])
            
            
            if print_message:
                perror, nerror, oerror, berror = self.diagnose(pcovered_new,ncovered_new^overlap_new,overlap_new,covered_new,Yhat_new)
                if print_message:
                    print('\niter = {}, alpha = {}, {}(obj) = {}(error) + {}(coverage) + {}(rejection)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(iter,round(alpha,2),round(obj_new,3),(FP+FN)/self.N, (self.fairness_reg * (covered_new.sum()/self.N)), (self.contradiction_reg*(contras_new/self.N)), (TP+TN+0.0)/self.N,sum(covered_new)/self.N, nfeatures,perror,nerror,oerror,berror ))
                    print('prs = {}, nrs = {}'.format(prs_new, nrs_new))
            if random() <= alpha:
                #look here maybe
                prs_curr, nrs_curr, obj_curr, pcovered_curr, ncovered_curr, overlap_curr, covered_curr, Yhat_curr = prs_new[:],nrs_new[:],obj_new,pcovered_new[:],ncovered_new[:],overlap_new[:],covered_new[:], Yhat_new[:]
                err_curr = err_new
                contras_curr = contras_new
                Yhat_soft_curr = Yhat_soft_new



            self.prs_min = prs_opt
            self.nrs_min = nrs_opt
            self.pcovered_opt = pcovered_opt
            self.ncovered_opt = ncovered_opt
        
            



        return self.maps,accuracy_min,covered_min

    def diagnose(self, pcovered, ncovered, overlapped, covered, Yhat):
        perror = sum(self.Y[pcovered]!=Yhat[pcovered])
        nerror = sum(self.Y[ncovered]!=Yhat[ncovered])
        oerror = sum(self.Y[overlapped]!=Yhat[overlapped])
        berror = sum(self.Y[~covered]!=Yhat[~covered])
        return perror, nerror, oerror, berror
    

    def compute_obj(self,pcovered,ncovered, confs, group=None, use_paccept=True):

        Yhat = self.Yb.copy()  # will cover all cases where model does not have recommendation
        Yhat_rules = self.Yb.copy()
        Yhat_rules[ncovered] = 0
        Yhat_rules[pcovered] = 1
        Yhat = Yhat.astype(float)

        conf_model = confs
        agreement = (Yhat_rules == self.Yb)
        if use_paccept:
            self.Paccept = self.fA(self.conf_human, conf_model, agreement)
        else:
            self.Paccept = np.ones(len(Yhat))
        rejection = self.Paccept <= 0.5
        numRejects = 0
        incorrectRejects = 0
        correctRejects = 0
        numRejects += sum(self.Yb[ncovered & rejection] == 1)
        numRejects += sum(self.Yb[pcovered & rejection] == 0)
        incorrectRejects += sum((self.Yb[ncovered & rejection] == 1) & (self.Y[ncovered & rejection] != 1))
        incorrectRejects += sum((self.Yb[pcovered & rejection] != 1) & (self.Y[pcovered & rejection] == 1))
        correctRejects += sum((self.Yb[ncovered & rejection] == 1) & (self.Y[ncovered & rejection] == 1))
        correctRejects += sum((self.Yb[pcovered & rejection] != 1) & (self.Y[pcovered & rejection] != 1))
    
    
        #Yhat[ncovered] = (Yhat[ncovered] * (1 - np.maximum(np.zeros(sum(ncovered)),(1-self.asym_accept)*self.Paccept[ncovered]))) + ((np.maximum(np.zeros(sum(ncovered)),(1-self.asym_accept)*self.Paccept[ncovered])) * 0)  # covers cases where model predicts negative
        #Yhat[pcovered] = (self.Yb.copy()[pcovered] * (1 - np.minimum(np.ones(sum(pcovered)),(1+self.asym_accept)*self.Paccept[pcovered]))) + ((np.minimum(np.ones(sum(pcovered)),(1+self.asym_accept)*self.Paccept[pcovered])) * 1)  # covers cases where model predicts positive
        
        Yhat[ncovered] = (Yhat[ncovered] * (1 - self.Paccept[ncovered])) + ((self.Paccept[ncovered]) * 0)  # covers cases where model predicts negative
        Yhat[pcovered] = (self.Yb.copy()[pcovered] * (1 - self.Paccept[pcovered])) + ((self.Paccept[pcovered]) * 1)  # covers cases where model predicts positive

        
        
        Yhat_soft = Yhat.copy()
        Yhat = Yhat.astype(float)
        Yhat[ncovered] = Yhat[ncovered].round()
        Yhat[pcovered] = Yhat[pcovered].round()
        TP,FP,TN,FN = getConfusion(Yhat,self.Y)

        if group is not None:
            TP,FP,TN,FN = getConfusion(Yhat[group],self.Y[group])





        return  Yhat,TP,FP,TN,FN, numRejects, Yhat_soft
    
    def propose_rs(self, prs_in,nrs_in,pcovered,ncovered,overlapped, covered,Yhat, Yhat_soft, contras, vt,print_message = False, use_paccept=True):
        prs = prs_in.copy()
        nrs = nrs_in.copy()
        Yhat = pd.Series(Yhat)
        if len(prs_in) > 0:
            pconfs = np.max(self.p_precision_matrix[:,prs_in],axis = 1)
        else: 
            pconfs = np.zeros(len(pcovered))
        if len(nrs_in) > 0:
            nconfs = np.max(self.n_precision_matrix[:,nrs_in],axis = 1)
        else:
            nconfs = np.zeros(len(ncovered))
        
        #overlapped_ind = np.where(overlapped)[0]
        p = np.sum(self.pRMatrix[:,prs],axis = 1)>0
        n = np.sum(self.nRMatrix[:,nrs],axis = 1)>0
        covered = p | n
        ex = -1
        conf_model = nconfs
        conf_model[p] = pconfs[p]

        rulePreds = self.Yb.copy()
        rulePreds[ncovered] = 0
        rulePreds[pcovered] = 1
        incorr = np.where((rulePreds!=self.Y) & covered)[0] # correct interpretable models
        incorrb = np.where((rulePreds!=self.Y) & ~covered)[0] 
        asymCosts = self.Y.replace({0: self.asym_loss[1], 1: self.asym_loss[0]})
        agreement = (rulePreds == self.Yb)
        if use_paccept:
            self.Paccept = self.fA(self.conf_human, conf_model, agreement)
        else:
            self.Paccept = np.ones(len(self.Y))
        err = (np.abs(self.Y - Yhat) * self.Paccept) + (np.abs(self.Y - self.Yb) * (1-self.Paccept)) * asymCosts
        contras = np.where((rulePreds != self.Yb) & covered & (self.Paccept < self.contradiction_reg))[0] 
        err[contras] += self.contradiction_reg

        #if random() <= 0.5: #randomly allow for top 5% of errors or take max error only
        #    max_errs = np.where((err >= np.quantile(err, 0.95)))[0]
        #elif random() <= 0.5:
        #    max_errs = np.where((err >= max(err)))[0]
        #else: 
        max_errs = np.where((err >= 0))[0]
        




        if sum(covered) == self.N and not(self.force_complete_coverage): # covering all examples.
            if print_message:
                print('===== already covering all examples ===== ')
            # print('0')
            move = ['cut']
            self.actions.append(0)
            if len(prs)==0:
                sign = [0]
            elif len(nrs)==0:
                sign = [1]
            else:
                sign = [int(random()<0.5)]
        else:
            self.actions.append(3)
            if print_message:
                print(' ===== decrease objective ===== ')
            # old version ex = sample(list(incorr) + list(incorrb),1)[0] #sample
            try:
                ex = choices(population=max_errs, weights=err[max_errs], k=1)[0]
            except:
                ex = sample(list(max_errs), 1)[0]
            #ex = sample(list(max_errs), 1)[0]  
            #to_draw = list(set(incorr).union(set(incorrb)).union(set(contras)))
            #ex = sample(to_draw, 1)[0]

            if (ex in incorr) or (ex in contras):  # incorrectly classified by interpretable model
                rs_indicator = (p[ex]).astype(int)  # covered by prules
                if self.force_complete_coverage:
                    if rs_indicator:
                        move = ['cut']
                        sign = [rs_indicator]
                    else:
                        move = ['add']
                        sign = [1]
                else:
                    if random() < 0.5 or (ex not in incorr):
                        # print('7')
                        move = ['cut']
                        sign = [rs_indicator]
                    else:
                        # print('8')
                        move = ['cut', 'add']
                        sign = [rs_indicator, rs_indicator]
            else:  # incorrectly classified by the human/not covered
                # print('9')
                move = ['add']
                sign = [int(self.Y[ex] == 1)]

        #if (ex in incorr) or (ex in contras):
        #    print('signs: {}, prs_old:{}, nrs_old:{}'.format(sign, prs, nrs))
        for j in range(len(move)):
            if sign[j]==1:
                prs = self.action(move[j],sign[j],ex,prs,Yhat,pcovered)
            else:
                nrs = self.action(move[j],sign[j],ex,nrs,Yhat,ncovered)
        #if (ex in incorr) or (ex in contras):
        #    print('prs_new:{}, nrs_new:{}'.format(prs, nrs))
        

        p = np.sum(self.pRMatrix[:,prs],axis = 1)>0
        n = np.sum(self.nRMatrix[:,nrs],axis = 1)>0
        o = np.multiply(p,n)
        return prs, nrs, p, n ^ o, o, np.logical_xor(p,n) + o

    def action(self,move, rs_indicator, ex, rules_in ,Yhat,covered):
        rules = rules_in.copy()
        if rs_indicator==1:
            RMatrix = self.pRMatrix
            # error = self.perror
            supp = self.psupp
        else:
            RMatrix = self.nRMatrix
            # error = self.nerror
            supp = self.nsupp
        Y = self.Y if rs_indicator else 1- self.Y
        if move=='cut' and len(rules)>0:
            # print('======= cut =======')
            """ cut """
            if random()<0.25 and ex >=0:
                candidate = list(set(np.where(RMatrix[ex,:]==1)[0]).intersection(rules))
                if len(candidate)==0:
                    candidate = rules
                cut_rule = sample(candidate,1)[0]
            else:
                p = []
                all_sum = np.sum(RMatrix[:,rules],axis = 1)
                for index,rule in enumerate(rules):
                    Yhat= ((all_sum - np.array(RMatrix[:,rule]))>0).astype(int)
                    TP,FP,TN,FN  = getConfusion(Yhat,Y)
                    p.append(TP.astype(float)/(TP+FP+1))
                    # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p,0,0)
                p = np.array(list(accumulate(p)))
                if p[-1]==0:
                    cut_rule = sample(rules,1)[0]
                else:
                    p = p/p[-1]
                    index = find_lt(p,random())
                    cut_rule = rules[index]
            rules.remove(cut_rule)
        elif move == 'add' and ex>=0: 
            # print('======= add =======')
            """ add """
            score_max = -self.N *10000000
            if self.Y[ex]*rs_indicator + (1 - self.Y[ex])*(1 - rs_indicator)==1:
                # select = list(np.where(RMatrix[ex] & (error +self.alpha*self.N < self.beta * supp))[0]) # fix
                select = list(np.where(RMatrix[ex])[0])
            else:
                # select = list(np.where( ~RMatrix[ex]& (error +self.alpha*self.N < self.beta * supp))[0])
                select = list(np.where( ~RMatrix[ex])[0])
            self.select = select
            if len(select)>0:
                if random()<0.25:
                    add_rule = sample(select,1)[0]
                else: 

                    for ind in select:
                        z = np.logical_or(RMatrix[:,ind],Yhat)
                        TP,FP,TN,FN = getConfusion(z,self.Y)
                        score = FP+FN -self.beta * sum(RMatrix[~covered ,ind])
                        if score > score_max:
                            score_max = score
                            add_rule = ind
                if add_rule not in rules:
                    rules.append(add_rule)
        else: 
            candidates = [x for x in range(RMatrix.shape[1])]
            if rs_indicator:
                select = list(set(candidates).difference(rules))
            else:
                select = list(set(candidates).difference(rules))
            self.supp = supp
            self.select = select
            self.candidates = candidates
            self.rules = rules
            if random()<0.25:
                add_rule = sample(select, 1)[0]
            else:
                # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                Yhat_neg_index = np.where(~covered)[0]
                mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(), np.array(Y[Yhat_neg_index]))
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat,axis = 1)
                FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                TN = np.sum(Y[Yhat_neg_index]==0)-FP
                FN = sum(Y[Yhat_neg_index]) - TP
                score = (FP + FN)+ self.beta * (TN + FN)
                # score = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select] # using precision as the criteria
                add_rule = select[sample(list(np.where(score==min(score))[0]),1)[0]] 
            if add_rule not in rules:
                rules.append(add_rule)
        return rules

    def print_rules(self, rules_max):
        for rule_index in rules_max:
            print(self.rules[rule_index])

    def predict_text(self,df,Y,Yb):
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]
        if len(prules):
            p = [[] for rule in prules]
            for i,rule in enumerate(prules):
                p[i] = np.array((np.sum(df[:,list(rule)],axis=1)==len(rule)).flatten().tolist()[0]).astype(int)
            p = (np.sum(p,axis=0)>0).astype(int)
        else:
            p = np.zeros(len(Y))
        if len(nrules):
            n = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = np.array((np.sum(df[:,list(rule)],axis=1)==len(rule)).flatten().tolist()[0]).astype(int)
            n = (np.sum(n,axis=0)>0).astype(int)
        else:
            n = np.zeros(len(Y))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = [x for x in range(len(Y)) if x in pind or x in nind]
        Yhat = Yb
        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat,covered,Yb

    def predict(self, df, Yb, with_reset=False, conf_human=None, p_yb=None, p_y=None):
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]
        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1-df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df,dfn],axis = 1)
        if len(prules):
            p = [[] for rule in prules]
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            p = (np.sum(p,axis=0)>0).astype(int)
        else:
            p = np.zeros(len(Yb))
        if len(nrules):
            n = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            n = (np.sum(n,axis=0)>0).astype(int)
        else:
            n = np.zeros(len(Yb))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = [x for x in range(len(Yb)) if x in pind or x in nind]
        Yhat = Yb.copy()
        Yhat[nind] = 0
        Yhat[pind] = 1

        if with_reset:
            reset = self.expected_loss_filter(df, Yhat, conf_human, p_yb=p_yb, p_y=p_y)


            Yhat[reset] = Yb[reset]
            covered = [i for i in covered if i not in np.where(reset==True)[0]]


        return Yhat,covered,Yb

    def expected_loss_filter(self, x, y_rules, conf_human, prs=None, nrs=None, p_yb=None, p_y=None, e_human_responses=None, conf_model=None):
        if prs is None:
            prs = self.prs_min
        if nrs is None:
            nrs = self.nrs_min
        if e_human_responses is None:
            e_human_responses = p_yb.argmax(axis=1)
        if conf_model is None:
            conf_model, agreement = self.get_model_conf_agreement(x, e_human_responses, prs_min=prs, nrs_min=nrs)
        agreement0 = (y_rules == 0) #hypothetical if human chooses 0
        agreement1 = (y_rules == 1) #hypothetical if human chooses 1
        conf_human0 = conf_human.copy()
        conf_human1 = conf_human.copy()
        conf_human0[e_human_responses == 1] = -conf_human0[e_human_responses == 1] #confidence inverted if human expected to choose 1 because situation is hypothetical human chooses 0
        conf_human1[e_human_responses == 0] = -conf_human1[e_human_responses == 0] #confidence inverted if human expected to choose 0 because situation is hypothetical human chooses 1

        p_a = (p_yb[:, 0]*self.fA(conf_human0, conf_model, agreement0)) + (p_yb[:, 1]*self.fA(conf_human1, conf_model, agreement1))

        

        e_loss_from_accept = p_a*((p_y[:, 0]*(y_rules == 1).astype(int)*self.asym_loss[1]) + p_y[:, 1]*(y_rules == 0).astype(int)*self.asym_loss[0])
        e_loss_from_reject = (1-p_a)*((p_y[:,0]*p_yb[:,1]*self.asym_loss[1]) + p_y[:, 1]*p_yb[:,0]*self.asym_loss[0])
        e_loss_from_contradict = self.contradiction_reg*((p_yb[:,0]*(y_rules==1).astype(int)) + p_yb[:,1]*(y_rules==0).astype(int))

        e_loss_from_advising = e_loss_from_accept + e_loss_from_reject + e_loss_from_contradict

        e_loss_from_withholding = p_y[:,0]*p_yb[:,1]*self.asym_loss[1] + p_y[:,1]*p_yb[:,0]*self.asym_loss[0]

        reset = e_loss_from_withholding < e_loss_from_advising

        return reset

        


    def predictSoft(self, df, Yb, conf_human, fA, with_reset=False, p_yb=None, p_y=None):
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]
        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1-df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df,dfn],axis = 1)
        if len(prules):
            p = [[] for rule in prules]
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            p = (np.sum(p,axis=0)>0).astype(int)
        else:
            p = np.zeros(len(Yb))
        if len(nrules):
            n = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            n = (np.sum(n,axis=0)>0).astype(int)
        else:
            n = np.zeros(len(Yb))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        pcovered = [x for x in range(len(Yb)) if x in pind]
        ncovered = [x for x in range(len(Yb)) if x in nind]
        covered = [x for x in range(len(Yb)) if x in pind or x in nind]
        Yhat = np.array([i for i in Yb])
        Yhat = Yhat.astype(float)

        conf_model, agreement = self.get_model_conf_agreement(df, Yb)
        
        paccept = fA(conf_human, conf_model, agreement)
        
        Yhat[nind] = Yhat[nind] * (1 - paccept[nind]) + (paccept[nind]) * 0  # covers cases where model predicts negative
        Yhat[pind] = (Yb.copy()[pind] * (1 - paccept[pind])) + ((paccept[pind]) * 1)  # covers cases where model predicts positive

        if with_reset:
            reset = self.expected_loss_filter(df, Yhat, conf_human, p_yb=p_yb, p_y=p_y)

            Yhat[reset] = Yb[reset]
            covered = [i for i in covered if i not in np.where(reset==True)[0]]

        

        


        
        return Yhat,covered,Yb
    
    def make_lite(self):
        self.df = None
        self.Y = None
        self.pRMatrix = None
        self.nRMatrix = None
        self.p_precision_matrix = None
        self.n_precision_matrix = None

        return 0 

    def predictHumanInLoop(self, df, Yb, conf_human, fA, with_reset=False, p_yb=None, p_y=None):
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]
        dfn = 1-df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df,dfn],axis = 1)
        if len(prules):
            p = [[] for rule in prules]
            pconfs = [[] for rule in prules]
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            p = (np.sum(p,axis=0)>0).astype(bool)
        else:
            p = np.zeros(len(Yb)).astype(bool)
        if len(nrules):
            n = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            n = (np.sum(n,axis=0)>0).astype(bool)
        else:
            n = np.zeros(len(Yb)).astype(bool)

        
        conf_model, agreement = self.get_model_conf_agreement(df, Yb)
        
        paccept = fA(conf_human, conf_model, agreement)
        accept = bernoulli.rvs(paccept, size=len(paccept)).astype(bool)

        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        acceptind = list(np.where(accept)[0])
        pind = intersection(pind, acceptind)
        nind = intersection(nind, acceptind)
        covered = Yb.copy()
        covered[:] = -1
        covered[n] = 0
        covered[p] = 1
        Yhat = np.array([i for i in Yb])
        Yhat[nind] = 0
        Yhat[pind] = 1

        if with_reset:
            reset = self.expected_loss_filter(df, Yhat, conf_human, p_yb=p_yb, p_y=p_y)


            Yhat[reset] = Yb[reset]
            covered[reset] = 0

        return Yhat,covered,Yb

def accumulate(iterable, func=operator.add):
    'Return running totals'
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
    else:
        return 0


def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN


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

def binary_code(df,collist,Nlevel):
    for col in collist:
        for q in range(1,Nlevel,1):
            threshold = df[col].quantile(float(q)/Nlevel)
            df[col+'_geq_'+str(int(q))+'q'] = (df[col] >= threshold).astype(float)
    df.drop(collist,axis = 1, inplace = True)







    
