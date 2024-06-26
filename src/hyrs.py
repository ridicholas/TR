import pandas as pd
#from fim import fpgrowth, fim
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


class hyrs(object):
    def __init__(self, binary_data, Y, Yb):
        self.df = binary_data
        self.Y = Y
        self.N = float(len(Y))
        self.Yb = Yb

    def set_parameters(self, alpha=1, beta=0.1, coverage_reg=0, contradiction_reg=0, force_complete_coverage=False, asym_loss = [1,1]):
        """
        asym_loss = [loss from False Negatives (ground truth positive), loss from False Positives (ground truth negative)]
        """
        # input al and bl are lists
        self.alpha = alpha
        self.beta = beta
        self.coverage_reg = 0
        self.contradiction_reg = contradiction_reg
        self.force_complete_coverage = force_complete_coverage
        self.asym_loss = asym_loss

    def generate_rulespace(self, supp, maxlen, N, need_negcode=False, njobs=5, method='fpgrowth', criteria='IG',
                           add_rules=[]):
        if method == 'fpgrowth':
            if need_negcode:
                df = 1 - self.df
                df.columns = [name.strip() + 'neg' for name in self.df.columns]
                df = pd.concat([self.df, df], axis=1)
            else:
                df = 1 - self.df
            pindex = np.where(self.Y == 1)[0]
            nindex = np.where(self.Y != 1)[0]
            itemMatrix = [[item for item in df.columns if row[item] == 1] for i, row in df.iterrows()]
            prules = fpgrowth([itemMatrix[i] for i in pindex], supp=supp, zmin=1, zmax=maxlen)
            prules = [np.sort(x[0]).tolist() for x in prules]
            nrules = fpgrowth([itemMatrix[i] for i in nindex], supp=supp, zmin=1, zmax=maxlen)
            nrules = [np.sort(x[0]).tolist() for x in nrules]
        else:
            print('Using random forest to generate rules ...')
            prules = []
            for length in range(max(min(2,maxlen), 1), maxlen + 1, 1):
                n_estimators = 250 * length  # min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length)
                clf.fit(self.df, self.Y)
                for n in range(n_estimators):
                    prules.extend(extract_rules(clf.estimators_[n], self.df.columns))
            prules = [list(x) for x in set(tuple(np.sort(x)) for x in prules)]
            if self.force_complete_coverage:
                nrules = [[self.df.columns[0]], [self.df.columns[0] + 'neg']]
            else:
                nrules = []
                for length in range(max(min(2,maxlen), 1), maxlen + 1, 1):
                    n_estimators = 250 * length  # min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length)
                    clf.fit(self.df, 1 - self.Y)
                    for n in range(n_estimators):
                        nrules.extend(extract_rules(clf.estimators_[n], self.df.columns))
                nrules = [list(x) for x in set(tuple(np.sort(x)) for x in nrules)]
            df = 1 - self.df
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df, df], axis=1)
        self.prules, self.pRMatrix, self.psupp, self.pprecision, self.perror, self.p_precision_matrix = self.screen_rules(prules, df, self.Y, N,
                                                                                                 supp)
        if self.force_complete_coverage:
            self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror, self.n_precision_matrix = self.screen_rules(nrules, df, 1 - self.Y,
                                                                                                 N, 0)
        else:
            self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror, self.n_precision_matrix = self.screen_rules(nrules, df, 1 - self.Y,
                                                                                                 N, supp)

        # print '\tTook %0.3fs to generate %d rules' % (self.screen_time, len(self.rules))

    def screen_rules(self, rules, df, y, N, supp, criteria='precision', njobs=5, add_rules=[]):
        # print 'screening rules'
        start_time = time.time()
        itemInd = {}
        for i, name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr = list(accumulate(len_rules))
        indptr.insert(0, 0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data, indices, indptr), shape=(len(df.columns), len(rules)))
        # mat = sparse.csr_matrix.dot(df,ruleMatrix)
        mat = np.matrix(df) * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z = (mat == lenMatrix).astype(int)

        Zpos = [Z[i] for i in np.where(y > 0)][0]
        TP = np.array(np.sum(Zpos, axis=0).tolist()[0])
        supp_select = np.where(TP >= supp * sum(y) / 100)[0]
        # if len(supp_select)<=N:
        #     rules = [rules[i] for i in supp_select]
        #     RMatrix = np.array(Z[:,supp_select])
        #     rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules]
        #     supp = np.array(np.sum(Z,axis=0).tolist()[0])[supp_select]
        # else:
        FP = np.array(np.sum(Z, axis=0))[0] - TP
        
        p1 = TP.astype(float) / (TP + FP)
     
        if self.force_complete_coverage:
            supp_select = np.array([i for i in supp_select])
        else:
            supp_select = np.array([i for i in supp_select if p1[i] > np.mean(y)])
        select = np.argsort(p1[supp_select])[::-1][:N].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        RMatrix = np.array(Z[:, ind])
        rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules]
        supp = np.array(np.sum(Z, axis=0).tolist()[0])[ind]
        precision_matrix = np.array(np.multiply(Z[:,ind], p1[ind]))
        return rules, RMatrix, supp, p1[ind], FP[ind], precision_matrix
    

    def get_rule_training_precision(self, the_rule, pos, prules):
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

        
        p = p.astype(bool)

        
        
        covered = (np.sum(df_test[list(the_rule)],axis=1)==len(the_rule)).astype(bool)
        
        if pos:
            precision = (self.Y[covered] == 1).sum()/sum(covered)
        else:
            if sum(covered & ~p) == 0:
                precision = 0
            else:
                precision = (self.Y[covered & ~p] == 0).sum()/sum(covered & ~p)
        
        return precision


    def get_model_conf_agreement(self, df, Yb):
        #get max confidence of each rule 
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]


        pprecisions = [self.get_rule_training_precision(i, pos=True, prules=prules) for i in prules]
        nprecisions = [self.get_rule_training_precision(i, pos=False, prules=prules) for i in nrules]
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

    def train(self, Niteration=5000, print_message=False, interpretability='size', T0=0.01):
        self.maps = []
        int_flag = int(interpretability == 'size')
        T0 = T0
        nprules = len(self.prules)
        pnrules = len(self.nrules)
        prs_curr = sample(list(range(nprules)), 3)
        if self.force_complete_coverage:
            nrs_curr = list(range(pnrules))
        else:
            nrs_curr = sample(list(range(pnrules)), 3)
        obj_curr = 1000000000
        obj_min = obj_curr
        self.maps.append([-1, obj_curr, prs_curr, nrs_curr, []])
        p = np.sum(self.pRMatrix[:, prs_curr], axis=1) > 0
        n = np.sum(self.nRMatrix[:, nrs_curr], axis=1) > 0
        overlap_curr = np.multiply(p, n)
        pcovered_curr = p ^ overlap_curr
        ncovered_curr = n ^ overlap_curr
        covered_curr = np.logical_xor(p, n)
        Yhat_curr, TP, FP, TN, FN, numRejects_curr = self.compute_obj(pcovered_curr, covered_curr)
        contras_curr = np.sum(Yhat_curr != self.Yb)
        #print(Yhat_curr, TP, FP, TN, FN)
        nfeatures = len(np.unique([con.split('_')[0] for i in prs_curr for con in self.prules[i]])) + len(
            np.unique([con.split('_')[0] for i in nrs_curr for con in self.nrules[i]]))
        obj_curr = (FN*self.asym_loss[0] + FP*self.asym_loss[1]) / self.N + (self.coverage_reg * (covered_curr.sum() / self.N)) + (
                self.contradiction_reg * (contras_curr / self.N)) + self.alpha * (
                           int_flag * (len(prs_curr) + len(nrs_curr)) + (1 - int_flag) * nfeatures) + self.beta * sum(
            ~covered_curr) / self.N
        self.actions = []
        for iter in range(Niteration):
            if iter > 0.75 * Niteration:
                prs_curr, nrs_curr, pcovered_curr, ncovered_curr, overlap_curr, covered_curr, Yhat_curr = prs_opt[
                                                                                                          :], nrs_opt[
                                                                                                              :], pcovered_opt[
                                                                                                                  :], ncovered_opt[
                                                                                                                      :], overlap_opt[
                                                                                                                          :], covered_opt[
                                                                                                                              :], Yhat_opt[
                                                                                                                                  :]
            prs_new, nrs_new, pcovered_new, ncovered_new, overlap_new, covered_new = self.propose_rs(prs_curr, nrs_curr,
                                                                                                     pcovered_curr,
                                                                                                     ncovered_curr,
                                                                                                     overlap_curr,
                                                                                                     covered_curr,
                                                                                                     Yhat_curr, obj_min,
                                                                                                     print_message)
            self.covered1 = covered_new[:]
            self.Yhat_curr = Yhat_curr
            # if sum(covered_new)<len(self.Y):
            #     # bbmodel.fit(self.df.iloc[~covered_new],self.Y[~covered_new])
            #     bbmodel.fit(self.df,self.Y)
            Yhat_new, TP, FP, TN, FN, numRejects_new = self.compute_obj(pcovered_new, covered_new)
            self.Yhat_new = Yhat_new
            contras_new = np.sum(self.Yhat_new != self.Yb)
            nfeatures = len(np.unique([con.split('_')[0] for i in prs_new for con in self.prules[i]])) + len(
                np.unique([con.split('_')[0] for i in nrs_new for con in self.nrules[i]]))
            obj_new = (FN*self.asym_loss[0] + FP*self.asym_loss[1]) / self.N + (self.coverage_reg * (covered_new.sum() / self.N)) + (
                    self.contradiction_reg * (contras_new / self.N)) + self.alpha * (
                              int_flag * (len(prs_new) + len(nrs_new)) + (1 - int_flag) * nfeatures) + self.beta * sum(
                ~covered_new) / self.N
            T = T0 ** (iter / Niteration)
            alpha = np.exp(float(-obj_new + obj_curr) / T)  # minimize
            #if (obj_new < self.maps[-1][1]) & ((len(prs_new) + len(nrs_new)) > 0):
            if (obj_new < self.maps[-1][1]):
                prs_opt, nrs_opt, obj_opt, pcovered_opt, ncovered_opt, overlap_opt, covered_opt, Yhat_opt = prs_new[
                                                                                                            :], nrs_new[
                                                                                                                :], obj_new, pcovered_new[
                                                                                                                             :], ncovered_new[
                                                                                                                                 :], overlap_new[
                                                                                                                                     :], covered_new[
                                                                                                                                         :], Yhat_new[
                                                                                                                                             :]
                perror, nerror, oerror, berror = self.diagnose(pcovered_new, ncovered_new, overlap_new, covered_new,
                                                               Yhat_new)
                accuracy_min = float(TP + TN) / self.N
                explainability_min = sum(covered_new) / self.N
                covered_min = covered_new
                print(
                    '\n**  max at iter = {} ** \n {}(obj) = {}(error) + {}(coverage) + {}(rejection) + {}(nrules) + {}(exp)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(
                        iter, round(obj_new, 3), (FN*self.asym_loss[0] + FP*self.asym_loss[1]) / self.N,
                        (self.coverage_reg * (covered_new.sum() / self.N)),
                        (self.contradiction_reg * (contras_new / self.N)), self.alpha * (len(prs_new) + len(nrs_new)),
                                                 self.beta * sum(~covered_new) / self.N, (TP + TN + 0.0) / self.N,
                                                 sum(covered_new) / self.N,
                        nfeatures, perror, nerror, oerror, berror))
                self.maps.append([iter, obj_new, prs_new, nrs_new])

            if print_message:
                perror, nerror, oerror, berror = self.diagnose(pcovered_new, ncovered_new, overlap_new, covered_new,
                                                               Yhat_new)
                print(
                    '\niter = {}, alpha = {}, {}(obj) = {}(error) + {}(coverage) + {}(rejection) {}(nrules) + {}(exp)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(
                        iter, round(alpha, 2), round(obj_new, 3), (FP + FN) / self.N,
                        (self.coverage_reg * (covered_new.sum() / self.N)),
                        (self.contradiction_reg * (contras_new / self.N)),
                                                                  self.alpha * (len(prs_new) + len(nrs_new)),
                                                                  self.beta * sum(~covered_new) / self.N,
                                                                  (TP + TN + 0.0) / self.N, sum(covered_new) / self.N,
                        nfeatures, perror, nerror, oerror, berror))
                print('covered_curr: {}'.format(covered_curr.sum()))
                print('covered_new: {}'.format(covered_new.sum()))
                print('covered_opt: {}'.format(covered_opt.sum()))
                print('prs = {}, nrs = {}'.format(prs_new, nrs_new))
            print(f'alpha: {alpha}')
            if random() <= alpha:
                prs_curr, nrs_curr, obj_curr, pcovered_curr, ncovered_curr, overlap_curr, covered_curr, Yhat_curr = prs_new[
                                                                                                                    :], nrs_new[
                                                                                                                        :], obj_new, pcovered_new[
                                                                                                                                     :], ncovered_new[
                                                                                                                                         :], overlap_new[
                                                                                                                                             :], covered_new[
                                                                                                                                                 :], Yhat_new[
                                                                                                                                                     :]
        self.prs_min = prs_opt
        self.nrs_min = nrs_opt
        return self.maps, accuracy_min, covered_min

    def diagnose(self, pcovered, ncovered, overlapped, covered, Yhat):
        perror = sum(self.Y[pcovered] != Yhat[pcovered])
        nerror = sum(self.Y[ncovered] != Yhat[ncovered])
        oerror = sum(self.Y[overlapped] != Yhat[overlapped])
        berror = sum(self.Y[~covered] != Yhat[~covered])
        return perror, nerror, oerror, berror

    def compute_obj(self, pcovered, covered):
        Yhat = np.zeros(int(self.N))
        Yhat[pcovered] = 1
        #rejection = self.paccept <= 0.5

        Yhat[~covered] = self.Yb[~covered]  # self.Y[~covered]#
        numRejects = 0
        #numRejects += sum(self.Yb[covered & rejection] != Yhat[covered & rejection])

        TP, FP, TN, FN = getConfusion(Yhat, self.Y)
        return Yhat, TP, FP, TN, FN, numRejects

    def propose_rs(self, prs_in, nrs_in, pcovered, ncovered, overlapped, covered, Yhat, vt, print_message=False):
        prs = prs_in.copy()
        nrs = nrs_in.copy()
        incorr = np.where(Yhat[covered] != self.Y[covered])[0]  # correct interpretable models
        incorrb = np.where(Yhat[~covered] != self.Y[~covered])[0]
        overlapped_ind = np.where(overlapped)[0]
        p = np.sum(self.pRMatrix[:, prs], axis=1)
        n = np.sum(self.nRMatrix[:, nrs], axis=1)
        ex = -1
        if sum(covered) == self.N and not(self.force_complete_coverage):  # covering all examples.
            if print_message:
                print('===== already covering all examples ===== ')
            # print('0')
            move = ['cut']
            self.actions.append(0)
            if len(prs) == 0:
                sign = [0]
            elif len(nrs) == 0:
                sign = [1]
            else:
                sign = [int(random() < 0.5)]
        elif (len(incorr) == 0 and (len(incorrb) == 0 or len(overlapped) == self.N) or sum(overlapped) > sum(covered)) and not(self.force_complete_coverage):
            if print_message:
                print(' ===== 1 ===== ')
            self.actions.append(1)
            # print('1')
            move = ['cut']
            sign = [int(random() < 0.5)]
            
        else:
            t = random()
            if True:  # try to decrease objective
                self.actions.append(3)
                if print_message:
                    print(' ===== decrease error ===== ')
                ex = sample(list(incorr) + list(incorrb), 1)[0]
                if ex in incorr:  # incorrectly classified by the interpretable model
                    
                    rs_indicator = (pcovered[ex]).astype(int)  # covered by prules
                    if self.force_complete_coverage:
                        if rs_indicator:
                            move = ['cut']
                            sign = [rs_indicator]
                        else:
                            move = ['add']
                            sign = [1]
                    elif random() < 0.5:
                        # print('7')
                        move = ['cut']
                        sign = [rs_indicator]
                        if self.force_complete_coverage:
                            move.append('add')
                            sign.append(rs_indicator)
                        
                    else:
                        # print('8')
                        move = ['cut', 'add']
                        sign = [rs_indicator, rs_indicator]
                else:  # incorrectly classified by the human
                    # print('9')
                    move = ['add']
                    sign = [int(self.Y[ex] == 1)]

            
        for j in range(len(move)):
            if sign[j] == 1:
                prs = self.action(move[j], sign[j], ex, prs, Yhat, pcovered)
            else:
                nrs = self.action(move[j], sign[j], ex, nrs, Yhat, ncovered)

        p = np.sum(self.pRMatrix[:, prs], axis=1) > 0
        n = np.sum(self.nRMatrix[:, nrs], axis=1) > 0
        o = np.multiply(p, n)
        return prs, nrs, p, n ^ o, o, np.logical_xor(p, n) + o

    def action(self, move, rs_indicator, ex, rules_in, Yhat, covered):
        rules = rules_in.copy()
        if rs_indicator == 1:
            RMatrix = self.pRMatrix
            # error = self.perror
            supp = self.psupp
        else:
            RMatrix = self.nRMatrix
            # error = self.nerror
            supp = self.nsupp
        Y = self.Y if rs_indicator else 1 - self.Y
        if move == 'cut' and len(rules) > 0:
            # print('======= cut =======')
            """ cut """
            if random() < 0.25 and ex >= 0:
                candidate = list(set(np.where(RMatrix[ex, :] == 1)[0]).intersection(rules))
                if len(candidate) == 0:
                    candidate = rules
                cut_rule = sample(candidate, 1)[0]
            else:
                p = []
                all_sum = np.sum(RMatrix[:, rules], axis=1)
                for index, rule in enumerate(rules):
                    Yhat = ((all_sum - np.array(RMatrix[:, rule])) > 0).astype(int)
                    TP, FP, TN, FN = getConfusion(Yhat, Y)
                    p.append(TP.astype(float) / (TP + FP + 1))
                    # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p, 0, 0)
                p = np.array(list(accumulate(p)))
                if p[-1] == 0:
                    cut_rule = sample(rules, 1)[0]
                else:
                    p = p / p[-1]
                    index = find_lt(p, random())
                    cut_rule = rules[index]
            rules.remove(cut_rule)
        elif move == 'add' and ex >= 0:
            # print('======= add =======')
            """ add """
            score_max = -self.N * 10000000
            if self.Y[ex] * rs_indicator + (1 - self.Y[ex]) * (1 - rs_indicator) == 1:
                # select = list(np.where(RMatrix[ex] & (error +self.alpha*self.N < self.beta * supp))[0]) # fix
                select = list(np.where(RMatrix[ex])[0])
            else:
                # select = list(np.where( ~RMatrix[ex]& (error +self.alpha*self.N < self.beta * supp))[0])
                select = list(np.where(~RMatrix[ex])[0])
            self.select = select
            if len(select) > 0:
                if random() < 0.25:
                    add_rule = sample(select, 1)[0]
                else:
                    # cover = np.sum(RMatrix[(~covered)&(~covered2), select],axis = 0)
                    # =============== Use precision as a criteria ===============
                    # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                    # mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),Y[Yhat_neg_index])
                    # TP = np.sum(mat,axis = 1)
                    # FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                    # TN = np.sum(Y[Yhat_neg_index]==0)-FP
                    # FN = sum(Y[Yhat_neg_index]) - TP
                    # p = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select]
                    # add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
                    # =============== Use objective function as a criteria ===============
                    for ind in select:
                        z = np.logical_or(RMatrix[:, ind], Yhat)
                        TP, FP, TN, FN = getConfusion(z, self.Y)
                        score = FP + FN - self.beta * sum(RMatrix[~covered, ind])
                        if score > score_max:
                            score_max = score
                            add_rule = ind
                if add_rule not in rules:
                    rules.append(add_rule)
        else:  # expand
            # print(['======= expand =======', len(rules)])
            # candidates = np.where(error < self.beta * supp-self.alpha*self.N)[0] # fix
            candidates = [x for x in range(RMatrix.shape[1])]
            if rs_indicator:
                select = list(set(candidates).difference(rules))
            else:
                select = list(set(candidates).difference(rules))
            # self.error = error
            self.supp = supp
            self.select = select
            self.candidates = candidates
            self.rules = rules
            if random() < 0.25:
                add_rule = sample(select, 1)[0]
            else:
                # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                Yhat_neg_index = np.where(~covered)[0]
                mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1, 1), select].transpose(),
                                  np.array(Y[Yhat_neg_index]))
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat, axis=1)
                FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1, 1), select], axis=0) - TP)
                TN = np.sum(Y[Yhat_neg_index] == 0) - FP
                FN = sum(Y[Yhat_neg_index]) - TP
                score = (FP + FN) + self.beta * (TN + FN)
                # score = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select] # using precision as the criteria
                add_rule = select[sample(list(np.where(score == min(score))[0]), 1)[0]]
            if add_rule not in rules:
                rules.append(add_rule)
        return rules

    def print_rules(self, rules_max):
        for rule_index in rules_max:
            print(self.rules[rule_index])

    def predict_text(self, df, Y, Yb):
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]
        if len(prules):
            p = [[] for rule in prules]
            for i, rule in enumerate(prules):
                p[i] = np.array((np.sum(df[:, list(rule)], axis=1) == len(rule)).flatten().tolist()[0]).astype(int)
            p = (np.sum(p, axis=0) > 0).astype(int)
        else:
            p = np.zeros(len(Y))
        if len(nrules):
            n = [[] for rule in nrules]
            for i, rule in enumerate(nrules):
                n[i] = np.array((np.sum(df[:, list(rule)], axis=1) == len(rule)).flatten().tolist()[0]).astype(int)
            n = (np.sum(n, axis=0) > 0).astype(int)
        else:
            n = np.zeros(len(Y))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = [x for x in range(len(Y)) if x in pind or x in nind]
        Yhat = Yb
        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat, covered, Yb
    
    def expected_loss_filter(self, x, y_rules, conf_human, p_y=None, e_human_responses=None, conf_model=None, fA=None, asym_loss=[1,1], contradiction_reg=0.0):
   
        conf_model, agreement = self.get_model_conf_agreement(x, e_human_responses)
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

    def predict(self, df, Yb):
        prules = [self.prules[i] for i in self.prs_min]
        nrules = [self.nrules[i] for i in self.nrs_min]
        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1 - df  # df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df, dfn], axis=1)
        if len(prules):
            p = [[] for rule in prules]
            for i, rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)], axis=1) == len(rule)).astype(int)
            p = (np.sum(p, axis=0) > 0).astype(bool)
        else:
            p = np.zeros(len(Yb)).astype(bool)
        if len(nrules):
            n = [[] for rule in nrules]
            for i, rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)], axis=1) == len(rule)).astype(int)
            n = (np.sum(n, axis=0) > 0).astype(bool)
        else:
            n = np.zeros(len(Yb)).astype(bool)
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        # covered = [x for x in range(len(Yb)) if x in pind or x in nind]
        Yhat = np.array([i for i in Yb])
        covered = Yb.copy()
        covered[:] = -1
        covered[n] = 0
        covered[p] = 1

        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat, covered, Yb

    def humanifyPreds(self, preds, Yb, conf_human, fA, X):
        conf_model, agreement = self.get_model_conf_agreement(X, Yb)
        paccept = fA(conf_human, conf_model, agreement)
        accept = bernoulli.rvs(paccept, size=len(paccept)).astype(bool)    
        finalPreds = Yb.copy()
        finalPreds[accept] = preds[accept]
        return finalPreds
    
    def make_lite(self):
        self.df = None
        self.Y = None
        self.pRMatrix = None
        self.nRMatrix = None
        self.p_precision_matrix = None
        self.n_precision_matrix = None

        return 0 


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
        return int(i - 1)
    else:
        return 0


def getConfusion(Yhat, Y):
    if len(Yhat) != len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y), np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y) - FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP, FP, TN, FN


def extract_rules(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

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
        lineage.append((features[parent].strip() + suffix))

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


def binary_code(df, collist, Nlevel):
    for col in collist:
        for q in range(1, Nlevel, 1):
            threshold = df[col].quantile(float(q) / Nlevel)
            df[col + '_geq_' + str(int(q)) + 'q'] = (df[col] >= threshold).astype(float)
    df.drop(collist, axis=1, inplace=True)



