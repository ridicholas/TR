from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, uniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

class Human(object):
    def __init__(self, name, X, y, dataset=None, decision_bias=False, alteration='') -> None:
        self.name = name
        self.alteration=alteration
        self.X = X.dropna(axis=1)
        self.y = y
        self.scaler = StandardScaler().fit(self.X)
        self.model = LogisticRegression().fit(self.scaler.transform(self.X), y)
        self.beta = 0.5
        self.dataset = dataset
        if dataset == 'heart_disease':
            self.confVal = 0.6
        elif dataset == 'fico':
            self.confVal = 0.3
        elif dataset == 'hr':
            self.confVal = 0.8
        elif dataset == 'adult':
            self.confVal = 0.6
        self.decision_bias = decision_bias
        self.learning_indexes = None


    
    def get_confidence(self, X):
        if self.dataset == 'heart_disease':
            return self.heart_confidence_transformation(X=X, t_type=self.name)
        elif self.dataset == 'fico':
            return self.fico_confidence_transformation(X=X, t_type=self.name)
        elif self.dataset == 'hr':
            return self.hr_confidence_transformation(X=X, t_type=self.name)
        elif self.dataset == 'adult':
            return self.adult_confidence_transformation(X=X, t_type=self.name)
        
    def set_confVal(self, val):
        self.confVal = val


    #remember to update back to female for heart! 
    def get_decisions(self, X, y):
        decisions = y.copy()
        if self.decision_bias:
            if self.dataset == 'heart_disease':
                model_confidences = np.ones(X.shape[0])
                ###asymmetric case version###############
                #model_confidences[(X['age54.0'] == 0) | (X['sex_Male'] == 0)] = 0
                #########################################
                ###feature decision and confidence bias###
                #if not(hasattr(self, 'alteration')) or self.alteration == '' or self.alteration == '_dec_bias' or self.alteration == 'quick' or self.alteration == 'quick'or self.alteration.contains('discretion'):
                    #model_confidences[(X['age54.0'] == 0)] = 0
                #else:
                    #model_confidences[(X['sex_Male'] == 1)] = 0
                if ('case1' in self.alteration) or ('case2' in self.alteration):
                    model_confidences[(X['sex_Male'] == 1)] = 0
                elif 'asym' in self.alteration:
                    model_confidences[(y == 0) | (X['age54.0'] == 1) ] = 0 #model_confidences[(X['age50.0'] == 1)] = 0  #model_confidences[(X['age54.0'] == 0) | (X['sex_Male'] == 0)] = 0
                else:
                    model_confidences[(X['age50.0'] == 1)] = 0
                
                #########################################
            if self.dataset == 'fico':
                model_confidences = np.ones(X.shape[0]) 
                model_confidences[(X['NumSatisfactoryTrades24.0'] == 1)] = 0
            if self.dataset == 'adult':
                model_confidences = np.ones(X.shape[0])
                model_confidences[(X['race_White'] == 1)] = 0
            if self.dataset == 'hr':
                model_confidences = np.ones(X.shape[0])
                model_confidences[(X['﻿Age32.0'] == 0)] = 0
            
                
                
        else:
            model_confidences = np.abs(self.model.predict_proba(self.scaler.transform(X))[:, 1] - 0.5)*2
        #low accuracy 60%
        low = bernoulli.rvs(p=0.4, size=len(decisions)).astype(bool)
        #high accuracy 100%
        if self.decision_bias == True:


            #for all other
            if 'case2' in self.alteration:
                high = bernoulli.rvs(p=0.1, size=len(decisions)).astype(bool)
            else:
                high = bernoulli.rvs(p=0.05, size=len(decisions)).astype(bool)

        else:
            high = bernoulli.rvs(p=0.00, size=len(decisions)).astype(bool)
        decisions[high] = 1-decisions[high]
        decisions[(model_confidences > self.confVal) & low] = 1-decisions[(model_confidences > self.confVal) & low]
        #if self.decision_bias & (self.dataset == 'heart_disease'):
            #mid = bernoulli.rvs(p=0.15, size=len(decisions)).astype(bool)
            #decisions[(X['sex_Male'] == 0) & (X['age54.0'] == 0)] = y[(X['sex_Male'] == 0) & (X['age54.0'] == 0)]
            #decisions[(X['sex_Male'] == 0) & (X['age54.0'] == 0) & mid] = 1-decisions[(X['sex_Male'] == 0) & (X['age54.0'] == 0) & mid]

            #mid = bernoulli.rvs(p=0.15, size=len(decisions)).astype(bool)
            #decisions[(X['sex_Male'] == 1) & (X['age54.0'] == 1)] = y[(X['sex_Male'] == 1) & (X['age54.0'] == 1)]
            #decisions[(X['sex_Male'] == 1) & (X['age54.0'] == 1) & mid] = 1-decisions[(X['sex_Male'] == 1) & (X['age54.0'] == 1) & mid]


        return decisions
    
    def get_soft_decisions(self, X, y):
        decisions = y.copy()
        if self.decision_bias:
            if self.dataset == 'heart_disease':
                model_confidences = np.ones(X.shape[0])
                ###asymmetric case version###############
                #model_confidences[(X['age54.0'] == 0) | (X['sex_Male'] == 0)] = 0
                #########################################
                ###feature decision and confidence bias###
                #if not(hasattr(self, 'alteration')) or self.alteration == '' or self.alteration == '_dec_bias' or self.alteration == 'quick' or self.alteration == 'quick'or self.alteration.contains('discretion'):
                    #model_confidences[(X['age54.0'] == 0)] = 0
                #else:
                    #model_confidences[(X['sex_Male'] == 1)] = 0
                if ('case1' in self.alteration) or ('case2' in self.alteration):
                    model_confidences[(X['sex_Male'] == 1)] = 0
                else:
                    model_confidences[(X['age54.0'] == 0)] = 0
                
                #########################################
            if self.dataset == 'fico':
                model_confidences = np.ones(X.shape[0]) 
                model_confidences[(X['NumSatisfactoryTrades24.0'] == 1)] = 0
            if self.dataset == 'adult':
                model_confidences = np.ones(X.shape[0])
                model_confidences[(X['race_White'] == 1)] = 0
            if self.dataset == 'hr':
                model_confidences = np.ones(X.shape[0])
                model_confidences[(X['﻿Age32.0'] == 0)] = 0
            
                
                
        else:
            model_confidences = np.abs(self.model.predict_proba(self.scaler.transform(X))[:, 1] - 0.5)*2
        #low accuracy 60%
        low = bernoulli.rvs(p=0.4, size=len(decisions)).astype(bool)
        #high accuracy 100%
        if self.decision_bias == True:


            #for all other
            high = np.zeros(len(decisions))+0.05 #bernoulli.rvs(p=0.05, size=len(decisions)).astype(bool)

        else:
            high = np.zeros(len(decisions)) #bernoulli.rvs(p=0.00, size=len(decisions)).astype(bool)
        decisions[high] = 1-decisions[high]
        decisions[(model_confidences > self.confVal) & low] = 1-decisions[(model_confidences > self.confVal) & low]
        #if self.decision_bias & (self.dataset == 'heart_disease'):
            #mid = bernoulli.rvs(p=0.15, size=len(decisions)).astype(bool)
            #decisions[(X['sex_Male'] == 0) & (X['age54.0'] == 0)] = y[(X['sex_Male'] == 0) & (X['age54.0'] == 0)]
            #decisions[(X['sex_Male'] == 0) & (X['age54.0'] == 0) & mid] = 1-decisions[(X['sex_Male'] == 0) & (X['age54.0'] == 0) & mid]

            #mid = bernoulli.rvs(p=0.15, size=len(decisions)).astype(bool)
            #decisions[(X['sex_Male'] == 1) & (X['age54.0'] == 1)] = y[(X['sex_Male'] == 1) & (X['age54.0'] == 1)]
            #decisions[(X['sex_Male'] == 1) & (X['age54.0'] == 1) & mid] = 1-decisions[(X['sex_Male'] == 1) & (X['age54.0'] == 1) & mid]


        return decisions

    def get_final_decisions(self, X, c_model, advice):
        c_human = self.get_confidence(X)
        agreement = (self.get_decisions(X) == advice)
        p_a = self.ADB(c_human, c_model, agreement)
        final_decisions = p_a*advice + (1-p_a)*self.get_decisions(X)
        return final_decisions
    
    def ADB(self, c_human, c_model, agreement, delta=5, beta=0.5, k=0.63, gamma=0.95, asym_scaler=0, asym_scaling=0):
        # from will you accept the AI recommendation
        
        if hasattr(self, 'beta'):
            beta = self.beta

        if type(asym_scaling) == int:
            asym_scaling = np.zeros(len(c_human))

        

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
        
        prob[asym_scaling == 1] = prob[asym_scaling == 1]*(1-asym_scaler)
        prob[asym_scaling == 0] = prob[asym_scaling == 0]*(1+asym_scaler)
        prob[prob > 1] = 1
        prob[prob < 0] = 0
        df = pd.DataFrame({'c_human': c_human, 'c_human_new': c_human_new,
                        'conf': conf, 'c_model': c_model, 'agreement': agreement, 'prob': prob})

        return prob
    
    
        



        
    
    def heart_confidence_transformation(self, X=None, t_type=None):
        #if self.decision_bias:
        #    if self.dataset == 'heart_disease':
        #        start_confidences = np.ones(X.shape[0])
        #        start_confidences[X['age54.0'] == 0] = 0       
        #else:
        start_confidences = np.abs(self.model.predict_proba(self.scaler.transform(X))[:, 1] - 0.5)*2

        if t_type==None or t_type=='calibrated':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences > self.confVal] = 0.2
            confidences[start_confidences <= self.confVal] = 1
        if t_type=='miscalibrated':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences > self.confVal] = 1
            confidences[start_confidences <= self.confVal] = 0.2
        if t_type=='biased':
            confidences = np.ones(X.shape[0])
            if self.decision_bias == False:
                confidences[(X['sex_Male'] == 0) & (start_confidences <= self.confVal)] = 0.9 #overconfident
                confidences[(X['sex_Male'] == 0) & (start_confidences > self.confVal)] = 1 #appropriate
                confidences[(X['sex_Male'] == 1) & (start_confidences <= self.confVal)] = 0.9 #overconfident
                confidences[(X['sex_Male'] == 1) & (start_confidences > self.confVal)] = 0.2 #appropriate/under
            else:
                ####for asymmetric case study#############
                #confidences[(X['age54.0'] == 0)] = 1
                #confidences[(X['age54.0'] == 1)] = 0.3
                ##########################################
                ####for feature decision and confidence bias#############
                if not(hasattr(self, 'alteration')) or self.alteration == '' or self.alteration == '_dec_bias' or self.alteration == 'case2bia' or ('discretion' in self.alteration) or ('asym' in self.alteration):
                    confidences[(X['sex_Male'] == 0) & (start_confidences <= self.confVal)] = 0.9
                    confidences[(X['sex_Male'] == 0) & (start_confidences > self.confVal)] = 1
                    confidences[(X['sex_Male'] == 1) & (start_confidences <= self.confVal)] = 0.9
                    confidences[(X['sex_Male'] == 1) & (start_confidences > self.confVal)] = 0.2
                elif self.alteration == 'case2_cal':
                    confidences[(X['sex_Male'] == 1) & (start_confidences <= self.confVal)] = 0.95
                    confidences[(X['sex_Male'] == 1) & (start_confidences > self.confVal)] = 0.95
                    confidences[(X['sex_Male'] == 0) & (start_confidences <= self.confVal)] = 0.2
                    confidences[(X['sex_Male'] == 0) & (start_confidences > self.confVal)] = 0.2
                else: #case2 
                    confidences[(X['sex_Male'] == 1) & (start_confidences <= self.confVal)] = 0.95
                    confidences[(X['sex_Male'] == 1) & (start_confidences > self.confVal)] = 0.95
                    confidences[(X['sex_Male'] == 0) & (start_confidences <= self.confVal)] = 0.95
                    confidences[(X['sex_Male'] == 0) & (start_confidences > self.confVal)] = 0.95
                #########################################################

                ####for regular case study#######################
                #confidences[(X['age54.0'] == 0)] = 1
                #confidences[(X['age54.0'] == 1)] = 0.3
                #confidences[(X['sex_Male'] == 0) & (start_confidences > self.confVal)] = 0.9
                #confidences[(X['sex_Male'] == 1) & (start_confidences <= self.confVal)] = 0.9


        if t_type=='offset_02':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.8
            confidences[start_confidences > self.confVal] = 0.4
        if t_type=='offset_01':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.9
            confidences[start_confidences > self.confVal] = 0.3
        if t_type=='offset_03':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.7
            confidences[start_confidences > self.confVal] = 0.3
        if t_type=='offset_05':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.5
            confidences[start_confidences > self.confVal] = 0.5
        if t_type=='offset_08':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.2
            confidences[start_confidences > self.confVal] = 0.8


        
        return confidences
    
    def hr_confidence_transformation(self, X=None, t_type=None):

            start_confidences = np.abs(self.model.predict_proba(self.scaler.transform(X))[:, 1] - 0.5)*2

            if t_type==None or t_type=='calibrated':
                
                confidences = np.ones(X.shape[0])
                confidences[start_confidences > self.confVal] = 0.2
                confidences[start_confidences <= self.confVal] = 1
            if t_type=='miscalibrated':
                
                confidences = np.ones(X.shape[0])
                confidences[start_confidences > self.confVal] = 1
                confidences[start_confidences <= self.confVal] = 0.2
            if t_type=='biased':
                confidences = np.ones(X.shape[0])
                confidences[(X['Gender_Male'] == 0) & (start_confidences <= self.confVal)] = 0.9
                confidences[(X['Gender_Male'] == 0) & (start_confidences > self.confVal)] = 1
                confidences[(X['Gender_Male'] == 1) & (start_confidences <= self.confVal)] = 0.9
                confidences[(X['Gender_Male'] == 1) & (start_confidences > self.confVal)] = 0.2

            if t_type=='offset_02':
                confidences = np.ones(X.shape[0])
                confidences[start_confidences <= self.confVal] = 0.8
                confidences[start_confidences > self.confVal] = 0.4
            if t_type=='offset_01':
                
                confidences = np.ones(X.shape[0])
                confidences[start_confidences <= self.confVal] = 0.9
                confidences[start_confidences > self.confVal] = 0.3
            if t_type=='offset_03':
                confidences = np.ones(X.shape[0])
                confidences[start_confidences <= self.confVal] = 0.7
                confidences[start_confidences > self.confVal] = 0.3
            if t_type=='offset_05':
                confidences = np.ones(X.shape[0])
                confidences[start_confidences <= self.confVal] = 0.5
                confidences[start_confidences > self.confVal] = 0.5
            if t_type=='offset_08':
                confidences = np.ones(X.shape[0])
                confidences[start_confidences <= self.confVal] = 0.2
                confidences[start_confidences > self.confVal] = 0.8


            
            return confidences
        
    def fico_confidence_transformation(self, X=None, t_type=None):


        start_confidences = np.abs(self.model.predict_proba(self.scaler.transform(X))[:, 1] - 0.5)*2

        if t_type==None or t_type=='calibrated':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences > self.confVal] = 0.2
            confidences[start_confidences <= self.confVal] = 1
        if t_type=='miscalibrated':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences > self.confVal] = 1
            confidences[start_confidences <= self.confVal] = 0.2
        if t_type=='biased':
            confidences = np.ones(X.shape[0])
            confidences[(X['ExternalRiskEstimate65.0'] == 1) & (start_confidences <= self.confVal)] = 0.9
            confidences[(X['ExternalRiskEstimate65.0'] == 1) & (start_confidences > self.confVal)] = 1
            confidences[(X['ExternalRiskEstimate65.0'] == 0) & (start_confidences <= self.confVal)] = 0.9
            confidences[(X['ExternalRiskEstimate65.0'] == 0) & (start_confidences > self.confVal)] = 0.2

            #if self.decision_bias==True:
            #    
            #    confidences[(X['ExternalRiskEstimate65.0'] == 1) & (X['NumSatisfactoryTrades24.0'] == 0)] = 0.2 #weak
            #    confidences[(X['ExternalRiskEstimate65.0'] == 1) & (X['NumSatisfactoryTrades24.0'] == 1)] = 0.2 #strong
            #    confidences[(X['ExternalRiskEstimate65.0'] == 0) & (X['NumSatisfactoryTrades24.0'] == 0)] = 1 #strong
            #    confidences[(X['ExternalRiskEstimate65.0'] == 0) & (X['NumSatisfactoryTrades24.0'] == 1)] = 1 #strong

        if t_type=='offset_02':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.8
            confidences[start_confidences > self.confVal] = 0.4
        if t_type=='offset_01':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.9
            confidences[start_confidences > self.confVal] = 0.3
        if t_type=='offset_03':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.7
            confidences[start_confidences > self.confVal] = 0.3
        if t_type=='offset_05':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.5
            confidences[start_confidences > self.confVal] = 0.5
        if t_type=='offset_08':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.2
            confidences[start_confidences > self.confVal] = 0.8


        
        return confidences
    

    def adult_confidence_transformation(self, X=None, t_type=None):
        if self.decision_bias:
            if self.dataset == 'adult':
                start_confidences = np.ones(X.shape[0])
                start_confidences[X['race_White'] == 1] = 0       
        else:
            start_confidences = np.abs(self.model.predict_proba(self.scaler.transform(X))[:, 1] - 0.5)*2

        if t_type==None or t_type=='calibrated':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences > self.confVal] = 0.2
            confidences[start_confidences <= self.confVal] = 1
        if t_type=='miscalibrated':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences > self.confVal] = 1
            confidences[start_confidences <= self.confVal] = 0.2
        if t_type=='biased':
            confidences = np.ones(X.shape[0])
            if self.decision_bias == False:
                confidences[(X['race_White'] == 1) & (start_confidences <= self.confVal)] = 0.2
                confidences[(X['race_White'] == 1) & (start_confidences > self.confVal)] = 1
                confidences[(X['race_White'] == 0) & (start_confidences <= self.confVal)] = 0.9
                confidences[(X['race_White'] == 0) & (start_confidences > self.confVal)] = 0.2
            else:
                ####for asymmetric case study#############
                #confidences[(X['age54.0'] == 0)] = 1
                #confidences[(X['age54.0'] == 1)] = 0.3
                ##########################################
                ####for feature decision and confidence bias#############
                confidences[(X['sex_Male'] == 0) & (X['race_White'] == 1)] = 0.2 #weak
                confidences[(X['sex_Male'] == 1) & (X['race_White'] == 0)] = 0.2 #strong
                confidences[(X['sex_Male'] == 0) & (X['race_White'] == 0)] = 1 #weak
                confidences[(X['sex_Male'] == 1) & (X['race_White'] == 1)] = 1 #strong
                #########################################################

                ####for regular case study#######################
                #confidences[(X['age54.0'] == 0)] = 1
                #confidences[(X['age54.0'] == 1)] = 0.3
                #confidences[(X['sex_Male'] == 0) & (start_confidences > self.confVal)] = 0.9
                #confidences[(X['sex_Male'] == 1) & (start_confidences <= self.confVal)] = 0.9


        if t_type=='offset_02':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.8
            confidences[start_confidences > self.confVal] = 0.4
        if t_type=='offset_01':
            
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.9
            confidences[start_confidences > self.confVal] = 0.3
        if t_type=='offset_03':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.7
            confidences[start_confidences > self.confVal] = 0.3
        if t_type=='offset_05':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.5
            confidences[start_confidences > self.confVal] = 0.5
        if t_type=='offset_08':
            confidences = np.ones(X.shape[0])
            confidences[start_confidences <= self.confVal] = 0.2
            confidences[start_confidences > self.confVal] = 0.8


        
        return confidences
    
    

