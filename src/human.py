from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, uniform

class Human(object):
    def __init__(self, name, X, y) -> None:
        self.name = name
        self.X = X.dropna(axis=1)
        self.y = y
        self.model = LogisticRegression().fit(self.X, y)
        self.custom_decision_func = None
        self.custom_confidence_func = None


    
    def get_confidence(self, X):
        return self.heart_confidence_transformation(X=X, t_type=self.name)

    def get_decisions(self, X, y):
        return self.heart_decision_transformation(X, y, t_type=self.name)

    def get_final_decisions(self, X, c_model, advice):
        c_human = self.get_confidence(X)
        agreement = (self.get_decisions(X) == advice)
        p_a = self.ADB(c_human, c_model, agreement)
        final_decisions = p_a*advice + (1-p_a)*self.get_decisions(X)
        return final_decisions
    
    def ADB(self, c_human, c_model, agreement, delta=5, beta=0.05, k=0.63, gamma=0.95):
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

        return prob

    def heart_decision_transformation(self, X=None, y=None, t_type=None):

        return self.slightly_miscalibrated_decisions_heart_1(X, y)

        
    
    def heart_confidence_transformation(self, X=None, t_type=None):
        if t_type==None or t_type=='calibrated':
            confidences = np.zeros(X.shape[0])
            confidences[X['sex_Male'] == 0] = 0.5
            confidences[X['sex_Male'] == 1] = 0.9

        if t_type=='miscalibrated':
            confidences = np.zeros(X.shape[0])
            confidences[X['sex_Male'] == 0] = 0.9
            confidences[X['sex_Male'] == 1] = 0.5
        if t_type=='slightly_miscalibrated':
            confidences = self.slightly_miscalibrated_confidences_heart_1(X)
        
        return confidences
    
    #dataset specific behaviors
    def slightly_miscalibrated_decisions_heart_1(self, X, y):
        decisions = y.copy()
        #low accuracy on females, 50%
        flip_females = bernoulli.rvs(p=0.5, size=len(decisions)).astype(bool)
        #higher accuracy on males, 90%
        flip_males = bernoulli.rvs(p=0.1, size=len(decisions)).astype(bool)
        decisions[(X['sex_Male'] == 0) & flip_females] = 1-decisions[(X['sex_Male'] == 0) & flip_females]
        decisions[(X['sex_Male'] == 1) & flip_males] = 1-decisions[(X['sex_Male'] == 1) & flip_males]

        return decisions
    
    def slightly_miscalibrated_confidences_heart_1(self, X):
        confidences = np.zeros(X.shape[0])
        confidences[X['age54.0'] == 0] = np.random.randint(97,100,len(confidences[X['age54.0'] == 0]))/100
        confidences[X['age54.0'] == 1] = np.random.randint(0,20,len(confidences[X['age54.0'] == 1]))/100

        return confidences



    

