from run import *
import sys, getopt

#making sure wd is file directory so hardcoded paths work
os.chdir("..")

run('heart_disease', 0, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 1, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 2, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 3, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 4, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)


run('heart_disease', 0, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 1, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 2, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 3, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 4, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)


run('heart_disease', 0, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 1, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 2, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 3, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 4, 'biased', runtype='standard', which_models=['hyrs','tr'], contradiction_reg=0.6, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)




