from run import *
import sys, getopt

#making sure wd is file directory so hardcoded paths work
os.chdir("..")


run('heart_disease', 0, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=True, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 1, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 2, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 3, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 4, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 5, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 6, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 7, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 8, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)
run('heart_disease', 9, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretionTrue', use_true=True, subsplit=1, shared_human=True)


run('heart_disease', 0, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=True, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 1, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 2, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 3, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 4, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 5, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 6, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 7, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 8, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)
run('heart_disease', 9, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion1', use_true=False, subsplit=1, shared_human=True)



run('heart_disease', 0, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=True, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 1, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 2, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 3, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 4, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 5, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 6, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 7, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 8, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)
run('heart_disease', 9, 'biased', runtype='standard', which_models=['tr,hyrs'], contradiction_reg=0.0, remake_humans=False, human_decision_bias=False, custom_name='_discretion02', use_true=False, subsplit=0.2, shared_human=True)




