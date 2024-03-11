from run import *
import sys, getopt

#making sure wd is file directory so hardcoded paths work
os.chdir("..")



def main(argv):
    """
    specify run parameters. d is dataset, i is run_num,  h is human name (type), r is runtype, c is cost, n is remake humans, p is custom name, brs is brs, hyrs is hyrs, tr is tr
    """

    
    print(os.getcwd())

    opts, args = getopt.getopt(argv, "d:i:h:r:c:n:p:b:w:")

    which_models = []

    for opt, arg in opts:
        if opt == "-d":
            dataset = arg
        elif opt == "-i":
            run_num = int(arg)
        elif opt == "-h":
            human_name = arg
        elif opt == "-r":
            runtype = arg
        elif opt == "-c":
            contradiction_reg = float(arg)
        elif opt == "-n":
            if arg == 'False':
                remake_humans = False
            else:
                remake_humans=True
        elif opt == "-p":
            custom_name = arg
        elif opt == "-b":
            if arg == 'False':
                decision_bias = False
            else:
                decision_bias=True
        elif opt == "-w":
            models = arg.split(',')
            for model in models:
                which_models.append(model)
            

    
    #print(remake_humans)
    #print(type(remake_humans))
    #print(which_models)

    
    run(dataset, run_num, human_name, runtype=runtype, which_models=which_models, contradiction_reg=contradiction_reg, remake_humans=remake_humans, human_decision_bias=decision_bias, custom_name=custom_name)



if __name__ == "__main__":
   main(sys.argv[1:])