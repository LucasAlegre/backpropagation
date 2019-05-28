import argparse
import random
import pandas as pd
import numpy as np
from backpropagation.util import stratified_k_cross_validation, txt2dataframe
from backpropagation.nn import NN

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="Backpropagation - Aprendizado de Máquina 2019/1 UFRGS")

    prs.add_argument("-s",    dest="seed",         required=False, default=None,                help="The random seed.\n", type=int)
    prs.add_argument("-d",    dest="data",         required=False, default='datasets/wine.csv', help="The dataset .csv file.\n")
    prs.add_argument("-c",    dest="class_column", required=False, default='class',             help="The column of the .csv to be predicted.\n")
    prs.add_argument("-sep",  dest="sep",          required=False, default=',',                 help=".csv separator.\n")
    prs.add_argument("-k",    dest="num_folds",    required=False, default=10,                  help="The number of folds used on cross validation.\n", type=int)
    prs.add_argument('-drop', nargs='+',           required=False, default=[],                  help="Columns to drop from .csv.\n")
    prs.add_argument('-nn',   dest='network',      required=True,                               help="Neural Network structure.\n")
    prs.add_argument('-w',    dest='weights',      required=False, default=None,                help="Initial weights.\n")
    prs.add_argument("-v",    action='store_true', required=False, default=False,               help="View reural network image.\n")

    args = prs.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.data.endswith('.txt'):
        df = txt2dataframe(args.data)
    else:
        df = pd.read_csv(args.data, sep=args.sep)
    class_column = args.class_column

    for column in args.drop:
        df.drop(column, inplace=True, axis=1)

    nn = NN(args.network, initial_weights=args.weights)
    nn.propagate(df.values[0][:-1])

    #stratified_k_cross_validation(nn, df, class_column, k=args.num_folds)
    
    if args.v:
        nn.view_architecture('NeuralNetwork')
