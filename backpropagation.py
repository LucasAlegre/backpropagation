import os
import argparse
import random
import pandas as pd
import numpy as np
from backpropagation.util import stratified_k_cross_validation, txt2dataframe, to_one_hot, parse_dataframe
from backpropagation.nn import NN

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="Backpropagation - Aprendizado de Máquina 2019/1 UFRGS")
    prs.add_argument("-s",              dest="seed",          required=False, default=None,                help="The random seed.\n", type=int)
    prs.add_argument("-d",              dest="data",          required=False, default='datasets/wine.csv', help="The dataset .csv file.\n")
    prs.add_argument("-c",              dest="class_column",  required=False, default='class',             help="The column of the .csv to be predicted.\n")
    prs.add_argument("-sep",            dest="sep",           required=False, default=',',                 help=".csv separator.\n")
    prs.add_argument("-k",              dest="num_folds",     required=False, default=10,                  help="The number of folds used on cross validation.\n", type=int)
    prs.add_argument("-e",              dest="epochs",        required=False, default=100,                 help="Amount of epochs for training the neural network.", type=int)
    prs.add_argument("-mb",             dest="batch_size",    required=False, default=None,                help="Mini-batch size used for training.", type=int)
    prs.add_argument('-drop',           nargs='+',            required=False, default=[],                  help="Columns to drop from .csv.\n")
    prs.add_argument('-nn',             nargs='+',            required=True,                               help="Neural Network structure.\n")
    prs.add_argument('-w',              dest='weights',       required=False, default=None,                help="Initial weights.\n")
    prs.add_argument('-alpha',          dest='alpha',         required=False, default=0.001,               help="Learning rate.\n", type=float)
    prs.add_argument('-beta',           dest='beta',          required=False, default=0.9,                 help="Efective direction rate used on the Momentum Method.\n", type=float)
    prs.add_argument('-regularization', dest='regularization',required=False, default=0.0,                 help="Regularization factor.\n", type=float)
    prs.add_argument("-numerical",      action='store_true',  required=False, default=False,               help="Calculate the gradients numerically.\n")
    prs.add_argument('-opt',            dest='opt',           required=False, default='SGD',               help="Optimizer [SGD, Momentum, Adam].\n")
    prs.add_argument("-log",            action='store_true',  required=False, default=False,               help="Generate log file.\n")
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

    if args.data.endswith('.txt'):  # naive dataset txt
        x = df.drop([c for c in df.columns if c.startswith('y')], axis=1).values
        y = df.drop([c for c in df.columns if c.startswith('x')], axis=1).values
        class_values = None
    else:
        normalized_df = parse_dataframe(df, class_column, normalize='minmax')
        class_values = pd.get_dummies(df[class_column]).columns.values

    if args.nn[0].endswith('.txt'):
        architecture = args.nn[0]
    else:
        architecture = [int(n) for n in args.nn]

    nn = NN(architecture=architecture, 
            initial_weights=args.weights, 
            optimizer=args.opt,
            alpha=args.alpha, 
            beta=args.beta,
            regularization_factor=args.regularization,
            class_column=class_column, 
            class_values=class_values, 
            epochs=args.epochs, 
            batch_size=args.batch_size)

    if args.numerical:
        nn.train_numerically(x, y)
    else:
        if args.data.endswith('.txt'):
            nn.train(x, y, x, y)
        else:
            log_name = 'logs/{}_{}_a{}_{}_mb{}_reg{}'.format(os.path.basename(args.data), args.opt, args.alpha, '-'.join(n for n in args.nn), 
                                                            args.batch_size, args.regularization) if args.log else None
            stratified_k_cross_validation(nn, normalized_df, class_column, k=args.num_folds, log_name=log_name)
