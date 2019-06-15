import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import brewer2mpl
import seaborn as sns
import argparse
import glob
from itertools import cycle

sns.set(rc={'figure.figsize':(12,9)})
sns.set(font_scale = 2)

bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors
colors = cycle(colors)

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument('-f', nargs='+', required=False, help="csv file\n")
prs.add_argument('-l', nargs='+', required=False, help="labels\n")
args = prs.parse_args()

labels = cycle(args.l)

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")

for file in args.f:
    main_df = pd.DataFrame()
    for f in glob.glob(file+'*'):
        df = pd.read_csv(f)
        if main_df.empty:
            main_df = df
        else:
            main_df = pd.concat((main_df, df))

    mean_trainloss = main_df.groupby('epoch').mean()['train-loss']
    std_trainloss = main_df.groupby('epoch').std()['train-loss']
    mean_testloss = main_df.groupby('epoch').mean()['test-loss']
    std_testloss = main_df.groupby('epoch').std()['test-loss']
    epoch = main_df.groupby('epoch').epoch.mean().keys()
    cor = next(colors)
    label = next(labels)

    plt.plot(mean_trainloss, color=cor, linewidth=2, label='train loss'+label)
    plt.fill_between(epoch, mean_trainloss + std_trainloss, mean_trainloss - std_trainloss, alpha=0.3, color=cor)
    plt.plot(mean_testloss, color=cor, linestyle='--', linewidth=2, label='validation loss'+label)
    plt.fill_between(epoch, mean_testloss + std_testloss, mean_testloss - std_testloss, alpha=0.3, color=cor)

plt.legend()
plt.show()

""" 
for f in files:
    boxplots = []
    for i in ntrees:
            df = pd.read_csv('results/{}_n_{}.csv'.format(f[0], i))
            for j in range(len(df)):
                    boxplots.append({'x': str(i), 'y': df['f1'][j]})
    boxplots = pd.DataFrame(boxplots)

    ax = sns.boxplot(x='x', y='y', data=boxplots, linewidth=2.5, order=[str(x) for x in ntrees])
    ax = sns.swarmplot(x='x', y='y', data=boxplots, color=".2", size=6, order=[str(x) for x in ntrees])

    ax.set(xlabel='Number of Trees', ylabel='F1 score')

    plt.show()

    ax.get_figure().savefig('results/{}'.format(f[0]) + '.pdf', bbox_inches='tight') """