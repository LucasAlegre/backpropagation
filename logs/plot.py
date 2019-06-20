import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import glob
from itertools import cycle

sns.set(rc={'figure.figsize':(12,9)}, font_scale=2, style='darkgrid')

colors = sns.color_palette('colorblind', 4)
sns.set_palette(colors)
colors = cycle(colors)

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument('-f', nargs='+', required=False, help="csv file\n")
prs.add_argument('-l', nargs='+', required=False, help="labels\n")
prs.add_argument('-n', required=False, help='name\n')
args = prs.parse_args()

labels = cycle(args.l)

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")

boxplots = []
for file in args.f:
    cor = next(colors)
    label = next(labels)
    
    main_df = pd.DataFrame()
    for f in glob.glob(file+'*'):
        df = pd.read_csv(f)
        boxplots.append({'x': label, 'y': df['f1_score'][0]})
        if main_df.empty:
            main_df = df
        else:
            main_df = pd.concat((main_df, df))

    mean_trainloss = main_df.groupby('epoch').mean()['train-loss']
    std_trainloss = main_df.groupby('epoch').std()['train-loss']
    mean_testloss = main_df.groupby('epoch').mean()['test-loss']
    std_testloss = main_df.groupby('epoch').std()['test-loss']
    epoch = main_df.groupby('epoch').epoch.mean().keys()

    plt.plot(mean_trainloss, linewidth=2, label=label+' train loss', color=cor)
    plt.fill_between(epoch, mean_trainloss + std_trainloss, mean_trainloss - std_trainloss, alpha=0.3, color=cor)
    plt.plot(mean_testloss, linestyle='--', linewidth=2, label=label + ' validation loss', color=cor)
    plt.fill_between(epoch, mean_testloss + std_testloss, mean_testloss - std_testloss, alpha=0.3, color=cor)
plt.ylim([0,5])
plt.legend()
plt.savefig(args.n+'plot.pdf', bbox_inches='tight')
plt.show()


## Boxplots F1-score
boxplots = pd.DataFrame(boxplots)
ax = plt.subplots()

ax = sns.boxplot(x='x', y='y', data=boxplots, linewidth=2.5)
ax = sns.swarmplot(x='x', y='y', data=boxplots, color=".2", size=6)

ax.set(xlabel='Architecture', ylabel='F1 score')
plt.ylim([0.6,1.001])

plt.savefig(args.n+'boxplot.pdf', bbox_inches='tight')
plt.show()
