import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import brewer2mpl
import seaborn as sns
import argparse
import glob

sns.set(rc={'figure.figsize':(12,9)})
sns.set(font_scale = 2)

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument('-f', required=False, help="csv file\n")
args = prs.parse_args()

main_df = pd.DataFrame()
for f in glob.glob(args.f+'*'):
    df = pd.read_csv(f)
    if main_df.empty:
        main_df = df
    else:
        main_df = pd.concat((main_df, df))

print(main_df)
plt.figure()
plt.plot(df.groupby('epoch').mean()['train-loss'])
plt.plot(df.groupby('epoch').mean()['test-loss'])
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