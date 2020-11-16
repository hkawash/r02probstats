# %%
# https://seaborn.pydata.org/generated/seaborn.stripplot.html
#
# sigma の値を設定して実行

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="whitegrid")
# tips = sns.load_dataset("tips")

# 重要パラメータ
# sigma = np.sqrt(4)  # 水準共通の母標準偏差 (ttest用)
# sigma = np.sqrt(2)  # 水準共通の母標準偏差 (Ftest用)
sigma = np.sqrt(0.2)  # 水準共通の母標準偏差

# 固定パラメータ
N = 3  # 水準数
mean = [9.7, 9.9, 10.7]  # 母平均
var = sigma*sigma  # 水準共通の母分散
n = [10 for i in range(N)]  # サンプルサイズ
group_str = ['group' + str(i+1) for i in range(N)]  # グループ名

data = np.array([np.random.normal(mean[i], sigma, n[i]) for i in range(N)]).flatten()
group = np.array([np.full(n[i], i+1) for i in range(N)]).flatten()

# データフレーム作成
df = pd.DataFrame()
df['data'] = data
df['group'] = group.astype(int)
# print(df)

# 各グループの平均
meanlist = df.groupby('group').mean().values.flatten()
print(meanlist)
# print(df.groupby('group').describe())
grouplist = [df[df['group'] == i+1]['data'].values for i in range(N)]

do_anova = False
do_pair = True

def save_figs(fname, title, df, orient='h'):
    if orient == 'h':
        xlim = [6, 16]
        xname = 'data'
        yname = 'group'
        figsize = (8, 4)
    else:
        ylim = [6, 16]
        xname = 'group'
        yname = 'data'
        figsize = (4, 8)

    with PdfPages(fname) as pdf_pages:
        plt.figure(figsize=figsize)
        ax = sns.stripplot(x=xname, y=yname, data=df, jitter=False, orient=orient)
        ax.set_xlim(xlim) if orient == 'h' else ax.set_ylim(ylim) 
        ax.set_title(title)
        pdf_pages.savefig()
        plt.show()

        plt.figure(figsize=figsize)
        ax = sns.boxplot(x=xname, y=yname, data=df, orient=orient)
        ax.set_xlim(xlim) if orient == 'h' else ax.set_ylim(ylim) 
        ax.set_title(title)
        pdf_pages.savefig()
        plt.show()

        # plt.figure(figsize=figsize)
        # ax = sns.violinplot(x=xname, y=yname, data=df, orient=orient)
        # ax.set_xlim(xlim) if orient == 'h' else ax.set_ylim(ylim) 
        # ax.set_title(title)
        # pdf_pages.savefig()
        # plt.show()


np.save('ndata-var{:.2f}.npy'.format(var), data)

if do_anova:
    # Fとp値
    f, p = sp.stats.f_oneway(grouplist[0], grouplist[1], grouplist[2])
    print((f, p))

    title_str = "F = {:.2f}, p = {:.2e}".format(f, p, meanlist[0]) + \
        ", mean: ({:.2f}, {:.2f}, {:.2f})".format(meanlist[0], meanlist[1], meanlist[2])

    save_figs(f'anova-var{var:.2f}.pdf', title_str, df, 'h')


# % 2020追加
if do_pair:
    # Fとp値
    f, p = sp.stats.f_oneway(grouplist[0], grouplist[2])
    t, p2 = sp.stats.ttest_ind(grouplist[0], grouplist[2], equal_var=True)
    print((f, p))
    print((t, p2))

    title_str = "F = {:.2f}, p = {:.2e}".format(f, p, meanlist[0]) + \
        ", mean: ({:.2f}, {:.2f})".format(meanlist[0], meanlist[2])

    save_figs(f'pair-var{var:.2f}.pdf', title_str, df[df['group'] != 2], None)


# %%
