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
# sigma = np.sqrt(2)  # 水準共通の母標準偏差
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

# Fとp値
f, p = sp.stats.f_oneway(grouplist[0], grouplist[1], grouplist[2])
print((f, p))

title_str = "F = {:.2f}, p = {:.2e}".format(f, p, meanlist[0]) + \
      ", mean: ({:.2f}, {:.2f}, {:.2f})".format(meanlist[0], meanlist[1], meanlist[2])

xlim = [6, 16]
with PdfPages('anova-var{:.2f}.pdf'.format(var)) as pdf_pages:
    ax = sns.stripplot(x='data', y='group', data=df, jitter=False, orient='h')
    ax.set_xlim(xlim)
    ax.set_title(title_str)
    pdf_pages.savefig()
    plt.show()

    ax = sns.boxplot(x='data', y='group', data=df, orient='h')
    ax.set_xlim(xlim)
    ax.set_title(title_str)
    pdf_pages.savefig()
    plt.show()

    ax = sns.violinplot(x='data', y='group', data=df, orient='h')
    ax.set_xlim(xlim)
    ax.set_title(title_str)
    pdf_pages.savefig()
    plt.show()


np.save('ndata-var{:.2f}.npy'.format(var), data)
