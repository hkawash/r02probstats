# %%
# sigma の値を設定して実行

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="whitegrid")

# 重要パラメータ
# sigma = np.sqrt(2)  # 水準共通の母標準偏差
sigma = np.sqrt(0.2)  # 水準共通の母標準偏差

# anova-data (2019): anova用
# ttest-data (2020): ttest用だが 3群データ入っている（群1, 3を利用）

# 固定パラメータ
N_level = 2  # 水準数 (2: ttest, 3: anova)
n_eachlevel = 10

#---
N_orig = 3 # データの群数
var = sigma*sigma  # 水準共通の母分散

data_dir = 'anova-data' if N_level == 3 else 'ttest-data'
data = np.load(f'{data_dir}/ndata-var{var:.2f}.npy')
print(data.shape)

n = [n_eachlevel for i in range(N_orig)]  # サンプルサイズ

group = np.array([np.full(n[i], i+1) for i in range(N_orig)]).flatten()
print(group)
# データフレーム作成
df = pd.DataFrame()
df['data'] = data
df['group'] = group.astype(int)  # ここまでは3群データ
if N_level == 2:
    df = df[df['group'].isin([1,3])]
    df['group'] = df['group'].map({1: 1, 3: 2})
print(df)

# 各グループの平均
meanlist = df.groupby('group').mean().values.flatten()
print(meanlist)
grouplist = [df[df['group'] == i+1]['data'].values for i in range(N_level)]


def save_figs(fname, title, df, orient='h'):

    if orient == 'h':
        xname = 'data'
        yname = 'group'
        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

    else:
        xname = 'group'
        yname = 'data'
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.6, hspace=0.4)

    data_lim = [6, 16]

    sns.stripplot(x=xname, y=yname, data=df, linewidth=1, jitter=False, orient=orient, ax=ax[0])
    ax[0].set_xlim(data_lim) if orient == 'h' else ax[0].set_ylim(data_lim)
    ax[0].set_title(title_str)
    # pdf_pages.savefig()
    # plt.show()

    # fig, ax = plt.subplots(figsize=(8, 2))
    sns.boxplot(x=xname, y=yname, data=df, orient=orient, ax=ax[1])
    ax[1].set_xlim(data_lim) if orient == 'h' else ax[1].set_ylim(data_lim)
    # ax[1].set_title(title_str)
    # pdf_pages.savefig()

    plt.show()
    fig.savefig(fname, bbox_inches='tight')


if N_level == 3:
    print("meandiff(2-1): {:.2f}, (3-2): {:.2f}, (3-1): {:.2f}" \
        .format(meanlist[1]-meanlist[0], meanlist[2]-meanlist[1], meanlist[2]-meanlist[0]))

    # Fとp値
    f, p = sp.stats.f_oneway(grouplist[0], grouplist[1], grouplist[2])
    print((f, p))

    title_str = "F = {:.2f}, p = {:.2e}".format(f, p) + \
        ", mean: ({:.2f}, {:.2f}, {:.2f})".format(meanlist[0], meanlist[1], meanlist[2])

    save_figs(f'anova-var{var:.2f}.png', title_str, df, orient='h')
else:
    print("meandiff(2-1): {:.2f}".format(meanlist[1]-meanlist[0]))

    # tとp値
    t, p_t = sp.stats.ttest_ind(grouplist[0], grouplist[1])
    print((t, p_t))
    f, p_f = sp.stats.f_oneway(grouplist[0], grouplist[1])
    print((f, p_f))

    title_str = "t = {:.2f}, p = {:.2e}".format(t, p_t) + \
        ", mean: ({:.2f}, {:.2f})".format(meanlist[0], meanlist[1])

    save_figs(f'ttest-var{var:.2f}.png', title_str, df, orient='v')

