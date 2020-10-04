# %%
# たこ焼き器所持率のカイ二乗検定
import pandas as pd
import scipy as sp
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
from joblib import Parallel, delayed


# 帰無仮説が本当に正しいか否か
H0 = False

fntsize = 22  # プロットのフォントサイズ
fntsize_ticklabel = 18

# 仮想的な母集団
if H0:
    pplsize = np.array([
        [17600, 26400],
        [2400, 3600]
    ])
else:
    pplsize = np.array([
        [17600, 26400],
        [600, 5400]
    ])
    # 10 point diff
    # pplsize = np.array([
    #     [17600, 26400],
    #     [1800, 4200]
    # ])
# pplsize = np.array([
#     [17, 26],
#     [2, 3]
# ])

print(pplsize.sum())

# 仮想的な母集団を準備
ppl = []
for pref in range(2):
    for tako in range(2):
        ppl.extend([[pref, tako]] * pplsize[pref][tako])
df_ppl = pd.DataFrame(ppl, columns=['pref', 'tako'])


def sampling(df_ppl, n, verbose=False):
    """ 標本抽出して chi-squared を計算 """
    # 標本抽出
    df_smp = df_ppl.sample(n=n)
    # groupby is faster than crosstab
    # df_cross = pd.crosstab(index=df_smp['pref'], columns=df_smp['tako'])
    df_cross = df_smp.groupby(['pref', 'tako']).size().unstack(fill_value=0)
    # compute chi2 and p-value
    chi2, p, dof, ef = chi2_contingency(df_cross, correction=False)
    if verbose:
        # print(df_smp)
        print(df_cross)
        print(chi2, p, dof, ef)
    return chi2


chi2 = sampling(df_ppl, n=50, verbose=True)
chi2 = sampling(df_ppl, n=200, verbose=True)
print(chi2)

# %%
num_it = 1000  # 反復回数
n_list = [50, 100, 150, 200, 250, 300]
# n_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
upper_p_list = []
lower_p_list = []
for n in n_list:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    chi2list = np.array(
        Parallel(n_jobs=-1)([
            delayed(sampling)(df_ppl, n=n) for _ in range(num_it)
        ])
    )
    upper5percentile = sp.stats.chi2.ppf(0.95, df=1)
    print(upper5percentile)
    upper_p = chi2list[chi2list > upper5percentile].shape[0] / num_it
    lower_p = chi2list[chi2list <= upper5percentile].shape[0] / num_it
    print(n, upper_p, lower_p)
    # 相対度数をプロット
    ax.hist(chi2list, density=True, alpha=0.6, bins=20)
    # 密度関数も重ねる
    x = np.arange(0, 20, 0.1)
    y = sp.stats.chi2.pdf(x, df=1)
    ax.plot(x, y)
    ax.set_xlabel('カイ二乗値', fontsize=fntsize)
    ax.set_ylabel('相対度数', fontsize=fntsize)
    ax.set_xlim([0, 20])
    xticks = range(0, 25, 5)
    yticks = np.arange(0.0, 1.4, 0.2)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    xticklabels = [str(x) for x in xticks]
    yticklabels = ['{:.1f}'.format(x) for x in yticks]
    # print(xticklabels)
    # print(yticklabels)
    ax.set_xticklabels(xticklabels, fontsize=fntsize_ticklabel)
    ax.set_yticklabels(yticklabels, fontsize=fntsize_ticklabel)
    # 画像保存
    figfile = 'n{}_H0{}.png'.format(n, H0)
    plt.savefig(figfile)
    print(figfile)
    plt.show()
    upper_p_list.append(upper_p)
    lower_p_list.append(lower_p)

#%%
# サンプルサイズによる alpha (もしくは beta) の変化をプロット
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig, ax = plt.subplots(1, 1, figsize=(20, 8))
if H0:
    print('alpha:', upper_p_list)
    ax.bar(n_list, upper_p_list, width=20)
    ax.set_ylim([0, 0.1])
    ax.set_ylabel('alpha', fontsize=fntsize)
    yticks = np.arange(0, 0.12, 0.02)
    yticklabels = ['{:.2f}'.format(x) for x in yticks]
    figfile = 'alpha_H0{}.png'.format(H0)
    np.savetxt('alpha.txt', np.array([n_list, upper_p_list]).T)
else:
    print('beta:', lower_p_list)
    ax.bar(n_list, lower_p_list, width=20)
    ax.set_ylim([0, 1])
    ax.set_ylabel('beta', fontsize=fntsize)
    yticks = np.arange(0, 1.2, 0.2)
    yticklabels = ['{:.1f}'.format(x) for x in yticks]
    figfile = 'beta_H0{}.png'.format(H0)
    np.savetxt('beta.txt', np.array([n_list, lower_p_list]).T)

# 軸のラベルを設定
xticklabels = [str(x) for x in n_list]
print(xticklabels)

ax.set_xticks(n_list)
ax.set_yticks(yticks)
ax.set_xticklabels(xticklabels, fontsize=fntsize_ticklabel)
ax.set_yticklabels(yticklabels, fontsize=fntsize_ticklabel)

ax.set_xlabel('n', fontsize=fntsize)

plt.savefig(figfile)
plt.show()

# %%
