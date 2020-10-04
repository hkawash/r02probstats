# %%
# sigma の値を設定して実行

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 重要パラメータ
# sigma = np.sqrt(2)  # 水準共通の母標準偏差
sigma = np.sqrt(0.2)  # 水準共通の母標準偏差


var = sigma*sigma  # 水準共通の母分散

data = np.load('plot/ndata-var{:.2f}.npy'.format(var))
# print(data)
sns.set(style="whitegrid")

# 固定パラメータ
N = 3  # 水準数
n = [10 for i in range(N)]  # サンプルサイズ

group = np.array([np.full(n[i], i+1) for i in range(N)]).flatten()

# データフレーム作成
df = pd.DataFrame()
df['data'] = data
df['group'] = group.astype(int)
# print(df)

# 各グループの平均
meanlist = df.groupby('group').mean().values.flatten()
print(meanlist)
print("meandiff(2-1): {:.2f}, (3-2): {:.2f}, (3-1): {:.2f}" \
    .format(meanlist[1]-meanlist[0], meanlist[2]-meanlist[1], meanlist[2]-meanlist[0]))
grouplist = [df[df['group'] == i+1]['data'].values for i in range(N)]

# Fとp値
f, p = sp.stats.f_oneway(grouplist[0], grouplist[1], grouplist[2])
print((f, p))

title_str = "F = {:.2f}, p = {:.2e}".format(f, p, meanlist[0]) + \
      ", mean: ({:.2f}, {:.2f}, {:.2f})".format(meanlist[0], meanlist[1], meanlist[2])

xlim = [6, 16]

# with PdfPages('anova-var{:.2f}.pdf'.format(var)) as pdf_pages:
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
plt.subplots_adjust(wspace=0.4, hspace=0.6)

sns.stripplot(x='data', y='group', data=df, linewidth=1, jitter=False, orient='h', ax=ax[0])
ax[0].set_xlim(xlim)
ax[0].set_title(title_str)
# pdf_pages.savefig()
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 2))
sns.boxplot(x='data', y='group', data=df, orient='h', ax=ax[1])
ax[1].set_xlim(xlim)
# ax[1].set_title(title_str)
# pdf_pages.savefig()

plt.show()
fig.savefig('anova-var{:.2f}.png'.format(var))

