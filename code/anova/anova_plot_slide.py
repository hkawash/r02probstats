# %%
# スライドのデータ例をプロット

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="whitegrid")

# 分散分析スライドのデータ
slidedata = [[9.5, 9.7, 10.1, 9.8, 9.3],
    [10.1, 10.5, 9.6, 9.3],
    [11.3, 10.7, 10.2]]
N = len(slidedata)  # 水準数
n = [len(slidedata[i]) for i in range(N)]  # サンプルサイズ
# data = np.array(slidedata).flatten()
# group = np.array([np.full(n[i], i+1) for i in range(N)]).flatten()
# サイズが違うので上のように flatten はできない
data = np.concatenate(slidedata, 0)
group = np.concatenate([np.full(n[i], i+1) for i in range(N)], 0)
print(n)
print(data)
print(group)
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

xlim = [8, 12]

plt.subplots_adjust(wspace=0.4, hspace=0.6)

# fig, ax = plt.subplots(2, 1, figsize=(8, 4))
# sns.stripplot(x='data', y='group', data=df, linewidth=1, jitter=False, orient='h', ax=ax[0])
# ax[0].set_xlim(xlim)
# ax[0].set_title(title_str)
# sns.boxplot(x='data', y='group', data=df, orient='h', ax=ax[1], showmeans=True)
# ax[1].set_xlim(xlim)

fig, ax = plt.subplots(figsize=(8, 4))
sns.stripplot(x='data', y='group', data=df, linewidth=1, jitter=False, orient='h', ax=ax)
ax.set_xlim(xlim)
ax.set_title(title_str)

plt.show()
fig.savefig('anova-slidedata.png')

