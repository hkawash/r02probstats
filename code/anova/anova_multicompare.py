# %%
# 多重検定の実験
# 帰無仮説に従うデータを生成し，第一種の過誤が生じる確率を，頻度ベースで調べる
# sigma の値を設定して実行

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# 重要パラメータ
# sigma = np.sqrt(2)  # 水準共通の母標準偏差
sigma = np.sqrt(0.2)  # 水準共通の母標準偏差

# 固定パラメータ
N = 3  # 水準数
mean = 10  # 母平均（全ての群で等しいとする）
var = sigma*sigma  # 水準共通の母分散
n = [10 for i in range(N)]  # サンプルサイズ
# group_str = ['group' + str(i+1) for i in range(N)]  # グループ名

# 各グループの平均
# meanlist = df.groupby('group').mean().values.flatten()
# print(meanlist)
# print(df.groupby('group').describe())

alpha = 0.05

num_repeat = 100
num_mc_rjct = 0
num_F_rjct = 0
for c in range(num_repeat):
    data_mat = np.array([np.random.normal(mean, sigma, n[i]) for i in range(N)])
    # print(data_mat)
    data = data_mat.flatten()
    group = np.array([np.full(n[i], i+1) for i in range(N)]).flatten()

    # データフレーム作成
    df = pd.DataFrame()
    df['data'] = data
    df['group'] = group.astype(int)
    # print(df)

    f, pF = sp.stats.f_oneway(data_mat[0], data_mat[1], data_mat[2])
    f01, p01 = sp.stats.ttest_ind(data_mat[0], data_mat[1])
    f12, p12 = sp.stats.ttest_ind(data_mat[1], data_mat[2])
    f20, p20 = sp.stats.ttest_ind(data_mat[2], data_mat[0])

    if p01 < alpha or p12 < alpha or p20 < alpha:
        print("p = ({:.2}, {:.2}, {:.2})".format(p01, p12, p20))
        num_mc_rjct += 1
    if pF < alpha:
        print("pF= {:.2}".format(pF))
        num_F_rjct += 1

print("ratio mc_rjct = {}/{} = {}".format(num_mc_rjct, num_repeat, num_mc_rjct/num_repeat))
print("ratio F_rjct = {}/{} = {}".format(num_F_rjct, num_repeat, num_F_rjct/num_repeat))

