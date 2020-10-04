# %% 異なる平均を持つ正規分布でもよいことを確認
import numpy as np
import matplotlib.pyplot as plt

sigma = 2
mu = [1, 2]
# mu = [8, 12]
n = [10, 15]

num_it = 1000
s2r_list = []
for it in range(num_it):
    x = [np.random.randn(n[i], 1) * sigma + mu[i] for i in range(2)]

    # %%
    xbar = [x[i].mean() for i in range(2)]
    # print(xbar)
    z = [(x[i] - xbar[i])/sigma for i in range(2)]
    s2 = [np.sum((x[i] - xbar[i])**2)/(n[i]-1) for i in range(2)]
    # print(s2)
    s2r_list.append(s2[0]/s2[1])


# plt.plot(s2r_list)
plt.hist(s2r_list, bins=16)
plt.xlim([0, 8])


# %%

height = np.array([[161, 158, 165, 150, 170],
                  [162, 160, 168, 153, 174]])
mean = height.mean(axis=1)
mean[1] - mean[0]

diff = height[1] - height[0]
print(diff)
print(diff.mean())
