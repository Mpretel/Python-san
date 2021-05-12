#%%
from scipy.stats import wilcoxon
import numpy as np



d = np.array([6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75])
x = np.array([125,115,130,140,140,115,140,125,140,135])
y = np.array([110,122,125,120,140,124,123,137,135,145])

z = x-y

w, p = wilcoxon(x,y)
print([w, p])
w, p = wilcoxon(z)
print([w, p])
w, p = wilcoxon(d)
print([w, p])
# %%
