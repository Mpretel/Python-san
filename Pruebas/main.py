# %%
# from funs import myprint
from funs2 import *

myprint()
hola()

# %%
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

conmat = np.ones((11,6,6))

media = stats.describe(conmat).mean
var = stats.describe(conmat).variance

# plt.imshow(con[:,:,0],cmap='Reds')
plt.imshow(media,cmap='Reds')
plt.colorbar()
plt.show()
plt.imshow(var,cmap='Reds')
plt.colorbar()
plt.show()
# %%
