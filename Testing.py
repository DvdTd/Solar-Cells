import pde
import matplotlib.pyplot as plt
from matplotlib import cm

fig, axs = plt.subplots(nrows=1, ncols=15, sharex=True, sharey=True, figsize=(15,15))
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()
print(axs)

