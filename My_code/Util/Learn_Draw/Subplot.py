import matplotlib.pyplot as plt
import numpy as np


np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].hist(data[0], label = "scatter")
axs[1, 0].scatter(data[0], data[1], label = "scatter")
axs[0, 1].plot(data[0], data[1], label = "scatter")
axs[1, 1].hist2d(data[0], data[1] ,label = "scatter")

axs[1, 0].legend()
plt.show()
