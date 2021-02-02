import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = './result/hist_gmem.dat'

# Read from data
data = pd.read_csv(filename, sep=' ', header=None, skiprows=1)
x = data[0]
p = data[1]
x = np.asarray(x)
p = np.asarray(p)

# Normalized p(x)
p = p / p[0]

# Plot exp(-x) distribution
x_coor = np.linspace(0, x.max(), 1000, endpoint=True)

plt.bar(x, p, width=(x[1] - x[0]), label="random number distribution")
plt.plot(x_coor, np.exp(-x_coor), '-r', label="analytical")
plt.title("Random Number with Distribution " + r'$e^{-x}$', fontsize=16)
plt.ylim(0, 1)
plt.xlim(x.min(), x.max())
plt.xlabel("x", fontsize=14)
plt.legend(loc="upper right")
plt.show()
