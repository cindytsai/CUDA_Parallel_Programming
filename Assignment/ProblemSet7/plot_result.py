import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
data = pd.read_csv("./result/MC_CPUonly/integration.txt", sep=' ', names=col_names, header=0)

N = np.asarray(data["N"])
N_log = np.log2(N)
SS = np.asarray(data["SS"])
SSerror = np.asarray(data["SSerror"])
MA = np.asarray(data["MA"])
MAerror = np.asarray(data["MAerror"])


plt.subplot(2, 1, 1)
plt.errorbar(N_log, SS, yerr=SSerror, fmt='.b', label='Simple Sampling')
plt.legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.title("High Dimensional Integration with Different Method", fontsize=16)


plt.subplot(2, 1, 2)
plt.errorbar(N_log, MA, yerr=MAerror, fmt='.g', label='Metropolis Algorithm ' + r'$\alpha$' + "=0.25")
plt.legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.xlabel("Sample Points  " + r'$2^n$', fontsize=14)

plt.show()

print(data[["N", "SS", "SSerror", "MA", "MAerror"]])
