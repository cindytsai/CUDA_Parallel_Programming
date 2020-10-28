import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get N
alpha = "0_2"
col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
N = np.asarray(data["N"])
N_log = np.log2(N)

# Plot Metropolis Algrorithm Result, with different alpha
plt.title("High Dimensional Integration with Metropolis Algorithm", fontsize=16)

# alpha = "0_2"
# col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
# data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
# MA = np.asarray(data["MA"])
# MAerror = np.asarray(data["MAerror"])
# plt.errorbar(N_log, MA, yerr=MAerror, fmt='.-', label='Metropolis Algorithm ' + r'$\alpha$' + "=0.2")

# alpha = "0_25"
# col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
# data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
# MA = np.asarray(data["MA"])
# MAerror = np.asarray(data["MAerror"])
# plt.errorbar(N_log, MA, yerr=MAerror, fmt='.-', label='Metropolis Algorithm ' + r'$\alpha$' + "=0.25")

# alpha = "0_3"
# col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
# data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
# MA = np.asarray(data["MA"])
# MAerror = np.asarray(data["MAerror"])
# plt.errorbar(N_log, MA, yerr=MAerror, fmt='.-', label='Metropolis Algorithm ' + r'$\alpha$' + "=0.3")

alpha = "0_5"
col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
MA = np.asarray(data["MA"])
MAerror = np.asarray(data["MAerror"])
plt.errorbar(N_log, MA, yerr=MAerror, fmt='.-', label='Metropolis Algorithm ' + r'$\alpha$' + "=0.5")

alpha = "0_8"
col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
MA = np.asarray(data["MA"])
MAerror = np.asarray(data["MAerror"])
plt.errorbar(N_log, MA, yerr=MAerror, fmt='.-', label='Metropolis Algorithm ' + r'$\alpha$' + "=0.8")

alpha = "1_0"
col_names = ["N", "SS", "sign1", "SSerror", "MA", "sign3", "MAerror"]
data = pd.read_csv("./result/integration_" + alpha + ".txt", sep=' ', names=col_names, header=0)
MA = np.asarray(data["MA"])
MAerror = np.asarray(data["MAerror"])
plt.errorbar(N_log, MA, yerr=MAerror, fmt='.-', label='Metropolis Algorithm ' + r'$\alpha$' + "=1.0")

plt.legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.xlabel("Sample Points  " + r'$2^n$', fontsize=14)


plt.show()

# print(data[["N", "SS", "SSerror", "MA", "MAerror"]])
