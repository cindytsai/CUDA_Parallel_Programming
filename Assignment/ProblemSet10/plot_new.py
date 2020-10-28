import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read Data
folder = "./result/"
filename = "phi_new.txt"
data = pd.read_csv(folder + filename, sep=' ', header=None)

phi_X = np.asarray(data[0])
r_X = np.asarray(data[1])

phi_D = np.asarray(data[2])
r_D = np.asarray(data[3])

# Choose the non-redundant range to plot
plot_point = int(np.ceil(len(phi_X) / 2))
phi_X = phi_X[2:plot_point + 1]
r_X = r_X[2:plot_point + 1]

print(phi_X)
print(r_X)

plot_point = int(np.ceil(len(phi_D) / 2))
phi_D = phi_D[2:plot_point + 1]
r_D = r_D[2:plot_point + 1]

print(phi_D)
print(r_D)

r_Exact = np.linspace(r_X.min(), r_D.max(), num=1000, endpoint=True)
phi_exact = (1. / (4. * np.pi)) * ((1.0 / r_Exact) - 1.0)

plt.scatter(r_X, phi_X, label="X axis")
plt.scatter(r_D, phi_D, label="Diagnol")
plt.plot(r_Exact, phi_exact, 'c-', label="ExactSolution")
plt.title("Exact and Simulation Solution", fontsize=16)
plt.xlabel("r", fontsize=14)
plt.ylabel(r"$\phi(r)$" + " - " + r"$\phi(1)$", fontsize=14)
plt.legend()
plt.show()
