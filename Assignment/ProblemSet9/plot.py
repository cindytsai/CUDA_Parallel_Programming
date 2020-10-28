import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read results from Metroplolis
folder = "./result/DifferentT/"
filename = "allE.txt"
data = pd.read_csv(folder + filename, sep=' ', header=0)

T = data["T"]
E = data["E"]
Eerr = data["Eerror"]

# Read results from Exact Solution
folder = "./result/DifferentT/"
filename = "exactE.txt"
data = pd.read_csv(folder + filename, sep=' ', header=0)

Eexact = data["E"]

plt.title(r"$\langle$" + "E" + r"$\rangle$", fontsize=16)
plt.errorbar(T, E, yerr=Eerr, fmt='o', ecolor='b', color='b', label="Metroplolis")
plt.scatter(T, Eexact, label="Exact", color='r')
plt.xlabel("Temperature", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.legend()
plt.show()

# Read results from Metroplolis
folder = "./result/DifferentT/"
filename = "allM.txt"
data = pd.read_csv(folder + filename, sep=' ', header=0)

T = data["T"]
M = data["M"]
Merr = data["Merror"]

# Read results from Exact Solution
folder = "./result/DifferentT/"
filename = "exactM.txt"
data = pd.read_csv(folder + filename, sep=' ', header=0)

Mexact = data["M"]


plt.title(r"$\langle$" + "M" + r"$\rangle$", fontsize=16)
plt.errorbar(T, M, yerr=Merr, fmt='o', ecolor='b', color='b', label="Metroplolis")
plt.scatter(T, Mexact, label="Exact", color='r')
plt.xlabel("Temperature", fontsize=14)
plt.ylabel("Magnetization", fontsize=14)
plt.legend()
plt.show()
