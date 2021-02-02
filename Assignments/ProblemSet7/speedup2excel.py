import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read from data #TODO
algorithm = "MA"
ngpu = "2GPU"
sampleNum = "81920000"

# For Simple Sampling # TODO
col_names = ["N", "BlockSize", "GridSize", "SSGPU", "SSGPUerror", "SSCPU", "sign", "SSCPUerror", "SpeedUp"]
# For Metropolis Algorithm #TODO
# col_names = ["N", "BlockSize", "GridSize", "MAGPU", "MAGPUerror", "MACPU", "sign", "MACPUerror", "SpeedUp"]

data = pd.read_csv("./result/MC_GPU_CPU/" + algorithm + "_" + ngpu + "_" + sampleNum + ".txt", sep=' ', names=col_names, header=0)

# Get data as numpy array
N = np.asarray(data[col_names[0]])
BlockSize = np.asarray(data[col_names[1]])
GridSize = np.asarray(data[col_names[2]])
GPU = np.asarray(data[col_names[3]])
GPUerror = np.asarray(data[col_names[4]])
CPU = np.asarray(data[col_names[5]])
CPUerror = np.asarray(data[col_names[7]])
SpeedUp = np.asarray(data[col_names[8]])


SpeedUpGrid = SpeedUp.reshape(10, 5)
SpeedUpGrid = SpeedUpGrid.T
print(SpeedUpGrid)

# Output data as excel file
SpeedUpData = pd.DataFrame(SpeedUpGrid)
SpeedUpData.to_excel("temp.xlsx", index=False)
