import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read from data
algorithm = "MA"
ngpu = "2GPU"
sampleNum = "81920000"

col_names = ["N", "BlockSize", "GridSize", "SSGPU", "SSGPUerror", "SSCPU", "sign", "SSCPUerror", "SpeedUp"]
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

# Error = |(GPU - CPU)| / CPU
Error = (GPU - CPU) / CPU
ErrorGrid = Error.reshape(10, 5)
ErrorGrid = ErrorGrid.T
ErrorGrid = np.abs(ErrorGrid)
print(ErrorGrid)

# Output data as excel file
ErrorData = pd.DataFrame(ErrorGrid)
ErrorData.to_excel("temp.xlsx", index=False)
