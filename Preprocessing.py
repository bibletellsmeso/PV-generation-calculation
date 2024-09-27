import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import sys


Sim_time = 96 # 15min
Weather_Forecast = np.genfromtxt('Hawaii weather.csv', delimiter=',', skip_header=1, dtype=np.float32)
Ir = Weather_Forecast[:,2]
Tem = Weather_Forecast[:,3]
real_PV = np.loadtxt("PV_for_scheduling.txt")

emtpy_Predict_PV = np.full(Sim_time, np.nan)
empty_Ir = np.full(Sim_time, np.nan)
empty_Tem = np.full(Sim_time, np.nan)

POA = np.zeros(Sim_time)
Ta = np.zeros(Sim_time)

for i in range(96):
    POA[i] = Ir[15*i]
    Ta[i] = Tem[15*i]
    
# hour = np.zeros(Sim_time)
# for i in range(23):
#     hour[i] = np.repeat(i, 4)
   
x = np.array(range(24))
hour = np.repeat(x, 4)
    
y = np.array([0, 15, 30, 45])
min = np.tile(y, 24)