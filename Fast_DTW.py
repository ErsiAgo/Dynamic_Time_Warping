#
import os
import pandas as pd
import random
from copy import deepcopy
import numpy as np
import plotly.graph_objects as go
from dtaidistance import dtw
from numpy.core._multiarray_umath import ndarray
from plotly.subplots import make_subplots
from scipy import interpolate
import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Retrieve current working directory
cwd = os.getcwd()
cwd

# List directory
os.listdir('.')

# Import the data
df = pd.read_excel(r'C:\Users\EA\Desktop\Paper\data.xlsx')
#print(df)

series1 = pd.Series(df['G1'] + 0.307808)
series2 = pd.Series(df['G2'] - 68.187345)
series3 = pd.Series(df['G3'] - 59.33976)


#time1 = np.linspace(start=0, stop=1, num=50)
#time2 = time1[0:40]

#x1 = 3 * np.sin(np.pi * time1) + 1.5 * np.sin(4*np.pi * time1)
#x2 = 3 * np.sin(np.pi * time2 + 0.5) + 1.5 * np.sin(4*np.pi * time2 + 0.5)

distance, warp_path = fastdtw(series2, series3, dist=euclidean)

fig, ax = plt.subplots(figsize=(16, 12))
# Remove the border and axes ticks
fig.patch.set_visible(False)
ax.axis('off')

for [map_x, map_y] in warp_path:
    ax.plot([map_x, map_y], [series2[map_x], series3[map_y]], '-k')

ax.plot(series2, color='blue', marker='o', markersize=2, linewidth=2)
ax.plot(series3, color='red', marker='o', markersize=2, linewidth=2)
ax.tick_params(axis="both", which="major", labelsize=18)
ax.set(xlim=(-300, 3000))
plt.show()