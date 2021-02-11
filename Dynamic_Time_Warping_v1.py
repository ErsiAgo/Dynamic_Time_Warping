#
# Import os
import os
import matplotlib.pyplot as plt
import pandas as pd
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
print(df)


def plot_graphs():
    # Plotting
    plt.plot(df['G1'] + 0.307808)
    plt.plot(df['G2'] - 68.187345)
    plt.plot(df['G3'] - 59.33976)

    # plt.plot(df['G1']+0)
    # plt.plot(df['G2']-0)
    # plt.plot(df['G3']-0)

    # Adding labels and axis
    ax = plt.gca()
    plt.grid()
    # plt.title('Rotor Angle plot')
    plt.xlabel('Time (sec)')
    plt.ylabel('Rotor Angle (Deg)')
    ax.set(xlim=(-300, 3000))
    ax.xaxis.set_ticklabels(np.arange(0, 3.5, 0.5).round(1))
    plt.show()


#plot_graphs()

series1 = pd.Series(df['G1'] + 0.307808)
series2 = pd.Series(df['G2'] - 68.187345)
series3 = pd.Series(df['G3'] - 59.33976)


##Fill DTW Matrix
def fill_dtw_cost_matrix(s1, s2):
    l_s_1, l_s_2 = len(s1), len(s2)
    cost_matrix = np.zeros((l_s_1 + 1, l_s_2 + 1))
    for i in range(l_s_1 + 1):
        for j in range(l_s_2 + 1):
            cost_matrix[i, j] = np.inf
    cost_matrix[0, 0] = 0

    for i in range(1, l_s_1 + 1):
        for j in range(1, l_s_2 + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            # take last min from the window
            prev_min = np.min([cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]])
            cost_matrix[i, j] = cost + prev_min
    return cost_matrix


##Call DTW function
dtw_cost_matrix = fill_dtw_cost_matrix(series1, series2)

# plt.plot(series1)
# plt.plot(series2)
#plt.plot(dtw_cost_matrix)

# plt.plot(series1)

plt.show()
