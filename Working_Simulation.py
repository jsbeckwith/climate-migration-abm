from Working_RunModel import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

simulations = 10  # number of simulations run

output = []  # list of model output from each simulation run
for sim in range(simulations):
    output.append(slr_adaptation())  # run model

# Visualize output

# plot population that has not retreated as a time series for each run
# plot percent reduction in population, and year to reach 95% of reduction
retreat_metrics = np.empty((simulations, 2))
fig_pop = plt.figure()
plt.ylabel('Population')
plt.xlabel('Time (years)')
for sim in range(simulations):
    population = output[sim][1].Population
    plt.plot(population, label='run ' + str(sim))
reduction_ax.set_xlabel('Run Number')
plt.show()