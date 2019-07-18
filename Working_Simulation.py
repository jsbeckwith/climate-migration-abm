from Working_RunModel import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

simulations = 10  # number of simulations run

output = []  # list of model output from each simulation run
for sim in range(simulations):
    output.append(runClimateMigrationModel())  # run model

# after first run nothing changes (?)
# how to calculate new demographics !?