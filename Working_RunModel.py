from Working_AdaptationModel import *
import sys
import numpy as np

steps = 20  # number of time steps to run model
num_counties = 4

def runClimateMigrationModel():

    # initialize model
    model = ClimateMigrationModel(num_counties, 0)
    print('created model obj')
    model.addAgents()
    print('added agents')
    model.initialize_networks()
    print('set networks')
    # model.calculateCurrentClimate()
    print('initial climate data')
    model.datacollector.collect(model)  # collect initial model state variables
    # note - can store model configuration using 'pickle' for repeated runs
    for i in range(steps):
        model.step()
    
    model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
    return model_attributes

data = runClimateMigrationModel()
print(data)
