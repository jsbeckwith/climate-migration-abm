from Working_AdaptationModel import *
import sys
import numpy as np

init_population = 180  # initial population in simulation area
steps = 20  # number of time steps to run model
num_counties = 4

def runClimateMigrationModel():

    # initialize model
    model = ClimateMigrationModel(num_counties, 0)
    print('created model obj')
    model.addAgents()
    print('added agents')
    model.setNetworks()
    print('set networks')
    model.calculateCurrentClimate()
    print('initial climate data')
    model.datacollector.collect(model)  # collect initial model state variables
    model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
    print(model_attributes)

    for i in range(steps):
        model.step()
    
    return model_attributes

data = runClimateMigrationModel()
print(data)
