from Working_AdaptationModel import *
import sys
import numpy as np

init_population = 180  # initial population in simulation area
steps = 47  # number of time steps to run model
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

    for i in range(steps):
        print(i)
        model.step()
        print(model.county_population_list)
        print(model.county_migration_rates)
        print(model.cum_county_migration_rates)
    
    model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
    return model_attributes

data = runClimateMigrationModel()
print(data)
