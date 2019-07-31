from Working_AdaptationModel import *
import sys
import numpy as np

steps = 5 # number of time steps to run model
num_counties = 74

def runClimateMigrationModel(collect_data):

    # initialize model
    model = ClimateMigrationModel(num_counties, 0)
    print('created model obj')
    model.addAgents()
    print('added agents')
    model.initialize_networks()
    print('set networks')
    model.datacollector.collect(model)  # collect initial model state variables
    # note - can store model configuration using 'pickle' for repeated runs
    for i in range(steps):
        print('step', i)
        model.step()
    
    if collect_data:
        model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
        model_attributes.to_csv('output/729_2.csv')
        model_attributes['County Influx'].to_csv('output/730_pool.csv')
        model_attributes['County Population'].to_csv('output/730_pool_pop.csv')

runClimateMigrationModel(False)