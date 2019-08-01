from Working_AdaptationModel import *
import sys
import numpy as np
import time

steps = 47 # number of time steps to run model
num_counties = 74

def runClimateMigrationModel(collect_data):
    start = time.time()
    # initialize model
    model = ClimateMigrationModel(num_counties, 0)
    print('created model obj')
    model.addAgents()
    print('added agents')
    print(model.num_agents)
    model.initialize_networks()
    print('set networks')
    model_creation = time.time()
    print(model_creation - start)

    model.datacollector.collect(model)  # collect initial model state variables
    # note - can store model configuration using 'pickle' for repeated runs
    step1 = model_creation
    for i in range(steps):
        print('step', i)
        model.step()
        step2 = time.time()
        print(step2 - step1)
        step1 = step2
    
    five_steps = time.time()

    if collect_data:
        model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
        model_attributes.to_csv('output/731_1000_47_climatewth.csv')
        model_attributes['County Influx'].to_csv('output/731_1000_flux_47_climatewth.csv')
        model_attributes['County Population'].to_csv('output/731_1000_pop_47_climatewth.csv')

    print('elapsed time (s):', five_steps - start)

runClimateMigrationModel(True)