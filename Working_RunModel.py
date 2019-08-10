from Working_AdaptationModel import *
import sys
import numpy as np
import time

steps = 47 # number of time steps to run model
num_counties = 74

sys.stdout = open('output.txt', 'wt')

def runClimateMigrationModel(collect_data, filename):
    start = time.time()
    # initialize model
    model = ClimateMigrationModel(num_counties, 0)
    print('created model obj')
    model.add_agents()
    print('added agents')
    print(model.num_agents)
    model.initialize_networks()
    print('set networks')
    model_creation = time.time()
    print(model_creation - start)

    model.datacollector.collect(model)  # collect initial model state variables
    step1 = model_creation
    model.update_income_counts()
    for i in range(steps):
        print('step', i)
        model.step()
        step2 = time.time()
        print(step2 - step1)
        step1 = step2
    model.update_income_counts()

    five_steps = time.time()

    print(model.G.edges(data=True)) # figure out how to export
    if collect_data:
        model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
        model_attributes.to_csv('output/' + filename + '.csv')
        model_attributes['County Influx'].to_csv('output/'+ filename + '_flux.csv')
        model_attributes['County Population'].to_csv('output/' + filename + '_pop.csv')

    print('elapsed time (s):', five_steps - start)

runClimateMigrationModel(True, '0809_heat75_and_dry200')