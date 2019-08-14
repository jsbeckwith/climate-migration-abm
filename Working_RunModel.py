from Working_AdaptationModel import *
import sys
import numpy as np
import time

steps = 5 # number of time steps to run model
num_counties = 74

# sys.stdout = open('output47_climate_fix.txt', 'wt')

def runClimateMigrationModel(collect_data, filename, network_type):
    start = time.time()
    # initialize model
    model = ClimateMigrationModel(num_counties, 0, network_type=network_type)
    print('created model obj')
    model.add_agents()
    print('added agents')
    print(model.num_agents)
    if network_type == 'income':
        model.initialize_all_income_networks()
    elif network_type == 'age':
        model.initialize_all_age_networks()
    elif network_type == 'income_age':
        model.initialize_all_income_age_networks()
    else:
        model.initialize_all_networks()
    print('set ', network_type, ' networks')
    model.initialize_all_families()
    print('set family')
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

runClimateMigrationModel(False, '0810_heat75_and_dry200', 'random')