from Working_AdaptationModel import *
import sys
import numpy as np
import time

def runClimateMigrationModel(collect_data, filename, preferences, network_type, climate_threshold, init_time):
    print(preferences, network_type, climate_threshold)
    start = time.time()
    # initialize model
    model = ClimateMigrationModel(num_counties=74, preferences=preferences, \
        network_type=network_type, climate_threshold=climate_threshold, init_time=0)
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
        model.initialize_all_random_networks()
    print('set', network_type, 'networks')
    networks_set = time.time()
    print(networks_set - start)
    model.initialize_all_families()
    print('set family')
    family_set = time.time()
    print(family_set - networks_set)
    model_creation = time.time()
    print('time to initialize model (s):')
    print(model_creation - start)

    model.datacollector.collect(model)  # collect initial model state variables
    step1 = model_creation
    model.update_income_counts()
    for i in range(47):
        print('step', i)
        model.step()
        step2 = time.time()
        print(step2 - step1)
        step1 = step2
    model.update_income_counts()

    five_steps = time.time()

    if collect_data:
        model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe
        model_attributes.to_csv('output/' + filename + '.csv')
        model_attributes['County Influx'].to_csv('output/'+ filename + '_flux.csv')
        model_attributes['County Population'].to_csv('output/' + filename + '_pop.csv')
    
    print(model.G.edges(data=True)) # figure out how to export
    model.get_preference_distribution()
    
    print('elapsed time (s):', five_steps - start)

def single_run():
    sys.stdout = open('clim_pref.txt', 'wt')
    runClimateMigrationModel(True, 'clim_pref', preferences=True, network_type='random', climate_threshold=[1, 51], init_time= 0)

def multiple_run_network():
    sys.stdout = open('no_pref_51', 'wt')
    # threshold_list = [[0, 50, 200], [0, 100, 250], [0, 200, 300]]
    # network_list = ['income', 'age', 'income_age']
    # for network in network_list:
    for i in range(10):
        filename = 'no_pref_'
        filename += str(i)
        print('RUN', i,)
        runClimateMigrationModel(True, filename, False, 'random', [1, 51], init_time=0)

multiple_run_network()