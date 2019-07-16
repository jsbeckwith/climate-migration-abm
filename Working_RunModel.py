from Working_AdaptationModel import *
import sys
import numpy as np

init_population = 100  # initial population in simulation area
steps = 10  # number of time steps to run model
num_counties = 3

def slr_adaptation():

    # initialize model
    model = AdaptationModel(init_population, num_counties, 0)
    model.datacollector.collect(model)  # collect initial model state variables

    model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe

    #for n in model.nodes:
        #print(model.G.node[n]['dry'])
    return model_attributes

    

data = slr_adaptation()
print(data)
