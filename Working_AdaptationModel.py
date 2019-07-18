from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from census_data import *

import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ACCESS NODE OF AGENT BY USING .pos
# ACCESS AGENTS IN NODE BY USING self.G.node[n]['agent']

tick = 0  # counter for current time step

class ClimateMigrationModel(Model):
    def __init__(self, num_agents, num_nodes, init_time=0):
        super().__init__()
        self.num_agents = num_agents  # number of agents in simulation area
        self.schedule = SimultaneousActivation(self)  # agents are updated in a random order at each time step
        self.G = createGraph()  # agents are located on a network
        self.num_nodes = num_nodes
        self.nodes = self.G.nodes()
        self.grid = NetworkGrid(self.G)
        self.agent_list = []  # list of all agents - necessary ??
        self.county_population_list = [0] * self.num_nodes
        self.county_migration_rates = [0] * self.num_nodes
        self.climateData = {0: {}, 1: {}, 2: {}, 3: {}}
        global tick
        tick = init_time

        # Store model level attributes using a DataCollector object
        self.datacollector = DataCollector(
            model_reporters={"County Population": lambda m1: m1.county_population_list}
        )

    def addAgents(self):
        # Create Agents -- agents are placed randomly on the grid and added to the model schedule
        # to access node data: G.node[n][attr]
        list_of_nodes = list(self.G.nodes())
        
        """
        populationList = [0]*self.num_nodes
        for j in range(self.num_nodes):
            populationList[j] = self.G.node[j]['total_18+'] 
        """
        populationList = [750, 50, 60, 60]
        populationList = np.cumsum(populationList)
        
        m = 0
        i = 0
        while i < populationList[m]:
            ageListM = [self.G.node[m]['mpct1'], self.G.node[m]['mpct2'], self.G.node[m]['mpct3'], self.G.node[m]['mpct4']]
            ageListF = [self.G.node[m]['fpct1'], self.G.node[m]['fpct2'], self.G.node[m]['fpct3'], self.G.node[m]['fpct4']]
            cumAgeM = np.cumsum(ageListM)
            cumAgeF = np.cumsum(ageListF)

            a = Person(i, self)
            self.grid.place_agent(a, list_of_nodes[m])
            
            if random.random() < 0.5:
                a.gender = 1 # male
            
            randNum = random.random()
            if a.gender == 0:
                if randNum < cumAgeM[0]:
                    a.age = 0
                elif randNum < cumAgeM[1]:
                    a.age = 1
                elif randNum < cumAgeM[2]:
                    a.age = 2
                else:
                    a.age = 3
            else:
                if randNum < cumAgeF[0]:
                    a.age = 0
                elif randNum < cumAgeF[1]:
                    a.age = 1
                elif randNum < cumAgeF[2]:
                    a.age = 2
                else:
                    a.age = 3
            
            i += 1
            if i == populationList[m]:
                if m < len(populationList) - 1:
                    m += 1
            self.agent_list.append(a)
            self.schedule.add(a)
        
        self.updateCountyPopulation()

    def setNetworks(self):
        for a in self.schedule.agents:
            a.connections = random.sample(self.G.node[a.pos]['agent'], 3)
            a.connections += random.sample(self.schedule.agents, 4)

    def updateCountyPopulation(self):
        for n in self.nodes:
            self.county_population_list[n] = len(self.G.node[n]['agent'])

    def updateClimate(self):
        for n in self.nodes:
            self.G.node[n]['heat2013'] += self.G.node[n]['dheat_dyear']
            self.G.node[n]['rain2013'] += self.G.node[n]['drain_dyear']
            self.G.node[n]['dry2013'] += self.G.node[n]['ddry_dyear']

    def calculateCurrentClimate(self):
        for n in self.nodes:
            self.climateData[n]['heat'] = self.G.node[n]['heat2013']
            self.climateData[n]['rain'] = self.G.node[n]['rain2013']
            self.climateData[n]['dry'] = self.G.node[n]['dry2013']

    def step(self):
        global tick
        self.schedule.step()  # update agents
        self.updateCountyPopulation()
        self.updateClimate()
        self.calculateCurrentClimate()
        self.datacollector.collect(self)  # collect model level attributes for current time step
        tick += 1


class Person(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id  # unique identification number
        self.age = 0
        self.gender = 0
        self.probability = 0
        # self.married = 0
        # self.income = 0
        self.connections = []  # list of all connected agents
        self.connectedLocations = []
        self.rankedCounties = []
        self.adaptive_capacity = random.normalvariate(0, 1)  # ability to implement adaptation actions
    
    def step(self):
        self.updateNetworkLocations()
        self.calculateMigrationProbability()  # calculate new adaptative capacity

    def advance(self):
        self.make_decision()  # implement adaptation actions

    def updateNetworkLocations(self):
        self.connectedLocations = []
        for i in range(7):
            self.connectedLocations.append(self.connections[i].pos)
    
    def rankCountiesByNetwork(self):
        countyList = [0]*self.model.num_nodes
        for n in self.connectedLocations:
            countyList[n] += 1
        self.rankedCounties = countyList # use index to get county/node id of max

    def calculateMigrationProbability(self):
        if self.age == 0:
            self.probability += 0.042
        if self.age == 1:
            self.probability += 0.03
        if self.age == 2:
            self.probability += 0.013
        if self.age == 3:
            self.probability += 0.009
        if self.gender == 0:
            self.probability += 0
        if self.gender == 1:
            self.probability += 0.001

    def calculate_adaptive_capacity(self):
        self.adaptive_capacity = random.random()

    def make_decision(self):
        currentHeat = self.model.climateData[self.pos]['heat']
        if currentHeat > 30:
            if random.random() < self.probability:
                self.rankCountiesByNetwork()
                maxNetworkCounty = self.rankedCounties.index(max(self.rankedCounties))
                for i in range(4):
                    if currentHeat <= self.model.climateData[i]['heat']:
                        self.rankedCounties.pop(i)
                maxNetworkCounty = self.rankedCounties.index(max(self.rankedCounties)) # MAX WILL RETURN FIRST VALUE IF A TIE
                self.model.grid.move_agent(self, maxNetworkCounty)

def createGraph():
    # create a perfectly connected graph of all counties (k^78)
    # G_counties = nx.complete_graph(78)
    # use pandas to manipulate county demographic data & climate data
    # add attributes to nodes via dictionaries as in line (109)

    nodeData = mergeClimateDemographic()
    G = nx.complete_graph(4)

    nx.set_node_attributes(G, nodeData)
    # need to figure out a better way to do this - eventually
    nx.set_edge_attributes(G, {(0, 1): 2294, (0, 2): 2591, (0, 3): 826, (1, 2): 394, (1, 3): 2347, (2, 3): 2532}, 'distance')

    return G