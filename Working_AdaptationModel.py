from census_data import *
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ACCESS NODE OF AGENT BY USING .pos
# ACCESS AGENTS IN NODE BY USING G.node[n]['agent']
# ACCESS NODE DATA USING G.node[n]['attr']

tick = 0  # counter for current time step

class ClimateMigrationModel(Model):
    def __init__(self, num_nodes, init_time=0):
        super().__init__()
        global tick
        tick = init_time
        self.num_agents = 0
        self.agent_index = 0
        self.schedule = SimultaneousActivation(self) 
        self.G = createGraph()
        self.num_nodes = num_nodes
        self.nodes = self.G.nodes()
        self.grid = NetworkGrid(self.G)
        self.county_population_list = [0] * self.num_nodes
        self.county_migration_rates = [0] * self.num_nodes
        self.cum_county_migration_rates = [0] * self.num_nodes
        self.deaths = [0] * self.num_nodes
        self.births = [0] * self.num_nodes
        self.climateData = {0: {}, 1: {}, 2: {}, 3: {}} # model-level climate data necessary? or just easier to track?
        self.datacollector = DataCollector(model_reporters={   
                                "County Population": lambda m1: list(m1.county_population_list), 
                                "County Migration Rates": lambda m2: m2.county_migration_rates,
                                "Deaths": lambda m3: m3.deaths, "Births": lambda m4: m4.births, 
                                "Total Population": lambda m5: m5.num_agents})

    def addAgents(self):
        """
        populationList = [0]*self.num_nodes
        for j in range(self.num_nodes):
            populationList[j] = self.G.node[j]['total_18+'] 
        """
        populationList = [323, 26, 28, 30]
        self.county_population_list = populationList
        populationList = list(np.cumsum(populationList))
        self.num_agents = populationList[3]
        self.agent_index = populationList[3]

        m = 0
        i = 0

        while i < populationList[m]:
            a = Household(i, self)
            self.grid.place_agent(a, list(self.nodes)[m])
            self.schedule.add(a)
            
            a.originalPos = a.pos
            a.initialize_agent()
            i += 1

            if i == populationList[m] and m < self.num_nodes - 1:
                m += 1

    def initialize_networks(self):
        for a in self.schedule.agents:
            a.connections = random.sample(self.G.node[a.pos]['agent'], 3)
            a.connections += random.sample(self.schedule.agents, random.randint(1, 4))

    def updateCountyPopulation(self):
        self.deaths = [0]*self.num_nodes
        self.births = [0]*self.num_nodes
        
        for a in self.schedule.agents:
            deathVariable = random.randint(-5, 10)
            if a.age > 78 + deathVariable:
                self.deaths[a.pos] += 1
                self.grid._remove_agent(a, a.pos)
                self.schedule.remove(a)
                self.num_agents -= 1
        
        for m in self.county_population_list:
            toAdd = m//30 # birth rate, completely unfounded
            self.num_agents += toAdd
            for i in range(toAdd):
                self.agent_index += 1
                a = Household(self.agent_index, self)
                self.grid.place_agent(a, self.county_population_list.index(m))
                a.originalPos = a.pos
                a.age = 18
                a.initialize_agent()
                self.schedule.add(a)
                self.births[a.pos] += 1
                
        for n in self.nodes:
            self.county_population_list[n] = len(self.G.node[n]['agent'])

    def updateClimate(self):
        for n in self.nodes:
            self.G.node[n]['climate'][0] += self.G.node[n]['climate'][3]
            self.G.node[n]['climate'][5] += self.G.node[n]['climate'][7]
            self.G.node[n]['climate'][9] += self.G.node[n]['climate'][11]

    def calculateCurrentClimate(self):
        for n in self.nodes:
            self.climateData[n]['heat'] = self.G.node[n]['climate'][0]
            self.climateData[n]['rain'] = self.G.node[n]['climate'][5]
            self.climateData[n]['dry'] = self.G.node[n]['climate'][9]
    
    def calculateMigrationRates(self):
        self.county_migration_rates = [0]*self.num_nodes
        for n in self.nodes:
            for a in self.G.node[n]['agent']:
                if a.originalPos != n:
                    self.cum_county_migration_rates[n] += 1
                    self.county_migration_rates[n] += 1
                    # could eventually factor in edges to get a better picture of Where ?
    
    def step(self):
        global tick
        self.schedule.step() 
        self.updateCountyPopulation()
        self.updateClimate()
        self.calculateCurrentClimate()
        self.calculateMigrationRates()
        self.datacollector.collect(self)  
        tick += 1


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id  
        self.originalPos = None
        self.age = 0
        self.probability = 0
        self.household = 0 # 0 = married, 1 = other, 2 = alone, 3 = not alone
        self.income = 0
        self.tenure = 0 # 0 = own, 1 = rent
        self.children = 0 # 0 = none, 1 = some
        self.connections = []  # list of all connected agents
        self.connectedLocations = []
        self.rankedCounties = []
        self.adaptive_capacity = 0  # ability to implement adaptation actions
    
    def initialize_agent(self):
        self.initialize_age(random.random())
        self.initialize_income(random.random())
        self.initialize_tenure(random.random())
        self.initialize_household(random.random())
        if self.household != 3:
            self.initialize_children(random.random())

    def initialize_age(self, rand_num):
        age_list = self.model.G.node[self.pos]['age']
        if rand_num < age_list[0]:
            self.age = random.randint(18, 24)
        elif rand_num < age_list[1]:
            self.age = random.randint(25, 44)
        elif rand_num < age_list[2]:
            self.age = random.randint(45, 64)
        else:
            self.age = random.randint(65, 80)

    def initialize_income(self, rand_num):
        if self.age < 25:
            income_list = self.model.G.node[self.pos]['u25income']
        elif self.age < 45:
            income_list = self.model.G.node[self.pos]['income2544']
        elif self.age < 65:
            income_list = self.model.G.node[self.pos]['income4564']
        else:
            income_list = self.model.G.node[self.pos]['income65a']
        
        if rand_num < income_list[0]:
            self.income = 0
        elif rand_num < income_list[1]:
            self.income = 1
        elif rand_num < income_list[2]:
            self.income = 2
        elif rand_num < income_list[3]:
            self.income = 3
        elif rand_num < income_list[4]:
            self.income = 4
        elif rand_num < income_list[5]:
            self.income = 5
        elif rand_num < income_list[6]:
            self.income = 6
        elif rand_num < income_list[7]:
            self.income = 7
        elif rand_num < income_list[8]:
            self.income = 8
        else:
            self.income = 9

    def initialize_tenure(self, rand_num):
        tenure_list = self.model.G.node[self.pos]['tenure']
        if rand_num > tenure_list[self.income]:
            self.tenure = 1
    
    def initialize_household(self, rand_num):
        if self.age < 35:
            if self.tenure == 0:
                household_list = self.model.G.node[self.pos]['own1534']
            else:
                household_list = self.model.G.node[self.pos]['rent1534']
        elif self.age < 65:
            if self.tenure == 0:
                household_list = self.model.G.node[self.pos]['own3564']
            else:
                household_list = self.model.G.node[self.pos]['rent3564']
        else:
            if self.tenure == 0:
                household_list = self.model.G.node[self.pos]['own65a']
            else:
                household_list = self.model.G.node[self.pos]['rent65a']
        
        if rand_num < household_list[0]:
            self.household = 0
        elif rand_num < household_list[1]:
            self.household = 1
        elif rand_num < household_list[2]:
            self.household = 2
        else:
            self.household = 3
    
    def initialize_children(self, rand_num):
        children_list = self.model.G.node[self.pos]['children']
        if rand_num < children_list[self.household]:
            self.children = 1

    def initialize_network(self):
        self.connections = random.sample(self.model.G.node[self.pos]['agent'], 3)
        self.connections += random.sample(self.model.schedule.agents, random.randint(1, 4))

    def step(self):
        self.updateAge()
        self.calculateMigrationProbability()  # calculate new adaptative capacity

    def advance(self):
        self.make_decision()  # implement adaptation actions

    def updateAge(self):
        self.age += 1

    def updateNetworkLocations(self):
        self.connectedLocations = []
        for i in range(len(self.connections)):
            self.connectedLocations.append(self.connections[i].pos)
    
    def rankCountiesByNetwork(self):
        countyList = [0]*self.model.num_nodes
        for n in self.connectedLocations:
            countyList[n] += 1
        self.rankedCounties = countyList # use index to get county/node id of max

    def calculateMigrationProbability(self):
        if self.age < 25:
            self.probability += 0.042
        elif self.age < 45:
            self.probability += 0.03
        elif self.age < 65:
            self.probability += 0.013
        else:
            self.probability += 0.009

    def calculate_adaptive_capacity(self):
        self.adaptive_capacity = random.random()

    def make_decision(self):
        currentHeat = self.model.climateData[self.pos]['heat']
        if currentHeat > 30:
            if random.random() < self.probability:
                self.calculate_adaptive_capacity()
                if self.adaptive_capacity > 0.5:
                    self.updateNetworkLocations()
                    self.rankCountiesByNetwork()
                    for i in range(len(self.rankedCounties)):
                        if currentHeat <= self.model.climateData[i]['heat']:
                            self.rankedCounties[i] = -1
                    maxNetworkCounty = self.rankedCounties.index(max(self.rankedCounties)) # MAX WILL RETURN FIRST VALUE IF A TIE
                    self.model.grid.move_agent(self, maxNetworkCounty)

def createGraph():
    # create a perfectly connected graph of all counties (k^78)
    # G_counties = nx.complete_graph(78)
    # use pandas to manipulate county demographic data & climate data
    # add attributes to nodes via dictionaries

    nodeData = mergeDicts()
    G = nx.complete_graph(4)

    nx.set_node_attributes(G, nodeData)
    # need to figure out a better way to do this - eventually
    nx.set_edge_attributes(G, {(0, 1): 2294, (0, 2): 2591, (0, 3): 826, (1, 2): 394, (1, 3): 2347, (2, 3): 2532}, 'distance')

    return G