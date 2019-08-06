from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from statistics import mean

import random
import math
import pickle
import networkx as nx
import numpy as np
import pandas as pd

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
        self.county_climate_ranking = []
        self.county_population_list = [0] * self.num_nodes
        self.county_influx = [0] * self.num_nodes
        self.deaths = []
        self.births = []
        self.datacollector = DataCollector(model_reporters={   
                                "County Population": lambda m1: list(m1.county_population_list), 
                                "County Influx": lambda m2: m2.county_influx,
                                "Deaths": lambda m3: m3.deaths, "Births": lambda m4: m4.births, 
                                "Total Population": lambda m5: m5.num_agents})

    def addAgents(self):
        cumulative_population_list = get_cumulative_population_list()
        self.county_population_list = get_population_list()
        self.num_agents = cumulative_population_list[73]
        self.agent_index = cumulative_population_list[73]

        m = 0
        i = 0

        while i < cumulative_population_list[m]:
            a = Household(i, self)
            self.grid.place_agent(a, list(self.nodes)[m])
            self.schedule.add(a)
            
            a.originalPos = a.pos
            a.initialize_agent()
            i += 1

            if i == cumulative_population_list[m] and m < self.num_nodes - 1:
                m += 1

    def initialize_networks(self):
        for a in self.schedule.agents:
            print(a.unique_id)
            a.connections = random.sample(self.G.node[a.pos]['agent'], 3)
            a.connections += random.sample(self.schedule.agents, random.randint(1, 3))


    def updateCountyPopulation(self):
        self.deaths = [0]*self.num_nodes
        self.births = [0]*self.num_nodes
        
        for a in self.schedule.agents:
            # better/more accurate way to do this? lol
            deathVariable = random.randint(-5, 10)
            if a.age > 78 + deathVariable:
                self.deaths[a.pos] += 1
                self.grid._remove_agent(a, a.pos)
                self.schedule.remove(a)
                self.num_agents -= 1
        
        for m in range(self.num_nodes):
            # better/more accurate way to do this? lol
            current_population = self.county_population_list[m]
            toAdd = current_population//30 # birth rate
            self.num_agents += toAdd
            for i in range(toAdd):
                
                self.agent_index += 1
                
                a = Household(self.agent_index, self)
                self.grid.place_agent(a, m)
                self.schedule.add(a)

                a.originalPos = a.pos
                a.age = 18
                a.initialize_agent()
                a.initialize_network()
                
                self.births[a.pos] += 1
                
        for n in self.nodes:
            self.county_population_list[n] = len(self.G.node[n]['agent'])

    def updateClimate(self):
        # explain numbers ? also, more sophisticated than linear ?
        for n in self.nodes:
            self.G.node[n]['climate'][1] += self.G.node[n]['climate'][3]
            self.G.node[n]['climate'][4] += self.G.node[n]['climate'][6]
            self.G.node[n]['climate'][7] += self.G.node[n]['climate'][9]

    def rank_by_climate(self):
        # necessary ? - how to collect/show model-level data is the bigger question
        heat_data = []
        dry_data = []
        heat_dry_data = []
        
        for n in self.nodes:
            heat_data.append(self.G.node[n]['climate'][1])
            dry_data.append(self.G.node[n]['climate'][7])
        
        max_heat = max(heat_data)
        max_dry = max(dry_data)
        heat_data = [(e/max_heat) for e in heat_data]
        dry_data = [(e/max_dry) for e in dry_data]
        
        for i in range(self.num_nodes):
            heat_dry_data.append(heat_data[i] + dry_data[i])
        
        heat_dry_data = np.array(heat_dry_data)
        county_climate_rank = np.argsort(heat_dry_data)
        self.county_climate_ranking = list(county_climate_rank)
    
    def step(self):
        global tick
        self.rank_by_climate()
        self.schedule.step() 
        self.updateCountyPopulation()
        self.updateClimate()
        self.datacollector.collect(self)  
        tick += 1


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id  
        self.originalPos = None
        self.age = 0
        self.probability = [0] * 2 # age, tenure, children
        self.household = 0 # 0 = married, 1 = other, 2 = alone, 3 = not alone
        self.income = 0
        self.tenure = 0 # 0 = own, 1 = rent
        self.children = 0 # 0 = none, 1 = some
        self.connections = []  # list of all connected agents
        self.counties_by_network = []
        self.adaptive_capacity = 0  # ability to implement adaptation actions
    
    def initialize_agent(self):
        self.initialize_age(random.random())
        self.initialize_income(random.random())
        self.initialize_tenure(random.random())
        # self.initialize_household(random.random())
        # if self.household != 3:
            # self.initialize_children(random.random())

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
            self.income = 1
        elif rand_num < income_list[1]:
            self.income = 2
        elif rand_num < income_list[2]:
            self.income = 3
        elif rand_num < income_list[3]:
            self.income = 4
        elif rand_num < income_list[4]:
            self.income = 5
        elif rand_num < income_list[5]:
            self.income = 6
        elif rand_num < income_list[6]:
            self.income = 7
        elif rand_num < income_list[7]:
            self.income = 8
        elif rand_num < income_list[8]:
            self.income = 9
        else:
            self.income = 10

    def initialize_tenure(self, rand_num):
        tenure_list = self.model.G.node[self.pos]['tenure']
        if rand_num > tenure_list[self.income-1]:
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
        self.connections += random.sample(self.model.schedule.agents, random.randint(1, 3))

    def step(self):
        self.updateAge()
        self.calculateMigrationProbability()

    def advance(self):
        self.make_climate_decision()

    def updateAge(self):
        self.age += 1
    
    def rank_counties_by_network(self):
        connected_locations = []
        self.counties_by_network = [0]*self.model.num_nodes
        for i in range(len(self.connections)):
            connected_locations.append(self.connections[i].pos)
        for n in connected_locations:
            self.counties_by_network[n] += 1

    def calculateMigrationProbability(self):
        # data from census report 2005-2010
        if self.age < 25:
            self.probability[0] += 0.179
        elif self.age < 29:
            self.probability[0] += 0.248
        elif self.age < 45:
            self.probability[0] += 0.156
        elif self.age < 65:
            self.probability[0] += 0.082
        elif self.age < 75:
            self.probability[0] += 0.064
        else:
            self.probability[0] += 0.046
        
        if self.tenure == 0:
            self.probability[1] += 0.084
        else:
            self.probability[1] += 0.212

        # figure out heuristic for children, household type
        # random distribution ?

    def calculate_adaptive_capacity(self):
        # how might it change with age, household type?
        self.adaptive_capacity = self.income/10
        if self.age > 65:
            self.adaptive_capacity -= self.age/600
        # if children, decrease adaptive capacity
        # if alone, increase

    def make_decision(self):
        if random.random() < mean(self.probability):
            
            to_choose = []
            to_move = None
            radius = (self.adaptive_capacity) * 3000

            for i in range(self.model.num_nodes):
                if i != self.pos:
                    distance = self.model.G.get_edge_data(self.pos, i)['distance']
                    if distance < radius:
                        for j in range(3000//distance):
                            to_choose.append(i)

            self.rank_counties_by_network()

            for i in range(len(self.counties_by_network)):
                if i != self.pos:
                    for j in range(self.counties_by_network[i]):
                        to_choose.append(i)
            
            if to_choose:
                to_move = random.choice(to_choose)
                self.model.county_influx[self.pos] -= 1
                self.model.county_influx[to_move] += 1
                self.model.grid.move_agent(self, to_move)
    
    def make_climate_decision(self):
        if random.random() < mean(self.probability):
            to_choose = []
            to_move = None
            radius = (self.adaptive_capacity) * 3000

            for i in range(self.model.num_nodes):
                if i != self.pos:
                    distance = self.model.G.get_edge_data(self.pos, i)['distance']
                    if distance < radius:
                        for j in range(3000//distance):
                            to_choose.append(i)
            
            self.rank_counties_by_network()

            for i in range(len(self.counties_by_network)):
                if i != self.pos:
                    for j in range(self.counties_by_network[i]):
                        to_choose.append(i)
            
            # need threshold - otherwise everyone will leave

            current_county_climate_rank = self.model.county_climate_ranking.index(self.pos)
            if current_county_climate_rank > 50:
                for i in range(current_county_climate_rank):
                    if self.model.county_climate_ranking[i] in to_choose:
                        to_choose.append(i)
                
                for i in range(current_county_climate_rank, self.model.num_nodes):
                    county = self.model.county_climate_ranking[i]
                    if county in to_choose:
                        to_choose.remove(county)
                
            if to_choose:
                to_move = random.choice(to_choose)
                self.model.county_influx[self.pos] -= 1
                self.model.county_influx[to_move] += 1
                self.model.grid.move_agent(self, to_move)


def createGraph():
    with open('real_data_dict.pickle', 'rb') as node_data_file:
        node_data = pickle.load(node_data_file)

    with open('distance_dict.pickle', 'rb') as edge_data_file:
        edge_data = pickle.load(edge_data_file)

    G = nx.complete_graph(74)
    nx.set_node_attributes(G, node_data)
    nx.set_edge_attributes(G, edge_data, 'distance')
    return G

def get_population_list():
    populationData = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    popDict = populationData.to_dict('list')
    return popDict['total_pop_1000']

def get_cumulative_population_list():
    populationData = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    cumPop = populationData.cumsum(axis=0, skipna=True)
    cumDict = cumPop.to_dict('list')
    return cumDict['total_pop_1000']